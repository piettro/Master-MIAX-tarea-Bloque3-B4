"""Competition stage (assignment brief, task 1).

For every ``(V, H)`` combination, on exactly the data and chronological
90/10 split of the professor's starter notebook:

1. fit the baseline suite and train every network in the catalogue;
2. report train, validation and test MAE plus the parameter count;
3. select the cell winner on validation MAE and report its test MAE;
4. test every model against the best baseline with Diebold-Mariano;
5. re-train the winning network with extra seeds to show how much of its
   edge survives initialisation noise.

Results are appended cell by cell, so an interrupted run resumes where it
stopped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.data.splits import apply_split, chronological_split, overlap_gap
from src.data.windowing import create_windows
from src.evaluation.selection import CELL_KEYS, select_best
from src.evaluation.significance import diebold_mariano
from src.experiments.runner import ModelOutcome, run_baseline, run_network
from src.models.baselines import Baseline, default_baselines
from src.models.registry import DEFAULT_SPECS, ModelSpec, get_spec
from src.utils.config import ProjectConfig

logger = logging.getLogger(__name__)

RESULTS_FILE = "competition_results.csv"
SEEDS_FILE = "competition_seed_robustness.csv"


@dataclass(frozen=True)
class CompetitionOutputs:
    """Tables produced by :func:`run_competition`.

    Attributes:
        results: One row per model per cell.
        best: Validation-selected winner of each cell.
        seed_robustness: Test MAE across seeds of each winning network.
    """

    results: pd.DataFrame
    best: pd.DataFrame
    seed_robustness: pd.DataFrame


def cell_name(input_window: int, output_window: int) -> str:
    """Canonical directory/file stem of a cell.

    Args:
        input_window: ``V``.
        output_window: ``H``.

    Returns:
        ``"V{V}_H{H}"``.
    """
    return f"V{input_window}_H{output_window}"


def history_path(
    root: Path, input_window: int, output_window: int, model: str
) -> Path:
    """Location of a competition loss history.

    Args:
        root: Histories root directory.
        input_window: ``V``.
        output_window: ``H``.
        model: Model name.

    Returns:
        CSV path.
    """
    return root / "competition" / cell_name(input_window, output_window) / (
        f"{model}.csv"
    )


def _add_significance(
    outcomes: Sequence[ModelOutcome],
    output_window: int,
) -> None:
    """Annotate rows with a DM test against the best validation baseline.

    Args:
        outcomes: Outcomes of one cell, updated in place.
        output_window: ``H``, which sets the HAC lag window.
    """
    baselines = [o for o in outcomes if o.row["family"] == "baseline"]
    reference = min(baselines, key=lambda o: o.row["mae_val"])
    for outcome in outcomes:
        result = diebold_mariano(
            outcome.test_losses, reference.test_losses, output_window
        )
        outcome.row["best_baseline"] = reference.row["model"]
        outcome.row["dm_statistic"] = result.statistic
        outcome.row["dm_p_value"] = result.p_value


def _seed_robustness(
    winner: ModelOutcome,
    split,
    config: ProjectConfig,
    specs: Sequence[ModelSpec],
) -> dict:
    """Re-train a winning network with additional seeds.

    Args:
        winner: Outcome of the validation-selected network.
        split: The cell's data split.
        config: Project configuration.
        specs: Catalogue containing the winner's specification.

    Returns:
        Record with the mean and standard deviation of validation and test
        MAE over all seeds (the original one included).
    """
    row = winner.row
    val_scores = [row["mae_val"]]
    test_scores = [row["mae_test"]]
    spec = get_spec(row["model"], tuple(specs))
    for offset in range(1, config.training.n_seeds_final):
        extra = run_network(
            spec, split, config.training, row["input_window"],
            row["output_window"], config.training.seed + offset,
        )
        val_scores.append(extra.row["mae_val"])
        test_scores.append(extra.row["mae_test"])
    return {
        "input_window": row["input_window"],
        "output_window": row["output_window"],
        "model": row["model"],
        "n_seeds": len(test_scores),
        "mae_val_mean": float(np.mean(val_scores)),
        "mae_val_std": float(np.std(val_scores)),
        "mae_test_mean": float(np.mean(test_scores)),
        "mae_test_std": float(np.std(test_scores)),
    }


def _load_previous(path: Path) -> pd.DataFrame:
    """Read a partial results table if it exists.

    Args:
        path: CSV path.

    Returns:
        The table, or an empty frame.

    Raises:
        ValueError: If the file exists but cannot be parsed.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read partial results {path}: {exc}") from exc


def run_competition(
    returns: pd.DataFrame,
    config: ProjectConfig,
    specs: Sequence[ModelSpec] = DEFAULT_SPECS,
    baseline_factory: Callable[[], list[Baseline]] = default_baselines,
    resume: bool = True,
) -> CompetitionOutputs:
    """Run the competition over every configured window combination.

    Args:
        returns: Daily log returns, one column per asset.
        config: Project configuration.
        specs: Neural network catalogue.
        baseline_factory: Creates fresh baselines for each cell.
        resume: Skip cells already present in the saved results.

    Returns:
        Results, per-cell winners and seed robustness tables.

    Raises:
        ValueError: If ``specs`` is empty.
    """
    if not specs:
        raise ValueError("At least one model specification is required")

    paths = config.paths
    paths.tables.mkdir(parents=True, exist_ok=True)
    results_path = paths.tables / RESULTS_FILE
    seeds_path = paths.tables / SEEDS_FILE

    previous = _load_previous(results_path) if resume else pd.DataFrame()
    previous_seeds = _load_previous(seeds_path) if resume else pd.DataFrame()
    done = (
        set(map(tuple, previous[CELL_KEYS].drop_duplicates().to_numpy()))
        if not previous.empty else set()
    )
    rows = previous.to_dict("records")
    seed_rows = previous_seeds.to_dict("records")

    for V in config.data.input_windows:
        for H in config.data.output_windows:
            if (V, H) in done:
                logger.info("Skipping completed cell V=%d H=%d", V, H)
                continue
            logger.info("=== Competition cell V=%d H=%d ===", V, H)
            X, y = create_windows(returns, V, H)
            embargo = (
                overlap_gap(V, H) if config.data.embargo_validation else 0
            )
            indices = chronological_split(
                len(X), config.data.test_fraction,
                config.data.validation_fraction, embargo,
            )
            split = apply_split(X, y, indices)

            outcomes = [
                run_baseline(b, split, V, H) for b in baseline_factory()
            ]
            for spec in specs:
                outcomes.append(
                    run_network(
                        spec, split, config.training, V, H,
                        config.training.seed,
                        history_path(paths.histories, V, H, spec.name),
                    )
                )
            _add_significance(outcomes, H)
            _save_predictions(paths.outputs, V, H, split.y_test, outcomes)

            networks = [o for o in outcomes if o.network is not None]
            winner = min(networks, key=lambda o: o.row["mae_val"])
            seed_rows.append(
                _seed_robustness(winner, split, config, specs)
            )

            rows.extend(o.row for o in outcomes)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            pd.DataFrame(seed_rows).to_csv(seeds_path, index=False)

    results = pd.DataFrame(rows)
    results["params_to_samples"] = (
        results["n_params"] / results["n_train_samples"]
    )
    results["exceeds_param_rule"] = (
        results["params_to_samples"] > config.training.max_params_to_samples
    )
    results.to_csv(results_path, index=False)
    best = select_best(results, by="mae_val")
    return CompetitionOutputs(results, best, pd.DataFrame(seed_rows))


def _save_predictions(
    outputs_root: Path,
    input_window: int,
    output_window: int,
    y_test: np.ndarray,
    outcomes: Sequence[ModelOutcome],
) -> None:
    """Persist test targets and every model's test forecasts.

    The research stage reuses them to test its variants against the
    competition model without re-training it.

    Args:
        outputs_root: Root output directory.
        input_window: ``V``.
        output_window: ``H``.
        y_test: Test targets.
        outcomes: Outcomes of the cell.
    """
    directory = outputs_root / "predictions" / "competition"
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"y_test": y_test.astype(np.float32)}
    arrays.update(
        {o.row["model"]: o.test_predictions.astype(np.float32)
         for o in outcomes}
    )
    np.savez_compressed(
        directory / f"{cell_name(input_window, output_window)}.npz", **arrays
    )


def load_test_predictions(
    outputs_root: Path,
    input_window: int,
    output_window: int,
) -> dict[str, np.ndarray]:
    """Load the arrays saved by the competition for one cell.

    Args:
        outputs_root: Root output directory.
        input_window: ``V``.
        output_window: ``H``.

    Returns:
        Mapping with ``y_test`` and one entry per model.

    Raises:
        FileNotFoundError: If the competition has not produced the cell.
    """
    path = (
        outputs_root / "predictions" / "competition"
        / f"{cell_name(input_window, output_window)}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run the competition stage first"
        )
    with np.load(path) as data:
        return {key: data[key].astype(float) for key in data.files}
