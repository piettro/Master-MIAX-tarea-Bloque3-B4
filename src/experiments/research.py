"""Research stage (assignment brief, task 2): pre-processing ablation.

Two lessons from the correction lecture shape this stage:

* **change one thing at a time.** A group that combined fractional
  differentiation with a new split could not tell which change helped.
  Each variant below differs from the competition setting in exactly one
  aspect, and uses the competition's validation-selected network per cell;
* **measure every variant in the same units.** Fractional differentiation
  transforms the *inputs* only; the target stays the mean future log
  return, so validation and test MAE are directly comparable with the
  competition (the approach the professor accepted). The test set is the
  competition test set in every variant.

Variants:

``ffd_inputs``
    Network inputs are fractionally differentiated log prices with the
    smallest common ``d`` that makes most assets stationary on training
    data (Lopez de Prado, ch. 5), instead of log returns.
``embargo``
    Training windows sharing raw days with the first validation window
    are dropped (Lopez de Prado, ch. 7), so early stopping and model
    selection are not flattered by overlapping samples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.data.splits import (
    DataSplit,
    apply_split,
    chronological_split,
    overlap_gap,
)
from src.data.windowing import create_windows, window_dates
from src.evaluation.metrics import sample_absolute_errors
from src.evaluation.selection import select_best
from src.evaluation.significance import diebold_mariano
from src.experiments.competition import load_test_predictions
from src.experiments.runner import run_network
from src.models.registry import get_spec
from src.preprocessing.fracdiff import frac_diff_frame, select_common_d
from src.utils.config import ProjectConfig

logger = logging.getLogger(__name__)

RESULTS_FILE = "research_results.csv"
D_SELECTION_FILE = "research_fracdiff_d_selection.csv"
NEURAL_FAMILIES = ("dense", "recurrent", "convolutional", "mixed")


@dataclass(frozen=True)
class ResearchOutputs:
    """Tables produced by :func:`run_research`.

    Attributes:
        results: One row per cell per variant (reference included).
        d: Fractional differentiation order used by ``ffd_inputs``.
        d_selection: ADF sweep behind the choice of ``d``.
    """

    results: pd.DataFrame
    d: float
    d_selection: pd.DataFrame


def training_cutoff(
    returns: pd.DataFrame,
    config: ProjectConfig,
) -> pd.Timestamp:
    """Earliest validation start date over all cells.

    Choosing ``d`` with data strictly before this date guarantees it never
    sees a validation or test price in any cell.

    Args:
        returns: Daily log returns.
        config: Project configuration.

    Returns:
        The cut-off date.
    """
    starts = []
    for V in config.data.input_windows:
        for H in config.data.output_windows:
            dates = window_dates(returns.index, V, H)
            indices = chronological_split(
                len(dates), config.data.test_fraction,
                config.data.validation_fraction,
            )
            starts.append(dates["input_start"].iloc[indices.validation[0]])
    return min(starts)


def _drop_incomplete_training(split: DataSplit) -> DataSplit:
    """Remove training windows whose inputs contain ``NaN``.

    Fractional differentiation leaves the first ``window - 1`` days
    undefined; those days only affect the oldest training windows.

    Args:
        split: Split whose training inputs may hold ``NaN``.

    Returns:
        Split with complete training windows only.

    Raises:
        ValueError: If validation or test inputs contain ``NaN``.
    """
    if np.isnan(split.X_val).any() or np.isnan(split.X_test).any():
        raise ValueError(
            "Validation or test windows fall inside the fractional "
            "differentiation warm-up; lower the weight threshold"
        )
    keep = ~np.isnan(split.X_train).any(axis=(1, 2))
    logger.info("Dropped %d warm-up training windows", int((~keep).sum()))
    return replace(split, X_train=split.X_train[keep],
                   y_train=split.y_train[keep])


def run_research(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    competition_results: pd.DataFrame,
    config: ProjectConfig,
) -> ResearchOutputs:
    """Evaluate each single-change variant on every cell.

    Args:
        prices: Close prices aligned with ``returns`` (one extra leading
            day).
        returns: Daily log returns.
        competition_results: Results table of the competition stage.
        config: Project configuration.

    Returns:
        Research results and the fractional differentiation diagnostics.
    """
    rc = config.research
    cutoff = training_cutoff(returns, config)
    log_prices = np.log(prices)
    d, d_selection = select_common_d(
        log_prices.loc[log_prices.index < cutoff],
        rc.d_grid, rc.ffd_threshold, rc.adf_significance, rc.adf_max_lag,
        rc.stationary_share,
    )
    logger.info("Fractional differentiation order d=%.2f (cutoff %s)",
                d, cutoff.date())
    ffd_inputs = frac_diff_frame(log_prices, d, rc.ffd_threshold).reindex(
        returns.index
    )

    winners = select_best(
        competition_results, by="mae_val", families=NEURAL_FAMILIES
    )
    rows = []
    for record in winners.to_dict("records"):
        V, H = int(record["input_window"]), int(record["output_window"])
        spec = get_spec(record["model"])
        stored = load_test_predictions(config.paths.outputs, V, H)
        reference_losses = sample_absolute_errors(
            stored["y_test"], stored[spec.name]
        )
        rows.append({
            "input_window": V, "output_window": H, "model": spec.name,
            "variant": "log_returns (competition)",
            "mae_val": record["mae_val"], "mae_test": record["mae_test"],
            "dm_statistic": 0.0, "dm_p_value": 1.0,
        })

        _, y = create_windows(returns, V, H)
        X_ffd, _ = create_windows(ffd_inputs, V, H)
        X_raw, _ = create_windows(returns, V, H)
        plain = chronological_split(
            len(y), config.data.test_fraction,
            config.data.validation_fraction,
        )
        embargoed = chronological_split(
            len(y), config.data.test_fraction,
            config.data.validation_fraction, overlap_gap(V, H),
        )
        variants = {
            "ffd_inputs": _drop_incomplete_training(
                apply_split(X_ffd, y, plain)
            ),
            "embargo": apply_split(X_raw, y, embargoed),
        }
        for name, split in variants.items():
            logger.info("Research V=%d H=%d %s variant %s", V, H,
                        spec.name, name)
            outcome = run_network(
                spec, split, config.training, V, H, config.training.seed
            )
            test = diebold_mariano(outcome.test_losses, reference_losses, H)
            rows.append({
                "input_window": V, "output_window": H, "model": spec.name,
                "variant": name, "mae_val": outcome.row["mae_val"],
                "mae_test": outcome.row["mae_test"],
                "dm_statistic": test.statistic, "dm_p_value": test.p_value,
            })

    results = pd.DataFrame(rows)
    reference = results.groupby(["input_window", "output_window"])[
        "mae_test"
    ].transform("first")
    results["test_change_pct"] = 100 * (results["mae_test"] / reference - 1)
    config.paths.tables.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.paths.tables / RESULTS_FILE, index=False)
    d_selection.to_csv(config.paths.tables / D_SELECTION_FILE)
    return ResearchOutputs(results, d, d_selection)
