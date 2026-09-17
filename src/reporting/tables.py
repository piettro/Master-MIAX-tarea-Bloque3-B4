"""Result tables in CSV, Markdown and plain text.

The brief asks for one table per window combination (train and test MAE,
number of parameters, best model highlighted) and for the competition
matrix. The professor additionally asked every team to e-mail the matrix
as copy-pasteable text, which :func:`write_competition_matrix` produces.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.evaluation.selection import competition_matrix

logger = logging.getLogger(__name__)

CELL_COLUMNS = [
    "model", "family", "mae_train", "mae_val", "mae_test",
    "directional_accuracy_test", "n_params", "params_to_samples",
    "epochs_run", "best_epoch", "dm_p_value",
]


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories.

    Args:
        path: Destination.
        text: Content.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Cannot write {path}: {exc}") from exc


def to_markdown(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    """Render a frame as a GitHub Markdown table.

    Args:
        frame: Table to render.
        floatfmt: Float format passed to ``tabulate``.

    Returns:
        Markdown string.
    """
    return frame.to_markdown(index=False, floatfmt=floatfmt)


def write_cell_tables(results: pd.DataFrame, directory: Path) -> Path:
    """Write one table per ``(V, H)`` combination.

    The validation-selected model is marked with ``**`` in Markdown; the
    lowest test MAE is also flagged so the reader can see when validation
    and test disagree.

    Args:
        results: Full competition results.
        directory: Output directory.

    Returns:
        Path of the combined Markdown file.
    """
    sections = ["# Competition results per window combination\n",
                "`*` selected on validation MAE (reported result); "
                "`(min test)` lowest test MAE, shown for information only. "
                "`dm_p_value`: Diebold-Mariano test against the best "
                "validation baseline.\n"]
    cell_directory = directory / "cells"
    cell_directory.mkdir(parents=True, exist_ok=True)
    cells = results.groupby(["input_window", "output_window"], sort=True)
    for (V, H), cell in cells:
        table = cell[CELL_COLUMNS].sort_values("mae_val").reset_index(
            drop=True
        )
        best_val = table["mae_val"].idxmin()
        best_test = table["mae_test"].idxmin()
        table.to_csv(cell_directory / f"V{V}_H{H}.csv", index=False)
        display = table.copy()
        display["model"] = [
            ("* " if i == best_val else "") + model
            + (" (min test)" if i == best_test and i != best_val else "")
            for i, model in enumerate(display["model"])
        ]
        sections.append(f"\n## V={V}, H={H}\n\n{to_markdown(display)}\n")
    path = directory / "competition_cells.md"
    _write_text(path, "".join(sections))
    return path


def _labelled_matrix(best: pd.DataFrame) -> pd.DataFrame:
    """Matrix whose cells read ``"<test MAE> (<model>)"``.

    Args:
        best: Winner per cell.

    Returns:
        String matrix with ``V`` as a column.
    """
    mae_matrix = competition_matrix(best, "mae_test")
    model_matrix = competition_matrix(best, "model")
    combined = mae_matrix.map(lambda v: f"{v:.6f}") + " (" \
        + model_matrix.astype(str) + ")"
    return combined.reset_index()


def write_competition_matrix(
    best: pd.DataFrame,
    seed_robustness: pd.DataFrame,
    directory: Path,
    best_networks: pd.DataFrame | None = None,
) -> str:
    """Write the ``#inputs x #outputs`` competition matrix.

    Args:
        best: Validation-selected winner per cell, baselines included.
        seed_robustness: Seed dispersion of each cell's best network.
        directory: Output directory.
        best_networks: Optional validation-selected best *network* per
            cell, written as a second matrix.

    Returns:
        The plain-text matrix (tab separated), ready to paste in an e-mail.
    """
    directory.mkdir(parents=True, exist_ok=True)
    mae_matrix = competition_matrix(best, "mae_test")
    model_matrix = competition_matrix(best, "model")
    mae_matrix.to_csv(directory / "competition_matrix_test_mae.csv")
    model_matrix.to_csv(directory / "competition_matrix_models.csv")

    text = mae_matrix.to_csv(sep="\t", float_format="%.6f")
    _write_text(directory / "competition_matrix.txt", text)

    detail = best[[
        "input_window", "output_window", "model", "family", "mae_val",
        "mae_test", "n_params", "best_baseline", "dm_p_value",
    ]]
    if not seed_robustness.empty:
        # Seeds are only re-run for the best *network* of each cell, so a
        # baseline winner correctly receives empty seed columns.
        detail = detail.merge(
            seed_robustness[[
                "input_window", "output_window", "model", "n_seeds",
                "mae_test_mean", "mae_test_std",
            ]],
            on=["input_window", "output_window", "model"], how="left",
        )
    detail.to_csv(directory / "competition_best_models.csv", index=False)
    markdown = (
        "# Competition matrix\n\nTest MAE of the model selected on "
        "validation among all models, baselines included (rows: input "
        "window V, columns: output window H).\n\n"
        + _labelled_matrix(best).to_markdown(index=False)
        + "\n\n## Winners\n\n" + to_markdown(detail) + "\n"
    )
    if best_networks is not None:
        networks = best_networks[[
            "input_window", "output_window", "model", "family", "mae_val",
            "mae_test", "n_params", "best_baseline", "dm_p_value",
        ]].merge(
            seed_robustness[[
                "input_window", "output_window", "model", "n_seeds",
                "mae_test_mean", "mae_test_std",
            ]],
            on=["input_window", "output_window", "model"], how="left",
        ) if not seed_robustness.empty else best_networks
        networks.to_csv(directory / "competition_best_networks.csv",
                        index=False)
        markdown += (
            "\n## Best neural network per cell\n\nSame selection rule "
            "restricted to the four network families; `dm_p_value` tests "
            "it against the cell's best validation baseline and the seed "
            "columns show its test MAE over independent re-trainings.\n\n"
            + _labelled_matrix(best_networks).to_markdown(index=False)
            + "\n\n" + to_markdown(networks) + "\n"
        )
    _write_text(directory / "competition_matrix.md", markdown)
    return text


def write_frame(
    frame: pd.DataFrame, path_stem: Path, index: bool = False
) -> None:
    """Write a frame as CSV and Markdown side by side.

    Args:
        frame: Table to write.
        path_stem: Destination without extension.
        index: Include the index as a column.
    """
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path_stem.with_suffix(".csv"), index=index)
    shown = frame.reset_index() if index else frame
    _write_text(path_stem.with_suffix(".md"), to_markdown(shown) + "\n")
