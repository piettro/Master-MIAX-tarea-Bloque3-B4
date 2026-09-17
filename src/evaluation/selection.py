"""Model selection and competition matrices.

The single most important correction to the original submission: the
winner of each ``(V, H)`` cell is chosen on **validation** MAE and only then
is its **test** MAE reported. The original notebook took ``idxmin`` of
``MAE_test``, which the professor explicitly called cheating ("we must
always decide with validation").
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

CELL_KEYS = ["input_window", "output_window"]
REQUIRED_COLUMNS = {"model", "family", *CELL_KEYS, "mae_val", "mae_test"}


def _check_results(results: pd.DataFrame) -> None:
    """Validate a results frame.

    Args:
        results: Candidate frame.

    Raises:
        ValueError: If it is empty or lacks required columns.
    """
    missing = REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(f"Results lack columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("Results frame is empty")


def select_best(
    results: pd.DataFrame,
    by: str = "mae_val",
    families: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Pick the lowest-``by`` model in each ``(V, H)`` cell.

    Args:
        results: One row per trained model per cell.
        by: Selection column; must be a validation metric for honest
            reporting.
        families: Optional whitelist of families to choose from.

    Returns:
        One row per cell, sorted by ``(V, H)``.

    Raises:
        ValueError: If ``by`` is missing or no row survives the filter.
    """
    _check_results(results)
    if by not in results.columns:
        raise ValueError(f"Unknown selection column {by!r}")
    pool = results
    if families is not None:
        pool = results[results["family"].isin(families)]
    if pool.empty:
        raise ValueError(f"No results for families {families}")
    best = pool.loc[pool.groupby(CELL_KEYS)[by].idxmin()]
    return best.sort_values(CELL_KEYS).reset_index(drop=True)


def competition_matrix(
    best: pd.DataFrame,
    value: str = "mae_test",
) -> pd.DataFrame:
    """Pivot per-cell winners into the ``#inputs x #outputs`` matrix.

    Rows are input windows and columns output windows, the layout agreed
    at the end of the correction lecture.

    Args:
        best: Output of :func:`select_best`.
        value: Column to place in each cell.

    Returns:
        Matrix indexed by ``V`` with ``H`` columns.

    Raises:
        ValueError: If ``value`` is missing.
    """
    if value not in best.columns:
        raise ValueError(f"Unknown value column {value!r}")
    matrix = best.pivot(
        index="input_window", columns="output_window", values=value
    )
    matrix.index.name = "V \\ H"
    matrix.columns.name = None
    return matrix
