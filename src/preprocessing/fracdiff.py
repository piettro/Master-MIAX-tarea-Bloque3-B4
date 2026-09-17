"""Fixed-width fractional differentiation (Lopez de Prado 2018, ch. 5).

Used by the research stage as an *input* transformation only. The target is
still the mean future log return, so MAE stays in the same units as the
competition and the two are directly comparable -- the approach the
correction lecture endorsed, after warning that transforming the target
makes errors for different ``d`` incomparable.

NOTE: added -- the original ``preprocessing.py`` differenced raw prices
while claiming that ``d = 1`` reproduces log returns; that only holds for
*log* prices, which is what this module differences.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

_MAX_WEIGHTS = 100_000


def frac_diff_weights(d: float, threshold: float) -> np.ndarray:
    """Compute truncated fractional differentiation weights.

    ``w_0 = 1`` and ``w_k = -w_{k-1} (d - k + 1) / k``; the expansion stops
    once ``|w_k|`` falls below ``threshold``.

    Args:
        d: Differentiation order; ``0`` is the identity, ``1`` the first
            difference.
        threshold: Truncation level in ``(0, 1)``.

    Returns:
        Weights in chronological order (the last one, ``1``, multiplies the
        most recent observation).

    Raises:
        ValueError: If ``d`` is negative, ``threshold`` is outside
            ``(0, 1)``, or the expansion does not truncate.
    """
    if d < 0:
        raise ValueError(f"d must be non-negative, got {d}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must lie in (0, 1), got {threshold}")

    weights = [1.0]
    for k in range(1, _MAX_WEIGHTS):
        next_weight = -weights[-1] * (d - k + 1) / k
        if abs(next_weight) < threshold:
            return np.array(weights[::-1], dtype=float)
        weights.append(next_weight)
    raise ValueError(
        f"Weights for d={d} did not fall below {threshold}; raise threshold"
    )


def frac_diff_ffd(series: pd.Series, d: float, threshold: float) -> pd.Series:
    """Apply fixed-width fractional differentiation to one series.

    Args:
        series: Input series, typically a log price.
        d: Differentiation order.
        threshold: Weight truncation level.

    Returns:
        Differenced series on the input index; the first ``width - 1``
        values are ``NaN``.

    Raises:
        ValueError: If the series has non-finite values or is shorter than
            the weight window.
    """
    values = series.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Series contains non-finite values")
    weights = frac_diff_weights(d, threshold)
    width = len(weights)
    if len(values) < width:
        raise ValueError(
            f"Series of length {len(values)} is shorter than the {width}-tap "
            f"window required by d={d}"
        )
    out = np.full(len(values), np.nan)
    out[width - 1:] = np.correlate(values, weights, mode="valid")
    return pd.Series(out, index=series.index, name=series.name)


def frac_diff_frame(
    frame: pd.DataFrame,
    d: float,
    threshold: float,
) -> pd.DataFrame:
    """Apply :func:`frac_diff_ffd` to every column.

    Args:
        frame: Panel of series (e.g. log prices), one column per asset.
        d: Common differentiation order.
        threshold: Weight truncation level.

    Returns:
        Differenced panel with leading ``NaN`` rows kept, so it stays
        aligned with ``frame``.
    """
    return pd.DataFrame(
        {col: frac_diff_ffd(frame[col], d, threshold) for col in frame},
        index=frame.index,
    )


def adf_p_value(series: pd.Series, max_lag: int) -> float:
    """Augmented Dickey-Fuller p-value with a fixed lag order.

    Args:
        series: Series to test; ``NaN`` values are dropped.
        max_lag: Lag order (``autolag=None`` keeps statistics comparable
            across orders).

    Returns:
        The ADF p-value.

    Raises:
        ValueError: If fewer than ``max_lag + 10`` observations remain.
    """
    clean = series.dropna().to_numpy()
    if len(clean) < max_lag + 10:
        raise ValueError(
            f"ADF test needs at least {max_lag + 10} observations, "
            f"got {len(clean)}"
        )
    return float(adfuller(clean, maxlag=max_lag, autolag=None)[1])


def select_common_d(
    log_prices: pd.DataFrame,
    d_grid: tuple[float, ...],
    threshold: float,
    significance: float,
    max_lag: int,
    stationary_share: float,
) -> tuple[float, pd.DataFrame]:
    """Find the smallest common ``d`` that makes most assets stationary.

    Must be called on training-period data only, otherwise the choice of
    ``d`` would peek at validation and test prices.

    Args:
        log_prices: Log prices restricted to the training period.
        d_grid: Candidate orders, tested in increasing order.
        threshold: Weight truncation level.
        significance: ADF significance level.
        max_lag: ADF lag order.
        stationary_share: Required share of assets passing the ADF test.

    Returns:
        ``(d, summary)`` where ``summary`` has one row per tested ``d`` with
        the share of stationary assets and the window length.

    Raises:
        ValueError: If ``d_grid`` is empty or no order meets the share.
    """
    if not d_grid:
        raise ValueError("d_grid must not be empty")

    rows = []
    for d in sorted(d_grid):
        width = len(frac_diff_weights(d, threshold))
        if width + max_lag + 10 > len(log_prices):
            logger.warning("Skipping d=%.2f: %d-tap window too long", d, width)
            continue
        differenced = frac_diff_frame(log_prices, d, threshold)
        p_values = [adf_p_value(differenced[c], max_lag) for c in differenced]
        share = float(np.mean(np.array(p_values) < significance))
        rows.append({"d": d, "window": width, "stationary_share": share})
        logger.info(
            "d=%.2f window=%d stationary share=%.0f%%", d, width, 100 * share
        )
        if share >= stationary_share:
            return d, pd.DataFrame(rows).set_index("d")

    raise ValueError(
        f"No d in {list(d_grid)} makes {stationary_share:.0%} of assets "
        f"stationary at the {significance:.0%} level"
    )
