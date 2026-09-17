"""Diebold-Mariano test for equal predictive accuracy.

NOTE: added -- in the correction lecture the professor asked for hypothesis
tests before claiming that one model beats another ("if you run a test it
is no longer your opinion"), since most MAE gaps between models were in the
fifth decimal.

Targets are means over ``H`` overlapping days, so loss differentials are
autocorrelated up to lag ``H - 1``. The long-run variance is therefore
estimated with Newey-West (Bartlett) weights, which are guaranteed
non-negative, and the Harvey-Leybourne-Newbold small-sample correction is
applied with a Student-t reference distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Outcome of a two-sided Diebold-Mariano test.

    Attributes:
        statistic: HLN-corrected DM statistic; negative means model A has
            the lower loss.
        p_value: Two-sided p-value.
        mean_loss_difference: Mean of ``loss_a - loss_b``.
    """

    statistic: float
    p_value: float
    mean_loss_difference: float


def newey_west_variance(series: np.ndarray, max_lag: int) -> float:
    """Long-run variance of the sample mean's numerator.

    Args:
        series: Demeaned-internally 1-D series.
        max_lag: Number of autocovariance lags with Bartlett weights.

    Returns:
        ``gamma_0 + 2 * sum_k w_k * gamma_k``.

    Raises:
        ValueError: If ``max_lag`` is negative or not smaller than the
            series length.
    """
    n = len(series)
    if not 0 <= max_lag < n:
        raise ValueError(f"max_lag must lie in [0, {n}), got {max_lag}")
    centred = series - series.mean()
    variance = float(centred @ centred) / n
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        autocov = float(centred[lag:] @ centred[:-lag]) / n
        variance += 2.0 * weight * autocov
    return variance


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int,
) -> DieboldMarianoResult:
    """Test ``H0: E[loss_a] = E[loss_b]``.

    Args:
        loss_a: Per-sample losses of model A.
        loss_b: Per-sample losses of model B, aligned with ``loss_a``.
        horizon: Forecast horizon ``H``; sets the lag window to ``H - 1``.

    Returns:
        The test result. When both loss series are identical the statistic
        is 0 and the p-value 1.

    Raises:
        ValueError: If the series lengths differ, ``horizon`` is not
            positive, or there are too few samples.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape or loss_a.ndim != 1:
        raise ValueError("Loss series must be 1-D and of equal length")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    n = len(loss_a)
    if n <= 2 * horizon:
        raise ValueError(f"Need more than {2 * horizon} samples, got {n}")

    diff = loss_a - loss_b
    mean_diff = float(diff.mean())
    variance = newey_west_variance(diff, horizon - 1)
    if variance <= 0 or not np.isfinite(variance):
        return DieboldMarianoResult(0.0, 1.0, mean_diff)

    dm = mean_diff / np.sqrt(variance / n)
    correction = np.sqrt(
        (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    )
    statistic = float(dm * correction)
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    return DieboldMarianoResult(statistic, p_value, mean_diff)
