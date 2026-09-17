"""Daily portfolio back-testing with drifting weights and trading costs.

Conventions (all fixes over the original notebook):

* portfolio P&L uses **simple** returns ``P_t / P_{t-1} - 1``. The original
  compounded ``cumprod(1 + log_return)``, which mixes return definitions;
* a decision taken with information up to the close of day ``d`` is traded
  at that close and earns returns from day ``d + 1`` onwards, so there is no
  look-ahead;
* between rebalances weights drift with prices -- a genuine buy-and-hold --
  instead of being implicitly reset to equal weights every day;
* uninvested weight is held in cash at zero return.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    """Output of :func:`simulate_portfolio`.

    Attributes:
        daily_returns: Net portfolio simple return per trading day.
        weights: Post-trade weights at each rebalance, one row per decision.
        turnover: Sum of one-way traded notional over all rebalances.
    """

    daily_returns: pd.Series
    weights: pd.DataFrame
    turnover: float

    @property
    def equity_curve(self) -> pd.Series:
        """Cumulative wealth starting from 1."""
        return (1.0 + self.daily_returns).cumprod()


def equal_weights(n_assets: int) -> np.ndarray:
    """Equal allocation across all assets.

    Args:
        n_assets: Number of assets.

    Returns:
        Array of ``1 / n_assets``.

    Raises:
        ValueError: If ``n_assets`` is not positive.
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be positive, got {n_assets}")
    return np.full(n_assets, 1.0 / n_assets)


def top_k_long_only_weights(predictions: np.ndarray, k: int) -> np.ndarray:
    """Equal-weight the ``k`` assets with the highest positive forecast.

    Assets with a non-positive forecast are never bought; if none is
    positive the portfolio moves to cash, which is how the forecast is
    allowed to express a bearish view.

    Args:
        predictions: Forecast mean log return per asset.
        k: Maximum number of holdings.

    Returns:
        Weights summing to 1 (or 0 when fully in cash).

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    predictions = np.asarray(predictions, dtype=float)
    ranked = np.argsort(predictions)[::-1][:k]
    chosen = ranked[predictions[ranked] > 0]
    weights = np.zeros_like(predictions)
    if len(chosen):
        weights[chosen] = 1.0 / len(chosen)
    return weights


def decision_dates(
    trading_dates: pd.DatetimeIndex,
    start: str,
    end: str,
    every: int,
) -> pd.DatetimeIndex:
    """Rebalance dates for an evaluation window.

    The first decision is the last close strictly before ``start`` (so the
    portfolio is invested on the first trading day of the period); later
    decisions follow every ``every`` trading days, excluding the final day
    of the period, whose trade could not earn any return.

    Args:
        trading_dates: All available trading dates.
        start: First calendar day of the evaluation period.
        end: Last calendar day of the evaluation period.
        every: Trading days between decisions.

    Returns:
        Sorted decision dates.

    Raises:
        ValueError: If ``every`` is not positive or the period has no
            trading day or no prior close.
    """
    if every < 1:
        raise ValueError(f"every must be positive, got {every}")
    dates = pd.DatetimeIndex(trading_dates).sort_values()
    period = dates[(dates >= start) & (dates <= end)]
    prior = dates[dates < start]
    if period.empty or prior.empty:
        raise ValueError(f"No trading days around period {start}..{end}")
    later = period[every - 1:-1:every]
    return pd.DatetimeIndex([prior[-1], *later])


def simulate_portfolio(
    simple_returns: pd.DataFrame,
    target_weights: Mapping[pd.Timestamp, np.ndarray],
    transaction_cost_bps: float = 0.0,
) -> BacktestResult:
    """Run a daily back-test.

    Args:
        simple_returns: Asset simple returns for the evaluation days only.
        target_weights: Post-trade weights keyed by decision date. A
            decision dated ``d`` applies to returns strictly after ``d``.
        transaction_cost_bps: One-way cost in basis points of traded
            notional, deducted from the first return after the trade.

    Returns:
        Daily net returns, the weight history and total turnover.

    Raises:
        ValueError: If inputs are empty, misaligned, contain missing
            values, or the first decision does not precede the first day.
    """
    if simple_returns.empty or not target_weights:
        raise ValueError("Returns and target weights must not be empty")
    if simple_returns.isna().any().any():
        raise ValueError("Simple returns contain missing values")
    n_assets = simple_returns.shape[1]
    schedule = sorted(target_weights.items())
    for date, weights in schedule:
        if len(weights) != n_assets or np.any(np.asarray(weights) < 0):
            raise ValueError(f"Invalid weights for decision {date}")
        if np.sum(weights) > 1 + 1e-9:
            raise ValueError(f"Weights for decision {date} exceed 100%")
    if schedule[0][0] >= simple_returns.index[0]:
        raise ValueError("The first decision must precede the first day")

    cost_rate = transaction_cost_bps / 10_000.0
    current = np.zeros(n_assets)
    pending = 0
    turnover = 0.0
    net_returns = []
    applied = []

    for date, row in zip(simple_returns.index, simple_returns.to_numpy()):
        cost = 0.0
        while pending < len(schedule) and schedule[pending][0] < date:
            decided_on, target = schedule[pending]
            traded = float(np.abs(np.asarray(target) - current).sum())
            turnover += traded
            cost += traded * cost_rate
            current = np.asarray(target, dtype=float).copy()
            applied.append(pd.Series(current, name=decided_on))
            pending += 1
        gross = float(current @ row)
        net_returns.append(gross - cost)
        # Drift: invested value grows with prices, cash stays flat.
        growth = 1.0 + gross
        if growth > 0:
            current = current * (1.0 + row) / growth

    if pending < len(schedule):
        logger.warning(
            "%d decision(s) after the last trading day were ignored",
            len(schedule) - pending,
        )

    weights = pd.DataFrame(applied)
    weights.columns = simple_returns.columns
    return BacktestResult(
        daily_returns=pd.Series(
            net_returns, index=simple_returns.index, name="return"
        ),
        weights=weights,
        turnover=turnover,
    )


def performance_summary(
    daily_returns: pd.Series,
    trading_days_per_year: int,
) -> dict[str, float]:
    """Standard risk/return statistics of a daily return series.

    Args:
        daily_returns: Net simple returns.
        trading_days_per_year: Annualisation constant.

    Returns:
        ``total_return``, ``annualised_return``, ``annualised_volatility``,
        ``sharpe_ratio`` (zero risk-free rate) and ``max_drawdown``
        (negative number).

    Raises:
        ValueError: If the series is empty.
    """
    if daily_returns.empty:
        raise ValueError("daily_returns must not be empty")
    wealth = (1.0 + daily_returns).cumprod()
    total = float(wealth.iloc[-1] - 1.0)
    years = len(daily_returns) / trading_days_per_year
    volatility = float(daily_returns.std(ddof=1) * np.sqrt(
        trading_days_per_year
    ))
    mean_annual = float(daily_returns.mean() * trading_days_per_year)
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "total_return": total,
        "annualised_return": float((1.0 + total) ** (1.0 / years) - 1.0),
        "annualised_volatility": volatility,
        "sharpe_ratio": mean_annual / volatility if volatility > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
    }
