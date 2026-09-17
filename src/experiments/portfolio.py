"""Portfolio stage (assignment brief, task 2): 2025 back-test.

"Given the best model for a 90-day output window, implement two portfolios:
one that does not use the predictions and one that does. Compare them over
2025."

* The model is the network with the lowest **validation** MAE among the
  ``H = 90`` cells of the competition.
* It is re-trained on every window whose *target* ends before 2025, so no
  2025 price leaks into its weights (the competition model's training set
  ends years earlier, which would handicap it for no reason).
* ``equal_weight_buy_and_hold`` buys the 23 assets in equal proportions at
  the last 2024 close and never trades again.
* ``model_top_k`` rebalances every ``rebalance_every`` trading days into the
  ``top_k`` assets with the highest positive 90-day forecast (cash if none),
  paying transaction costs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.splits import DataSplit, overlap_gap
from src.data.windowing import (
    create_input_windows,
    create_windows,
    window_dates,
)
from src.evaluation.backtest import (
    decision_dates,
    equal_weights,
    performance_summary,
    simulate_portfolio,
    top_k_long_only_weights,
)
from src.evaluation.metrics import mae
from src.evaluation.selection import select_best
from src.models.registry import get_spec
from src.training.trainer import TrainedNetwork, train_network
from src.utils.config import ProjectConfig

logger = logging.getLogger(__name__)

NEURAL_FAMILIES = ("dense", "recurrent", "convolutional", "mixed")
METRICS_FILE = "portfolio_2025_metrics.csv"
HOLDINGS_FILE = "portfolio_2025_holdings.csv"
CURVES_FILE = "portfolio_2025_equity_curves.csv"


@dataclass(frozen=True)
class PortfolioOutputs:
    """Artefacts of :func:`run_portfolio`.

    Attributes:
        metrics: One row per portfolio.
        equity_curves: Daily wealth of each portfolio (starts at 1).
        holdings: Assets selected at each model rebalance.
        model_name: Network driving the model portfolio.
        input_window: Its look-back ``V``.
        network: The re-trained network.
    """

    metrics: pd.DataFrame
    equity_curves: pd.DataFrame
    holdings: pd.DataFrame
    model_name: str
    input_window: int
    network: TrainedNetwork


def _pre_period_split(
    returns: pd.DataFrame,
    input_window: int,
    output_window: int,
    config: ProjectConfig,
) -> DataSplit:
    """Chronological train/validation split of windows ending pre-period.

    Args:
        returns: Daily log returns.
        input_window: ``V``.
        output_window: ``H``.
        config: Project configuration.

    Returns:
        Split with empty test arrays.

    Raises:
        ValueError: If too few windows precede the evaluation period.
    """
    X, y = create_windows(returns, input_window, output_window)
    dates = window_dates(returns.index, input_window, output_window)
    usable = np.flatnonzero(dates["target_end"] < config.portfolio.start)
    n_val = int(np.ceil(config.data.validation_fraction * len(usable)))
    embargo = (
        overlap_gap(input_window, output_window)
        if config.data.embargo_validation else 0
    )
    n_train = len(usable) - n_val - embargo
    if n_train < 1:
        raise ValueError("Not enough windows before the evaluation period")
    train, val = usable[:n_train], usable[-n_val:]
    empty_X = np.empty((0,) + X.shape[1:])
    empty_y = np.empty((0, y.shape[1]))
    logger.info(
        "Portfolio model trained on %d windows (last target %s), "
        "validated on %d", n_train,
        dates["target_end"].iloc[train[-1]].date(), n_val,
    )
    return DataSplit(X[train], y[train], X[val], y[val], empty_X, empty_y)


def run_portfolio(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    competition_results: pd.DataFrame,
    config: ProjectConfig,
) -> PortfolioOutputs:
    """Build and compare the two 2025 portfolios.

    Args:
        prices: Close prices.
        returns: Daily log returns derived from ``prices``.
        competition_results: Results table of the competition stage.
        config: Project configuration.

    Returns:
        Metrics, equity curves, holdings and the re-trained network.

    Raises:
        ValueError: If the competition has no network for the configured
            horizon.
    """
    pc = config.portfolio
    horizon_rows = competition_results[
        competition_results["output_window"] == pc.output_window
    ]
    if horizon_rows.empty:
        raise ValueError(f"No competition results for H={pc.output_window}")
    overall = horizon_rows.loc[horizon_rows["mae_val"].idxmin()]
    best = select_best(horizon_rows, families=NEURAL_FAMILIES)
    choice = best.loc[best["mae_val"].idxmin()]
    V = int(choice["input_window"])
    spec = get_spec(choice["model"])
    logger.info(
        "Portfolio model: %s V=%d (val MAE %.6f); best overall on "
        "validation at H=%d: %s V=%d (%.6f)", spec.name, V,
        choice["mae_val"], pc.output_window, overall["model"],
        overall["input_window"], overall["mae_val"],
    )

    split = _pre_period_split(returns, V, pc.output_window, config)
    network = train_network(
        spec, split, config.training, config.training.seed,
        config.paths.histories / "portfolio" / f"{spec.name}_V{V}.csv",
    )

    decisions = decision_dates(
        returns.index, pc.start, pc.end, pc.rebalance_every
    )
    inputs = create_input_windows(returns, V)
    input_end = pd.Series(np.arange(len(inputs)), index=returns.index[V - 1:])
    forecasts = network.predict(inputs[input_end.loc[decisions].to_numpy()])

    simple = prices.pct_change().loc[pc.start:pc.end]
    n_assets = simple.shape[1]
    model_weights = {
        date: top_k_long_only_weights(forecast, pc.top_k)
        for date, forecast in zip(decisions, forecasts)
    }
    portfolios = {
        "equal_weight_buy_and_hold": simulate_portfolio(
            simple, {decisions[0]: equal_weights(n_assets)},
            pc.transaction_cost_bps,
        ),
        f"model_top{pc.top_k}_{spec.name}_V{V}": simulate_portfolio(
            simple, model_weights, pc.transaction_cost_bps
        ),
    }

    metrics = pd.DataFrame(
        {
            name: {
                **performance_summary(
                    result.daily_returns, pc.trading_days_per_year
                ),
                "turnover": result.turnover,
                "n_rebalances": len(result.weights),
            }
            for name, result in portfolios.items()
        }
    ).T
    metrics.index.name = "portfolio"

    holdings = pd.DataFrame(
        {
            "decision_date": decisions.date,
            "holdings": [
                ", ".join(prices.columns[w > 0]) or "cash"
                for w in model_weights.values()
            ],
            "top_forecast": forecasts.max(axis=1),
        }
    )
    curves = pd.DataFrame(
        {name: r.equity_curve for name, r in portfolios.items()}
    )

    _log_forecast_accuracy(returns, network, V, config)
    config.paths.tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.paths.tables / METRICS_FILE)
    holdings.to_csv(config.paths.tables / HOLDINGS_FILE, index=False)
    curves.to_csv(config.paths.tables / CURVES_FILE)
    return PortfolioOutputs(metrics, curves, holdings, spec.name, V, network)


def _log_forecast_accuracy(
    returns: pd.DataFrame,
    network: TrainedNetwork,
    input_window: int,
    config: ProjectConfig,
) -> None:
    """Log the re-trained model's MAE on windows starting in the period.

    Args:
        returns: Daily log returns.
        network: Re-trained network.
        input_window: ``V``.
        config: Project configuration.
    """
    pc = config.portfolio
    X, y = create_windows(returns, input_window, pc.output_window)
    dates = window_dates(returns.index, input_window, pc.output_window)
    mask = (dates["target_start"] >= pc.start).to_numpy() & (
        dates["target_start"] <= pc.end
    ).to_numpy()
    if mask.any():
        logger.info(
            "Re-trained model MAE on %d windows starting in %s..%s: %.6f",
            int(mask.sum()), pc.start, pc.end,
            mae(y[mask], network.predict(X[mask])),
        )
