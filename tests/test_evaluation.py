"""Tests for metrics, significance, selection and back-testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import (
    decision_dates,
    equal_weights,
    performance_summary,
    simulate_portfolio,
    top_k_long_only_weights,
)
from src.evaluation.metrics import (
    directional_accuracy,
    mae,
    sample_absolute_errors,
)
from src.evaluation.selection import competition_matrix, select_best
from src.evaluation.significance import diebold_mariano, newey_west_variance


def test_mae_and_sample_errors_agree():
    y = np.array([[0.0, 1.0], [2.0, 3.0]])
    p = np.array([[1.0, 1.0], [2.0, 5.0]])
    assert mae(y, p) == pytest.approx(0.75)
    np.testing.assert_allclose(sample_absolute_errors(y, p), [0.5, 1.0])


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        mae(np.zeros((2, 2)), np.zeros((2, 3)))


def test_directional_accuracy_ignores_zero_targets():
    y = np.array([[1.0, -1.0, 0.0]])
    p = np.array([[2.0, 1.0, 5.0]])
    assert directional_accuracy(y, p) == pytest.approx(0.5)


def test_newey_west_lag_zero_is_variance():
    x = np.random.default_rng(0).normal(size=500)
    assert newey_west_variance(x, 0) == pytest.approx(np.var(x))


def test_dm_identical_losses_not_significant():
    loss = np.abs(np.random.default_rng(0).normal(size=300))
    result = diebold_mariano(loss, loss.copy(), horizon=5)
    assert result.p_value == 1.0
    assert result.statistic == 0.0


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    base = np.abs(rng.normal(size=400))
    better = base * 0.5
    result = diebold_mariano(better, base, horizon=1)
    assert result.statistic < 0
    assert result.p_value < 0.01


def test_dm_validates_inputs():
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(10), np.ones(11), 1)
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(10), np.ones(10), 30)


def _results_frame():
    return pd.DataFrame({
        "model": ["a", "b", "a", "b"],
        "family": ["dense", "baseline", "dense", "baseline"],
        "input_window": [5, 5, 10, 10],
        "output_window": [1, 1, 1, 1],
        "mae_val": [0.10, 0.20, 0.30, 0.25],
        "mae_test": [0.50, 0.40, 0.30, 0.35],
    })


def test_selection_uses_validation_not_test():
    best = select_best(_results_frame())
    # Cell V=5: "a" wins on validation even though "b" has lower test MAE.
    assert best.loc[best["input_window"] == 5, "model"].item() == "a"
    assert best.loc[best["input_window"] == 10, "model"].item() == "b"


def test_selection_family_filter():
    best = select_best(_results_frame(), families=["dense"])
    assert set(best["model"]) == {"a"}


def test_competition_matrix_layout():
    matrix = competition_matrix(select_best(_results_frame()))
    assert list(matrix.index) == [5, 10]
    assert list(matrix.columns) == [1]
    assert matrix.loc[5, 1] == pytest.approx(0.50)


def _simple_returns():
    index = pd.bdate_range("2025-01-01", periods=4)
    return pd.DataFrame(
        {"A": [0.10, 0.00, -0.05, 0.02], "B": [0.00, 0.10, 0.00, 0.00]},
        index=index,
    )


def test_buy_and_hold_matches_price_relatives():
    returns = _simple_returns()
    start = pd.Timestamp("2024-12-31")
    result = simulate_portfolio(returns, {start: equal_weights(2)})
    growth = (1 + returns).prod()
    assert result.equity_curve.iloc[-1] == pytest.approx(growth.mean())


def test_decision_applies_from_next_day_only():
    returns = _simple_returns()
    first_day = returns.index[0]
    weights = {
        pd.Timestamp("2024-12-31"): np.array([0.0, 1.0]),
        first_day: np.array([1.0, 0.0]),
    }
    result = simulate_portfolio(returns, weights)
    # Day 1 still earns B's return; A's +10% on day 1 must not be captured.
    assert result.daily_returns.iloc[0] == pytest.approx(0.0)
    assert result.daily_returns.iloc[1] == pytest.approx(0.0)
    assert result.daily_returns.iloc[2] == pytest.approx(-0.05)


def test_transaction_costs_reduce_first_return():
    returns = _simple_returns()
    start = pd.Timestamp("2024-12-31")
    free = simulate_portfolio(returns, {start: equal_weights(2)})
    costly = simulate_portfolio(returns, {start: equal_weights(2)}, 100)
    assert free.daily_returns.iloc[0] - costly.daily_returns.iloc[0] == \
        pytest.approx(0.01)


def test_simulation_rejects_late_first_decision():
    returns = _simple_returns()
    with pytest.raises(ValueError):
        simulate_portfolio(returns, {returns.index[0]: equal_weights(2)})


def test_top_k_long_only_weights():
    weights = top_k_long_only_weights(np.array([0.3, -0.1, 0.2, 0.1]), 2)
    np.testing.assert_allclose(weights, [0.5, 0, 0.5, 0])
    cash = top_k_long_only_weights(np.array([-0.3, -0.1]), 2)
    assert cash.sum() == 0


def test_decision_dates_start_before_period():
    dates = pd.bdate_range("2024-12-20", "2025-02-28")
    decisions = decision_dates(dates, "2025-01-01", "2025-12-31", 10)
    assert decisions[0] == pd.Timestamp("2024-12-31")
    assert (decisions[1:] >= pd.Timestamp("2025-01-01")).all()
    assert decisions[-1] < dates[-1]


def test_performance_summary_values():
    returns = pd.Series([0.01, -0.02, 0.03])
    summary = performance_summary(returns, 252)
    assert summary["total_return"] == pytest.approx(1.01 * 0.98 * 1.03 - 1)
    assert summary["max_drawdown"] == pytest.approx(-0.02)
