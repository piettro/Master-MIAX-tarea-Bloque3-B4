"""Tests for loading, windowing, splitting and scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data.loaders import (
    compute_log_returns,
    load_close_prices,
    validate_prices,
)
from src.data.scaling import InputScaler, TargetScaler
from src.data.splits import (
    apply_split,
    chronological_split,
    overlap_gap,
)
from src.data.windowing import (
    create_input_windows,
    create_windows,
    window_dates,
)


def professor_create_time_series_data(data, input_window, output_window):
    """Verbatim logic of the starter notebook, used as the test oracle."""
    X, y = [], []
    array = data.values if isinstance(data, pd.DataFrame) else data
    for i in range(len(array) - input_window - output_window + 1):
        X.append(array[i:i + input_window])
        if output_window > 0:
            y.append(np.mean(
                array[i + input_window:i + input_window + output_window],
                axis=0,
            ))
        else:
            y.append(array[i + input_window - 1])
    return np.array(X), np.array(y)


@pytest.mark.parametrize("V,H", [(5, 1), (10, 5), (30, 30), (3, 0)])
def test_create_windows_matches_professor(synthetic_returns, V, H):
    X, y = create_windows(synthetic_returns, V, H)
    X_ref, y_ref = professor_create_time_series_data(synthetic_returns, V, H)
    np.testing.assert_allclose(X, X_ref)
    np.testing.assert_allclose(y, y_ref, atol=1e-15)


def test_create_windows_rejects_short_series():
    with pytest.raises(ValueError):
        create_windows(np.zeros((5, 2)), 4, 3)


def test_create_windows_rejects_1d():
    with pytest.raises(ValueError):
        create_windows(np.zeros(10), 2, 1)


def test_input_windows_include_latest_day(synthetic_returns):
    windows = create_input_windows(synthetic_returns, 7)
    assert windows.shape == (len(synthetic_returns) - 6, 7, 4)
    np.testing.assert_allclose(
        windows[-1], synthetic_returns.to_numpy()[-7:]
    )


def test_window_dates_align_with_samples(synthetic_returns):
    V, H = 5, 3
    dates = window_dates(synthetic_returns.index, V, H)
    X, y = create_windows(synthetic_returns, V, H)
    assert len(dates) == len(X)
    k = 10
    idx = synthetic_returns.index
    assert dates.loc[k, "input_end"] == idx[k + V - 1]
    target = synthetic_returns.loc[
        dates.loc[k, "target_start"]:dates.loc[k, "target_end"]
    ]
    np.testing.assert_allclose(target.mean().to_numpy(), y[k])


def test_split_reproduces_professor_test_set():
    n = 16_167
    X = np.arange(n)
    _, X_test_ref = train_test_split(
        X, test_size=0.1, shuffle=False, random_state=42
    )
    indices = chronological_split(n, 0.10, 0.10)
    np.testing.assert_array_equal(indices.test, X_test_ref)


def test_split_is_chronological_and_disjoint():
    indices = chronological_split(1000, 0.1, 0.2)
    assert indices.train[-1] < indices.validation[0]
    assert indices.validation[-1] < indices.test[0]
    assert indices.validation[-1] + 1 == indices.test[0]


def test_embargo_only_shrinks_training():
    plain = chronological_split(1000, 0.1, 0.2)
    gap = overlap_gap(30, 90)
    embargoed = chronological_split(1000, 0.1, 0.2, embargo=gap)
    np.testing.assert_array_equal(plain.test, embargoed.test)
    np.testing.assert_array_equal(plain.validation, embargoed.validation)
    assert embargoed.validation[0] - embargoed.train[-1] == gap + 1


def test_split_validates_arguments():
    with pytest.raises(ValueError):
        chronological_split(100, 1.5, 0.1)
    with pytest.raises(ValueError):
        chronological_split(5, 0.5, 0.9, embargo=10)


def test_apply_split_rejects_mismatch():
    with pytest.raises(ValueError):
        apply_split(np.zeros((10, 2, 1)), np.zeros((9, 1)),
                    chronological_split(10, 0.2, 0.2))


def test_target_scaling_preserves_mae_ranking():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.01, (200, 3)) + np.array([0.001, -0.002, 0.0])
    p = y + rng.normal(0, 0.003, y.shape)
    scaler = TargetScaler().fit(y)
    scaled_mae = np.mean(np.abs(scaler.transform(y) - scaler.transform(p)))
    assert scaled_mae * scaler.scale_ == pytest.approx(np.mean(np.abs(y - p)),
                                                       rel=1e-5)
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(p)), p, atol=1e-8
    )


def test_input_scaler_standardises_training_channels():
    rng = np.random.default_rng(2)
    X = rng.normal(5, 3, (100, 4, 2))
    scaled = InputScaler().fit(X).transform(X)
    np.testing.assert_allclose(scaled.mean(axis=(0, 1)), 0, atol=1e-5)
    np.testing.assert_allclose(scaled.std(axis=(0, 1)), 1, atol=1e-5)


def test_scalers_require_fit():
    with pytest.raises(RuntimeError):
        InputScaler().transform(np.zeros((1, 1, 1)))
    with pytest.raises(RuntimeError):
        TargetScaler().inverse_transform(np.zeros((1, 1)))


def test_log_returns_match_starter_formula(synthetic_prices):
    returns = compute_log_returns(synthetic_prices)
    expected = np.log(synthetic_prices).diff().dropna()
    pd.testing.assert_frame_equal(returns, expected)


def test_validate_prices_rejects_bad_panels(synthetic_prices):
    with pytest.raises(ValueError):
        validate_prices(synthetic_prices, ["AAA", "ZZZ"])
    broken = synthetic_prices.copy()
    broken.iloc[3, 0] = -1.0
    with pytest.raises(ValueError):
        validate_prices(broken, ["AAA"])


def test_load_close_prices_reads_snapshot(tmp_path, synthetic_prices):
    path = tmp_path / "prices.parquet"
    synthetic_prices.to_parquet(path)
    loaded = load_close_prices(path, ["BBB", "AAA"], "2000-01-01")
    assert list(loaded.columns) == ["BBB", "AAA"]
    assert loaded.index.is_monotonic_increasing
