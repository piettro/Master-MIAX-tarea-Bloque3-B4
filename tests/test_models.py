"""Tests for baselines, fractional differentiation and architectures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    BuyAndHoldBaseline,
    HistoricalMeanBaseline,
    LinearRegressionBaseline,
    WindowMeanBaseline,
    ZeroReturnBaseline,
    default_baselines,
)
from src.preprocessing.fracdiff import (
    frac_diff_ffd,
    frac_diff_weights,
    select_common_d,
)


@pytest.fixture()
def windows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6, 3))
    y = rng.normal(size=(50, 3))
    return X, y


def test_buy_and_hold_repeats_last_observation(windows):
    X, _ = windows
    np.testing.assert_array_equal(BuyAndHoldBaseline().predict(X), X[:, -1])


def test_simple_baselines(windows):
    X, y = windows
    assert np.all(ZeroReturnBaseline().predict(X) == 0)
    np.testing.assert_allclose(WindowMeanBaseline().predict(X), X.mean(1))
    hist = HistoricalMeanBaseline().fit(X, y)
    np.testing.assert_allclose(hist.predict(X[:2]), [y.mean(0)] * 2)
    assert hist.n_params == 3


def test_linear_regression_can_represent_buy_and_hold(windows):
    """The professor's argument: OLS nests persistence exactly."""
    X, _ = windows
    y = X[:, -1, :]
    model = LinearRegressionBaseline().fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)
    assert model.n_params == 6 * 3 * 3 + 3


def test_baselines_require_fit(windows):
    X, _ = windows
    with pytest.raises(RuntimeError):
        HistoricalMeanBaseline().predict(X)
    with pytest.raises(RuntimeError):
        LinearRegressionBaseline().predict(X)
    assert len({b.name for b in default_baselines()}) == 5


def test_frac_diff_weights_limits():
    np.testing.assert_allclose(frac_diff_weights(1.0, 1e-5), [-1.0, 1.0])
    assert len(frac_diff_weights(0.0, 1e-5)) == 1
    with pytest.raises(ValueError):
        frac_diff_weights(-0.1, 1e-3)


def test_frac_diff_of_log_prices_with_d1_is_log_return(synthetic_prices):
    log_prices = np.log(synthetic_prices["AAA"])
    differenced = frac_diff_ffd(log_prices, 1.0, 1e-5).dropna()
    expected = log_prices.diff().dropna()
    np.testing.assert_allclose(differenced, expected)


def test_select_common_d_finds_stationary_order(synthetic_prices):
    d, summary = select_common_d(
        np.log(synthetic_prices), (0.0, 0.5, 1.0), 1e-3, 0.05, 5, 0.75
    )
    assert d in (0.5, 1.0)
    assert summary["stationary_share"].iloc[-1] >= 0.75


def test_frac_diff_rejects_non_finite():
    with pytest.raises(ValueError):
        frac_diff_ffd(pd.Series([1.0, np.nan, 2.0]), 0.5, 1e-2)


def test_every_spec_builds_with_expected_output_shape():
    from src.models.registry import DEFAULT_SPECS, FAMILIES

    assert {spec.family for spec in DEFAULT_SPECS} == set(FAMILIES)
    for V in (5, 90):
        for spec in DEFAULT_SPECS:
            model = spec.build((V, 23), 23)
            assert model.output_shape == (None, 23), spec.name
            assert model.count_params() > 0


def test_causal_convolution_ignores_future_steps():
    import keras

    from src.models.architectures import build_cnn

    model = build_cnn((12, 2), 2, filters=(4,), kernel_size=3)
    conv = keras.Model(model.input, model.get_layer("block1_conv").output)
    x = np.random.default_rng(0).normal(size=(1, 12, 2)).astype("float32")
    changed = x.copy()
    changed[:, -1, :] += 10.0
    np.testing.assert_allclose(
        conv(x).numpy()[:, :-1], conv(changed).numpy()[:, :-1], atol=1e-6
    )


def test_unknown_spec_raises():
    from src.models.registry import get_spec

    with pytest.raises(KeyError):
        get_spec("does_not_exist")
