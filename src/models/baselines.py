"""Simple reference forecasters every neural network must beat.

All baselines operate on raw log-return windows and share the same
``fit`` / ``predict`` / ``n_params`` interface as the trained networks.

Naming follows the correction lecture. The professor described "Buy and
Hold" as the forecaster that repeats the previous day -- a linear model
with weight 1 on the last observation and 0 elsewhere -- and pointed out
that a linear regression scoring worse than it has not converged, since it
can represent it exactly. The original submission used that name for the
constant-zero forecast instead; both are kept, under unambiguous names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LinearRegression


def _check_inputs(X: np.ndarray) -> None:
    """Validate a ``(N, V, C)`` input tensor.

    Args:
        X: Candidate inputs.

    Raises:
        ValueError: If ``X`` is not three-dimensional or is empty.
    """
    if X.ndim != 3 or X.shape[0] == 0:
        raise ValueError(f"Expected non-empty (N, V, C) inputs, got {X.shape}")


class Baseline(ABC):
    """Common interface of non-neural forecasters."""

    name: str = "baseline"
    family: str = "baseline"

    def fit(self, X: np.ndarray, y: np.ndarray) -> Baseline:
        """Fit the forecaster (no-op unless overridden).

        Args:
            X: Training inputs ``(N, V, C)``.
            y: Training targets ``(N, C)``.

        Returns:
            The fitted forecaster.
        """
        _check_inputs(X)
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast mean future log returns.

        Args:
            X: Inputs ``(N, V, C)``.

        Returns:
            Predictions ``(N, C)``.
        """

    @property
    def n_params(self) -> int:
        """Number of fitted parameters."""
        return 0


class BuyAndHoldBaseline(Baseline):
    """Persistence: the forecast is the last observed daily return."""

    name = "buy_and_hold"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the last timestep of every window.

        Args:
            X: Inputs ``(N, V, C)``.

        Returns:
            ``X[:, -1, :]``.
        """
        _check_inputs(X)
        return X[:, -1, :].astype(float)


class ZeroReturnBaseline(Baseline):
    """Random-walk prices: the expected future log return is zero."""

    name = "zero_return"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return zeros.

        Args:
            X: Inputs ``(N, V, C)``.

        Returns:
            Zeros of shape ``(N, C)``.
        """
        _check_inputs(X)
        return np.zeros((X.shape[0], X.shape[2]))


class WindowMeanBaseline(Baseline):
    """The forecast is the mean return of the input window."""

    name = "window_mean"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average each window over time.

        Args:
            X: Inputs ``(N, V, C)``.

        Returns:
            ``X.mean(axis=1)``.
        """
        _check_inputs(X)
        return X.mean(axis=1)


class HistoricalMeanBaseline(Baseline):
    """The forecast is each asset's mean target over the training set."""

    name = "historical_mean"

    def __init__(self) -> None:
        """Create an unfitted forecaster."""
        self.mean_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> HistoricalMeanBaseline:
        """Store per-asset training means.

        Args:
            X: Training inputs ``(N, V, C)``.
            y: Training targets ``(N, C)``.

        Returns:
            The fitted forecaster.
        """
        _check_inputs(X)
        self.mean_ = np.asarray(y, dtype=float).mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Repeat the training means.

        Args:
            X: Inputs ``(N, V, C)``.

        Returns:
            Constant predictions ``(N, C)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        _check_inputs(X)
        if self.mean_ is None:
            raise RuntimeError("HistoricalMeanBaseline is not fitted")
        return np.tile(self.mean_, (X.shape[0], 1))

    @property
    def n_params(self) -> int:
        """One mean per asset."""
        return 0 if self.mean_ is None else int(self.mean_.size)


class LinearRegressionBaseline(Baseline):
    """Ordinary least squares on the flattened window (closed form)."""

    name = "linear_regression"

    def __init__(self) -> None:
        """Create an unfitted regression."""
        self.model_ = LinearRegression()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressionBaseline:
        """Fit OLS on ``(N, V * C)`` features.

        Args:
            X: Training inputs ``(N, V, C)``.
            y: Training targets ``(N, C)``.

        Returns:
            The fitted regression.
        """
        _check_inputs(X)
        self.model_.fit(X.reshape(len(X), -1), y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the fitted regression.

        Args:
            X: Inputs ``(N, V, C)``.

        Returns:
            Predictions ``(N, C)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        _check_inputs(X)
        if not self._fitted:
            raise RuntimeError("LinearRegressionBaseline is not fitted")
        return self.model_.predict(X.reshape(len(X), -1))

    @property
    def n_params(self) -> int:
        """Coefficients plus intercepts."""
        if not self._fitted:
            return 0
        return int(self.model_.coef_.size + self.model_.intercept_.size)


def default_baselines() -> list[Baseline]:
    """Instantiate the full baseline suite.

    Returns:
        Fresh, unfitted baseline instances.
    """
    return [
        BuyAndHoldBaseline(),
        ZeroReturnBaseline(),
        WindowMeanBaseline(),
        HistoricalMeanBaseline(),
        LinearRegressionBaseline(),
    ]
