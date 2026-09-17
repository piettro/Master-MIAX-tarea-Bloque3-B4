"""Scalers fitted on training data only.

Daily log returns are of order ``1e-2`` and 90-day mean returns of order
``1e-4``. Feeding such magnitudes straight into a network forced the
original submission down to a ``1e-5`` learning rate. The fix keeps the
objective intact:

* inputs are standardised per asset (statistics from training windows);
* targets are centred per asset and divided by **one global scalar**.
  ``|(y - m) / s - (p - m) / s| = |y - p| / s``, so minimising MAE on the
  scaled targets is exactly minimising MAE in log-return units, as the brief
  requires. A per-asset scale would silently re-weight assets.
"""

from __future__ import annotations

import numpy as np

_MIN_STD = 1e-12


class InputScaler:
    """Per-asset standardisation of ``(N, V, C)`` input tensors."""

    def __init__(self) -> None:
        """Create an unfitted scaler."""
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> InputScaler:
        """Estimate channel means and standard deviations.

        Args:
            X: Training inputs of shape ``(N, V, C)``; NaNs are ignored.

        Returns:
            The fitted scaler.

        Raises:
            ValueError: If ``X`` is not three-dimensional.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected (N, V, C) inputs, got {X.shape}")
        self.mean_ = np.nanmean(X, axis=(0, 1))
        self.std_ = np.maximum(np.nanstd(X, axis=(0, 1)), _MIN_STD)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise inputs with the fitted statistics.

        Args:
            X: Inputs of shape ``(N, V, C)``.

        Returns:
            Float32 standardised inputs.

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("InputScaler must be fitted before transform")
        return ((X - self.mean_) / self.std_).astype(np.float32)


class TargetScaler:
    """Per-asset centring and global scalar scaling of ``(N, C)`` targets."""

    def __init__(self) -> None:
        """Create an unfitted scaler."""
        self.mean_: np.ndarray | None = None
        self.scale_: float | None = None

    def fit(self, y: np.ndarray) -> TargetScaler:
        """Estimate per-asset means and the global standard deviation.

        Args:
            y: Training targets of shape ``(N, C)``.

        Returns:
            The fitted scaler.

        Raises:
            ValueError: If ``y`` is not two-dimensional.
        """
        if y.ndim != 2:
            raise ValueError(f"Expected (N, C) targets, got {y.shape}")
        self.mean_ = y.mean(axis=0)
        self.scale_ = float(max(np.std(y - self.mean_), _MIN_STD))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Scale targets.

        Args:
            y: Targets of shape ``(N, C)``.

        Returns:
            Float32 scaled targets.

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        self._check_fitted()
        return ((y - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Map scaled predictions back to log-return units.

        Args:
            y_scaled: Scaled values of shape ``(N, C)``.

        Returns:
            Values in the original units (float64).

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        self._check_fitted()
        return np.asarray(y_scaled, dtype=float) * self.scale_ + self.mean_

    def _check_fitted(self) -> None:
        """Raise if :meth:`fit` has not been called.

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TargetScaler must be fitted before use")
