"""Forecast accuracy metrics.

MAE, averaged over samples *and* the 23 assets, is the official metric of
the brief. Everything is computed in log-return units.
"""

from __future__ import annotations

import numpy as np


def _check_pair(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate a target/prediction pair.

    Args:
        y_true: Targets.
        y_pred: Predictions.

    Raises:
        ValueError: If shapes differ or the arrays are empty.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot score empty arrays")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error over every sample and asset.

    Args:
        y_true: Targets ``(N, C)``.
        y_pred: Predictions ``(N, C)``.

    Returns:
        The scalar MAE.
    """
    _check_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def sample_absolute_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Absolute error of each sample, averaged across assets.

    This is the loss series fed to the Diebold-Mariano test; its mean is
    :func:`mae`.

    Args:
        y_true: Targets ``(N, C)``.
        y_pred: Predictions ``(N, C)``.

    Returns:
        Array of shape ``(N,)``.
    """
    _check_pair(y_true, y_pred)
    return np.abs(y_pred - y_true).reshape(len(y_true), -1).mean(axis=1)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of sample-asset pairs whose predicted sign is correct.

    Zero targets are excluded because their sign is undefined.

    Args:
        y_true: Targets ``(N, C)``.
        y_pred: Predictions ``(N, C)``.

    Returns:
        Hit rate in ``[0, 1]``, or ``nan`` if every target is zero.
    """
    _check_pair(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))
