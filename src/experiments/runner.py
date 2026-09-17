"""Fit-and-score helpers shared by every experiment stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.data.splits import DataSplit
from src.evaluation.metrics import (
    directional_accuracy,
    mae,
    sample_absolute_errors,
)
from src.models.baselines import Baseline
from src.models.registry import ModelSpec
from src.training.trainer import TrainedNetwork, train_network
from src.utils.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelOutcome:
    """Scores of one model on one ``(V, H)`` split.

    Attributes:
        row: Flat record destined for the results table.
        test_predictions: Test forecasts ``(N_test, C)``.
        test_losses: Per-sample absolute test errors (for DM tests).
        network: The trained network, or ``None`` for baselines.
    """

    row: dict
    test_predictions: np.ndarray
    test_losses: np.ndarray
    network: TrainedNetwork | None = None


def score_predictor(
    predict: Callable[[np.ndarray], np.ndarray],
    split: DataSplit,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute MAE on every partition and test diagnostics.

    Args:
        predict: Function mapping raw inputs to log-return forecasts.
        split: Data partitions.

    Returns:
        ``(metrics, test_predictions)`` where ``metrics`` holds
        ``mae_train``, ``mae_val``, ``mae_test`` and
        ``directional_accuracy_test``.
    """
    test_predictions = predict(split.X_test)
    metrics = {
        "mae_train": mae(split.y_train, predict(split.X_train)),
        "mae_val": mae(split.y_val, predict(split.X_val)),
        "mae_test": mae(split.y_test, test_predictions),
        "directional_accuracy_test": directional_accuracy(
            split.y_test, test_predictions
        ),
    }
    return metrics, test_predictions


def run_baseline(
    baseline: Baseline,
    split: DataSplit,
    input_window: int,
    output_window: int,
) -> ModelOutcome:
    """Fit and score a baseline.

    Args:
        baseline: Unfitted baseline instance.
        split: Data partitions.
        input_window: ``V`` of the split.
        output_window: ``H`` of the split.

    Returns:
        The baseline's outcome.
    """
    baseline.fit(split.X_train, split.y_train)
    metrics, predictions = score_predictor(baseline.predict, split)
    row = {
        "model": baseline.name,
        "family": baseline.family,
        "input_window": input_window,
        "output_window": output_window,
        **metrics,
        "n_params": baseline.n_params,
        "n_train_samples": len(split.X_train),
        "epochs_run": np.nan,
        "best_epoch": np.nan,
        "hit_epoch_limit": False,
        "seconds": 0.0,
        "seed": np.nan,
    }
    return ModelOutcome(
        row, predictions, sample_absolute_errors(split.y_test, predictions)
    )


def run_network(
    spec: ModelSpec,
    split: DataSplit,
    config: TrainingConfig,
    input_window: int,
    output_window: int,
    seed: int,
    history_path: Path | None = None,
) -> ModelOutcome:
    """Train and score a neural network.

    Args:
        spec: Architecture recipe.
        split: Data partitions.
        config: Training hyper-parameters.
        input_window: ``V`` of the split.
        output_window: ``H`` of the split.
        seed: Training seed.
        history_path: Optional CSV destination of the loss history.

    Returns:
        The network's outcome, including the trained model.
    """
    network = train_network(spec, split, config, seed, history_path)
    metrics, predictions = score_predictor(network.predict, split)
    row = {
        "model": spec.name,
        "family": spec.family,
        "input_window": input_window,
        "output_window": output_window,
        **metrics,
        "n_params": network.n_params,
        "n_train_samples": len(split.X_train),
        "epochs_run": network.epochs_run,
        "best_epoch": network.best_epoch,
        "hit_epoch_limit": network.hit_epoch_limit,
        "seconds": round(network.seconds, 1),
        "seed": seed,
    }
    logger.info(
        "V=%-3d H=%-3d %-18s val=%.6f test=%.6f params=%-7d epochs=%d "
        "(best %d) %.0fs",
        input_window, output_window, spec.name, metrics["mae_val"],
        metrics["mae_test"], network.n_params, network.epochs_run,
        network.best_epoch, network.seconds,
    )
    return ModelOutcome(
        row,
        predictions,
        sample_absolute_errors(split.y_test, predictions),
        network,
    )
