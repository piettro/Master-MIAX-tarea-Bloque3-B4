"""Training of one neural network on one data split.

Responsibilities: seeding, input/target scaling, compilation with Adam and
MAE, early stopping on validation MAE, persistence of the loss history and a
convergence diagnostic. Evaluation (metrics on every partition) is handled
by :mod:`src.experiments.runner`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from src.data.scaling import InputScaler, TargetScaler
from src.data.splits import DataSplit
from src.models.registry import ModelSpec
from src.utils.config import TrainingConfig
from src.utils.seeding import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainedNetwork:
    """A fitted network together with its preprocessing and diagnostics.

    Attributes:
        model: The Keras model with the best validation weights restored.
        input_scaler: Scaler fitted on training inputs.
        target_scaler: Scaler fitted on training targets.
        history: Per-epoch ``loss`` and ``val_loss`` in log-return units.
        best_epoch: 1-based epoch with the lowest validation loss.
        epochs_run: Number of epochs actually trained.
        hit_epoch_limit: True if training ran until ``max_epochs`` with the
            best epoch close to the end, i.e. the curve was still improving
            and the model may not have converged.
        seconds: Wall-clock training time.
    """

    model: keras.Model
    input_scaler: InputScaler
    target_scaler: TargetScaler
    history: pd.DataFrame
    best_epoch: int
    epochs_run: int
    hit_epoch_limit: bool
    seconds: float

    @property
    def n_params(self) -> int:
        """Total number of model parameters."""
        return int(self.model.count_params())

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict mean future log returns for raw input windows.

        Args:
            X: Unscaled inputs ``(N, V, C)``.
            batch_size: Inference batch size.

        Returns:
            Predictions ``(N, C)`` in log-return units.
        """
        scaled = self.model.predict(
            self.input_scaler.transform(X), batch_size=batch_size, verbose=0
        )
        return self.target_scaler.inverse_transform(scaled)


def _callbacks(config: TrainingConfig) -> list[keras.callbacks.Callback]:
    """Assemble the training callbacks.

    Args:
        config: Training hyper-parameters.

    Returns:
        Early stopping, plus ReduceLROnPlateau when enabled.
    """
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            min_delta=config.min_delta,
            restore_best_weights=True,
        )
    ]
    if config.use_reduce_lr:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                min_lr=config.min_learning_rate,
            )
        )
    return callbacks


def train_network(
    spec: ModelSpec,
    split: DataSplit,
    config: TrainingConfig,
    seed: int,
    history_path: Path | None = None,
) -> TrainedNetwork:
    """Train ``spec`` on ``split`` and restore the best validation weights.

    Args:
        spec: Architecture recipe.
        split: Unscaled partitions; training inputs must be finite.
        config: Optimisation hyper-parameters.
        seed: Seed for weight initialisation and batch order.
        history_path: Optional CSV destination of the loss history.

    Returns:
        The trained network and its diagnostics.

    Raises:
        ValueError: If training or validation arrays contain non-finite
            values.
    """
    for label, array in (
        ("X_train", split.X_train),
        ("y_train", split.y_train),
        ("X_val", split.X_val),
        ("y_val", split.y_val),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} contains non-finite values")

    keras.backend.clear_session()
    set_global_seed(seed)

    input_scaler = InputScaler().fit(split.X_train)
    target_scaler = TargetScaler().fit(split.y_train)
    X_train = input_scaler.transform(split.X_train)
    X_val = input_scaler.transform(split.X_val)
    y_train = target_scaler.transform(split.y_train)
    y_val = target_scaler.transform(split.y_val)

    n_outputs = split.y_train.shape[1]
    model = spec.build(split.X_train.shape[1:], n_outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mean_absolute_error",
    )

    started = time.perf_counter()
    fit = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        shuffle=True,
        callbacks=_callbacks(config),
        verbose=config.verbose,
    )
    seconds = time.perf_counter() - started

    # Report the curve in log-return units so it is comparable with the
    # metric tables (the loss was computed on scaled targets).
    history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(fit.history["loss"]) + 1),
            "loss": np.array(fit.history["loss"]) * target_scaler.scale_,
            "val_loss": np.array(fit.history["val_loss"])
            * target_scaler.scale_,
        }
    )
    best_epoch = int(history["val_loss"].idxmin()) + 1
    epochs_run = len(history)
    hit_limit = (
        epochs_run == config.max_epochs
        and best_epoch > epochs_run - config.patience // 2
    )
    if hit_limit:
        logger.warning(
            "%s reached max_epochs=%d while still improving (best epoch "
            "%d): the curve may not have converged",
            spec.name, config.max_epochs, best_epoch,
        )

    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(history_path, index=False)

    return TrainedNetwork(
        model=model,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        history=history,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        hit_epoch_limit=hit_limit,
        seconds=seconds,
    )
