"""Keras architectures for the four required families.

Every builder returns an *uncompiled* functional model mapping
``(V, C)`` windows to ``C`` outputs; compilation (optimizer, MAE loss)
lives in :mod:`src.training.trainer`, so architecture and optimisation stay
independent.

Design notes taken from the course:

* the output layer is linear -- scaled returns are unbounded in sign;
* convolutional blocks follow Conv -> BatchNorm -> ReLU -> Pool, the order
  recommended in the CNN lecture (normalising after ReLU would re-create the
  negative values ReLU has just removed);
* ``padding="causal"`` so a feature at time ``t`` only sees ``<= t``.
  NOTE: improved over the original submission, whose docstring promised
  causal padding while the code used ``"same"``.
"""

from __future__ import annotations

from typing import Sequence

import keras
from keras import layers


def _conv_block(
    x: keras.KerasTensor,
    filters: int,
    kernel_size: int,
    name: str,
) -> keras.KerasTensor:
    """Apply Conv1D -> BatchNorm -> ReLU, then pool if the sequence allows.

    Args:
        x: Input tensor ``(batch, time, channels)``.
        filters: Number of convolution filters.
        kernel_size: Temporal kernel length.
        name: Prefix for layer names.

    Returns:
        Output tensor.
    """
    x = layers.Conv1D(
        filters, kernel_size, padding="causal", use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    if x.shape[1] is not None and x.shape[1] >= 4:
        x = layers.MaxPooling1D(2, name=f"{name}_pool")(x)
    return x


def _dense_head(
    x: keras.KerasTensor,
    units: Sequence[int],
    dropout: float,
    n_outputs: int,
    name: str,
) -> keras.KerasTensor:
    """Stack ReLU dense layers with dropout and a linear output layer.

    Args:
        x: Flat input tensor.
        units: Hidden widths; may be empty.
        dropout: Dropout rate after each hidden layer.
        n_outputs: Number of assets.
        name: Prefix for layer names.

    Returns:
        Output tensor of shape ``(batch, n_outputs)``.
    """
    for i, width in enumerate(units, start=1):
        x = layers.Dense(width, activation="relu", name=f"{name}_dense{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"{name}_dropout{i}")(x)
    return layers.Dense(n_outputs, name="output")(x)


def _recurrent_layer(cell: str, units: int, **kwargs) -> layers.Layer:
    """Instantiate an LSTM or GRU layer.

    Args:
        cell: ``"lstm"`` or ``"gru"``.
        units: Hidden state size.
        **kwargs: Forwarded to the Keras layer.

    Returns:
        The recurrent layer.

    Raises:
        ValueError: If ``cell`` is unknown.
    """
    if cell == "lstm":
        return layers.LSTM(units, **kwargs)
    if cell == "gru":
        return layers.GRU(units, **kwargs)
    raise ValueError(f"cell must be 'lstm' or 'gru', got {cell!r}")


def _validate_shape(input_shape: tuple[int, int], n_outputs: int) -> None:
    """Validate builder arguments.

    Args:
        input_shape: ``(V, C)``.
        n_outputs: Number of outputs.

    Raises:
        ValueError: If any dimension is not positive.
    """
    if len(input_shape) != 2 or min(input_shape) < 1 or n_outputs < 1:
        raise ValueError(
            f"Invalid input_shape={input_shape} or n_outputs={n_outputs}"
        )


def build_mlp(
    input_shape: tuple[int, int],
    n_outputs: int,
    hidden_units: Sequence[int] = (64, 32),
    dropout: float = 0.0,
) -> keras.Model:
    """Dense family: flatten the window and apply a multilayer perceptron.

    Args:
        input_shape: ``(V, C)``.
        n_outputs: Number of assets.
        hidden_units: Hidden layer widths.
        dropout: Dropout rate after each hidden layer.

    Returns:
        Uncompiled Keras model.
    """
    _validate_shape(input_shape, n_outputs)
    inputs = keras.Input(shape=input_shape, name="window")
    x = layers.Flatten(name="flatten")(inputs)
    outputs = _dense_head(x, hidden_units, dropout, n_outputs, "mlp")
    return keras.Model(inputs, outputs, name="mlp")


def build_recurrent(
    input_shape: tuple[int, int],
    n_outputs: int,
    cell: str = "lstm",
    units: Sequence[int] = (32,),
    dropout: float = 0.0,
) -> keras.Model:
    """Recurrent family: stacked LSTM or GRU layers.

    Args:
        input_shape: ``(V, C)``.
        n_outputs: Number of assets.
        cell: ``"lstm"`` or ``"gru"``.
        units: Hidden sizes; all but the last return full sequences.
        dropout: Input dropout of each recurrent layer.

    Returns:
        Uncompiled Keras model.

    Raises:
        ValueError: If ``units`` is empty.
    """
    _validate_shape(input_shape, n_outputs)
    if not units:
        raise ValueError("units must contain at least one layer size")
    inputs = keras.Input(shape=input_shape, name="window")
    x = inputs
    for i, size in enumerate(units, start=1):
        x = _recurrent_layer(
            cell, size, return_sequences=i < len(units), dropout=dropout,
            name=f"{cell}{i}",
        )(x)
    outputs = layers.Dense(n_outputs, name="output")(x)
    return keras.Model(inputs, outputs, name=cell)


def build_cnn(
    input_shape: tuple[int, int],
    n_outputs: int,
    filters: Sequence[int] = (32, 32),
    kernel_size: int = 3,
    dropout: float = 0.0,
) -> keras.Model:
    """Convolutional family: causal Conv1D blocks and a linear read-out.

    Args:
        input_shape: ``(V, C)``.
        n_outputs: Number of assets.
        filters: Filters per convolutional block.
        kernel_size: Temporal kernel length.
        dropout: Dropout before the output layer.

    Returns:
        Uncompiled Keras model.

    Raises:
        ValueError: If ``filters`` is empty.
    """
    _validate_shape(input_shape, n_outputs)
    if not filters:
        raise ValueError("filters must contain at least one block")
    inputs = keras.Input(shape=input_shape, name="window")
    x = inputs
    for i, n_filters in enumerate(filters, start=1):
        x = _conv_block(x, n_filters, kernel_size, f"block{i}")
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="cnn_dropout")(x)
    outputs = layers.Dense(n_outputs, name="output")(x)
    return keras.Model(inputs, outputs, name="cnn")


def build_conv_recurrent(
    input_shape: tuple[int, int],
    n_outputs: int,
    filters: int = 32,
    kernel_size: int = 3,
    cell: str = "gru",
    units: int = 32,
    dense_units: Sequence[int] = (32,),
    dropout: float = 0.0,
) -> keras.Model:
    """Mixed family: Conv1D feature extractor -> recurrent layer -> dense.

    Convolution summarises local patterns (and halves long sequences), the
    recurrent layer integrates them over time and the dense head maps the
    state to the 23 assets: all three layer types in one model.

    Args:
        input_shape: ``(V, C)``.
        n_outputs: Number of assets.
        filters: Convolution filters.
        kernel_size: Temporal kernel length.
        cell: ``"lstm"`` or ``"gru"``.
        units: Recurrent hidden size.
        dense_units: Hidden widths of the dense head.
        dropout: Dropout in the dense head.

    Returns:
        Uncompiled Keras model.
    """
    _validate_shape(input_shape, n_outputs)
    inputs = keras.Input(shape=input_shape, name="window")
    x = _conv_block(inputs, filters, kernel_size, "block1")
    x = _recurrent_layer(cell, units, name=f"{cell}1")(x)
    outputs = _dense_head(x, dense_units, dropout, n_outputs, "head")
    return keras.Model(inputs, outputs, name=f"conv_{cell}")


def build_multibranch(
    input_shape: tuple[int, int],
    n_outputs: int,
    filters: int = 16,
    kernel_sizes: Sequence[int] = (3, 7),
    recurrent_units: int = 16,
    dense_units: Sequence[int] = (32,),
    dropout: float = 0.0,
) -> keras.Model:
    """Mixed family: parallel convolutional, recurrent and dense branches.

    Each branch reads the same window with a different inductive bias
    (local patterns at several scales, sequential memory, and a global
    linear-nonlinear view); their embeddings are concatenated.

    Args:
        input_shape: ``(V, C)``.
        n_outputs: Number of assets.
        filters: Filters of each convolutional branch.
        kernel_sizes: One convolutional branch per kernel length.
        recurrent_units: GRU hidden size of the recurrent branch.
        dense_units: Hidden widths of the head after concatenation.
        dropout: Dropout in the head.

    Returns:
        Uncompiled Keras model.

    Raises:
        ValueError: If ``kernel_sizes`` is empty.
    """
    _validate_shape(input_shape, n_outputs)
    if not kernel_sizes:
        raise ValueError("kernel_sizes must contain at least one size")
    inputs = keras.Input(shape=input_shape, name="window")

    branches = []
    for k in kernel_sizes:
        x = _conv_block(inputs, filters, k, f"conv_k{k}")
        branches.append(layers.GlobalAveragePooling1D(name=f"gap_k{k}")(x))
    branches.append(layers.GRU(recurrent_units, name="gru_branch")(inputs))
    flat = layers.Flatten(name="flatten")(inputs)
    branches.append(
        layers.Dense(filters, activation="relu", name="dense_branch")(flat)
    )

    merged = layers.Concatenate(name="concat")(branches)
    outputs = _dense_head(merged, dense_units, dropout, n_outputs, "head")
    return keras.Model(inputs, outputs, name="multibranch")
