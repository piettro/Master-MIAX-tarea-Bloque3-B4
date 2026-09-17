"""Catalogue of the neural network specifications trained in each cell.

Two specifications per family give 8 networks x 16 window combinations =
128 trained models, twice the brief's minimum of 64 (one per family per
combination). Sizes are deliberately small: the correction lecture's rule
of thumb is roughly ten training samples per parameter, and financial data
never gets the "just add parameters" regime of internet-scale datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import keras

from src.models import architectures

FAMILIES = ("dense", "recurrent", "convolutional", "mixed")


@dataclass(frozen=True)
class ModelSpec:
    """Named, reproducible recipe for one neural network.

    Attributes:
        name: Unique identifier used in tables, figures and file names.
        family: One of :data:`FAMILIES`.
        builder: Function ``(input_shape, n_outputs, **kwargs) -> Model``.
        kwargs: Architecture hyper-parameters passed to ``builder``.
    """

    name: str
    family: str
    builder: Callable[..., keras.Model]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the family.

        Raises:
            ValueError: If ``family`` is not in :data:`FAMILIES`.
        """
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown family {self.family!r}")

    def build(
        self, input_shape: tuple[int, int], n_outputs: int
    ) -> keras.Model:
        """Instantiate a fresh, uncompiled model.

        Args:
            input_shape: ``(V, C)``.
            n_outputs: Number of assets.

        Returns:
            The Keras model.
        """
        return self.builder(input_shape, n_outputs, **self.kwargs)


DEFAULT_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("mlp_small", "dense", architectures.build_mlp,
              {"hidden_units": (32,), "dropout": 0.0}),
    ModelSpec("mlp_deep", "dense", architectures.build_mlp,
              {"hidden_units": (64, 32), "dropout": 0.2}),
    ModelSpec("lstm", "recurrent", architectures.build_recurrent,
              {"cell": "lstm", "units": (32,)}),
    ModelSpec("gru", "recurrent", architectures.build_recurrent,
              {"cell": "gru", "units": (32,)}),
    ModelSpec("cnn", "convolutional", architectures.build_cnn,
              {"filters": (32,), "kernel_size": 3}),
    ModelSpec("cnn_deep", "convolutional", architectures.build_cnn,
              {"filters": (32, 32), "kernel_size": 5, "dropout": 0.2}),
    ModelSpec("conv_gru", "mixed", architectures.build_conv_recurrent,
              {"filters": 32, "cell": "gru", "units": 32,
               "dense_units": (32,)}),
    ModelSpec("multibranch", "mixed", architectures.build_multibranch,
              {"filters": 16, "kernel_sizes": (3, 7),
               "recurrent_units": 16, "dense_units": (32,)}),
)


def get_spec(
    name: str, specs: tuple[ModelSpec, ...] = DEFAULT_SPECS
) -> ModelSpec:
    """Look a specification up by name.

    Args:
        name: Specification name.
        specs: Catalogue to search.

    Returns:
        The matching specification.

    Raises:
        KeyError: If no specification has that name.
    """
    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(
        f"Unknown model {name!r}; available: {[s.name for s in specs]}"
    )
