"""Deterministic seeding of every random source used by the project."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic_tf: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow.

    TensorFlow is imported lazily so that data-only code paths (and their
    unit tests) do not pay its start-up cost.

    Args:
        seed: Non-negative integer seed.
        deterministic_tf: Also request deterministic TensorFlow kernels,
            which makes CPU results bit-for-bit repeatable.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)
    if deterministic_tf:
        tf.config.experimental.enable_op_determinism()
    logger.debug("Global random seed set to %d", seed)
