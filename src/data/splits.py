"""Chronological train / validation / test partitions.

The competition test set is fixed by the professor's starter notebook::

    train_test_split(X, y, test_size=0.1, shuffle=False, random_state=42)

With ``shuffle=False`` the seed is irrelevant and scikit-learn assigns the
last ``ceil(0.1 * N)`` windows to test. :func:`chronological_split`
reproduces that boundary exactly and carves the validation block from the
tail of the training part, so the test set is untouched by any choice made
here.

Known caveat of the provided split (documented, not changed, because the
competition must use it verbatim): the last training windows share raw days
with the first test windows. The research stage measures the effect of an
embargo on the train/validation boundary instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    """Integer positions of each partition, in chronological order.

    Attributes:
        train: Training sample positions.
        validation: Validation sample positions.
        test: Test sample positions.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class DataSplit:
    """Arrays of one ``(V, H)`` experiment.

    Attributes:
        X_train: Training inputs ``(N_tr, V, C)``.
        y_train: Training targets ``(N_tr, C)``.
        X_val: Validation inputs.
        y_val: Validation targets.
        X_test: Test inputs.
        y_test: Test targets.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def overlap_gap(input_window: int, output_window: int) -> int:
    """Samples that must separate two windows so they share no raw day.

    Args:
        input_window: Look-back ``V``.
        output_window: Horizon ``H``.

    Returns:
        ``V + H - 1``.
    """
    return input_window + output_window - 1


def chronological_split(
    n_samples: int,
    test_fraction: float,
    validation_fraction: float,
    embargo: int = 0,
) -> SplitIndices:
    """Compute chronological partition positions.

    Args:
        n_samples: Total number of windows ``N``.
        test_fraction: Test share of ``N``; ``ceil`` rounding as in
            scikit-learn.
        validation_fraction: Validation share of the training block.
        embargo: Training windows dropped just before the validation block
            (see :func:`overlap_gap`); ``0`` disables the embargo.

    Returns:
        The three index arrays.

    Raises:
        ValueError: If fractions are outside ``(0, 1)``, ``embargo`` is
            negative, or a partition would be empty.
    """
    for name, value in (
        ("test_fraction", test_fraction),
        ("validation_fraction", validation_fraction),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0, 1), got {value}")
    if embargo < 0:
        raise ValueError(f"embargo must be non-negative, got {embargo}")

    n_test = math.ceil(test_fraction * n_samples)
    n_train_block = n_samples - n_test
    n_val = math.ceil(validation_fraction * n_train_block)
    n_train = n_train_block - n_val - embargo
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Cannot split {n_samples} samples into non-empty partitions "
            f"(train={n_train}, validation={n_val}, test={n_test})"
        )

    return SplitIndices(
        train=np.arange(n_train),
        validation=np.arange(n_train_block - n_val, n_train_block),
        test=np.arange(n_train_block, n_samples),
    )


def apply_split(
    X: np.ndarray,
    y: np.ndarray,
    indices: SplitIndices,
) -> DataSplit:
    """Materialise a :class:`DataSplit` from index arrays.

    Args:
        X: Inputs ``(N, V, C)``.
        y: Targets ``(N, C)``.
        indices: Partition positions.

    Returns:
        The partitioned arrays.

    Raises:
        ValueError: If ``X`` and ``y`` disagree on ``N``.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} samples but y has {len(y)}")
    return DataSplit(
        X_train=X[indices.train],
        y_train=y[indices.train],
        X_val=X[indices.validation],
        y_val=y[indices.validation],
        X_test=X[indices.test],
        y_test=y[indices.test],
    )
