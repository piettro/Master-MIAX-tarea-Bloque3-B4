"""Sliding-window construction of ``(X, y)`` samples.

For sample ``i`` with input window ``V`` and output window ``H``:

* ``X[i] = data[i : i + V]``                   shape ``(V, n_assets)``
* ``y[i] = mean(data[i + V : i + V + H])``      shape ``(n_assets,)``

This reproduces ``create_time_series_data`` from the professor's starter
notebook value-for-value (checked in ``tests/test_windowing.py``), but uses
stride tricks instead of a Python loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _as_2d_array(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return ``data`` as a 2-D float array.

    Args:
        data: Frame or array of shape ``(T, n_assets)``.

    Returns:
        The underlying 2-D float array.

    Raises:
        ValueError: If the input is not two-dimensional.
    """
    array = data.to_numpy(dtype=float) if isinstance(
        data, pd.DataFrame
    ) else np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {array.shape}")
    return array


def n_windows(n_timesteps: int, input_window: int, output_window: int) -> int:
    """Number of complete samples a series of given length yields.

    Args:
        n_timesteps: Series length ``T``.
        input_window: Look-back ``V``.
        output_window: Horizon ``H``.

    Returns:
        ``T - V - H + 1``.

    Raises:
        ValueError: If the windows are invalid or the series is too short.
    """
    if input_window < 1:
        raise ValueError(f"input_window must be >= 1, got {input_window}")
    if output_window < 0:
        raise ValueError(f"output_window must be >= 0, got {output_window}")
    count = n_timesteps - input_window - output_window + 1
    if count < 1:
        raise ValueError(
            f"A series of length {n_timesteps} is too short for "
            f"input_window={input_window}, output_window={output_window}"
        )
    return count


def create_windows(
    data: pd.DataFrame | np.ndarray,
    input_window: int,
    output_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build input sequences and mean-over-horizon targets.

    Args:
        data: Time series of shape ``(T, n_assets)``.
        input_window: Look-back ``V``.
        output_window: Horizon ``H``. ``0`` makes the target the last input
            value, as in the starter notebook.

    Returns:
        ``(X, y)`` with shapes ``(N, V, n_assets)`` and ``(N, n_assets)``
        where ``N = T - V - H + 1``.

    Raises:
        ValueError: If the windows do not fit in the series.
    """
    array = _as_2d_array(data)
    count = n_windows(len(array), input_window, output_window)

    # sliding_window_view appends the window axis last: (T-V+1, C, V).
    inputs = sliding_window_view(array, input_window, axis=0)[:count]
    X = np.ascontiguousarray(inputs.transpose(0, 2, 1))

    if output_window == 0:
        y = X[:, -1, :].copy()
    else:
        cumulative = np.vstack(
            [np.zeros((1, array.shape[1])), np.cumsum(array, axis=0)]
        )
        start = np.arange(count) + input_window
        y = (
            cumulative[start + output_window] - cumulative[start]
        ) / output_window
    return X, y


def create_input_windows(
    data: pd.DataFrame | np.ndarray,
    input_window: int,
) -> np.ndarray:
    """Build every input window, including those without a known target.

    Used at inference time, where the most recent windows matter most.

    Args:
        data: Time series of shape ``(T, n_assets)``.
        input_window: Look-back ``V``.

    Returns:
        Array of shape ``(T - V + 1, V, n_assets)``; row ``j`` ends at
        timestep ``j + V - 1``.

    Raises:
        ValueError: If the series is shorter than ``input_window``.
    """
    array = _as_2d_array(data)
    n_windows(len(array), input_window, 0)
    windows = sliding_window_view(array, input_window, axis=0)
    return np.ascontiguousarray(windows.transpose(0, 2, 1))


def window_dates(
    index: pd.DatetimeIndex,
    input_window: int,
    output_window: int,
) -> pd.DataFrame:
    """Calendar boundaries of every sample produced by :func:`create_windows`.

    Args:
        index: Dates of the underlying series.
        input_window: Look-back ``V``.
        output_window: Horizon ``H`` (must be at least 1).

    Returns:
        Frame with one row per sample and columns ``input_start``,
        ``input_end`` (the decision date), ``target_start`` and
        ``target_end``.

    Raises:
        ValueError: If ``output_window`` is smaller than one or the windows
            do not fit in ``index``.
    """
    if output_window < 1:
        raise ValueError("window_dates requires output_window >= 1")
    count = n_windows(len(index), input_window, output_window)
    positions = np.arange(count)
    dates = pd.DatetimeIndex(index)
    return pd.DataFrame(
        {
            "input_start": dates[positions],
            "input_end": dates[positions + input_window - 1],
            "target_start": dates[positions + input_window],
            "target_end": dates[positions + input_window + output_window - 1],
        }
    )
