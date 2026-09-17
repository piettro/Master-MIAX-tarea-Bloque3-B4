"""Shared fixtures: small synthetic markets that run in seconds."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    """Geometric random walks for four assets over 2018-2025 business days.

    Returns:
        Positive prices with a ``DatetimeIndex`` covering 2025, so the
        portfolio stage can run on them.
    """
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2018-01-01", "2025-12-31")
    drift = np.array([0.0004, 0.0001, -0.0001, 0.0002])
    log_returns = drift + 0.01 * rng.standard_normal((len(index), 4))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=["AAA", "BBB", "CCC",
                                                      "DDD"])


@pytest.fixture()
def synthetic_returns(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns of :func:`synthetic_prices`.

    Args:
        synthetic_prices: Price fixture.

    Returns:
        Daily log returns.
    """
    return np.log(synthetic_prices).diff().dropna()
