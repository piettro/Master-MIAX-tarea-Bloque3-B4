"""Loading of the 23-asset close-price panel and its log returns.

The professor's starter notebook downloads adjusted closes with
``yfinance`` on every run. Yahoo revises history retroactively, so a live
download is not reproducible; this module reads a Parquet snapshot that is
versioned with the repository and only downloads when asked to.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def download_close_prices(
    tickers: Sequence[str],
    start_date: str,
) -> pd.DataFrame:
    """Download adjusted daily close prices from Yahoo Finance.

    Mirrors the starter notebook: ``auto_adjust=True`` and columns with any
    missing value are dropped, so every retained asset spans the full
    sample.

    Args:
        tickers: Ticker symbols to request.
        start_date: First date, ``YYYY-MM-DD``.

    Returns:
        Prices indexed by date with one column per retained ticker.

    Raises:
        ValueError: If ``tickers`` is empty.
        RuntimeError: If the download fails or returns no usable column.
    """
    if not tickers:
        raise ValueError("tickers must not be empty")

    try:
        import yfinance as yf

        raw = yf.download(
            list(tickers), start=start_date, auto_adjust=True, progress=False
        )
    except Exception as exc:  # network, SSL or API-change failures
        raise RuntimeError(
            f"Yahoo Finance download failed: {exc}. Restore the snapshot "
            f"data/raw/close_prices.parquet from the repository instead."
        ) from exc

    if raw is None or raw.empty or "Close" not in raw:
        raise RuntimeError("Yahoo Finance returned no close prices")

    prices = raw["Close"].dropna(axis=1)
    dropped = sorted(set(tickers) - set(prices.columns))
    if dropped:
        logger.warning("Dropped tickers with incomplete history: %s", dropped)
    if prices.empty:
        raise RuntimeError("No ticker has a complete price history")
    return prices


def load_close_prices(
    path: Path,
    tickers: Sequence[str],
    start_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load the price snapshot, downloading it first if needed.

    Args:
        path: Location of the Parquet snapshot.
        tickers: Expected ticker columns.
        start_date: Download start date, used only when downloading.
        refresh: Force a fresh download that overwrites the snapshot.

    Returns:
        Close prices indexed by a sorted ``DatetimeIndex``.

    Raises:
        FileNotFoundError: If the snapshot is missing and cannot be
            downloaded.
        ValueError: If the snapshot is empty, lacks expected tickers or
            contains non-positive prices.
    """
    if refresh or not path.exists():
        logger.info("Downloading %d tickers from %s", len(tickers), start_date)
        try:
            prices = download_close_prices(tickers, start_date)
        except RuntimeError as exc:
            raise FileNotFoundError(
                f"Price snapshot {path} is missing and the download failed "
                f"({exc}). Place close_prices.parquet (dates as index, one "
                f"column per ticker) in {path.parent}."
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(path)
        logger.info("Saved price snapshot to %s", path)

    try:
        prices = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read price snapshot {path}: {exc}") from exc

    return validate_prices(prices, tickers)


def validate_prices(
    prices: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """Check and normalise a price panel.

    Args:
        prices: Candidate price frame.
        tickers: Tickers that must be present.

    Returns:
        The frame restricted to ``tickers`` (in that order) with a sorted
        ``DatetimeIndex``.

    Raises:
        ValueError: If the frame is empty, misses tickers, has missing
            values or non-positive prices.
    """
    if prices.empty:
        raise ValueError("Price panel is empty")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Price panel lacks tickers: {missing}")

    prices = prices.loc[:, list(tickers)].sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices.columns.name = None
    if prices.isna().any().any():
        raise ValueError("Price panel contains missing values")
    if (prices <= 0).any().any():
        raise ValueError("Price panel contains non-positive prices")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns exactly as the starter notebook does.

    Args:
        prices: Strictly positive close prices.

    Returns:
        ``log(P_t) - log(P_{t-1})`` with the first (undefined) row dropped.

    Raises:
        ValueError: If any price is non-positive.
    """
    if (prices <= 0).any().any():
        raise ValueError("Log returns require strictly positive prices")
    return np.log(prices).diff().dropna()
