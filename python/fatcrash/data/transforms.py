"""Data transformations: log prices, log returns, resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_prices(df: pd.DataFrame, col: str = "close") -> np.ndarray:
    """Compute log(price) series."""
    prices = df[col].values
    log_p = np.full_like(prices, np.nan, dtype=float)
    mask = prices > 0
    log_p[mask] = np.log(prices[mask])
    return log_p


def log_returns(df: pd.DataFrame, col: str = "close") -> np.ndarray:
    """Compute log returns: ln(p_t / p_{t-1})."""
    prices = df[col].values
    log_p = np.full_like(prices, np.nan, dtype=float)
    mask = prices > 0
    log_p[mask] = np.log(prices[mask])
    return np.diff(log_p)


def time_index(df: pd.DataFrame) -> np.ndarray:
    """Convert DatetimeIndex to float64 array (days from start)."""
    dates = df.index
    origin = dates[0]
    return np.array([(d - origin).total_seconds() / 86400.0 for d in dates])


def resample_ohlcv(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Resample OHLCV data to a different frequency."""
    agg = {}
    if "open" in df.columns:
        agg["open"] = "first"
    if "high" in df.columns:
        agg["high"] = "max"
    if "low" in df.columns:
        agg["low"] = "min"
    if "close" in df.columns:
        agg["close"] = "last"
    if "volume" in df.columns:
        agg["volume"] = "sum"

    return df.resample(freq).agg(agg).dropna()


def negative_returns(returns: np.ndarray) -> np.ndarray:
    """Extract negative returns (losses) for tail analysis."""
    return -returns[returns < 0]


def block_maxima(returns: np.ndarray, block_size: int = 21) -> np.ndarray:
    """Compute block maxima of absolute returns for GEV fitting.

    Default block_size=21 corresponds to monthly blocks for daily data.
    """
    n = len(returns)
    n_blocks = n // block_size
    if n_blocks == 0:
        return np.array([np.max(np.abs(returns))])

    trimmed = np.abs(returns[: n_blocks * block_size])
    blocks = trimmed.reshape(n_blocks, block_size)
    return np.max(blocks, axis=1)
