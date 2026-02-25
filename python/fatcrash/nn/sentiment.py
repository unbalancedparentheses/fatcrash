"""Volume-based sentiment proxy for HLPPL.

Provides hype features derived from price/volume data without requiring
external NLP sentiment feeds. Can optionally load pre-computed sentiment.

Reference: Cao, Shao, Yan, Geman (2025) "HLPPL", arXiv:2510.10878

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class SentimentFeatures:
    """Hype-related features for HLPPL sentiment stream."""

    volume_z: npt.NDArray[np.float64]  # z-scored volume
    volume_momentum: npt.NDArray[np.float64]  # rolling volume change
    abs_return_z: npt.NDArray[np.float64]  # z-scored absolute returns
    hype_index: npt.NDArray[np.float64]  # combined hype signal [0, 1]


def compute_sentiment_proxy(
    df: pd.DataFrame,
    volume_col: str = "volume",
    close_col: str = "close",
    window: int = 20,
) -> SentimentFeatures:
    """Compute volume-based sentiment features from OHLCV data.

    Args:
        df: DataFrame with at minimum close prices and volume columns
        volume_col: column name for volume data
        close_col: column name for close prices
        window: rolling window for z-scoring and momentum

    Returns:
        SentimentFeatures with four hype indicators
    """
    n = len(df)
    close = df[close_col].values.astype(np.float64)
    volume = df[volume_col].values.astype(np.float64)

    # Volume z-score (rolling)
    vol_mean = pd.Series(volume).rolling(window, min_periods=1).mean().values
    vol_std = pd.Series(volume).rolling(window, min_periods=1).std().values
    vol_std = np.where(vol_std < 1e-10, 1.0, vol_std)
    volume_z = (volume - vol_mean) / vol_std

    # Volume momentum: rate of change over window
    vol_series = pd.Series(volume)
    vol_lag = vol_series.shift(window).bfill().values
    vol_lag = np.where(vol_lag < 1e-10, 1.0, vol_lag)
    volume_momentum = (volume - vol_lag) / vol_lag

    # Absolute return z-score
    returns = np.diff(np.log(np.maximum(close, 1e-10)), prepend=np.log(max(close[0], 1e-10)))
    abs_returns = np.abs(returns)
    ret_mean = pd.Series(abs_returns).rolling(window, min_periods=1).mean().values
    ret_std = pd.Series(abs_returns).rolling(window, min_periods=1).std().values
    ret_std = np.where(ret_std < 1e-10, 1.0, ret_std)
    abs_return_z = (abs_returns - ret_mean) / ret_std

    # Replace NaNs (e.g. first element where std is undefined)
    volume_z = np.nan_to_num(volume_z, nan=0.0)
    volume_momentum = np.nan_to_num(volume_momentum, nan=0.0)
    abs_return_z = np.nan_to_num(abs_return_z, nan=0.0)

    # Combined hype index: average of clipped indicators, mapped to [0, 1]
    vz_clipped = np.clip(volume_z, -3, 3) / 6 + 0.5
    vm_clipped = np.clip(volume_momentum, -2, 2) / 4 + 0.5
    ar_clipped = np.clip(abs_return_z, -3, 3) / 6 + 0.5
    hype_index = np.clip((vz_clipped + vm_clipped + ar_clipped) / 3.0, 0, 1)

    return SentimentFeatures(
        volume_z=volume_z,
        volume_momentum=volume_momentum,
        abs_return_z=abs_return_z,
        hype_index=hype_index,
    )


def load_external_sentiment(
    path: str,
    date_col: str = "date",
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Load pre-computed sentiment scores from CSV.

    Expected columns: date (parseable), sentiment (float in [-1, 1]).
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df[sentiment_col] = df[sentiment_col].astype(float)
    return df
