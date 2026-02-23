"""Data ingestion from exchanges, CSV, and APIs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def from_csv(path: str | Path, date_col: str = "date", price_col: str = "close") -> pd.DataFrame:
    """Load OHLCV data from CSV.

    Returns DataFrame with DatetimeIndex and at least a 'close' column.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "date", price_col: "close"})
    df = df.set_index("date").sort_index()
    return df


def from_ccxt(
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV data via CCXT."""
    import ccxt

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["timestamp"])
    return df


def from_coingecko(
    coin_id: str = "bitcoin",
    vs_currency: str = "usd",
    days: int = 365,
) -> pd.DataFrame:
    """Fetch daily prices from CoinGecko (free, no API key)."""
    import requests

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["timestamp"])
    return df


def from_yahoo(
    ticker: str = "BTC-USD",
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch data from Yahoo Finance via the free yfinance-like API."""
    import httpx

    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Use Yahoo Finance v8 chart API
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": start_ts, "period2": end_ts, "interval": "1d"}
    headers = {"User-Agent": "fatcrash/0.1"}

    resp = httpx.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s"),
            "open": quotes["open"],
            "high": quotes["high"],
            "low": quotes["low"],
            "close": quotes["close"],
            "volume": quotes["volume"],
        }
    )
    df = df.set_index("date").sort_index()
    return df
