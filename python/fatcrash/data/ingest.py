"""Data ingestion from exchanges, CSV, and APIs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fatcrash.data import cache


def from_csv(path: str | Path, date_col: str = "date", price_col: str = "close") -> pd.DataFrame:
    """Load OHLCV data from CSV.

    Returns DataFrame with DatetimeIndex and at least a 'close' column.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    if date_col not in df.columns:
        raise ValueError(f"CSV missing date column '{date_col}'")
    if price_col not in df.columns:
        raise ValueError(f"CSV missing price column '{price_col}'")
    df = df.rename(columns={date_col: "date", price_col: "close"})
    df = df.set_index("date").sort_index()
    return df


def from_ccxt(
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 1000,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data via CCXT."""
    import ccxt

    scope = f"{timeframe}:{limit}:{exchange}"
    if use_cache:
        cached = cache.load_cached("ccxt", symbol, scope)
        if cached is not None:
            return cached

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["timestamp"])
    if use_cache:
        cache.save_cache(df, "ccxt", symbol, scope)

    return df


def from_coingecko(
    coin_id: str = "bitcoin",
    vs_currency: str = "usd",
    days: int = 365,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily prices from CoinGecko (free, no API key)."""
    import requests

    scope = f"{vs_currency}:{days}"
    if use_cache:
        cached = cache.load_cached("coingecko", coin_id, scope)
        if cached is not None:
            return cached

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    prices = data.get("prices")
    if not prices:
        raise ValueError(f"CoinGecko returned no prices for {coin_id}")

    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["timestamp"])
    if use_cache:
        cache.save_cache(df, "coingecko", coin_id, scope)
    return df


def from_yahoo(
    ticker: str = "BTC-USD",
    start: str = "2015-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch data from Yahoo Finance via the free yfinance-like API."""
    import httpx

    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")

    scope = f"1d:{start}:{end}"
    if use_cache:
        cached = cache.load_cached("yahoo", ticker, scope)
        if cached is not None:
            return cached

    # Use Yahoo Finance v8 chart API
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": start_ts, "period2": end_ts, "interval": "1d"}
    headers = {"User-Agent": "fatcrash/0.1"}

    resp = httpx.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    chart = data.get("chart", {})
    result = chart.get("result")
    if not result:
        error = chart.get("error")
        message = error.get("description") if isinstance(error, dict) else "unknown error"
        raise ValueError(f"Yahoo Finance returned no data for {ticker}: {message}")

    result = result[0]
    timestamps = result.get("timestamp")
    indicators = result.get("indicators", {})
    quote = indicators.get("quote")
    quotes = quote[0] if quote else None

    if not timestamps or not quotes:
        raise ValueError(f"Yahoo Finance returned empty data for {ticker} ({start} to {end})")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s"),
            "open": quotes.get("open"),
            "high": quotes.get("high"),
            "low": quotes.get("low"),
            "close": quotes.get("close"),
            "volume": quotes.get("volume"),
        }
    )
    df = df.set_index("date").sort_index()
    if use_cache:
        cache.save_cache(df, "yahoo", ticker, scope)
    return df


def from_fred(
    series: str | list[str],
    start: str = "2007-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch macro data from FRED (Federal Reserve Economic Data).

    Common series IDs:
        VIXCLS          - VIX (CBOE Volatility Index)
        T10Y2Y          - 10Y-2Y yield curve spread
        BAMLH0A0HYM2    - High yield spread
        DTWEXBGS        - Trade-weighted US Dollar Index
        GDP             - US GDP (quarterly)
        DGS10           - 10-Year Treasury yield
        CPIAUCSL        - CPI (inflation)
        UNRATE          - Unemployment rate
    """
    import httpx

    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")

    if isinstance(series, str):
        series = [series]

    scope = f"{start}:{end}:{','.join(series)}"
    if use_cache:
        cached = cache.load_cached("fred", "series", scope)
        if cached is not None:
            return cached

    frames = {}
    for sid in series:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        params = {"id": sid, "cosd": start, "coed": end}
        resp = httpx.get(url, params=params, headers={"User-Agent": "fatcrash/0.1"}, timeout=30)
        resp.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"])
        df = df.rename(columns={"DATE": "date"})
        df = df.set_index("date")
        col = [c for c in df.columns if c != "date"][0]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        frames[sid] = df[col].rename(sid.lower())

    result = pd.concat(frames.values(), axis=1).sort_index()
    if use_cache:
        cache.save_cache(result, "fred", "series", scope)
    return result


def from_sample(asset: str = "btc") -> pd.DataFrame:
    """Load bundled sample data (no internet needed).

    Available: btc, spy, gold, signals
    """
    sample_dir = Path(__file__).parent.parent.parent.parent / "data" / "sample"

    files = {
        "btc": "btc_daily.csv",
        "bitcoin": "btc_daily.csv",
        "spy": "spy_daily.csv",
        "spx": "spy_daily.csv",
        "sp500": "spy_daily.csv",
        "gold": "gold_daily.csv",
        "signals": "macro_signals.csv",
    }

    key = asset.lower()
    if key not in files:
        available = ", ".join(sorted(set(files.values())))
        raise ValueError(f"Unknown asset '{asset}'. Available: {available}")

    path = sample_dir / files[key]
    if not path.exists():
        raise FileNotFoundError(f"Sample data not found: {path}")

    return from_csv(path)
