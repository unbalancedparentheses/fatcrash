"""Local Parquet cache for price data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

CACHE_DIR = Path.home() / ".cache" / "fatcrash"


def _cache_key(source: str, symbol: str, scope: str) -> str:
    raw = f"{source}:{symbol}:{scope}"
    return hashlib.md5(raw.encode()).hexdigest()


def cache_path(source: str, symbol: str, scope: str = "1d") -> Path:
    """Get the cache file path for a given data source and scope."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(source, symbol, scope)
    return CACHE_DIR / f"{key}.parquet"


def load_cached(source: str, symbol: str, scope: str = "1d") -> pd.DataFrame | None:
    """Load data from cache if available."""
    path = cache_path(source, symbol, scope)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_cache(df: pd.DataFrame, source: str, symbol: str, scope: str = "1d") -> Path:
    """Save data to Parquet cache."""
    path = cache_path(source, symbol, scope)
    df.to_parquet(path)
    return path


def clear_cache() -> None:
    """Remove all cached files."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
