"""Tests for data transforms."""

import numpy as np
import pandas as pd
import pytest

from fatcrash.data.transforms import log_prices, log_returns, time_index, block_maxima


def _make_df(prices):
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"close": prices}, index=dates)


def test_log_returns():
    df = _make_df([100, 110, 105, 115])
    ret = log_returns(df)
    assert len(ret) == 3
    assert abs(ret[0] - np.log(110 / 100)) < 1e-10


def test_log_prices():
    df = _make_df([100, 200])
    lp = log_prices(df)
    assert abs(lp[0] - np.log(100)) < 1e-10
    assert abs(lp[1] - np.log(200)) < 1e-10


def test_time_index():
    df = _make_df([100, 200, 300])
    t = time_index(df)
    assert t[0] == 0.0
    assert abs(t[1] - 1.0) < 1e-10
    assert abs(t[2] - 2.0) < 1e-10


def test_block_maxima():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.02, 252)
    bm = block_maxima(returns, block_size=21)
    assert len(bm) == 12  # 252 / 21
    assert all(b > 0 for b in bm)
