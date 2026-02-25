"""Tests for HLPPL (Hyped LPPL) dual-stream transformer."""

import importlib

import numpy as np
import pandas as pd
import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch not installed")


def _make_ohlcv_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.lognormal(10, 1, n)
    return pd.DataFrame({
        "close": close,
        "volume": volume,
    })


class TestSentimentProxy:
    def test_output_fields(self):
        from fatcrash.nn.sentiment import compute_sentiment_proxy

        df = _make_ohlcv_df(100)
        sent = compute_sentiment_proxy(df)
        assert sent.volume_z.shape == (100,)
        assert sent.volume_momentum.shape == (100,)
        assert sent.abs_return_z.shape == (100,)
        assert sent.hype_index.shape == (100,)

    def test_hype_index_range(self):
        """Hype index should be in [0, 1]."""
        from fatcrash.nn.sentiment import compute_sentiment_proxy

        df = _make_ohlcv_df(200)
        sent = compute_sentiment_proxy(df)
        assert sent.hype_index.min() >= 0.0
        assert sent.hype_index.max() <= 1.0

    def test_no_nans_in_output(self):
        from fatcrash.nn.sentiment import compute_sentiment_proxy

        df = _make_ohlcv_df(100)
        sent = compute_sentiment_proxy(df)
        assert not np.any(np.isnan(sent.hype_index))


class TestHLPPLNetwork:
    def test_forward_pass_shape(self):
        """Forward pass should produce correct output shapes."""
        import torch
        from fatcrash.nn.hlppl import HLPPLNetwork

        model = HLPPLNetwork()
        temporal = torch.randn(2, 60, 5)  # batch=2, seq=60, features=5
        sentiment = torch.randn(2, 60, 4)

        score, t_feats, s_feats = model(temporal, sentiment)
        assert score.shape == (2,)
        assert t_feats.shape == (2, 64)  # temporal d_model
        assert s_feats.shape == (2, 32)  # sentiment d_model

    def test_score_in_01(self):
        """Bubble score should be in [0, 1] due to sigmoid."""
        import torch
        from fatcrash.nn.hlppl import HLPPLNetwork

        model = HLPPLNetwork()
        temporal = torch.randn(5, 30, 5)
        sentiment = torch.randn(5, 30, 4)

        with torch.no_grad():
            score, _, _ = model(temporal, sentiment)
        assert (score >= 0).all()
        assert (score <= 1).all()


class TestHLPPLTraining:
    def test_train_small(self):
        """Train on small synthetic data to verify pipeline."""
        from fatcrash.nn.hlppl import train_hlppl

        dfs = [_make_ohlcv_df(80, seed=i) for i in range(10)]
        labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]

        model = train_hlppl(dfs, labels, window=60, epochs=3, lr=1e-3, seed=42)
        assert model is not None

    def test_predict(self):
        """Predict on single DataFrame."""
        from fatcrash.nn.hlppl import predict_hlppl, train_hlppl

        dfs = [_make_ohlcv_df(80, seed=i) for i in range(6)]
        labels = [1, 0, 1, 0, 1, 0]

        model = train_hlppl(dfs, labels, window=60, epochs=2, seed=42)
        result = predict_hlppl(model, _make_ohlcv_df(80, seed=99), window=60)

        assert 0.0 <= result.bubble_score <= 1.0
        assert result.temporal_features.shape == (64,)
        assert result.sentiment_features.shape == (32,)


class TestHLPPLSignalConverter:
    def test_hlppl_signal(self):
        from fatcrash.aggregator.signals import hlppl_signal

        assert hlppl_signal(0.8) == pytest.approx(0.8)
        assert hlppl_signal(0.0) == pytest.approx(0.0)
        assert hlppl_signal(1.5) == pytest.approx(1.0)  # Clipped
        assert hlppl_signal(float("nan")) == pytest.approx(0.0)
