"""Tests for M-LNN (Mono-LPPLS Neural Network)."""

import importlib

import numpy as np
import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch not installed")


class TestMLNN:
    def test_fit_on_synthetic(self):
        """M-LNN should recover approximate params on clean synthetic data."""
        from fatcrash.nn.mlnn import fit_mlnn
        from fatcrash.nn.synthetic import generate_lppls_series

        series, params = generate_lppls_series(
            n_obs=120, tc=140.0, m=0.5, omega=8.0,
            a=10.0, b=-0.3, c1=0.02, c2=-0.01,
            rng=np.random.default_rng(42),
        )
        times = np.arange(120, dtype=np.float64)

        result = fit_mlnn(times, series, epochs=100, lr=1e-2, seed=42)

        assert result.method == "M-LNN"
        assert result.rss < 1.0  # Should fit reasonably
        assert 0.1 <= result.m <= 1.0
        assert 6.0 <= result.omega <= 13.0

    def test_variable_length_input(self):
        """M-LNN should work with different input lengths."""
        from fatcrash.nn.mlnn import fit_mlnn

        for n in [60, 120, 252]:
            times = np.arange(n, dtype=np.float64)
            # Simple exponential growth as proxy
            log_prices = 10.0 - 0.3 * np.power(n + 20 - times, 0.5)
            result = fit_mlnn(times, log_prices, epochs=20, seed=42)
            assert result.method == "M-LNN"
            assert not np.isnan(result.tc)

    def test_result_fields(self):
        """Check all fields are populated."""
        from fatcrash.nn.mlnn import fit_mlnn

        times = np.arange(100, dtype=np.float64)
        log_prices = 10.0 - 0.3 * np.power(120 - times, 0.5)
        result = fit_mlnn(times, log_prices, epochs=20, seed=42)

        assert isinstance(result.tc, float)
        assert isinstance(result.m, float)
        assert isinstance(result.omega, float)
        assert isinstance(result.a, float)
        assert isinstance(result.b, float)
        assert isinstance(result.c1, float)
        assert isinstance(result.c2, float)
        assert isinstance(result.rss, float)
        assert isinstance(result.is_bubble, bool)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_range(self):
        from fatcrash.nn.mlnn import fit_mlnn

        times = np.arange(100, dtype=np.float64)
        log_prices = 10.0 - 0.3 * np.power(120 - times, 0.5)
        result = fit_mlnn(times, log_prices, epochs=50, seed=42)
        assert 0.0 <= result.confidence <= 1.0
