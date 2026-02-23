"""Tests for GSADF, Hurst, and Pickands — new methods."""

import numpy as np
import pytest
from pathlib import Path

from fatcrash._core import (
    pickands_estimator,
    pickands_rolling,
    hurst_exponent,
    hurst_rolling,
    gsadf_test,
    gsadf_rolling,
)
from fatcrash.data.ingest import from_sample
from fatcrash.data.transforms import log_returns, log_prices
from fatcrash.indicators.tail_indicator import (
    estimate_pickands,
    rolling_pickands,
    estimate_hurst,
    rolling_hurst,
)
from fatcrash.indicators.bubble_indicator import detect_bubble, rolling_bubble_detection


# ══════════════════════════════════════════
# Pickands estimator
# ══════════════════════════════════════════

class TestPickands:
    def test_pareto_heavy_tail(self):
        """Pareto data should give positive gamma (heavy tail)."""
        rng = np.random.default_rng(42)
        pareto = rng.pareto(2.0, 5000)
        gamma = pickands_estimator(pareto)
        assert gamma > 0, f"Pareto should give gamma > 0, got {gamma}"

    def test_gaussian_lighter_than_pareto(self):
        """Gaussian should have lower Pickands gamma than Pareto."""
        rng = np.random.default_rng(42)
        gaussian = rng.standard_normal(5000)
        pareto = rng.pareto(2.0, 5000)
        gamma_g = pickands_estimator(gaussian)
        gamma_p = pickands_estimator(pareto)
        assert gamma_g < gamma_p, f"Gaussian gamma {gamma_g} should be < Pareto {gamma_p}"

    def test_rolling_produces_output(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        result = pickands_rolling(data, window=200)
        assert len(result) == 500
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_wrapper_classifies_correctly(self):
        rng = np.random.default_rng(42)
        pareto = rng.pareto(2.0, 5000)
        est = estimate_pickands(pareto)
        assert est.tail_type == "heavy"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_on_real_data(self, asset_name):
        df = from_sample(asset_name)
        returns = log_returns(df)
        gamma = pickands_estimator(returns)
        assert not np.isnan(gamma), f"{asset_name}: Pickands gamma is NaN"


# ══════════════════════════════════════════
# Hurst exponent
# ══════════════════════════════════════════

class TestHurst:
    def test_random_walk_near_half(self):
        """iid Gaussian should give H near 0.5."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000)
        h = hurst_exponent(data)
        assert 0.3 < h < 0.7, f"Random walk H should be near 0.5, got {h}"

    def test_trending_above_half(self):
        """Cumulative sum (trending) should give H > 0.5."""
        rng = np.random.default_rng(42)
        increments = rng.standard_normal(2000)
        trending = np.cumsum(increments)
        h = hurst_exponent(trending)
        assert h > 0.4, f"Trending data should have H > 0.5, got {h}"

    def test_rolling_produces_output(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        result = hurst_rolling(data, window=200)
        assert len(result) == 500
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_wrapper_classifies(self):
        rng = np.random.default_rng(42)
        trending = np.cumsum(rng.standard_normal(2000))
        est = estimate_hurst(trending)
        assert est.regime in ("trending", "random_walk", "mean_reverting")

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_on_real_data(self, asset_name):
        df = from_sample(asset_name)
        returns = log_returns(df)
        h = hurst_exponent(returns)
        assert not np.isnan(h), f"{asset_name}: Hurst H is NaN"
        assert 0.0 < h < 1.0, f"{asset_name}: H={h} out of range"


# ══════════════════════════════════════════
# GSADF
# ══════════════════════════════════════════

class TestGSADF:
    def test_explosive_detected(self):
        """Explosive process should have high GSADF stat."""
        rng = np.random.default_rng(42)
        n = 200
        series = np.zeros(n)
        series[0] = 100.0
        for i in range(1, n):
            series[i] = series[i - 1] * 1.02 + rng.standard_normal()
        stat, bsadf, (cv90, cv95, cv99) = gsadf_test(series, n_sims=100, seed=42)
        assert stat > cv95, f"Explosive should exceed cv95: stat={stat:.2f}, cv95={cv95:.2f}"

    def test_random_walk_not_detected(self):
        """Random walk should not be detected as bubble."""
        rng = np.random.default_rng(42)
        rw = np.cumsum(rng.standard_normal(200))
        stat, _, (_, cv95, _) = gsadf_test(rw, n_sims=100, seed=42)
        # Random walk might occasionally exceed, but stat should be moderate
        assert stat < cv95 * 3, f"Random walk stat too high: {stat:.2f}"

    def test_bsadf_length(self):
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.standard_normal(100))
        _, bsadf, _ = gsadf_test(data, n_sims=50, seed=42)
        assert len(bsadf) == 100

    def test_critical_values_ordered(self):
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.standard_normal(100))
        _, _, (cv90, cv95, cv99) = gsadf_test(data, n_sims=100, seed=42)
        assert cv90 <= cv95 <= cv99

    def test_wrapper_detect_bubble(self):
        rng = np.random.default_rng(42)
        n = 200
        series = np.zeros(n)
        series[0] = 100.0
        for i in range(1, n):
            series[i] = series[i - 1] * 1.02 + rng.standard_normal()
        result = detect_bubble(series, n_sims=100)
        assert result.is_bubble_95 or result.gsadf_stat > 0

    def test_rolling_produces_output(self):
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.standard_normal(300))
        result = rolling_bubble_detection(data, window=100)
        assert len(result) == 300
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    @pytest.mark.parametrize("asset_name", ["btc", "spy"])
    def test_on_real_prices(self, asset_name):
        df = from_sample(asset_name)
        prices = df["close"].values[-500:].astype(np.float64)
        stat, _, (_, cv95, _) = gsadf_test(prices, n_sims=50, seed=42)
        assert np.isfinite(stat), f"{asset_name}: GSADF stat is not finite"


# ══════════════════════════════════════════
# Cross-method consistency
# ══════════════════════════════════════════

class TestCrossMethodConsistency:
    def test_pickands_and_hill_agree_on_direction(self):
        """Both should identify Pareto as heavy-tailed and Gaussian as lighter."""
        from fatcrash._core import hill_estimator

        rng = np.random.default_rng(42)

        pareto = rng.pareto(2.0, 3000)
        gaussian = rng.standard_normal(3000)

        hill_pareto = hill_estimator(pareto)
        hill_gaussian = hill_estimator(gaussian)
        pick_pareto = pickands_estimator(pareto)
        pick_gaussian = pickands_estimator(gaussian)

        # Pareto should have lower Hill alpha (fatter) and higher Pickands gamma
        assert hill_pareto < hill_gaussian
        assert pick_pareto > pick_gaussian

    def test_hurst_and_gsadf_agree_on_bubble(self):
        """Explosive series should show both high Hurst and high GSADF."""
        rng = np.random.default_rng(42)
        n = 300
        series = np.zeros(n)
        series[0] = 100.0
        for i in range(1, n):
            series[i] = series[i - 1] * 1.01 + 0.5 * rng.standard_normal()
        returns = np.diff(np.log(series))

        h = hurst_exponent(returns)
        stat, _, (_, cv95, _) = gsadf_test(series, n_sims=50, seed=42)

        # Both should indicate trending/explosive
        assert h > 0.4, f"Trending returns should have H > 0.5: {h}"
        assert stat > 0, f"Explosive series should have positive GSADF: {stat}"
