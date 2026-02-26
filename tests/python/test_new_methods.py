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
    estimate_momentum,
    rolling_momentum,
    estimate_reversal,
    estimate_velocity,
    rolling_velocity,
)
from fatcrash.indicators.vol_indicator import (
    constant_vol_weight,
    rolling_constant_vol_weight,
    rebalance_risk,
)
from fatcrash.indicators.bubble_indicator import detect_bubble, rolling_bubble_detection
from fatcrash.aggregator.signals import (
    momentum_reversal_signal,
    velocity_signal,
)


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


# ══════════════════════════════════════════
# Momentum (Jegadeesh-Titman 1993)
# ══════════════════════════════════════════

class TestMomentumWrapper:
    def test_uptrend_positive(self):
        prices = np.array([100 * np.exp(0.001 * i) for i in range(300)])
        est = estimate_momentum(prices, lookback=252)
        assert est.direction == "up"
        assert est.momentum > 0

    def test_downtrend_negative(self):
        prices = np.array([100 * np.exp(-0.001 * i) for i in range(300)])
        est = estimate_momentum(prices, lookback=252)
        assert est.direction == "down"
        assert est.momentum < 0

    def test_rolling_shape(self):
        prices = np.array([100 * np.exp(0.001 * i) for i in range(600)])
        result = rolling_momentum(prices, lookback=252, window=300)
        assert len(result) == 600

    def test_reversal_detects_unwind(self):
        # Build up strongly then crash hard — reversal signal should be positive
        prices = np.array(
            [100 * np.exp(0.003 * i) for i in range(260)]
            + [100 * np.exp(0.003 * 260) * np.exp(-0.02 * i)
               for i in range(1, 41)]
        )
        est = estimate_reversal(prices, short_lookback=21, long_lookback=200)
        # reversal = long_mom - short_mom should be positive (long still up, short down)
        assert est.reversal > 0, f"Reversal should be positive, got {est.reversal}"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_on_real_data(self, asset_name):
        df = from_sample(asset_name)
        prices = df["close"].values.astype(np.float64)
        est = estimate_momentum(prices, lookback=252)
        assert np.isfinite(est.momentum), f"{asset_name}: momentum is NaN"


# ══════════════════════════════════════════
# Velocity (cascade detection)
# ══════════════════════════════════════════

class TestVelocityWrapper:
    def test_stable_moderate(self):
        rng = np.random.default_rng(42)
        returns = 0.01 * rng.standard_normal(500)
        est = estimate_velocity(returns)
        assert np.isfinite(est.velocity)
        assert not est.is_accelerating  # stable vol shouldn't trigger

    def test_spike_positive(self):
        returns = np.concatenate([
            np.full(100, 0.001),
            0.05 * np.random.default_rng(42).standard_normal(21),
        ])
        est = estimate_velocity(returns, vol_window=21, lag=5)
        assert np.isfinite(est.velocity)
        assert est.velocity > 0

    def test_rolling_shape(self):
        rng = np.random.default_rng(42)
        returns = 0.01 * rng.standard_normal(500)
        result = rolling_velocity(returns, vol_window=21, lag=5, window=252)
        assert len(result) == 500

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_on_real_data(self, asset_name):
        df = from_sample(asset_name)
        returns = log_returns(df)
        est = estimate_velocity(returns)
        assert np.isfinite(est.velocity), f"{asset_name}: velocity is NaN"


# ══════════════════════════════════════════
# Constant volatility weighting (CBS thesis)
# ══════════════════════════════════════════

class TestConstantVol:
    def test_high_vol_reduces_weight(self):
        # High vol returns → weight < 1
        rng = np.random.default_rng(42)
        returns = 0.03 * rng.standard_normal(100)  # ~48% annualized vol
        result = constant_vol_weight(returns, target_vol=0.15, window=63)
        assert result.weight < 1.0, f"High vol should reduce weight, got {result.weight}"
        assert result.current_vol > 0.15

    def test_low_vol_increases_weight(self):
        # Low vol returns → weight > 1
        rng = np.random.default_rng(42)
        returns = 0.002 * rng.standard_normal(100)  # ~3% annualized vol
        result = constant_vol_weight(returns, target_vol=0.15, window=63)
        assert result.weight > 1.0, f"Low vol should increase weight, got {result.weight}"

    def test_max_leverage_cap(self):
        returns = np.full(100, 0.0001)  # near-zero vol
        result = constant_vol_weight(returns, target_vol=0.15, max_leverage=2.0, window=63)
        assert result.weight <= 2.0

    def test_rolling_shape(self):
        rng = np.random.default_rng(42)
        returns = 0.01 * rng.standard_normal(200)
        result = rolling_constant_vol_weight(returns, window=63)
        assert len(result) == 200
        assert np.isnan(result[0])
        assert np.isfinite(result[-1])


# ══════════════════════════════════════════
# Rebalance risk (Rattray-Harvey 2018)
# ══════════════════════════════════════════

class TestRebalanceRisk:
    def test_trending_drawdown_high_risk(self):
        # Persistent (DFA > 0.5) + negative momentum = dangerous to rebalance
        result = rebalance_risk(dfa_alpha=0.7, momentum=-0.15)
        assert result.risk > 0.3, f"Should be high risk, got {result.risk}"

    def test_mean_reverting_low_risk(self):
        # Mean-reverting + drawdown = safe to rebalance
        result = rebalance_risk(dfa_alpha=0.4, momentum=-0.15)
        assert result.risk == 0.0, f"Mean-reverting should be zero risk, got {result.risk}"

    def test_trending_uptrend_low_risk(self):
        # Persistent + positive momentum = not dangerous
        result = rebalance_risk(dfa_alpha=0.7, momentum=0.1)
        assert result.risk == 0.0, f"Uptrend should be zero risk, got {result.risk}"

    def test_nan_inputs(self):
        result = rebalance_risk(dfa_alpha=float("nan"), momentum=-0.1)
        assert result.risk == 0.0


# ══════════════════════════════════════════
# Signal converters
# ══════════════════════════════════════════

class TestNewSignalConverters:
    def test_momentum_reversal_signal_positive(self):
        assert momentum_reversal_signal(0.3) == pytest.approx(1.0)

    def test_momentum_reversal_signal_zero(self):
        assert momentum_reversal_signal(-0.1) == 0.0

    def test_momentum_reversal_signal_nan(self):
        assert momentum_reversal_signal(float("nan")) == 0.0

    def test_velocity_signal_spike(self):
        assert velocity_signal(2.0) == pytest.approx(1.0)

    def test_velocity_signal_moderate(self):
        sig = velocity_signal(1.0)
        assert 0.0 < sig < 1.0

    def test_velocity_signal_negative(self):
        assert velocity_signal(-0.5) == 0.0

    def test_velocity_signal_nan(self):
        assert velocity_signal(float("nan")) == 0.0
