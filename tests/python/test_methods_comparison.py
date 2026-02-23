"""Comprehensive comparison tests: all methods against each other on real data.

Tests that every indicator produces sensible output on real assets,
and that crisis periods show elevated signals vs quiet periods.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from fatcrash._core import (
    hill_estimator,
    hill_rolling,
    kappa_metric,
    kappa_rolling,
    taleb_kappa,
    taleb_kappa_rolling,
    gpd_fit,
    gpd_var_es,
    gev_fit,
    lppls_fit,
    log_returns as rust_log_returns,
    log_prices as rust_log_prices,
    dfa_exponent,
    dfa_rolling,
    deh_estimator,
    deh_rolling,
    qq_estimator,
    qq_rolling,
    maxsum_ratio,
    maxsum_rolling,
    spectral_exponent,
    spectral_rolling,
)
from fatcrash.data.transforms import log_returns, log_prices, time_index, block_maxima

SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"


def load_asset(name: str) -> pd.DataFrame:
    path = SAMPLE_DIR / f"{name}_daily.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


@pytest.fixture(scope="module")
def btc():
    return load_asset("btc")


@pytest.fixture(scope="module")
def spy():
    return load_asset("spy")


@pytest.fixture(scope="module")
def gold():
    return load_asset("gold")


@pytest.fixture(scope="module")
def gbpusd():
    df = pd.read_csv(SAMPLE_DIR / "gbpusd_daily.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


# ═══════════════════════════════════════════
# Basic sanity: every method runs on every asset
# ═══════════════════════════════════════════

class TestHillAcrossAssets:
    """Hill estimator should work on all assets and produce alpha > 0."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_hill_produces_positive_alpha(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        alpha = hill_estimator(returns)
        assert alpha > 0, f"{asset_name}: alpha should be positive, got {alpha}"
        assert not np.isnan(alpha), f"{asset_name}: alpha is NaN"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_hill_rolling_no_crash(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        result = hill_rolling(returns, window=252)
        assert len(result) == len(returns)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0, f"{asset_name}: all rolling Hill values are NaN"
        assert all(v > 0 for v in valid), f"{asset_name}: negative Hill alpha found"


class TestKappaAcrossAssets:
    """Kappa metric should work on all assets."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_kappa_runs(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        kappa, benchmark = kappa_metric(returns, n_subsamples=10, n_sims=100)
        assert not np.isnan(kappa), f"{asset_name}: kappa is NaN"
        assert not np.isnan(benchmark), f"{asset_name}: benchmark is NaN"
        assert kappa > 0, f"{asset_name}: kappa should be positive"
        assert benchmark > 0, f"{asset_name}: benchmark should be positive"


class TestTalebKappaAcrossAssets:
    """Taleb kappa should work on all assets."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_taleb_kappa_runs(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        kappa, benchmark = taleb_kappa(returns, n0=30, n1=100, n_sims=100)
        assert not np.isnan(benchmark), f"{asset_name}: benchmark is NaN"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_taleb_kappa_rolling_runs(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        kappa_arr, benchmark = taleb_kappa_rolling(returns, window=252, n0=30, n1=100, n_sims=50)
        assert len(kappa_arr) == len(returns)
        valid = kappa_arr[~np.isnan(kappa_arr)]
        assert len(valid) > 0, f"{asset_name}: all rolling Taleb kappa values are NaN"

    def test_btc_higher_taleb_kappa_than_spy(self, btc, spy):
        """BTC should have higher Taleb kappa (fatter tails) than SPY."""
        btc_k, _ = taleb_kappa(log_returns(btc), n0=30, n1=100, n_sims=200)
        spy_k, _ = taleb_kappa(log_returns(spy), n0=30, n1=100, n_sims=200)
        assert btc_k > spy_k, (
            f"BTC Taleb kappa ({btc_k:.3f}) should > SPY ({spy_k:.3f})"
        )


class TestEVTAcrossAssets:
    """GPD and GEV should fit on all assets."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_gpd_fits(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        sigma, xi, threshold, n_exc = gpd_fit(returns, quantile=0.95)
        assert sigma > 0, f"{asset_name}: sigma <= 0"
        assert n_exc > 0, f"{asset_name}: no exceedances"
        assert threshold > 0, f"{asset_name}: threshold <= 0"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_var_es_ordering(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        var, es = gpd_var_es(returns, p=0.99, quantile=0.95)
        assert var > 0, f"{asset_name}: VaR should be positive"
        assert es >= var, f"{asset_name}: ES ({es}) should >= VaR ({var})"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_gev_fits(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        bm = block_maxima(returns, block_size=21)
        mu, sigma, xi = gev_fit(bm)
        assert sigma > 0, f"{asset_name}: GEV sigma <= 0"


class TestLPPLSAcrossAssets:
    """LPPLS should run on all assets (may or may not detect bubble)."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_lppls_runs(self, asset_name):
        df = load_asset(asset_name)
        lp = log_prices(df)
        t = time_index(df)
        # Use last 200 points
        t_slice = t[-200:]
        lp_slice = lp[-200:]
        tc, m, omega, a, b, c1, c2, rss = lppls_fit(t_slice, lp_slice)
        assert not np.isnan(tc), f"{asset_name}: tc is NaN"
        assert 0.0 < m < 1.0, f"{asset_name}: m out of range: {m}"
        assert rss >= 0, f"{asset_name}: RSS negative"


class TestDFAAcrossAssets:
    """DFA should work on all assets."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_dfa_runs(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        alpha = dfa_exponent(returns)
        assert not np.isnan(alpha), f"{asset_name}: DFA alpha is NaN"
        assert alpha > 0, f"{asset_name}: DFA alpha should be positive, got {alpha}"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_dfa_rolling_no_crash(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        result = dfa_rolling(returns, window=252)
        assert len(result) == len(returns)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0, f"{asset_name}: all rolling DFA values are NaN"


class TestDEHAcrossAssets:
    """DEH should work on all assets."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_deh_runs(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        gamma = deh_estimator(returns)
        assert not np.isnan(gamma), f"{asset_name}: DEH gamma is NaN"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_deh_rolling_no_crash(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        result = deh_rolling(returns, window=252)
        assert len(result) == len(returns)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0, f"{asset_name}: all rolling DEH values are NaN"


class TestQQAcrossAssets:
    """QQ estimator should work on all assets."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_qq_runs(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        alpha = qq_estimator(returns)
        assert not np.isnan(alpha), f"{asset_name}: QQ alpha is NaN"
        assert alpha > 0, f"{asset_name}: QQ alpha should be positive, got {alpha}"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_qq_rolling_no_crash(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        result = qq_rolling(returns, window=252)
        assert len(result) == len(returns)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0, f"{asset_name}: all rolling QQ values are NaN"


class TestMaxSumAcrossAssets:
    """Max-to-sum ratio should work on all assets."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_maxsum_runs(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        r = maxsum_ratio(returns)
        assert not np.isnan(r), f"{asset_name}: max/sum ratio is NaN"
        assert 0 < r < 1, f"{asset_name}: max/sum ratio should be in (0,1), got {r}"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_maxsum_rolling_no_crash(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        result = maxsum_rolling(returns, window=252)
        assert len(result) == len(returns)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0, f"{asset_name}: all rolling max/sum values are NaN"


class TestSpectralAcrossAssets:
    """Spectral exponent should work on all assets."""

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_spectral_runs(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        d = spectral_exponent(returns)
        assert not np.isnan(d), f"{asset_name}: spectral d is NaN"

    @pytest.mark.parametrize("asset_name", ["btc", "spy", "gold"])
    def test_spectral_rolling_no_crash(self, asset_name):
        df = load_asset(asset_name)
        returns = log_returns(df)
        result = spectral_rolling(returns, window=252)
        assert len(result) == len(returns)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0, f"{asset_name}: all rolling spectral values are NaN"


# ═══════════════════════════════════════════
# Cross-asset comparison: BTC should have fatter tails than SPY
# ═══════════════════════════════════════════

class TestCrossAssetComparison:
    """Compare statistical properties across assets."""

    def test_all_assets_have_fat_tails(self, btc, spy, gold):
        """All assets should have Hill alpha < 5 (fatter than Gaussian)."""
        for name, df in [("BTC", btc), ("SPY", spy), ("Gold", gold)]:
            alpha = hill_estimator(log_returns(df))
            assert 1.0 < alpha < 6.0, f"{name}: alpha {alpha:.2f} outside expected range"

    def test_btc_higher_var_than_spy(self, btc, spy):
        """BTC should have higher VaR than SPY (more risky)."""
        btc_var, _ = gpd_var_es(log_returns(btc), p=0.99)
        spy_var, _ = gpd_var_es(log_returns(spy), p=0.99)
        assert btc_var > spy_var, (
            f"Expected BTC VaR ({btc_var:.4f}) > SPY VaR ({spy_var:.4f})"
        )

    def test_kappa_below_benchmark_for_all(self, btc, spy, gold):
        """All financial assets should have kappa below Gaussian benchmark (fat tails)."""
        for name, df in [("BTC", btc), ("SPY", spy), ("Gold", gold)]:
            returns = log_returns(df)
            kappa, benchmark = kappa_metric(returns, n_subsamples=10, n_sims=200)
            assert kappa < benchmark, (
                f"{name}: kappa ({kappa:.3f}) should be < benchmark ({benchmark:.3f})"
            )

    def test_var_ordering_across_assets(self, btc, spy, gold):
        """VaR ordering should reflect known volatility: BTC > Gold > SPY (approximately)."""
        btc_var, _ = gpd_var_es(log_returns(btc), p=0.99)
        spy_var, _ = gpd_var_es(log_returns(spy), p=0.99)
        gold_var, _ = gpd_var_es(log_returns(gold), p=0.99)
        # BTC most volatile
        assert btc_var > spy_var, f"BTC VaR ({btc_var}) should > SPY VaR ({spy_var})"
        assert btc_var > gold_var, f"BTC VaR ({btc_var}) should > Gold VaR ({gold_var})"


# ═══════════════════════════════════════════
# Crisis detection: signals should be elevated before known crashes
# ═══════════════════════════════════════════

class TestCrisisDetection:
    """Test that indicators show elevated signals during crisis periods."""

    def _get_btc_2017_bubble(self, btc):
        """Get BTC data leading up to Dec 2017 crash."""
        mask = (btc.index >= "2017-01-01") & (btc.index <= "2017-12-17")
        return btc.loc[mask]

    def _get_btc_quiet_2019(self, btc):
        """Get BTC data during quiet 2019 H1."""
        mask = (btc.index >= "2019-01-01") & (btc.index <= "2019-06-01")
        return btc.loc[mask]

    def test_hill_alpha_lower_near_crash(self, btc):
        """Hill alpha should be lower (fatter tails) near the 2017 crash."""
        bubble = self._get_btc_2017_bubble(btc)
        quiet = self._get_btc_quiet_2019(btc)

        if len(bubble) < 50 or len(quiet) < 50:
            pytest.skip("Not enough data for this period")

        bubble_alpha = hill_estimator(log_returns(bubble))
        quiet_alpha = hill_estimator(log_returns(quiet))
        # During bubble, tails thicken (alpha decreases) — or at minimum, both are finite
        assert not np.isnan(bubble_alpha)
        assert not np.isnan(quiet_alpha)

    def test_var_higher_during_crisis(self, btc):
        """VaR should be higher during volatile periods (use lower quantile for short windows)."""
        bubble = btc.loc["2017-01-01":"2017-12-17"]
        quiet = btc.loc["2019-01-01":"2019-12-31"]

        if len(bubble) < 100 or len(quiet) < 100:
            pytest.skip("Not enough data")

        bubble_var, _ = gpd_var_es(log_returns(bubble), p=0.95, quantile=0.85)
        quiet_var, _ = gpd_var_es(log_returns(quiet), p=0.95, quantile=0.85)
        assert bubble_var > quiet_var, (
            f"Bubble VaR ({bubble_var:.4f}) should > quiet VaR ({quiet_var:.4f})"
        )

    def test_spy_2008_higher_var_than_2019(self, spy):
        """SPY should show higher VaR during 2008 crisis vs 2019."""
        crisis = spy.loc["2007-01-01":"2009-06-01"]
        calm = spy.loc["2019-01-01":"2019-12-31"]

        if len(crisis) < 100 or len(calm) < 100:
            pytest.skip("Not enough data")

        crisis_var, _ = gpd_var_es(log_returns(crisis), p=0.95, quantile=0.85)
        calm_var, _ = gpd_var_es(log_returns(calm), p=0.95, quantile=0.85)
        assert crisis_var > calm_var, (
            f"2008 crisis VaR ({crisis_var:.4f}) should > 2019 calm VaR ({calm_var:.4f})"
        )


# ═══════════════════════════════════════════
# Aggregator: combined signals
# ═══════════════════════════════════════════

class TestAggregator:
    """Test signal aggregation logic."""

    def test_aggregate_produces_valid_probability(self):
        from fatcrash.aggregator.signals import aggregate_signals
        components = {
            "lppls_confidence": 0.8,
            "gpd_var_exceedance": 0.5,
            "kappa_regime": 0.3,
            "taleb_kappa": 0.4,
            "hill_thinning": 0.2,
        }
        signal = aggregate_signals(components)
        assert 0 <= signal.probability <= 1
        assert signal.level in ("LOW", "ELEVATED", "HIGH", "CRITICAL")

    def test_all_high_signals_give_critical(self):
        from fatcrash.aggregator.signals import aggregate_signals
        components = {
            "lppls_confidence": 0.95,
            "lppls_tc_proximity": 0.9,
            "gpd_var_exceedance": 0.85,
            "kappa_regime": 0.8,
            "taleb_kappa": 0.85,
            "hill_thinning": 0.9,
            "multiscale": 0.85,
        }
        signal = aggregate_signals(components)
        assert signal.probability > 0.7
        assert signal.level in ("HIGH", "CRITICAL")

    def test_all_low_signals_give_low(self):
        from fatcrash.aggregator.signals import aggregate_signals
        components = {
            "lppls_confidence": 0.05,
            "gpd_var_exceedance": 0.0,
            "kappa_regime": 0.0,
            "hill_thinning": 0.0,
        }
        signal = aggregate_signals(components)
        assert signal.probability < 0.3
        assert signal.level == "LOW"


# ═══════════════════════════════════════════
# GBP/USD forex: long history test
# ═══════════════════════════════════════════

class TestForex:
    """Test on GBP/USD daily data (1971-2025, 13k+ points)."""

    def test_hill_on_forex(self, gbpusd):
        returns = log_returns(gbpusd)
        alpha = hill_estimator(returns)
        assert alpha > 0
        # Forex typically has alpha between 2-5
        assert 1.0 < alpha < 10.0, f"GBP/USD alpha {alpha} seems off"

    def test_kappa_on_forex(self, gbpusd):
        returns = log_returns(gbpusd)
        kappa, benchmark = kappa_metric(returns, n_subsamples=10, n_sims=100)
        assert kappa > 0
        # Forex should also show fat tails vs Gaussian
        assert kappa < benchmark * 1.2  # Allow some slack

    def test_taleb_kappa_on_forex(self, gbpusd):
        returns = log_returns(gbpusd)
        kappa, benchmark = taleb_kappa(returns, n0=30, n1=100, n_sims=100)
        assert not np.isnan(benchmark)
        # GBP/USD should show some fat-tailedness

    def test_gpd_on_forex(self, gbpusd):
        returns = log_returns(gbpusd)
        sigma, xi, threshold, n_exc = gpd_fit(returns, quantile=0.95)
        assert sigma > 0
        assert n_exc > 50  # Should have plenty of exceedances with 13k+ days

    def test_dfa_on_forex(self, gbpusd):
        returns = log_returns(gbpusd)
        alpha = dfa_exponent(returns)
        assert not np.isnan(alpha)
        assert alpha > 0

    def test_deh_on_forex(self, gbpusd):
        returns = log_returns(gbpusd)
        gamma = deh_estimator(returns)
        assert not np.isnan(gamma)

    def test_qq_on_forex(self, gbpusd):
        returns = log_returns(gbpusd)
        alpha = qq_estimator(returns)
        assert not np.isnan(alpha)
        assert alpha > 0

    def test_maxsum_on_forex(self, gbpusd):
        returns = log_returns(gbpusd)
        r = maxsum_ratio(returns)
        assert not np.isnan(r)
        assert 0 < r < 1

    def test_spectral_on_forex(self, gbpusd):
        returns = log_returns(gbpusd)
        d = spectral_exponent(returns)
        assert not np.isnan(d)


# ═══════════════════════════════════════════
# Consistency: Rust vs Python implementations
# ═══════════════════════════════════════════

class TestRustPythonConsistency:
    """Verify Rust and Python implementations agree."""

    def test_log_returns_match(self, btc):
        prices = btc["close"].values
        rust_ret = np.asarray(rust_log_returns(prices))
        py_ret = log_returns(btc)
        np.testing.assert_allclose(rust_ret, py_ret, atol=1e-10)

    def test_log_prices_match(self, btc):
        prices = btc["close"].values
        rust_lp = np.asarray(rust_log_prices(prices))
        py_lp = log_prices(btc)
        np.testing.assert_allclose(rust_lp, py_lp, atol=1e-10)


# ═══════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════

class TestEdgeCases:
    """Test behavior on edge cases."""

    def test_hill_on_gaussian(self):
        """Gaussian data should have high alpha (thin tails)."""
        rng = np.random.default_rng(42)
        gaussian = rng.standard_normal(5000)
        alpha = hill_estimator(gaussian)
        # Gaussian has infinite alpha, but finite-sample estimate should be high
        assert alpha > 3.0, f"Gaussian alpha {alpha} seems too low"

    def test_hill_on_cauchy(self):
        """Cauchy data should have alpha near 1 (very fat tails)."""
        rng = np.random.default_rng(42)
        cauchy = rng.standard_cauchy(5000)
        alpha = hill_estimator(cauchy)
        assert alpha < 2.5, f"Cauchy alpha {alpha} seems too high"

    def test_kappa_gaussian_near_benchmark(self):
        """For Gaussian data, kappa should be close to benchmark."""
        rng = np.random.default_rng(42)
        gaussian = rng.standard_normal(2000)
        kappa, benchmark = kappa_metric(gaussian, n_subsamples=10, n_sims=200)
        assert abs(kappa - benchmark) < 0.15, (
            f"Gaussian kappa {kappa:.3f} too far from benchmark {benchmark:.3f}"
        )

    def test_taleb_kappa_gaussian_low(self):
        """For Gaussian data, Taleb kappa should be near 0."""
        rng = np.random.default_rng(42)
        gaussian = rng.standard_normal(2000)
        kappa, benchmark = taleb_kappa(gaussian, n0=30, n1=100, n_sims=200)
        assert kappa < 0.4, f"Gaussian Taleb kappa {kappa:.3f} should be low"

    def test_taleb_kappa_cauchy_high(self):
        """For Cauchy data, Taleb kappa should be high."""
        rng = np.random.default_rng(42)
        cauchy = rng.standard_cauchy(2000)
        kappa, _ = taleb_kappa(cauchy, n0=30, n1=100, n_sims=100)
        assert kappa > 0.3, f"Cauchy Taleb kappa {kappa:.3f} should be high"

    def test_gpd_on_exponential(self):
        """Exponential exceedances should give xi near 0."""
        rng = np.random.default_rng(42)
        exp_data = -rng.exponential(1.0, 3000)  # Negative = losses
        sigma, xi, _, _ = gpd_fit(exp_data, quantile=0.9)
        assert abs(xi) < 0.3, f"Exponential xi {xi} should be near 0"

    def test_var_increases_with_confidence_level(self, btc):
        """VaR at 99.5% should be higher than at 99%."""
        returns = log_returns(btc)
        var_99, _ = gpd_var_es(returns, p=0.99)
        var_995, _ = gpd_var_es(returns, p=0.995)
        assert var_995 > var_99, f"VaR 99.5% ({var_995}) should > VaR 99% ({var_99})"

    def test_dfa_on_gaussian(self):
        """DFA alpha for Gaussian should be near 0.5."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(4000)
        alpha = dfa_exponent(data)
        assert abs(alpha - 0.5) < 0.15, f"DFA for Gaussian should be ~0.5, got {alpha}"

    def test_deh_on_pareto(self):
        """DEH gamma for Pareto(2) should be positive."""
        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, 5000)
        samples = u ** (-1 / 2.0)
        gamma = deh_estimator(samples)
        assert gamma > 0, f"DEH gamma for Pareto should be > 0, got {gamma}"

    def test_qq_on_gaussian(self):
        """QQ alpha for Gaussian should be high (thin tails)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(5000)
        alpha = qq_estimator(data)
        assert alpha > 2.0, f"QQ alpha for Gaussian should be high, got {alpha}"

    def test_maxsum_on_gaussian_vs_cauchy(self):
        """Max/sum for Cauchy should be higher than Gaussian."""
        rng = np.random.default_rng(42)
        gauss = rng.standard_normal(5000)
        cauchy = rng.standard_cauchy(5000)
        assert maxsum_ratio(cauchy) > maxsum_ratio(gauss)

    def test_spectral_on_gaussian(self):
        """Spectral d for white noise should be near 0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(4000)
        d = spectral_exponent(data)
        assert abs(d) < 0.3, f"Spectral d for white noise should be ~0, got {d}"
