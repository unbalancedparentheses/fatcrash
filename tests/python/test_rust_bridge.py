"""Tests for Rust bridge functions."""

import numpy as np
import pytest

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
    log_returns,
    log_prices,
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
    momentum_score,
    momentum_rolling,
    momentum_reversal,
    momentum_reversal_rolling,
    price_velocity,
    price_velocity_rolling,
)


class TestUtils:
    def test_log_returns(self):
        prices = np.array([100.0, 110.0, 105.0])
        ret = log_returns(prices)
        assert len(ret) == 2
        assert abs(ret[0] - np.log(1.1)) < 1e-10

    def test_log_prices(self):
        prices = np.array([100.0, 200.0])
        lp = log_prices(prices)
        assert abs(lp[0] - np.log(100)) < 1e-10


class TestHill:
    def test_pareto_tail(self):
        rng = np.random.default_rng(42)
        # Pareto with alpha=3
        u = rng.uniform(0, 1, 5000)
        samples = u ** (-1 / 3.0)
        alpha = hill_estimator(samples, k=70)
        assert abs(alpha - 3.0) < 1.0  # Rough check

    def test_rolling(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        result = hill_rolling(data, window=100)
        assert len(result) == 500
        assert np.isnan(result[0])
        assert not np.isnan(result[499])


class TestKappa:
    def test_gaussian_near_benchmark(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000)
        kappa, benchmark = kappa_metric(data, n_subsamples=10, n_sims=200)
        assert abs(kappa - benchmark) < 0.2

    def test_rolling(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        kappa_arr, benchmark = kappa_rolling(data, window=200, n_subsamples=5, n_sims=50)
        assert len(kappa_arr) == 500


class TestTalebKappa:
    def test_gaussian_near_zero(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000)
        kappa, benchmark = taleb_kappa(data, n0=30, n1=100, n_sims=200)
        assert kappa < 0.4, f"Gaussian taleb_kappa {kappa} should be near 0"
        assert not np.isnan(benchmark)

    def test_cauchy_higher_than_gaussian(self):
        rng = np.random.default_rng(77)
        gauss = rng.standard_normal(3000)
        cauchy = rng.standard_cauchy(3000)
        k_gauss, _ = taleb_kappa(gauss, n0=30, n1=150, n_sims=100)
        k_cauchy, _ = taleb_kappa(cauchy, n0=30, n1=150, n_sims=100)
        assert k_cauchy > k_gauss, (
            f"Cauchy kappa ({k_cauchy}) should > Gaussian kappa ({k_gauss})"
        )

    def test_rolling(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        kappa_arr, benchmark = taleb_kappa_rolling(data, window=250, n0=15, n1=50, n_sims=50)
        assert len(kappa_arr) == 500
        assert not np.isnan(benchmark)
        # First 249 should be NaN, rest should have values
        assert np.isnan(kappa_arr[0])
        valid = kappa_arr[~np.isnan(kappa_arr)]
        assert len(valid) > 0


class TestEVT:
    def test_gpd_fit(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000)
        sigma, xi, threshold, n_exc = gpd_fit(data, quantile=0.9)
        assert sigma > 0
        assert n_exc > 0

    def test_gpd_var_es(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000)
        var, es = gpd_var_es(data, p=0.99, quantile=0.9)
        assert var > 0
        assert es >= var

    def test_gev_fit(self):
        rng = np.random.default_rng(42)
        # Generate Gumbel samples
        data = -np.log(-np.log(rng.uniform(0, 1, 500)))
        mu, sigma, xi = gev_fit(data)
        assert sigma > 0


class TestDFA:
    def test_white_noise(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000)
        alpha = dfa_exponent(data)
        assert abs(alpha - 0.5) < 0.2, f"DFA for white noise should be ~0.5, got {alpha}"

    def test_persistent(self):
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.standard_normal(2000))
        alpha = dfa_exponent(data)
        assert alpha > 0.8, f"DFA for random walk should be > 0.8, got {alpha}"

    def test_rolling(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        result = dfa_rolling(data, window=100)
        assert len(result) == 500
        assert np.isnan(result[0])
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestDEH:
    def test_pareto(self):
        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, 5000)
        samples = u ** (-1 / 2.0)  # Pareto alpha=2 → gamma=0.5
        gamma = deh_estimator(samples, k=70)
        assert gamma > 0, f"DEH gamma should be positive for Pareto, got {gamma}"

    def test_rolling(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        result = deh_rolling(data, window=100)
        assert len(result) == 500
        assert np.isnan(result[0])
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestQQ:
    def test_pareto(self):
        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, 5000)
        samples = u ** (-1 / 3.0)  # Pareto alpha=3
        alpha = qq_estimator(samples, k=70)
        assert 1.0 < alpha < 6.0, f"QQ alpha for Pareto(3) should be near 3, got {alpha}"

    def test_rolling(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        result = qq_rolling(data, window=100)
        assert len(result) == 500
        assert np.isnan(result[0])
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestMaxSum:
    def test_gaussian_low_ratio(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(5000)
        r = maxsum_ratio(data)
        assert r < 0.05, f"Gaussian max/sum should be small, got {r}"

    def test_cauchy_higher_than_gaussian(self):
        rng = np.random.default_rng(42)
        gauss = rng.standard_normal(5000)
        cauchy = rng.standard_cauchy(5000)
        r_gauss = maxsum_ratio(gauss)
        r_cauchy = maxsum_ratio(cauchy)
        assert r_cauchy > r_gauss, (
            f"Cauchy ratio ({r_cauchy}) should > Gaussian ratio ({r_gauss})"
        )

    def test_rolling(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        result = maxsum_rolling(data, window=100)
        assert len(result) == 500
        assert np.isnan(result[0])
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestSpectral:
    def test_white_noise(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000)
        d = spectral_exponent(data)
        assert abs(d) < 0.3, f"Spectral d for white noise should be ~0, got {d}"

    def test_persistent(self):
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.standard_normal(2000))
        d = spectral_exponent(data)
        assert d > 0.3, f"Spectral d for random walk should be > 0.3, got {d}"

    def test_rolling(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        result = spectral_rolling(data, window=100)
        assert len(result) == 500
        assert np.isnan(result[0])
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestLPPLS:
    def test_synthetic_bubble(self):
        # Generate synthetic LPPLS data
        tc = 100.0
        m = 0.5
        omega = 6.0
        a = 10.0
        b = -0.5

        times = np.arange(0, 80, dtype=np.float64)
        dt = tc - times
        log_p = a + b * dt ** m

        result = lppls_fit(times, log_p, tc_range=(80.0, 120.0))
        assert len(result) == 9
        tc_est = result[0]
        r2 = result[8]
        # tc should be in reasonable range
        assert 70 < tc_est < 130
        # R² should be between 0 and 1
        assert 0.0 <= r2 <= 1.0


class TestMomentum:
    """Tests for momentum score and reversal functions."""

    def test_momentum_uptrend(self):
        prices = np.array([100 * np.exp(0.001 * i) for i in range(300)])
        mom = momentum_score(prices, lookback=252)
        assert np.isfinite(mom)
        assert mom > 0, f"Uptrend should have positive momentum, got {mom}"

    def test_momentum_downtrend(self):
        prices = np.array([100 * np.exp(-0.001 * i) for i in range(300)])
        mom = momentum_score(prices, lookback=252)
        assert np.isfinite(mom)
        assert mom < 0, f"Downtrend should have negative momentum, got {mom}"

    def test_momentum_short_data(self):
        prices = np.array([100.0, 101.0, 102.0])
        mom = momentum_score(prices, lookback=10)
        assert np.isnan(mom)

    def test_momentum_rolling(self):
        prices = np.array([100 * np.exp(0.001 * i) for i in range(600)])
        result = momentum_rolling(prices, lookback=252, window=300)
        assert len(result) == len(prices)
        # First 299 values should be NaN, rest should be finite
        assert np.isnan(result[0])
        assert np.isfinite(result[-1])
        assert result[-1] > 0

    def test_reversal_signal(self):
        # Uptrend then reversal
        prices = np.array(
            [100 * np.exp(0.002 * i) for i in range(250)]
            + [100 * np.exp(0.002 * 250) * np.exp(-0.01 * i) for i in range(1, 51)]
        )
        rev = momentum_reversal(prices, short_lookback=21, long_lookback=200)
        assert np.isfinite(rev)
        assert rev > 0, f"Reversal should be positive during unwind, got {rev}"

    def test_reversal_rolling(self):
        prices = np.array([100 * np.exp(0.001 * i) for i in range(600)])
        result = momentum_reversal_rolling(
            prices, short_lookback=21, long_lookback=252, window=300)
        assert len(result) == len(prices)
        assert np.isnan(result[0])
        assert np.isfinite(result[-1])


class TestVelocity:
    """Tests for price velocity (cascade detection)."""

    def test_stable_vol(self):
        rng = np.random.default_rng(42)
        returns = 0.01 * rng.standard_normal(500)
        vel = price_velocity(returns, vol_window=21, lag=5)
        assert np.isfinite(vel)
        # Stable vol: velocity should be moderate
        assert abs(vel) < 3.0, f"Stable vol should have moderate velocity, got {vel}"

    def test_vol_spike(self):
        # Low vol then high vol
        returns = np.concatenate([
            np.full(100, 0.001),
            0.05 * np.random.default_rng(42).standard_normal(21),
        ])
        vel = price_velocity(returns, vol_window=21, lag=5)
        assert np.isfinite(vel)
        assert vel > 0, f"Velocity should be positive during vol spike, got {vel}"

    def test_insufficient_data(self):
        returns = np.array([0.01] * 10)
        vel = price_velocity(returns, vol_window=21, lag=5)
        assert np.isnan(vel)

    def test_velocity_rolling(self):
        rng = np.random.default_rng(42)
        returns = 0.01 * rng.standard_normal(500)
        result = price_velocity_rolling(returns, vol_window=21, lag=5, window=252)
        assert len(result) == len(returns)
        assert np.isnan(result[0])
        assert np.isfinite(result[-1])
