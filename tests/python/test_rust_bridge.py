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
        assert len(result) == 8
        tc_est = result[0]
        # tc should be in reasonable range
        assert 70 < tc_est < 130
