"""Tests for LPPLS synthetic data generator."""

import numpy as np
import pytest

from fatcrash.nn.synthetic import (
    SyntheticDataset,
    add_ar1_noise,
    add_white_noise,
    generate_dataset,
    generate_lppls_series,
)


class TestGenerateLPPLSSeries:
    def test_output_shape(self):
        series, params = generate_lppls_series(n_obs=100)
        assert series.shape == (100,)

    def test_default_length(self):
        series, _ = generate_lppls_series()
        assert series.shape == (252,)

    def test_known_params_match_formula(self):
        """Clean series with known params should match LPPLS formula exactly."""
        tc, m, omega = 300.0, 0.5, 8.0
        a, b, c1, c2 = 10.0, -0.3, 0.02, -0.01

        series, params = generate_lppls_series(
            n_obs=252, tc=tc, m=m, omega=omega, a=a, b=b, c1=c1, c2=c2,
        )

        # Verify against manual LPPLS computation
        times = np.arange(252, dtype=np.float64)
        dt = tc - times
        dt_m = np.power(dt, m)
        log_dt = np.log(dt)
        expected = a + b * dt_m + dt_m * (c1 * np.cos(omega * log_dt) + c2 * np.sin(omega * log_dt))

        np.testing.assert_allclose(series, expected, rtol=1e-10)

    def test_params_returned(self):
        _, params = generate_lppls_series()
        assert "tc" in params
        assert "m" in params
        assert "omega" in params
        assert "a" in params
        assert "b" in params
        assert "c1" in params
        assert "c2" in params

    def test_param_ranges(self):
        """Sampled params should be within paper-specified ranges."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            _, params = generate_lppls_series(n_obs=252, rng=rng)
            assert 252 <= params["tc"] <= 302
            assert 0.1 <= params["m"] <= 0.9
            assert 6.0 <= params["omega"] <= 13.0
            assert params["b"] < 0  # Must be negative

    def test_no_nans(self):
        series, _ = generate_lppls_series()
        assert not np.any(np.isnan(series))


class TestNoise:
    def test_white_noise_changes_series(self):
        series, _ = generate_lppls_series(rng=np.random.default_rng(1))
        noisy = add_white_noise(series, alpha=0.1, rng=np.random.default_rng(2))
        assert not np.allclose(series, noisy)

    def test_white_noise_reasonable_amplitude(self):
        series, _ = generate_lppls_series(rng=np.random.default_rng(1))
        s_range = series.max() - series.min()
        noisy = add_white_noise(series, alpha=0.05, rng=np.random.default_rng(2))
        diff = np.abs(noisy - series)
        # Most noise should be within 3*alpha*range
        assert diff.mean() < 3 * 0.05 * s_range

    def test_ar1_noise_changes_series(self):
        series, _ = generate_lppls_series(rng=np.random.default_rng(1))
        noisy = add_ar1_noise(series, amplitude=0.03, rng=np.random.default_rng(2))
        assert not np.allclose(series, noisy)

    def test_ar1_noise_is_correlated(self):
        """AR(1) noise with phi=0.9 should show autocorrelation."""
        # Use a non-trivial series so amplitude scaling works
        series, _ = generate_lppls_series(n_obs=1000, rng=np.random.default_rng(1))
        noisy = add_ar1_noise(series, phi=0.9, amplitude=0.03, rng=np.random.default_rng(42))
        noise = noisy - series
        # Check autocorrelation at lag 1
        autocorr = np.corrcoef(noise[:-1], noise[1:])[0, 1]
        assert autocorr > 0.5  # Should be close to 0.9 for long series


class TestGenerateDataset:
    def test_shapes(self):
        ds = generate_dataset(n_samples=10, n_obs=50, seed=42)
        assert isinstance(ds, SyntheticDataset)
        assert ds.X.shape == (10, 50)
        assert ds.y.shape == (10, 3)
        assert len(ds.params) == 10

    def test_normalization_range(self):
        """Normalized series should be in [0, 1]."""
        ds = generate_dataset(n_samples=20, n_obs=100, seed=42)
        assert ds.X.min() >= -0.01  # Small float tolerance
        assert ds.X.max() <= 1.01

    def test_dtypes(self):
        ds = generate_dataset(n_samples=5, n_obs=50, seed=42)
        assert ds.X.dtype == np.float32
        assert ds.y.dtype == np.float32

    def test_noise_types(self):
        for noise in ("white", "ar1", "both"):
            ds = generate_dataset(n_samples=5, n_obs=50, noise_type=noise, seed=42)
            assert ds.X.shape == (5, 50)

    def test_invalid_noise_type(self):
        with pytest.raises(ValueError, match="Unknown noise_type"):
            generate_dataset(n_samples=5, noise_type="invalid")

    def test_reproducibility(self):
        ds1 = generate_dataset(n_samples=10, n_obs=50, seed=123)
        ds2 = generate_dataset(n_samples=10, n_obs=50, seed=123)
        np.testing.assert_array_equal(ds1.X, ds2.X)
        np.testing.assert_array_equal(ds1.y, ds2.y)
