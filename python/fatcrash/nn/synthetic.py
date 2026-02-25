"""Synthetic LPPLS data generator for P-LNN training.

Generates clean and noisy LPPLS time series with known parameters,
following the protocol in Nielsen, Sornette, Raissi (2024) Section 3.2.

Reference: arXiv:2405.12803

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class SyntheticDataset:
    """Dataset of synthetic LPPLS series for training."""

    X: npt.NDArray[np.float32]  # (n_samples, n_obs) normalized prices
    y: npt.NDArray[np.float32]  # (n_samples, 3) targets: [tc_norm, m, omega]
    params: list[dict]  # Full parameter dicts for each sample


def generate_lppls_series(
    n_obs: int = 252,
    tc: float | None = None,
    m: float | None = None,
    omega: float | None = None,
    a: float | None = None,
    b: float | None = None,
    c1: float | None = None,
    c2: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], dict]:
    """Generate a single clean LPPLS log-price series.

    Parameters that are None are sampled uniformly from paper-specified ranges.

    Returns:
        (series, params_dict) where series has shape (n_obs,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample parameters from paper ranges
    if tc is None:
        tc = rng.uniform(n_obs, n_obs + 50)
    if m is None:
        m = rng.uniform(0.1, 0.9)
    if omega is None:
        omega = rng.uniform(6.0, 13.0)
    if a is None:
        a = rng.uniform(5.0, 12.0)
    if b is None:
        b = rng.uniform(-1.0, -0.01)  # Must be negative
    if c1 is None:
        c1 = rng.uniform(-0.05, 0.05)
    if c2 is None:
        c2 = rng.uniform(-0.05, 0.05)

    times = np.arange(n_obs, dtype=np.float64)
    dt = tc - times
    dt = np.clip(dt, 1e-6, None)
    dt_m = np.power(dt, m)
    log_dt = np.log(dt)

    series = a + b * dt_m + dt_m * (c1 * np.cos(omega * log_dt) + c2 * np.sin(omega * log_dt))

    params = dict(tc=tc, m=m, omega=omega, a=a, b=b, c1=c1, c2=c2)
    return series, params


def add_white_noise(
    series: npt.NDArray[np.float64],
    alpha: float | None = None,
    alpha_range: tuple[float, float] = (0.01, 0.15),
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Add Gaussian white noise scaled as a fraction of series range.

    Args:
        series: clean LPPLS series
        alpha: noise fraction (if None, sampled from alpha_range)
        alpha_range: (min, max) for random alpha sampling
        rng: random generator
    """
    if rng is None:
        rng = np.random.default_rng()
    if alpha is None:
        alpha = rng.uniform(*alpha_range)

    amplitude = alpha * (series.max() - series.min())
    noise = rng.normal(0.0, amplitude, size=series.shape)
    return series + noise


def add_ar1_noise(
    series: npt.NDArray[np.float64],
    phi: float = 0.9,
    amplitude: float | None = None,
    amplitude_range: tuple[float, float] = (0.01, 0.05),
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Add AR(1) correlated noise to LPPLS series.

    Args:
        series: clean LPPLS series
        phi: autoregressive coefficient (default 0.9 per paper)
        amplitude: noise amplitude fraction (if None, sampled from range)
        amplitude_range: (min, max) for random amplitude sampling
        rng: random generator
    """
    if rng is None:
        rng = np.random.default_rng()
    if amplitude is None:
        amplitude = rng.uniform(*amplitude_range)

    n = len(series)
    scale = amplitude * (series.max() - series.min())
    innovations = rng.normal(0.0, scale, size=n)

    ar_noise = np.zeros(n)
    ar_noise[0] = innovations[0]
    for i in range(1, n):
        ar_noise[i] = phi * ar_noise[i - 1] + innovations[i]

    return series + ar_noise


def _min_max_normalize_np(series: npt.NDArray) -> npt.NDArray:
    """Per-series [0,1] normalization."""
    s_min = series.min()
    s_max = series.max()
    denom = s_max - s_min
    if denom < 1e-10:
        return np.zeros_like(series)
    return (series - s_min) / denom


def generate_dataset(
    n_samples: int = 100_000,
    n_obs: int = 252,
    noise_type: str = "white",
    seed: int = 42,
) -> SyntheticDataset:
    """Generate a full training dataset of synthetic LPPLS series.

    Args:
        n_samples: number of series to generate
        n_obs: observations per series (default 252 = 1 trading year)
        noise_type: "white", "ar1", or "both"
        seed: random seed for reproducibility

    Returns:
        SyntheticDataset with normalized X and parameter targets y
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, n_obs), dtype=np.float32)
    y = np.zeros((n_samples, 3), dtype=np.float32)
    params_list = []

    for i in range(n_samples):
        series, params = generate_lppls_series(n_obs=n_obs, rng=rng)

        # Add noise
        if noise_type == "white":
            noisy = add_white_noise(series, rng=rng)
        elif noise_type == "ar1":
            noisy = add_ar1_noise(series, rng=rng)
        elif noise_type == "both":
            noisy = add_white_noise(series, rng=rng)
            noisy = add_ar1_noise(noisy, rng=rng)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}. Use 'white', 'ar1', or 'both'.")

        # Normalize to [0,1]
        normalized = _min_max_normalize_np(noisy)
        X[i] = normalized.astype(np.float32)

        # Targets: tc normalized to [0,1] range relative to series length
        tc_norm = (params["tc"] - 0) / (n_obs + 50)  # Normalize to approx [0, 1]
        y[i] = [tc_norm, params["m"], params["omega"]]
        params_list.append(params)

    return SyntheticDataset(X=X, y=y, params=params_list)
