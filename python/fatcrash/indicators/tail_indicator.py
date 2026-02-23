"""Python wrapper for Rust Hill/Kappa/Pickands/Hurst tail estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fatcrash._core import (
    hill_estimator,
    hill_rolling,
    hurst_exponent,
    hurst_rolling,
    kappa_metric,
    kappa_rolling,
    pickands_estimator,
    pickands_rolling,
)


@dataclass
class TailEstimate:
    alpha: float
    is_fat_tail: bool  # alpha < 4 typically indicates fat tails


@dataclass
class KappaEstimate:
    kappa: float
    gaussian_benchmark: float
    is_fat_tail: bool  # kappa < benchmark


@dataclass
class PickandsEstimate:
    gamma: float
    tail_type: str  # "heavy", "light", or "bounded"


@dataclass
class HurstEstimate:
    h: float
    regime: str  # "trending", "random_walk", or "mean_reverting"


def estimate_tail_index(
    returns: npt.NDArray[np.float64],
    k: int | None = None,
) -> TailEstimate:
    """Estimate tail index alpha via Hill estimator."""
    alpha = hill_estimator(returns, k=k)
    return TailEstimate(alpha=alpha, is_fat_tail=alpha < 4.0)


def rolling_tail_index(
    returns: npt.NDArray[np.float64],
    window: int = 252,
    k: int | None = None,
) -> npt.NDArray[np.float64]:
    """Rolling Hill estimator."""
    return np.asarray(hill_rolling(returns, window, k=k))


def estimate_kappa(
    returns: npt.NDArray[np.float64],
    n_subsamples: int = 10,
) -> KappaEstimate:
    """Estimate Taleb's kappa metric."""
    kappa, benchmark = kappa_metric(returns, n_subsamples=n_subsamples)
    return KappaEstimate(
        kappa=kappa,
        gaussian_benchmark=benchmark,
        is_fat_tail=kappa < benchmark,
    )


def rolling_kappa(
    returns: npt.NDArray[np.float64],
    window: int = 252,
    n_subsamples: int = 10,
) -> tuple[npt.NDArray[np.float64], float]:
    """Rolling kappa metric. Returns (kappa_series, gaussian_benchmark)."""
    kappa_arr, benchmark = kappa_rolling(returns, window, n_subsamples=n_subsamples)
    return np.asarray(kappa_arr), benchmark


def estimate_pickands(
    returns: npt.NDArray[np.float64],
    k: int | None = None,
) -> PickandsEstimate:
    """Estimate tail index gamma via Pickands estimator.

    More robust than Hill -- works for all tail types.
    gamma > 0: heavy tails (Frechet), gamma = 0: light (Gumbel), gamma < 0: bounded (Weibull).
    """
    gamma = pickands_estimator(returns, k=k)
    if np.isnan(gamma):
        tail_type = "unknown"
    elif gamma > 0.05:
        tail_type = "heavy"
    elif gamma < -0.05:
        tail_type = "bounded"
    else:
        tail_type = "light"
    return PickandsEstimate(gamma=gamma, tail_type=tail_type)


def rolling_pickands(
    returns: npt.NDArray[np.float64],
    window: int = 252,
    k: int | None = None,
) -> npt.NDArray[np.float64]:
    """Rolling Pickands estimator."""
    return np.asarray(pickands_rolling(returns, window, k=k))


def estimate_hurst(
    data: npt.NDArray[np.float64],
) -> HurstEstimate:
    """Estimate Hurst exponent via R/S analysis.

    H > 0.5: trending (persistent), H = 0.5: random walk, H < 0.5: mean-reverting.
    """
    h = hurst_exponent(data)
    if np.isnan(h):
        regime = "unknown"
    elif h > 0.55:
        regime = "trending"
    elif h < 0.45:
        regime = "mean_reverting"
    else:
        regime = "random_walk"
    return HurstEstimate(h=h, regime=regime)


def rolling_hurst(
    data: npt.NDArray[np.float64],
    window: int = 252,
) -> npt.NDArray[np.float64]:
    """Rolling Hurst exponent."""
    return np.asarray(hurst_rolling(data, window))
