"""Python wrapper for Rust tail estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fatcrash._core import (
    deh_estimator,
    deh_rolling,
    dfa_exponent,
    dfa_rolling,
    hill_estimator,
    hill_rolling,
    hurst_exponent,
    hurst_rolling,
    kappa_metric,
    kappa_rolling,
    maxsum_ratio,
    maxsum_rolling,
    pickands_estimator,
    pickands_rolling,
    qq_estimator,
    qq_rolling,
    spectral_exponent,
    spectral_rolling,
    taleb_kappa,
    taleb_kappa_rolling,
)


@dataclass
class TailEstimate:
    alpha: float
    is_fat_tail: bool  # alpha < 4 typically indicates fat tails


@dataclass
class KappaEstimate:
    """Max-stability kappa: subsample-max ratio vs Gaussian benchmark."""
    kappa: float
    gaussian_benchmark: float
    is_fat_tail: bool  # kappa < benchmark


@dataclass
class TalebKappaEstimate:
    """Taleb's kappa: MAD convergence rate. 0 = Gaussian, 1 = Cauchy."""
    kappa: float
    gaussian_benchmark: float
    is_fat_tail: bool  # kappa > benchmark


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
    """Rolling max-stability kappa. Returns (kappa_series, gaussian_benchmark)."""
    kappa_arr, benchmark = kappa_rolling(returns, window, n_subsamples=n_subsamples)
    return np.asarray(kappa_arr), benchmark


def estimate_taleb_kappa(
    returns: npt.NDArray[np.float64],
    n0: int = 30,
    n1: int = 100,
) -> TalebKappaEstimate:
    """Estimate Taleb's kappa: MAD convergence rate.

    kappa = 0 → Gaussian, kappa → 1 → Cauchy.
    """
    kappa, benchmark = taleb_kappa(returns, n0=n0, n1=n1)
    return TalebKappaEstimate(
        kappa=kappa,
        gaussian_benchmark=benchmark,
        is_fat_tail=kappa > benchmark,
    )


def rolling_taleb_kappa(
    returns: npt.NDArray[np.float64],
    window: int = 252,
    n0: int = 30,
    n1: int = 100,
) -> tuple[npt.NDArray[np.float64], float]:
    """Rolling Taleb kappa. Returns (kappa_series, gaussian_benchmark)."""
    kappa_arr, benchmark = taleb_kappa_rolling(returns, window, n0=n0, n1=n1)
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


# ── DFA ──────────────────────────────────────────────────────


@dataclass
class DFAEstimate:
    """Detrended Fluctuation Analysis: alpha exponent."""
    alpha: float
    regime: str  # "persistent", "white_noise", or "anti_persistent"


def estimate_dfa(
    data: npt.NDArray[np.float64],
) -> DFAEstimate:
    """Estimate DFA exponent.

    alpha > 0.5: persistent, alpha = 0.5: white noise, alpha < 0.5: anti-persistent.
    """
    alpha = dfa_exponent(data)
    if np.isnan(alpha):
        regime = "unknown"
    elif alpha > 0.55:
        regime = "persistent"
    elif alpha < 0.45:
        regime = "anti_persistent"
    else:
        regime = "white_noise"
    return DFAEstimate(alpha=alpha, regime=regime)


def rolling_dfa(
    data: npt.NDArray[np.float64],
    window: int = 252,
) -> npt.NDArray[np.float64]:
    """Rolling DFA exponent."""
    return np.asarray(dfa_rolling(data, window))


# ── DEH ──────────────────────────────────────────────────────


@dataclass
class DEHEstimate:
    """Dekkers-Einmahl-de Haan moment estimator."""
    gamma: float
    tail_type: str  # "heavy", "light", or "bounded"


def estimate_deh(
    returns: npt.NDArray[np.float64],
    k: int | None = None,
) -> DEHEstimate:
    """Estimate extreme value index gamma via DEH moment estimator.

    Valid for all domains of attraction.
    gamma > 0: heavy (Frechet), gamma = 0: light (Gumbel), gamma < 0: bounded (Weibull).
    """
    gamma = deh_estimator(returns, k=k)
    if np.isnan(gamma):
        tail_type = "unknown"
    elif gamma > 0.05:
        tail_type = "heavy"
    elif gamma < -0.05:
        tail_type = "bounded"
    else:
        tail_type = "light"
    return DEHEstimate(gamma=gamma, tail_type=tail_type)


def rolling_deh(
    returns: npt.NDArray[np.float64],
    window: int = 252,
    k: int | None = None,
) -> npt.NDArray[np.float64]:
    """Rolling DEH estimator."""
    return np.asarray(deh_rolling(returns, window, k=k))


# ── QQ ──────────────────────────────────────────────────────


@dataclass
class QQEstimate:
    """QQ tail index estimator."""
    alpha: float
    is_fat_tail: bool  # alpha < 4


def estimate_qq(
    returns: npt.NDArray[np.float64],
    k: int | None = None,
) -> QQEstimate:
    """Estimate tail index alpha from QQ plot slope."""
    alpha = qq_estimator(returns, k=k)
    return QQEstimate(alpha=alpha, is_fat_tail=alpha < 4.0 if not np.isnan(alpha) else False)


def rolling_qq(
    returns: npt.NDArray[np.float64],
    window: int = 252,
    k: int | None = None,
) -> npt.NDArray[np.float64]:
    """Rolling QQ estimator."""
    return np.asarray(qq_rolling(returns, window, k=k))


# ── Max-to-Sum ──────────────────────────────────────────────


@dataclass
class MaxSumEstimate:
    """Maximum-to-Sum ratio diagnostic."""
    ratio: float
    is_infinite_variance: bool  # high ratio suggests alpha < 2


def estimate_maxsum(
    data: npt.NDArray[np.float64],
) -> MaxSumEstimate:
    """Compute max-to-sum ratio. High ratio = infinite variance hypothesis supported."""
    r = maxsum_ratio(data)
    return MaxSumEstimate(ratio=r, is_infinite_variance=r > 0.05 if not np.isnan(r) else False)


def rolling_maxsum(
    data: npt.NDArray[np.float64],
    window: int = 252,
) -> npt.NDArray[np.float64]:
    """Rolling max-to-sum ratio."""
    return np.asarray(maxsum_rolling(data, window))


# ── Spectral ────────────────────────────────────────────────


@dataclass
class SpectralEstimate:
    """Spectral exponent (long-memory parameter d)."""
    d: float
    regime: str  # "long_memory", "short_memory", or "anti_persistent"


def estimate_spectral(
    data: npt.NDArray[np.float64],
    bandwidth_exp: float = 0.65,
) -> SpectralEstimate:
    """Estimate spectral exponent d via GPH.

    d > 0: long memory, d = 0: short memory, d < 0: anti-persistent.
    """
    d = spectral_exponent(data, bandwidth_exp=bandwidth_exp)
    if np.isnan(d):
        regime = "unknown"
    elif d > 0.1:
        regime = "long_memory"
    elif d < -0.1:
        regime = "anti_persistent"
    else:
        regime = "short_memory"
    return SpectralEstimate(d=d, regime=regime)
