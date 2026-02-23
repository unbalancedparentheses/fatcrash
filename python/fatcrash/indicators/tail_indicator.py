"""Python wrapper for Rust Hill/Kappa tail estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fatcrash._core import hill_estimator, hill_rolling, kappa_metric, kappa_rolling


@dataclass
class TailEstimate:
    alpha: float
    is_fat_tail: bool  # alpha < 4 typically indicates fat tails


@dataclass
class KappaEstimate:
    kappa: float
    gaussian_benchmark: float
    is_fat_tail: bool  # kappa < benchmark


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
