"""Python wrapper for Rust EVT (GEV/GPD) functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fatcrash._core import gev_fit, gpd_fit, gpd_var_es


@dataclass
class GPDResult:
    sigma: float
    xi: float
    threshold: float
    n_exceedances: int


@dataclass
class GEVResult:
    mu: float
    sigma: float
    xi: float


@dataclass
class RiskMetrics:
    var: float
    es: float


def fit_gpd(
    returns: npt.NDArray[np.float64],
    quantile: float = 0.95,
) -> GPDResult:
    """Fit GPD to the tail of the return distribution."""
    sigma, xi, threshold, n_exc = gpd_fit(returns, quantile=quantile)
    return GPDResult(sigma=sigma, xi=xi, threshold=threshold, n_exceedances=n_exc)


def fit_gev(
    block_maxima: npt.NDArray[np.float64],
) -> GEVResult:
    """Fit GEV to block maxima."""
    mu, sigma, xi = gev_fit(block_maxima)
    return GEVResult(mu=mu, sigma=sigma, xi=xi)


def compute_var_es(
    returns: npt.NDArray[np.float64],
    p: float = 0.99,
    quantile: float = 0.95,
) -> RiskMetrics:
    """Compute VaR and Expected Shortfall from GPD tail fit."""
    var, es = gpd_var_es(returns, p=p, quantile=quantile)
    return RiskMetrics(var=var, es=es)


def rolling_var_es(
    returns: npt.NDArray[np.float64],
    window: int = 252,
    p: float = 0.99,
    quantile: float = 0.95,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rolling VaR and ES computation."""
    n = len(returns)
    var_arr = np.full(n, np.nan)
    es_arr = np.full(n, np.nan)

    for i in range(window, n):
        chunk = returns[i - window : i]
        try:
            var, es = gpd_var_es(chunk, p=p, quantile=quantile)
            var_arr[i] = var
            es_arr[i] = es
        except (ValueError, RuntimeError):
            pass

    return var_arr, es_arr
