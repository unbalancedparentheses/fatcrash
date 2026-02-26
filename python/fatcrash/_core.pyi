"""Type stubs for the Rust _core module."""

import numpy as np
import numpy.typing as npt

# Utils
def log_returns(prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def log_prices(prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

# Tail estimators
def hill_estimator(
    data: npt.NDArray[np.float64],
    k: int | None = None,
    use_abs: bool | None = True,
) -> float: ...

def hill_rolling(
    data: npt.NDArray[np.float64],
    window: int,
    k: int | None = None,
    use_abs: bool | None = True,
) -> npt.NDArray[np.float64]: ...

def kappa_metric(
    data: npt.NDArray[np.float64],
    n_subsamples: int | None = 10,
    n_sims: int | None = 1000,
) -> tuple[float, float]: ...

def kappa_rolling(
    data: npt.NDArray[np.float64],
    window: int,
    n_subsamples: int | None = 10,
    n_sims: int | None = 200,
) -> tuple[npt.NDArray[np.float64], float]: ...

def taleb_kappa(
    data: npt.NDArray[np.float64],
    n0: int | None = 30,
    n1: int | None = 100,
    n_sims: int | None = 500,
) -> tuple[float, float]: ...

def taleb_kappa_rolling(
    data: npt.NDArray[np.float64],
    window: int,
    n0: int | None = 30,
    n1: int | None = 100,
    n_sims: int | None = 100,
) -> tuple[npt.NDArray[np.float64], float]: ...

def pickands_estimator(
    data: npt.NDArray[np.float64],
    k: int | None = None,
    use_abs: bool | None = True,
) -> float: ...

def pickands_rolling(
    data: npt.NDArray[np.float64],
    window: int,
    k: int | None = None,
    use_abs: bool | None = True,
) -> npt.NDArray[np.float64]: ...

def hurst_exponent(
    data: npt.NDArray[np.float64],
) -> float: ...

def hurst_rolling(
    data: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]: ...

def dfa_exponent(
    data: npt.NDArray[np.float64],
) -> float: ...

def dfa_rolling(
    data: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]: ...

def deh_estimator(
    data: npt.NDArray[np.float64],
    k: int | None = None,
    use_abs: bool | None = True,
) -> float: ...

def deh_rolling(
    data: npt.NDArray[np.float64],
    window: int,
    k: int | None = None,
    use_abs: bool | None = True,
) -> npt.NDArray[np.float64]: ...

def qq_estimator(
    data: npt.NDArray[np.float64],
    k: int | None = None,
    use_abs: bool | None = True,
) -> float: ...

def qq_rolling(
    data: npt.NDArray[np.float64],
    window: int,
    k: int | None = None,
    use_abs: bool | None = True,
) -> npt.NDArray[np.float64]: ...

def maxsum_ratio(
    data: npt.NDArray[np.float64],
) -> float: ...

def maxsum_rolling(
    data: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]: ...

def spectral_exponent(
    data: npt.NDArray[np.float64],
    bandwidth_exp: float | None = 0.65,
) -> float: ...

def spectral_rolling(
    data: npt.NDArray[np.float64],
    window: int,
    bandwidth_exp: float | None = 0.65,
) -> npt.NDArray[np.float64]: ...

# Momentum
def momentum_score(
    prices: npt.NDArray[np.float64],
    lookback: int | None = 252,
) -> float: ...

def momentum_rolling(
    prices: npt.NDArray[np.float64],
    lookback: int | None = 252,
    window: int | None = 504,
) -> npt.NDArray[np.float64]: ...

def momentum_reversal(
    prices: npt.NDArray[np.float64],
    short_lookback: int | None = 21,
    long_lookback: int | None = 252,
) -> float: ...

def momentum_reversal_rolling(
    prices: npt.NDArray[np.float64],
    short_lookback: int | None = 21,
    long_lookback: int | None = 252,
    window: int | None = 504,
) -> npt.NDArray[np.float64]: ...

# Velocity
def price_velocity(
    returns: npt.NDArray[np.float64],
    vol_window: int | None = 21,
    lag: int | None = 5,
) -> float: ...

def price_velocity_rolling(
    returns: npt.NDArray[np.float64],
    vol_window: int | None = 21,
    lag: int | None = 5,
    window: int | None = 252,
) -> npt.NDArray[np.float64]: ...

# EVT
def gpd_fit(
    data: npt.NDArray[np.float64],
    quantile: float | None = 0.95,
) -> tuple[float, float, float, int]: ...

def gpd_var_es(
    data: npt.NDArray[np.float64],
    p: float | None = 0.99,
    quantile: float | None = 0.95,
) -> tuple[float, float]: ...

def gev_fit(
    data: npt.NDArray[np.float64],
) -> tuple[float, float, float]: ...

# LPPLS
def lppls_fit(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    tc_range: tuple[float, float] | None = None,
    pop_size: int | None = 50,
    n_generations: int | None = 40,
    seed: int | None = 42,
) -> tuple[float, float, float, float, float, float, float, float, float]: ...

def lppls_confidence(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    min_window: int | None = 60,
    max_window: int | None = 750,
    n_windows: int | None = 50,
    n_candidates: int | None = 30,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

# Multiscale
def multiscale_signals(
    signals_1d: npt.NDArray[np.float64],
    signals_3d: npt.NDArray[np.float64],
    signals_7d: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...

# Bubble detection
def gsadf_test(
    data: npt.NDArray[np.float64],
    min_window: int | None = None,
    n_sims: int | None = 200,
    seed: int | None = 42,
) -> tuple[float, npt.NDArray[np.float64], tuple[float, float, float]]: ...

def gsadf_rolling(
    data: npt.NDArray[np.float64],
    window: int,
    min_window: int | None = None,
) -> npt.NDArray[np.float64]: ...

# LPPLS solve_linear
def lppls_solve_linear(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    tc: float,
    m: float,
    omega: float,
) -> tuple[float, float, float, float, float] | None: ...
