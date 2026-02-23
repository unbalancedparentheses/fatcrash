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
) -> tuple[float, float, float, float, float, float, float, float]: ...

def lppls_confidence(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    min_window: int | None = 60,
    max_window: int | None = 750,
    n_windows: int | None = 50,
    n_candidates: int | None = 30,
) -> npt.NDArray[np.float64]: ...

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
