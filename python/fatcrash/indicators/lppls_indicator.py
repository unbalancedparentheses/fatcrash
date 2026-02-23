"""Python wrapper for Rust LPPLS functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fatcrash._core import lppls_confidence, lppls_fit


@dataclass
class LPPLSResult:
    tc: float  # Critical time (predicted crash date)
    m: float  # Power law exponent
    omega: float  # Log-periodic frequency
    a: float  # Linear param
    b: float  # Linear param
    c1: float  # Linear param
    c2: float  # Linear param
    rss: float  # Residual sum of squares
    is_bubble: bool  # Passes Sornette filter

    @property
    def c_amplitude(self) -> float:
        return (self.c1**2 + self.c2**2) ** 0.5

    @property
    def days_to_crash(self) -> float:
        """Days from end of fitting window to predicted tc."""
        return self.tc


def fit_lppls(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    tc_range: tuple[float, float] | None = None,
    pop_size: int = 50,
    n_generations: int = 40,
) -> LPPLSResult:
    """Fit LPPLS model to log-price time series."""
    tc, m, omega, a, b, c1, c2, rss = lppls_fit(
        times, log_prices, tc_range=tc_range, pop_size=pop_size, n_generations=n_generations
    )

    # Sornette filter check
    is_bubble = (
        0.1 <= m <= 0.9
        and 2.0 <= omega <= 25.0
        and b < 0.0
    )

    return LPPLSResult(
        tc=tc, m=m, omega=omega, a=a, b=b, c1=c1, c2=c2, rss=rss, is_bubble=is_bubble
    )


def compute_confidence(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    min_window: int = 60,
    max_window: int = 750,
    n_windows: int = 50,
    n_candidates: int = 30,
) -> npt.NDArray[np.float64]:
    """Compute DS LPPLS confidence indicator (rayon-parallelized)."""
    return np.asarray(
        lppls_confidence(
            times,
            log_prices,
            min_window=min_window,
            max_window=max_window,
            n_windows=n_windows,
            n_candidates=n_candidates,
        )
    )
