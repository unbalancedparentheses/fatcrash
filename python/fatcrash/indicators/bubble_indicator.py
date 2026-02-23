"""Python wrapper for Rust GSADF bubble detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fatcrash._core import gsadf_rolling, gsadf_test


@dataclass
class CriticalValues:
    cv_90: float
    cv_95: float
    cv_99: float


@dataclass
class GSADFResult:
    gsadf_stat: float
    bsadf_sequence: npt.NDArray[np.float64]
    critical_values: CriticalValues
    is_bubble_90: bool
    is_bubble_95: bool
    is_bubble_99: bool


def detect_bubble(
    prices: npt.NDArray[np.float64],
    min_window: int | None = None,
    n_sims: int = 200,
    seed: int = 42,
) -> GSADFResult:
    """Run GSADF test for explosive bubble detection (Phillips-Shi-Yu 2015).

    Args:
        prices: Price series (levels, not returns).
        min_window: Minimum regression window. If None, uses PSY rule of thumb.
        n_sims: Number of Monte Carlo simulations for critical values.
        seed: Random seed for reproducibility.

    Returns:
        GSADFResult with test statistic, BSADF sequence, and critical values.
    """
    stat, bsadf, (cv90, cv95, cv99) = gsadf_test(
        prices, min_window=min_window, n_sims=n_sims, seed=seed
    )
    cvs = CriticalValues(cv_90=cv90, cv_95=cv95, cv_99=cv99)
    return GSADFResult(
        gsadf_stat=stat,
        bsadf_sequence=np.asarray(bsadf),
        critical_values=cvs,
        is_bubble_90=stat > cv90,
        is_bubble_95=stat > cv95,
        is_bubble_99=stat > cv99,
    )


def rolling_bubble_detection(
    prices: npt.NDArray[np.float64],
    window: int = 252,
    min_window: int | None = None,
) -> npt.NDArray[np.float64]:
    """Rolling GSADF statistic for continuous bubble monitoring.

    Args:
        prices: Price series (levels, not returns).
        window: Rolling window size.
        min_window: Minimum regression window within each rolling window.

    Returns:
        Array of GSADF statistics (NaN for initial entries).
    """
    return np.asarray(gsadf_rolling(prices, window, min_window=min_window))
