"""Deep LPPLS — Neural network methods for LPPLS bubble detection.

Thin wrapper delegating to fatcrash.nn submodules. Provides backward-compatible
``fit_deep_lppls`` as a deprecated alias for ``fit_mlnn``.

References:
  - Nielsen, Sornette, Raissi (2024) "Deep Learning for LPPLS", arXiv:2405.12803
  - Cao, Shao, Yan, Geman (2025) "HLPPL", arXiv:2510.10878

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class DeepLPPLSResult:
    """Legacy result type for backward compatibility."""

    tc_pred: float
    m_pred: float
    omega_pred: float
    confidence: float
    reconstruction_loss: float
    physics_loss: float


def fit_deep_lppls(
    times: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    window_size: int = 120,
    epochs: int = 200,
    lr: float = 1e-2,
    physics_weight: float = 1.0,
) -> DeepLPPLSResult:
    """Fit M-LNN to a time series (deprecated, use fatcrash.nn.mlnn.fit_mlnn).

    This is a backward-compatible wrapper. The original implementation had
    incorrect architecture and normalization. New code should use fit_mlnn directly.
    """
    warnings.warn(
        "fit_deep_lppls is deprecated. Use fatcrash.nn.mlnn.fit_mlnn instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from fatcrash.nn.mlnn import fit_mlnn

    result = fit_mlnn(
        times=times,
        log_prices=log_prices,
        epochs=epochs,
        lr=lr,
    )

    return DeepLPPLSResult(
        tc_pred=result.tc,
        m_pred=result.m,
        omega_pred=result.omega,
        confidence=result.confidence,
        reconstruction_loss=result.rss,
        physics_loss=0.0,
    )
