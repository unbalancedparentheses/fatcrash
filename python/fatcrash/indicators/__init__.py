"""Tail, persistence, and bubble detection indicators."""

from fatcrash.indicators.tail_indicator import (
    estimate_deh,
    estimate_dfa,
    estimate_hurst,
    estimate_kappa,
    estimate_maxsum,
    estimate_pickands,
    estimate_qq,
    estimate_spectral,
    estimate_tail_index,
    estimate_taleb_kappa,
    rolling_deh,
    rolling_dfa,
    rolling_hurst,
    rolling_kappa,
    rolling_maxsum,
    rolling_pickands,
    rolling_qq,
    rolling_tail_index,
    rolling_taleb_kappa,
)

# Deep LPPLS (deprecated wrapper)
from fatcrash.indicators.deep_lppls import DeepLPPLSResult, fit_deep_lppls

__all__ = [
    "estimate_tail_index",
    "rolling_tail_index",
    "estimate_kappa",
    "rolling_kappa",
    "estimate_taleb_kappa",
    "rolling_taleb_kappa",
    "estimate_pickands",
    "rolling_pickands",
    "estimate_hurst",
    "rolling_hurst",
    "estimate_dfa",
    "rolling_dfa",
    "estimate_deh",
    "rolling_deh",
    "estimate_qq",
    "rolling_qq",
    "estimate_maxsum",
    "rolling_maxsum",
    "estimate_spectral",
    # Deep LPPLS
    "DeepLPPLSResult",
    "fit_deep_lppls",
]
