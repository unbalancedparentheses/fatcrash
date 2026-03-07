"""Tail, persistence, bubble detection, EVT, regime, and vol indicators."""

from fatcrash.indicators.tail_indicator import (
    estimate_deh,
    estimate_dfa,
    estimate_hurst,
    estimate_kappa,
    estimate_maxsum,
    estimate_momentum,
    estimate_pickands,
    estimate_qq,
    estimate_reversal,
    estimate_spectral,
    estimate_tail_index,
    estimate_taleb_kappa,
    estimate_velocity,
    rolling_deh,
    rolling_dfa,
    rolling_hurst,
    rolling_kappa,
    rolling_maxsum,
    rolling_momentum,
    rolling_pickands,
    rolling_qq,
    rolling_reversal,
    rolling_tail_index,
    rolling_taleb_kappa,
    rolling_velocity,
)

from fatcrash.indicators.evt_indicator import (
    fit_gpd,
    fit_gev,
    compute_var_es,
    rolling_var_es,
)

from fatcrash.indicators.bubble_indicator import (
    detect_bubble,
    rolling_bubble_detection,
)

from fatcrash.indicators.lppls_indicator import (
    fit_lppls,
    compute_confidence,
)

from fatcrash.indicators.regime_indicator import (
    estimate_realized_variance,
    estimate_rv_spike,
    estimate_jump_risk,
    estimate_jump_test,
    estimate_csd,
    estimate_csd_on_vol,
    estimate_hamilton,
    rolling_rv,
    rolling_csd,
    rolling_ar1_coeff,
)

from fatcrash.indicators.vol_indicator import (
    constant_vol_weight,
    rebalance_risk,
)

__all__ = [
    # Tail estimators
    "estimate_tail_index", "rolling_tail_index",
    "estimate_kappa", "rolling_kappa",
    "estimate_taleb_kappa", "rolling_taleb_kappa",
    "estimate_pickands", "rolling_pickands",
    "estimate_hurst", "rolling_hurst",
    "estimate_dfa", "rolling_dfa",
    "estimate_deh", "rolling_deh",
    "estimate_qq", "rolling_qq",
    "estimate_maxsum", "rolling_maxsum",
    "estimate_spectral",
    "estimate_momentum", "rolling_momentum",
    "estimate_reversal", "rolling_reversal",
    "estimate_velocity", "rolling_velocity",
    # EVT
    "fit_gpd", "fit_gev", "compute_var_es", "rolling_var_es",
    # Bubble detection
    "detect_bubble", "rolling_bubble_detection",
    # LPPLS
    "fit_lppls", "compute_confidence",
    # Regime detection
    "estimate_realized_variance", "estimate_rv_spike",
    "estimate_jump_risk", "estimate_jump_test",
    "estimate_csd", "estimate_csd_on_vol",
    "estimate_hamilton",
    "rolling_rv", "rolling_csd", "rolling_ar1_coeff",
    # Vol strategy
    "constant_vol_weight", "rebalance_risk",
]
