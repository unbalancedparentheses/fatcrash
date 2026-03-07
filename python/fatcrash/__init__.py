"""fatcrash — Crash detection via fat-tail statistics."""

__version__ = "0.1.0"

# Data
from fatcrash.data.ingest import from_sample, from_yahoo, from_coingecko, from_csv, from_fred, from_fred_macro
from fatcrash.data.transforms import log_returns, log_prices, time_index, block_maxima, negative_returns

# Tail estimators
from fatcrash.indicators.tail_indicator import (
    estimate_tail_index, rolling_tail_index,
    estimate_kappa, rolling_kappa,
    estimate_taleb_kappa, rolling_taleb_kappa,
    estimate_pickands, rolling_pickands,
    estimate_hurst, rolling_hurst,
    estimate_dfa, rolling_dfa,
    estimate_deh, rolling_deh,
    estimate_qq, rolling_qq,
    estimate_maxsum, rolling_maxsum,
    estimate_spectral,
    estimate_momentum, rolling_momentum,
    estimate_reversal, rolling_reversal,
    estimate_velocity, rolling_velocity,
)

# EVT
from fatcrash.indicators.evt_indicator import fit_gpd, fit_gev, compute_var_es, rolling_var_es

# Bubble detection
from fatcrash.indicators.bubble_indicator import detect_bubble, rolling_bubble_detection

# LPPLS
from fatcrash.indicators.lppls_indicator import fit_lppls, compute_confidence

# Regime detection
from fatcrash.indicators.regime_indicator import (
    estimate_realized_variance, estimate_rv_spike,
    estimate_jump_risk, estimate_jump_test,
    estimate_csd, estimate_csd_on_vol,
    estimate_hamilton,
    rolling_rv, rolling_csd, rolling_ar1_coeff,
)

# Vol strategy
from fatcrash.indicators.vol_indicator import constant_vol_weight, rebalance_risk

# Signal aggregation
from fatcrash.aggregator.signals import aggregate_signals, CrashSignal, RegimeSignal
from fatcrash.aggregator.calibration import calibrate_weights

__all__ = [
    # Data
    "from_sample", "from_yahoo", "from_coingecko", "from_csv", "from_fred", "from_fred_macro",
    "log_returns", "log_prices", "time_index", "block_maxima", "negative_returns",
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
    # Signal aggregation
    "aggregate_signals", "CrashSignal", "RegimeSignal", "calibrate_weights",
]
