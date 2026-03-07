"""Signal aggregation and calibration."""

from fatcrash.aggregator.signals import (
    aggregate_signals,
    CrashSignal,
    RegimeSignal,
    DEFAULT_WEIGHTS,
)
from fatcrash.aggregator.calibration import calibrate_weights

__all__ = [
    "aggregate_signals",
    "CrashSignal",
    "RegimeSignal",
    "DEFAULT_WEIGHTS",
    "calibrate_weights",
]
