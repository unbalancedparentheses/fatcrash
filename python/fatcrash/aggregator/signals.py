"""Signal aggregation: combine indicators into crash probability."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class CrashSignal:
    probability: float  # [0, 1]
    horizon_days: float  # Estimated days to event
    components: dict[str, float] = field(default_factory=dict)

    @property
    def level(self) -> str:
        if self.probability > 0.7:
            return "CRITICAL"
        elif self.probability > 0.5:
            return "HIGH"
        elif self.probability > 0.3:
            return "ELEVATED"
        else:
            return "LOW"


# Default weights from the plan
DEFAULT_WEIGHTS = {
    "lppls_confidence": 0.30,
    "lppls_tc_proximity": 0.15,
    "gpd_var_exceedance": 0.15,
    "kappa_regime": 0.10,
    "hill_thinning": 0.10,
    "deep_lppls": 0.10,
    "multiscale": 0.10,
}


def aggregate_signals(
    components: dict[str, float],
    weights: dict[str, float] | None = None,
) -> CrashSignal:
    """Combine individual indicator signals into a crash probability.

    Each component value should be in [0, 1].
    """
    w = weights or DEFAULT_WEIGHTS

    total_weight = 0.0
    weighted_sum = 0.0

    for name, value in components.items():
        if name in w and not np.isnan(value):
            weight = w[name]
            weighted_sum += weight * np.clip(value, 0.0, 1.0)
            total_weight += weight

    if total_weight == 0:
        return CrashSignal(probability=0.0, horizon_days=float("inf"), components=components)

    probability = weighted_sum / total_weight

    # Estimate horizon from LPPLS tc proximity if available
    horizon = components.get("lppls_tc_days", float("inf"))

    return CrashSignal(probability=probability, horizon_days=horizon, components=components)


def lppls_confidence_signal(confidence: float) -> float:
    """Convert LPPLS confidence [0,1] to signal [0,1]."""
    return np.clip(confidence, 0.0, 1.0)


def tc_proximity_signal(days_to_tc: float, max_days: float = 90.0) -> float:
    """Convert days-to-tc to urgency signal. Closer = higher."""
    if days_to_tc <= 0 or np.isnan(days_to_tc):
        return 1.0
    return np.clip(1.0 - days_to_tc / max_days, 0.0, 1.0)


def var_exceedance_signal(current_return: float, var: float) -> float:
    """Signal based on how much the current return exceeds VaR."""
    if np.isnan(var) or var == 0:
        return 0.0
    loss = -current_return
    if loss <= 0:
        return 0.0
    ratio = loss / var
    return np.clip(ratio, 0.0, 1.0)


def kappa_regime_signal(kappa: float, benchmark: float) -> float:
    """Signal from kappa deviation below Gaussian benchmark."""
    if np.isnan(kappa) or np.isnan(benchmark) or benchmark == 0:
        return 0.0
    ratio = kappa / benchmark
    # ratio < 1 means fatter tails than Gaussian
    return np.clip(1.0 - ratio, 0.0, 1.0)


def hill_thinning_signal(alpha: float, alpha_prev: float) -> float:
    """Signal from declining Hill alpha (thickening tails)."""
    if np.isnan(alpha) or np.isnan(alpha_prev):
        return 0.0
    if alpha_prev <= 0:
        return 0.0
    change = (alpha_prev - alpha) / alpha_prev
    return np.clip(change, 0.0, 1.0)
