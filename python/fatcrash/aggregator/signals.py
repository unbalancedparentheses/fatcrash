"""Signal aggregation: combine all indicators into crash probability.

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class CrashSignal:
    probability: float  # [0, 1]
    horizon_days: float  # Estimated days to event
    components: dict[str, float] = field(default_factory=dict)
    n_agreeing: int = 0  # How many independent methods agree

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


# Updated weights including NN methods
DEFAULT_WEIGHTS = {
    # Bubble detectors (highest weight — best accuracy)
    "lppls_confidence": 0.16,
    "lppls_tc_proximity": 0.06,
    "gsadf_bubble": 0.10,
    # NN bubble detectors
    "mlnn_signal": 0.05,
    "plnn_signal": 0.05,
    "hlppl_signal": 0.06,
    "dtcai_signal": 0.06,
    # Tail estimators
    "gpd_var_exceedance": 0.07,
    "kappa_regime": 0.05,
    "taleb_kappa": 0.04,
    "hill_thinning": 0.03,
    "pickands_thinning": 0.03,
    "deh_thinning": 0.03,
    "qq_thinning": 0.03,
    "maxsum_signal": 0.03,
    # Regime
    "hurst_trending": 0.03,
    "dfa_trending": 0.03,
    "spectral_memory": 0.03,
    # Other
    "multiscale": 0.06,
}


def aggregate_signals(
    components: dict[str, float],
    weights: dict[str, float] | None = None,
) -> CrashSignal:
    """Combine individual indicator signals into a crash probability.

    Each component value should be in [0, 1].

    Uses weighted average as base, then applies an agreement bonus:
    when multiple independent method categories agree (signal > 0.5),
    the probability is boosted. This rewards consensus across different
    approaches (bubble detection, tail analysis, regime detection).
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

    base_probability = weighted_sum / total_weight

    # Count how many independent method categories have elevated signals
    categories = {
        "bubble": ["lppls_confidence", "gsadf_bubble",
                    "mlnn_signal", "plnn_signal", "hlppl_signal", "dtcai_signal"],
        "tail": ["kappa_regime", "taleb_kappa", "hill_thinning", "pickands_thinning",
                 "gpd_var_exceedance", "deh_thinning", "qq_thinning", "maxsum_signal"],
        "regime": ["hurst_trending", "dfa_trending", "spectral_memory"],
        "structure": ["multiscale", "lppls_tc_proximity"],
    }

    n_agreeing = 0
    for cat, keys in categories.items():
        cat_signals = [components.get(k, 0.0) for k in keys if k in components]
        if cat_signals and max(cat_signals) > 0.5:
            n_agreeing += 1

    # Agreement bonus: if 3+ categories agree, boost probability
    if n_agreeing >= 3:
        agreement_bonus = 0.15
    elif n_agreeing >= 2:
        agreement_bonus = 0.05
    else:
        agreement_bonus = 0.0

    probability = min(1.0, base_probability + agreement_bonus)

    horizon = components.get("lppls_tc_days", float("inf"))

    return CrashSignal(
        probability=probability,
        horizon_days=horizon,
        components=components,
        n_agreeing=n_agreeing,
    )


# ── Signal converters ──────────────────────────────────────

def lppls_confidence_signal(confidence: float) -> float:
    """Convert LPPLS confidence [0,1] to signal [0,1]."""
    return np.clip(confidence, 0.0, 1.0)


def tc_proximity_signal(days_to_tc: float, max_days: float = 90.0) -> float:
    """Convert days-to-tc to urgency signal. Closer = higher."""
    if days_to_tc <= 0 or np.isnan(days_to_tc):
        return 0.0
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
    return np.clip(1.0 - ratio, 0.0, 1.0)


def taleb_kappa_signal(kappa: float, benchmark: float) -> float:
    """Signal from Taleb kappa exceeding Gaussian benchmark.

    Higher kappa = fatter tails = higher signal.
    """
    if np.isnan(kappa) or np.isnan(benchmark):
        return 0.0
    # kappa above benchmark means fatter than Gaussian
    excess = kappa - benchmark
    if excess <= 0:
        return 0.0
    # Scale: 0.3 excess → full signal
    return np.clip(excess / 0.3, 0.0, 1.0)


def hill_thinning_signal(alpha: float, alpha_prev: float) -> float:
    """Signal from declining Hill alpha (thickening tails)."""
    if np.isnan(alpha) or np.isnan(alpha_prev):
        return 0.0
    if alpha_prev <= 0:
        return 0.0
    change = (alpha_prev - alpha) / alpha_prev
    return np.clip(change, 0.0, 1.0)


def pickands_signal(gamma: float, gamma_prev: float) -> float:
    """Signal from increasing Pickands gamma (thickening tails)."""
    if np.isnan(gamma) or np.isnan(gamma_prev):
        return 0.0
    if gamma_prev == 0:
        return 0.0
    # Increasing gamma = heavier tails
    change = (gamma - gamma_prev) / abs(gamma_prev)
    return np.clip(change, 0.0, 1.0)


def gsadf_signal(gsadf_stat: float, cv95: float) -> float:
    """Signal from GSADF test. Above critical value = explosive bubble."""
    if np.isnan(gsadf_stat) or np.isnan(cv95) or cv95 == 0:
        return 0.0
    ratio = gsadf_stat / cv95
    if ratio < 0.5:
        return 0.0
    return np.clip((ratio - 0.5) / 1.5, 0.0, 1.0)


def hurst_signal(h: float) -> float:
    """Signal from Hurst exponent. H > 0.5 = trending = potential bubble buildup."""
    if np.isnan(h):
        return 0.0
    # Only signal when clearly trending (H > 0.55)
    if h <= 0.55:
        return 0.0
    return np.clip((h - 0.55) / 0.3, 0.0, 1.0)


def dfa_signal(alpha: float) -> float:
    """Signal from DFA exponent. alpha > 0.5 = persistent dynamics."""
    if np.isnan(alpha):
        return 0.0
    if alpha <= 0.55:
        return 0.0
    return np.clip((alpha - 0.55) / 0.3, 0.0, 1.0)


def deh_signal(gamma: float, gamma_prev: float) -> float:
    """Signal from increasing DEH gamma (thickening tails)."""
    if np.isnan(gamma) or np.isnan(gamma_prev):
        return 0.0
    if gamma_prev == 0:
        return 0.0
    change = (gamma - gamma_prev) / abs(gamma_prev)
    return np.clip(change, 0.0, 1.0)


def qq_signal(alpha: float, alpha_prev: float) -> float:
    """Signal from declining QQ alpha (thickening tails)."""
    if np.isnan(alpha) or np.isnan(alpha_prev):
        return 0.0
    if alpha_prev <= 0:
        return 0.0
    change = (alpha_prev - alpha) / alpha_prev
    return np.clip(change, 0.0, 1.0)


def maxsum_signal(ratio: float) -> float:
    """Signal from max-to-sum ratio. High ratio = infinite variance.

    For Gaussian, R_n ~ 0.01; for alpha < 2, R_n ~ 0.05+.
    """
    if np.isnan(ratio):
        return 0.0
    # Scale: 0.02 baseline, 0.10 = full signal
    return np.clip((ratio - 0.02) / 0.08, 0.0, 1.0)


def spectral_signal(d: float) -> float:
    """Signal from spectral exponent. d > 0 = long memory."""
    if np.isnan(d):
        return 0.0
    if d <= 0.05:
        return 0.0
    return np.clip((d - 0.05) / 0.4, 0.0, 1.0)


# ── NN signal converters ──────────────────────────────────

def mlnn_signal(confidence: float, is_bubble: bool) -> float:
    """Convert M-LNN result to signal [0,1].

    Uses confidence directly if bubble filter passes, halved otherwise.
    """
    if np.isnan(confidence):
        return 0.0
    if is_bubble:
        return np.clip(confidence, 0.0, 1.0)
    return np.clip(confidence * 0.5, 0.0, 1.0)


def plnn_signal(confidence: float, is_bubble: bool) -> float:
    """Convert P-LNN result to signal [0,1]."""
    if np.isnan(confidence):
        return 0.0
    if is_bubble:
        return np.clip(confidence, 0.0, 1.0)
    return np.clip(confidence * 0.5, 0.0, 1.0)


def hlppl_signal(bubble_score: float) -> float:
    """Convert HLPPL bubble score [0,1] directly to signal."""
    if np.isnan(bubble_score):
        return 0.0
    return np.clip(bubble_score, 0.0, 1.0)


def dtcai_signal(dtcai_score: float) -> float:
    """Convert DTCAI score [0,1] directly to signal.

    DTCAI = DTC * reliability, already in [0, 1].
    """
    if np.isnan(dtcai_score):
        return 0.0
    return np.clip(dtcai_score, 0.0, 1.0)
