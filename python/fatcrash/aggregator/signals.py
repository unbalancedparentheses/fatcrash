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


@dataclass
class RegimeSignal:
    """Market regime classification from macro/microstructure signals."""

    label: str  # "risk_on", "neutral", "risk_off"
    score: float  # Smoothed regime score
    confidence: float  # [0, 1]
    buckets: dict[str, float | None] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)


# Updated weights including NN methods
DEFAULT_WEIGHTS = {
    # Bubble detectors (highest weight — best accuracy)
    "lppls_confidence": 0.18,
    "lppls_tc_proximity": 0.06,
    "gsadf_bubble": 0.12,
    # NN bubble detectors
    "mlnn_signal": 0.06,
    "plnn_signal": 0.06,
    # Tail estimators
    "gpd_var_exceedance": 0.07,
    "kappa_regime": 0.05,
    "taleb_kappa": 0.04,
    "hill_thinning": 0.03,
    "pickands_thinning": 0.02,
    "deh_thinning": 0.02,
    "qq_thinning": 0.02,
    "maxsum_signal": 0.02,
    # Regime / momentum / velocity
    "hurst_trending": 0.03,
    "dfa_trending": 0.03,
    "spectral_memory": 0.02,
    "momentum_reversal": 0.04,
    "velocity_spike": 0.04,
    # Other
    "multiscale": 0.06,
    # Regime detection algorithms (implemented in Rust)
    "rv_spike": 0.03,
    "jump_risk_signal": 0.03,  # legacy BNS fraction — kept for backward compat
    "csd_warning": 0.03,
    "hamilton_stress": 0.03,
    # Market regime signals (placeholder weights — require external data sources)
    "vrp_signal": 0.0,
    "sofr_ois_z": 0.0,
    "ted_z": 0.0,
    "amihud_pct": 0.0,
    "xccy_basis_z": 0.0,
    "vix_slope_z": 0.0,
    "skew_z": 0.0,
    "move_z": 0.0,
    "vvix_z": 0.0,
    "ofr_fsi_z": 0.0,
    "credit_spread_z": 0.0,
    "ebp_z": 0.0,
    "yield_curve_z": 0.0,
    "eigenvalue_z": 0.0,
    "cot_z": 0.0,
    "etf_flows_z": 0.0,
    "covar_z": 0.0,
    "mes_z": 0.0,
    "srisk_z": 0.0,
    "fomc_tone_z": 0.0,
    "news_uncertainty_z": 0.0,
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
                    "mlnn_signal", "plnn_signal"],
        "tail": ["kappa_regime", "taleb_kappa", "hill_thinning", "pickands_thinning",
                 "gpd_var_exceedance", "deh_thinning", "qq_thinning", "maxsum_signal"],
        "regime": ["hurst_trending", "dfa_trending", "spectral_memory",
                   "momentum_reversal", "csd_warning", "hamilton_stress"],
        "structure": ["multiscale", "lppls_tc_proximity", "velocity_spike"],
        # Market regime signal families (macro/microstructure)
        "risk_premium": ["vrp_signal", "rv_spike", "jump_risk_signal"],
        "liquidity": ["sofr_ois_z", "ted_z", "amihud_pct", "xccy_basis_z"],
        "vol_regime": ["vix_slope_z", "skew_z", "move_z", "vvix_z"],
        "credit_macro": ["ofr_fsi_z", "credit_spread_z", "ebp_z", "yield_curve_z"],
        "structure_flows": ["eigenvalue_z", "cot_z", "etf_flows_z"],
        "contagion": ["covar_z", "mes_z", "srisk_z"],
        "sentiment": ["fomc_tone_z", "news_uncertainty_z"],
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


# ── Momentum & velocity signal converters ─────────────────

def momentum_reversal_signal(reversal: float) -> float:
    """Signal from momentum reversal (long momentum exceeds short).

    When long-term momentum is positive but short-term turns negative,
    this signals a potential trend break. The wider the divergence, the
    stronger the signal.

    Reference: Jegadeesh & Titman (1993), momentum reversal after 12+ months.
    """
    if np.isnan(reversal):
        return 0.0
    # reversal = long_mom - short_mom
    # Positive and large = trend breaking down
    if reversal <= 0:
        return 0.0
    # Scale: 0.1 = mild divergence, 0.3 = full signal
    return np.clip(reversal / 0.3, 0.0, 1.0)


def velocity_signal(velocity: float) -> float:
    """Signal from price velocity (volatility acceleration).

    Detects cascade dynamics where vol itself is spiking — the signature
    of forced liquidation (Volmageddon, repo crisis, etc.).

    Reference: Feb 5 2018 (XIV: 116% VIX spike), Sep 2019 repo blowup.
    """
    if np.isnan(velocity):
        return 0.0
    if velocity <= 0:
        return 0.0
    # Scale: 0.5 = vol increased 50% (notable), 2.0 = doubled (full signal)
    return np.clip(velocity / 2.0, 0.0, 1.0)


# ── Market regime signal converters ───────────────────────

def zscore_stress_signal(z: float) -> float:
    """Convert z-score to [0,1] stress signal. Higher z = more stress = higher signal.

    Used for funding spreads, credit spreads, vol indices, and stress indices.
    Maps z in [0, 4] to [0, 1]; below 0 returns 0.
    """
    if np.isnan(z):
        return 0.0
    return np.clip(z / 4.0, 0.0, 1.0)


def percentile_stress_signal(pct: float) -> float:
    """Convert percentile rank [0,1] to stress signal [0,1].

    Used for Amihud illiquidity and similar rank-based signals.
    """
    if np.isnan(pct):
        return 0.0
    return np.clip(pct, 0.0, 1.0)


def vrp_signal(implied_var: float, realized_var: float) -> float:
    """Signal from variance risk premium. High VRP = fear; negative VRP = acute stress.

    VRP = implied_var - realized_var. Both should be annualized.
    Returns high signal for both high positive VRP (fear) and negative VRP (acute crisis).
    """
    if np.isnan(implied_var) or np.isnan(realized_var):
        return 0.0
    vrp = implied_var - realized_var
    if vrp < 0:
        # Negative VRP: realized exceeds implied → acute stress
        return np.clip(-vrp / 0.05, 0.0, 1.0)
    # High positive VRP: fear premium
    return np.clip(vrp / 0.10, 0.0, 1.0)


def vix_slope_signal(slope: float) -> float:
    """Signal from VIX term structure slope. Negative slope (backwardation) = stress.

    slope = front_month - back_month (negative = backwardation = stress).
    """
    if np.isnan(slope):
        return 0.0
    if slope >= 0:
        return 0.0
    # Deeper backwardation = stronger signal
    return np.clip(-slope / 10.0, 0.0, 1.0)


def credit_spread_signal(spread_z: float) -> float:
    """Signal from credit spread z-score. Widening spreads = stress."""
    if np.isnan(spread_z):
        return 0.0
    return np.clip(spread_z / 3.0, 0.0, 1.0)


def jump_risk_signal_converter(jv: float, rv: float) -> float:
    """Signal from jump variance fraction. High JV/RV = jump-driven stress.

    Spikes during Flash Crash, Lehman, COVID when discontinuous moves dominate.
    """
    if np.isnan(jv) or np.isnan(rv) or rv <= 0:
        return 0.0
    fraction = jv / rv
    # Scale: 0.1 = some jumps, 0.5 = jump-dominated → full signal
    return np.clip(fraction / 0.5, 0.0, 1.0)


def rv_spike_signal(rv_short: float, rv_long: float) -> float:
    """Signal from realized variance spike: short-term RV vs long-term baseline.

    When current RV exceeds baseline by 2x+, volatility is spiking — the
    signature of regime change / crash onset. More robust than BNS JV/RV
    fraction at daily frequency.

    Args:
        rv_short: Short-window RV (e.g. 21-day).
        rv_long: Long-window RV baseline (e.g. 63-day or 126-day).

    Returns:
        Signal in [0, 1]. 0 = normal vol, 1 = vol spike ≥ 3x baseline.
    """
    if np.isnan(rv_short) or np.isnan(rv_long) or rv_long <= 0:
        return 0.0
    ratio = rv_short / rv_long
    if ratio <= 1.0:
        return 0.0
    # Scale: 1x = no spike, 2x = moderate (0.5), 3x+ = full signal (1.0)
    return np.clip((ratio - 1.0) / 2.0, 0.0, 1.0)


def csd_warning_signal(ar1_roc: float, var_roc: float) -> float:
    """Signal from Critical Slowing Down applied to volatility series.

    Dual increase in AR(1) of vol + variance of vol = approaching tipping point.
    Returns 1.0 if both positive, 0.3 if only one positive, 0.0 if neither.

    Note: CSD should be applied to the *volatility* series (rolling RV), not
    raw returns. Raw returns have near-zero autocorrelation, so AR(1) on
    returns is meaningless. Volatility, however, has strong autocorrelation
    (volatility clustering), so CSD on vol detects regime changes.
    """
    if np.isnan(ar1_roc) or np.isnan(var_roc):
        return 0.0
    ar1_up = ar1_roc > 0
    var_up = var_roc > 0
    if ar1_up and var_up:
        return 1.0
    if ar1_up or var_up:
        return 0.3
    return 0.0


def hamilton_stress_signal(prob_stressed: float) -> float:
    """Signal from Hamilton filter P(stressed).

    Direct probability output from 2-state HMM.
    """
    if np.isnan(prob_stressed):
        return 0.0
    return np.clip(prob_stressed, 0.0, 1.0)


def eigenvalue_signal(lambda_frac: float) -> float:
    """Signal from largest eigenvalue fraction of cross-asset correlation matrix.

    lambda_frac approaching 1.0 = all assets moving together = crisis regime.
    For n=4 assets, random baseline is ~0.25; stress is ~0.6+.
    """
    if np.isnan(lambda_frac):
        return 0.0
    # Scale from 0.3 (normal) to 0.8 (crisis)
    return np.clip((lambda_frac - 0.3) / 0.5, 0.0, 1.0)
