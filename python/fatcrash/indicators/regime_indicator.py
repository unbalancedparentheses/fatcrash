"""Market regime signal indicator: macro/microstructure regime classification.

Combines normalized signals from 7 thematic buckets into a single regime
score, then maps to risk_on / neutral / risk_off labels with confidence.

Also provides Python wrappers for Rust regime detection algorithms:
realized variance, jump risk decomposition, critical slowing down,
and Hamilton filter HMM.

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fatcrash._core import (
    bipower_variation,
    bipower_variation_rolling,
    csd_indicator,
    hamilton_fit,
    hamilton_filter,
    hamilton_smooth,
    jump_test,
    jump_variance,
    jump_variance_rolling,
    realized_variance,
    realized_variance_gk,
    realized_variance_parkinson,
    realized_variance_parkinson_rolling,
    realized_variance_rolling,
    rolling_ar1,
    rolling_variance,
)
from fatcrash.aggregator.signals import RegimeSignal


# ── Normalization ─────────────────────────────────────────

def rolling_zscore(
    x: npt.NDArray[np.float64],
    window: int = 756,
    min_obs: int = 63,
    clip: float = 4.0,
) -> npt.NDArray[np.float64]:
    """Rolling z-score normalization.

    Args:
        x: Input time series.
        window: Rolling window length in observations (default 756 = ~3Y daily).
        min_obs: Minimum observations before emitting a value.
        clip: Clip z-scores to [-clip, clip].

    Returns:
        Array of z-scores, NaN where insufficient data.
    """
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        w = x[start : i + 1]
        w = w[~np.isnan(w)]
        if len(w) < min_obs:
            continue
        mu = np.mean(w)
        sigma = np.std(w, ddof=1)
        if sigma == 0:
            continue
        result[i] = np.clip((x[i] - mu) / sigma, -clip, clip)
    return result


def rolling_percentile(
    x: npt.NDArray[np.float64],
    window: int = 756,
    min_obs: int = 126,
) -> npt.NDArray[np.float64]:
    """Rolling percentile rank normalization.

    Args:
        x: Input time series.
        window: Rolling window length.
        min_obs: Minimum observations before emitting.

    Returns:
        Array of percentile ranks in [0, 1], NaN where insufficient data.
    """
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        w = x[start : i + 1]
        w = w[~np.isnan(w)]
        if len(w) < min_obs:
            continue
        result[i] = np.searchsorted(np.sort(w), x[i]) / len(w)
    return result


# ── Bucket aggregation ────────────────────────────────────

def aggregate_bucket(
    signals: dict[str, float],
    method: str = "mean",
) -> float | None:
    """Aggregate signals within a thematic bucket.

    Args:
        signals: Dict of signal_name -> value (NaN or None = missing).
        method: Aggregation method ("mean" only for now).

    Returns:
        Bucket score (float) or None if >50% of signals are missing.
    """
    values = []
    for v in signals.values():
        if v is not None and not np.isnan(v):
            values.append(v)

    if len(values) == 0 or len(values) < len(signals) * 0.5:
        return None

    if method == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unknown aggregation method: {method}")


# ── Regime scoring ────────────────────────────────────────

DEFAULT_BUCKET_WEIGHTS = {
    "risk_premium": -0.30,
    "liquidity": -0.25,
    "volatility": -0.20,
    "credit_macro": -0.15,
    "structure_flows": -0.10,
    "contagion": 0.0,   # Placeholder until implemented
    "sentiment": 0.0,   # Placeholder until implemented
}

# Thresholds for regime labels
DEFAULT_RISK_ON_THRESHOLD = 0.50
DEFAULT_RISK_OFF_THRESHOLD = -0.50
DEFAULT_CONFIDENCE_DIVISOR = 2.0


def compute_regime_score(
    buckets: dict[str, float | None],
    weights: dict[str, float] | None = None,
    ema_alpha: float = 0.2,
    prev_score: float = 0.0,
) -> float:
    """Compute regime score from bucket values with EMA smoothing.

    Args:
        buckets: Dict of bucket_name -> score (None = missing).
        weights: Bucket weights (negative = more stress -> risk_off).
        ema_alpha: EMA smoothing factor (0.2 = ~4-day half-life).
        prev_score: Previous smoothed score for EMA.

    Returns:
        Smoothed regime score.
    """
    w = weights or DEFAULT_BUCKET_WEIGHTS

    available = []
    for name, score in buckets.items():
        if score is not None and name in w and w[name] != 0.0:
            available.append((w[name], score))

    if not available:
        return prev_score

    weight_sum = sum(abs(wt) for wt, _ in available)
    if weight_sum == 0:
        return prev_score

    raw_score = sum(wt / weight_sum * score for wt, score in available)

    # EMA smoothing
    smoothed = ema_alpha * raw_score + (1.0 - ema_alpha) * prev_score
    return smoothed


def label_regime(
    score: float,
    risk_on_threshold: float = DEFAULT_RISK_ON_THRESHOLD,
    risk_off_threshold: float = DEFAULT_RISK_OFF_THRESHOLD,
) -> str:
    """Map regime score to label."""
    if score >= risk_on_threshold:
        return "risk_on"
    elif score <= risk_off_threshold:
        return "risk_off"
    return "neutral"


def compute_confidence(
    score: float,
    divisor: float = DEFAULT_CONFIDENCE_DIVISOR,
) -> float:
    """Compute confidence from regime score magnitude."""
    return min(1.0, abs(score) / divisor)


def compute_regime(
    buckets: dict[str, float | None],
    weights: dict[str, float] | None = None,
    ema_alpha: float = 0.2,
    prev_score: float = 0.0,
    risk_on_threshold: float = DEFAULT_RISK_ON_THRESHOLD,
    risk_off_threshold: float = DEFAULT_RISK_OFF_THRESHOLD,
    confidence_divisor: float = DEFAULT_CONFIDENCE_DIVISOR,
    components: dict[str, float] | None = None,
) -> RegimeSignal:
    """Full regime computation: score -> label -> confidence.

    Args:
        buckets: Dict of bucket_name -> score (None = missing bucket).
        weights: Bucket weights.
        ema_alpha: EMA smoothing factor.
        prev_score: Previous smoothed score.
        risk_on_threshold: Score threshold for risk_on label.
        risk_off_threshold: Score threshold for risk_off label.
        confidence_divisor: Divisor for confidence calculation.
        components: Optional dict of individual signal values.

    Returns:
        RegimeSignal with label, score, confidence, and bucket breakdown.
    """
    score = compute_regime_score(buckets, weights, ema_alpha, prev_score)
    label = label_regime(score, risk_on_threshold, risk_off_threshold)
    confidence = compute_confidence(score, confidence_divisor)

    return RegimeSignal(
        label=label,
        score=score,
        confidence=confidence,
        buckets={k: v for k, v in buckets.items()},
        components=components or {},
    )


# ══════════════════════════════════════════════════════════
# Rust regime detection algorithm wrappers
# ══════════════════════════════════════════════════════════


# ── Realized Variance ─────────────────────────────────────


@dataclass
class RealizedVarianceEstimate:
    """Annualized realized variance."""
    rv: float
    rv_annualized_vol: float  # sqrt(rv)


def estimate_realized_variance(
    returns: npt.NDArray[np.float64],
    window: int = 21,
) -> RealizedVarianceEstimate:
    """Estimate annualized realized variance from returns."""
    rv = realized_variance(returns, window=window)
    return RealizedVarianceEstimate(rv=rv, rv_annualized_vol=np.sqrt(rv) if rv > 0 else 0.0)


def estimate_realized_variance_parkinson(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    window: int = 21,
) -> RealizedVarianceEstimate:
    """Parkinson RV from OHLC high/low (~5x more efficient)."""
    rv = realized_variance_parkinson(high, low, window=window)
    return RealizedVarianceEstimate(rv=rv, rv_annualized_vol=np.sqrt(rv) if rv > 0 else 0.0)


def estimate_realized_variance_gk(
    open_: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    window: int = 21,
) -> RealizedVarianceEstimate:
    """Garman-Klass RV from OHLC (most efficient estimator)."""
    rv = realized_variance_gk(open_, high, low, close, window=window)
    return RealizedVarianceEstimate(rv=rv, rv_annualized_vol=np.sqrt(rv) if rv > 0 else 0.0)


def rolling_rv(
    returns: npt.NDArray[np.float64],
    window: int = 21,
    step: int = 1,
) -> npt.NDArray[np.float64]:
    """Rolling realized variance."""
    return np.asarray(realized_variance_rolling(returns, window=window, step=step))


@dataclass
class RVSpikeEstimate:
    """RV spike: short-term vol vs long-term baseline."""
    rv_short: float
    rv_long: float
    ratio: float  # rv_short / rv_long


def estimate_rv_spike(
    returns: npt.NDArray[np.float64],
    short_window: int = 21,
    long_window: int = 126,
) -> RVSpikeEstimate:
    """Compare short-term RV to long-term baseline RV.

    More robust than BNS JV/RV fraction at daily frequency because it directly
    measures volatility regime change rather than jump/continuous decomposition.
    """
    rv_short = realized_variance(returns, window=min(len(returns), short_window))
    rv_long = realized_variance(returns, window=min(len(returns), long_window))
    ratio = rv_short / rv_long if rv_long > 0 and not np.isnan(rv_long) else 1.0
    return RVSpikeEstimate(rv_short=rv_short, rv_long=rv_long, ratio=ratio)


def rolling_rv_parkinson(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    window: int = 21,
    step: int = 1,
) -> npt.NDArray[np.float64]:
    """Rolling Parkinson RV."""
    return np.asarray(realized_variance_parkinson_rolling(high, low, window=window, step=step))


# ── Jump Risk ─────────────────────────────────────────────


@dataclass
class JumpRiskEstimate:
    """BNS jump risk decomposition."""
    rv: float          # Total realized variance
    bv: float          # Bipower variation (continuous component)
    jv: float          # Jump variance = max(RV - BV, 0)
    jump_fraction: float  # JV / RV


def estimate_jump_risk(
    returns: npt.NDArray[np.float64],
    window: int = 21,
) -> JumpRiskEstimate:
    """Decompose variance into continuous and jump components."""
    rv = realized_variance(returns, window=window)
    bv = bipower_variation(returns, window=window)
    jv = jump_variance(returns, window=window)
    jf = jv / rv if rv > 0 and not np.isnan(rv) else 0.0
    return JumpRiskEstimate(rv=rv, bv=bv, jv=jv, jump_fraction=jf)


@dataclass
class JumpTestResult:
    """BNS jump test result."""
    z_stat: float
    jv: float
    significant: bool  # z > 1.96 (5% level)


def estimate_jump_test(
    returns: npt.NDArray[np.float64],
    window: int = 63,
) -> JumpTestResult:
    """BNS z-test for jumps."""
    z, jv = jump_test(returns, window=window)
    return JumpTestResult(z_stat=z, jv=jv, significant=z > 1.96 if np.isfinite(z) else False)


def rolling_jv(
    returns: npt.NDArray[np.float64],
    window: int = 21,
    step: int = 1,
) -> npt.NDArray[np.float64]:
    """Rolling jump variance."""
    return np.asarray(jump_variance_rolling(returns, window=window, step=step))


def rolling_bv(
    returns: npt.NDArray[np.float64],
    window: int = 21,
    step: int = 1,
) -> npt.NDArray[np.float64]:
    """Rolling bipower variation."""
    return np.asarray(bipower_variation_rolling(returns, window=window, step=step))


# ── Critical Slowing Down ─────────────────────────────────


@dataclass
class CSDEstimate:
    """Critical Slowing Down indicator."""
    ar1_rising: bool
    var_rising: bool
    warning: bool  # Both rising simultaneously


def estimate_csd(
    data: npt.NDArray[np.float64],
    window: int = 252,
    roc_window: int = 63,
) -> CSDEstimate:
    """Check for critical slowing down (dual increase in AR(1) and variance).

    Note: For meaningful results, pass a *volatility* series (rolling RV),
    not raw returns. Raw returns have near-zero AR(1), so CSD on returns
    is uninformative. Use ``estimate_csd_on_vol`` for the recommended approach.
    """
    ar1_roc, var_roc, csd_sig = csd_indicator(data, window=window, roc_window=roc_window)
    ar1_roc = np.asarray(ar1_roc)
    var_roc = np.asarray(var_roc)
    csd_sig = np.asarray(csd_sig)

    last_ar1 = ar1_roc[~np.isnan(ar1_roc)]
    last_var = var_roc[~np.isnan(var_roc)]
    last_csd = csd_sig[~np.isnan(csd_sig)]

    ar1_up = float(last_ar1[-1]) > 0 if len(last_ar1) > 0 else False
    var_up = float(last_var[-1]) > 0 if len(last_var) > 0 else False
    warn = float(last_csd[-1]) > 0 if len(last_csd) > 0 else False

    return CSDEstimate(ar1_rising=ar1_up, var_rising=var_up, warning=warn)


def estimate_csd_on_vol(
    returns: npt.NDArray[np.float64],
    rv_window: int = 21,
    csd_window: int = 63,
    roc_window: int = 21,
) -> CSDEstimate | None:
    """CSD on the rolling realized variance series (the recommended approach).

    Raw returns have near-zero autocorrelation, making CSD meaningless.
    Volatility has strong autocorrelation (clustering), so CSD on the vol
    series detects true regime changes.

    Minimum data required: rv_window + csd_window + roc_window (~105 points).

    Args:
        returns: Log return series.
        rv_window: Window for computing rolling realized variance (default 21).
        csd_window: Window for AR(1) and variance in CSD (default 63).
        roc_window: Window for rate-of-change in CSD (default 21).

    Returns:
        CSDEstimate or None if insufficient data.
    """
    min_needed = rv_window + csd_window + roc_window + 1
    if len(returns) < min_needed:
        return None

    # Step 1: compute rolling RV series
    rv_series = np.asarray(realized_variance_rolling(returns, window=rv_window, step=1))

    # Drop NaN prefix from rolling RV
    valid_mask = ~np.isnan(rv_series)
    if valid_mask.sum() < csd_window + roc_window + 1:
        return None

    vol_series = rv_series[valid_mask]

    # Step 2: apply CSD to the vol series
    return estimate_csd(vol_series, window=csd_window, roc_window=roc_window)


def rolling_csd(
    data: npt.NDArray[np.float64],
    window: int = 252,
    roc_window: int = 63,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rolling CSD: returns (ar1_roc, var_roc, csd_signal) arrays."""
    ar1_roc, var_roc, csd_sig = csd_indicator(data, window=window, roc_window=roc_window)
    return np.asarray(ar1_roc), np.asarray(var_roc), np.asarray(csd_sig)


def rolling_ar1_coeff(
    data: npt.NDArray[np.float64],
    window: int = 252,
) -> npt.NDArray[np.float64]:
    """Rolling AR(1) coefficient."""
    return np.asarray(rolling_ar1(data, window=window))


def rolling_var(
    data: npt.NDArray[np.float64],
    window: int = 252,
) -> npt.NDArray[np.float64]:
    """Rolling variance."""
    return np.asarray(rolling_variance(data, window=window))


# ── Hamilton Filter ───────────────────────────────────────


@dataclass
class HamiltonEstimate:
    """Hamilton 2-state HMM fit result."""
    mu_normal: float
    sigma_normal: float
    mu_stressed: float
    sigma_stressed: float
    p00: float  # P(stay normal)
    p11: float  # P(stay stressed)
    prob_stressed: float  # Current P(stressed)
    filtered_probs: npt.NDArray[np.float64]


def estimate_hamilton(
    data: npt.NDArray[np.float64],
    n_restarts: int = 10,
) -> HamiltonEstimate:
    """Fit 2-state HMM via EM with random restarts."""
    mu0, s0, mu1, s1, p00, p11, probs = hamilton_fit(data, n_restarts=n_restarts)
    probs = np.asarray(probs)
    current = float(probs[-1]) if len(probs) > 0 and np.isfinite(probs[-1]) else np.nan
    return HamiltonEstimate(
        mu_normal=mu0, sigma_normal=s0,
        mu_stressed=mu1, sigma_stressed=s1,
        p00=p00, p11=p11,
        prob_stressed=current,
        filtered_probs=probs,
    )


def hamilton_filtered_probs(
    data: npt.NDArray[np.float64],
    mu: tuple[float, float],
    sigma: tuple[float, float],
    p00: float,
    p11: float,
) -> npt.NDArray[np.float64]:
    """Forward filter with known parameters."""
    return np.asarray(hamilton_filter(data, list(mu), list(sigma), p00, p11))


def hamilton_smoothed_probs(
    data: npt.NDArray[np.float64],
    mu: tuple[float, float],
    sigma: tuple[float, float],
    p00: float,
    p11: float,
) -> npt.NDArray[np.float64]:
    """Kim smoother with known parameters."""
    return np.asarray(hamilton_smooth(data, list(mu), list(sigma), p00, p11))
