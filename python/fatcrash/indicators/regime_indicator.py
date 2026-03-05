"""Market regime signal indicator: macro/microstructure regime classification.

Combines normalized signals from 7 thematic buckets into a single regime
score, then maps to risk_on / neutral / risk_off labels with confidence.

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

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
