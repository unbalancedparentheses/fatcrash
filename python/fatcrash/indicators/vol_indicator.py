"""Constant volatility weighting and trend-aware rebalancing signals.

Implements the constant volatility strategy from the CBS thesis on tail-risk
hedging, and the trend-aware rebalancing signal from Rattray, Harvey & Van
Hemert (2018).

References:
- Dalgaard, K.T. (2016). "Tail-Risk Hedging: An Empirical Study."
  Copenhagen Business School (MSc thesis). Finding: constant volatility
  strategy reduces drawdowns and earns higher returns than put monetization.
- Rattray, S., Harvey, C.R. & Van Hemert, O. (2018). "Strategic Rebalancing."
  Journal of Portfolio Management. Finding: mechanical rebalancing is negative
  convexity; a 10% trend-following allocation reduces drawdowns by ~5pp.

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ConstantVolResult:
    """Result of constant volatility targeting."""
    weight: float  # position size multiplier [0, max_leverage]
    current_vol: float  # annualized realized vol
    target_vol: float  # target annualized vol


def constant_vol_weight(
    returns: npt.NDArray[np.float64],
    target_vol: float = 0.15,
    window: int = 63,
    max_leverage: float = 2.0,
    annualize: float = 252.0,
) -> ConstantVolResult:
    """Compute position sizing for constant volatility targeting.

    Scales position inversely to realized volatility:
    weight = target_vol / realized_vol

    When vol is low, you take larger positions; when vol spikes, you
    cut exposure. This mechanically implements the CBS thesis finding
    that constant-vol targeting beats put-based tail hedging.

    Args:
        returns: Array of log returns.
        target_vol: Target annualized volatility (default 15%).
        window: Lookback window for realized vol (default 63 = 3 months).
        max_leverage: Maximum position size (default 2x).
        annualize: Trading days per year for annualization.

    Returns:
        ConstantVolResult with weight, current_vol, target_vol.
    """
    if len(returns) < window:
        return ConstantVolResult(weight=1.0, current_vol=float("nan"),
                                 target_vol=target_vol)

    recent = returns[-window:]
    realized = float(np.std(recent, ddof=1) * np.sqrt(annualize))

    if realized < 1e-10:
        weight = max_leverage
    else:
        weight = min(target_vol / realized, max_leverage)

    return ConstantVolResult(
        weight=max(0.0, weight),
        current_vol=realized,
        target_vol=target_vol,
    )


def rolling_constant_vol_weight(
    returns: npt.NDArray[np.float64],
    target_vol: float = 0.15,
    window: int = 63,
    max_leverage: float = 2.0,
    annualize: float = 252.0,
) -> npt.NDArray[np.float64]:
    """Rolling constant-volatility position weights.

    Returns array of position sizes [0, max_leverage] for each timestep.
    """
    n = len(returns)
    weights = np.full(n, np.nan)

    if n < window:
        return weights

    for i in range(window - 1, n):
        recent = returns[i + 1 - window:i + 1]
        realized = float(np.std(recent, ddof=1) * np.sqrt(annualize))
        if realized < 1e-10:
            weights[i] = max_leverage
        else:
            weights[i] = min(target_vol / realized, max_leverage)

    return np.clip(weights, 0.0, max_leverage)


@dataclass
class RebalanceRiskResult:
    """Trend-aware rebalancing risk assessment."""
    risk: float  # [0, 1] — higher = more dangerous to rebalance mechanically
    dfa_component: float
    momentum_component: float


def rebalance_risk(
    dfa_alpha: float,
    momentum: float,
) -> RebalanceRiskResult:
    """Assess risk of mechanical rebalancing given current regime.

    Based on Rattray, Harvey & Van Hemert (2018): mechanical rebalancing
    is negative convexity — it buys into drawdowns that continue. When
    DFA detects trending behavior (alpha > 0.5) and momentum is negative,
    rebalancing into the drawdown is dangerous.

    Args:
        dfa_alpha: DFA exponent. > 0.5 = persistent/trending.
        momentum: Trailing log return. Negative = drawdown in progress.

    Returns:
        RebalanceRiskResult with risk score [0, 1].
    """
    if np.isnan(dfa_alpha) or np.isnan(momentum):
        return RebalanceRiskResult(risk=0.0, dfa_component=0.0,
                                   momentum_component=0.0)

    # DFA component: trending markets make rebalancing dangerous
    dfa_risk = float(np.clip((dfa_alpha - 0.5) / 0.3, 0.0, 1.0))

    # Momentum component: negative momentum = active drawdown
    mom_risk = float(np.clip(-momentum / 0.2, 0.0, 1.0))

    # Combined: both must be present for high risk
    # (trending + drawdown = don't rebalance)
    risk = dfa_risk * mom_risk

    return RebalanceRiskResult(
        risk=risk,
        dfa_component=dfa_risk,
        momentum_component=mom_risk,
    )
