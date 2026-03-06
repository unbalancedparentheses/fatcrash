use serde::Serialize;
use std::collections::HashMap;

/// Aggregated crash signal with probability in [0, 1].
#[derive(Debug, Clone, Serialize)]
pub struct CrashSignal {
    pub probability: f64,
    pub horizon_days: f64,
    pub components: HashMap<String, f64>,
    pub n_agreeing: usize,
}

impl CrashSignal {
    pub fn level(&self) -> &'static str {
        if self.probability > 0.7 {
            "CRITICAL"
        } else if self.probability > 0.5 {
            "HIGH"
        } else if self.probability > 0.3 {
            "ELEVATED"
        } else {
            "LOW"
        }
    }
}

/// All 56 default weights, matching the Python DEFAULT_WEIGHTS exactly.
pub fn default_weights() -> HashMap<&'static str, f64> {
    let mut w = HashMap::new();
    // Bubble detectors
    w.insert("lppls_confidence", 0.22);
    w.insert("lppls_tc_proximity", 0.06);
    w.insert("gsadf_bubble", 0.16);
    // Tail estimators
    w.insert("gpd_var_exceedance", 0.0);
    w.insert("kappa_regime", 0.12);
    w.insert("taleb_kappa", 0.06);
    w.insert("hill_thinning", 0.0);
    w.insert("pickands_thinning", 0.02);
    w.insert("deh_thinning", 0.02);
    w.insert("qq_thinning", 0.02);
    w.insert("maxsum_signal", 0.0);
    // Regime / momentum / velocity
    w.insert("hurst_trending", 0.0);
    w.insert("dfa_trending", 0.03);
    w.insert("spectral_memory", 0.02);
    w.insert("momentum_reversal", 0.04);
    w.insert("velocity_spike", 0.04);
    // Other
    w.insert("multiscale", 0.06);
    // Regime detection algorithms
    w.insert("rv_spike", 0.03);
    w.insert("jump_risk_signal", 0.03);
    w.insert("csd_warning", 0.03);
    w.insert("hamilton_stress", 0.03);
    w.insert("amihud_spike", 0.05);
    // Market regime signals (placeholders)
    w.insert("vrp_signal", 0.0);
    w.insert("sofr_ois_z", 0.0);
    w.insert("ted_z", 0.0);
    w.insert("amihud_pct", 0.0);
    w.insert("xccy_basis_z", 0.0);
    w.insert("vix_slope_z", 0.0);
    w.insert("skew_z", 0.0);
    w.insert("move_z", 0.0);
    w.insert("vvix_z", 0.0);
    w.insert("ofr_fsi_z", 0.0);
    w.insert("credit_spread_z", 0.0);
    w.insert("ebp_z", 0.0);
    w.insert("yield_curve_z", 0.0);
    w.insert("eigenvalue_z", 0.0);
    w.insert("cot_z", 0.0);
    w.insert("etf_flows_z", 0.0);
    w.insert("covar_z", 0.0);
    w.insert("mes_z", 0.0);
    w.insert("srisk_z", 0.0);
    w.insert("fomc_tone_z", 0.0);
    w.insert("news_uncertainty_z", 0.0);
    w
}

/// All 11 signal categories for agreement counting.
pub fn categories() -> HashMap<&'static str, Vec<&'static str>> {
    let mut cats = HashMap::new();
    cats.insert(
        "bubble",
        vec![
            "lppls_confidence",
            "gsadf_bubble",
        ],
    );
    cats.insert(
        "tail",
        vec![
            "kappa_regime",
            "taleb_kappa",
            "hill_thinning",
            "pickands_thinning",
            "gpd_var_exceedance",
            "deh_thinning",
            "qq_thinning",
            "maxsum_signal",
        ],
    );
    cats.insert(
        "regime",
        vec![
            "hurst_trending",
            "dfa_trending",
            "spectral_memory",
            "momentum_reversal",
            "csd_warning",
            "hamilton_stress",
        ],
    );
    cats.insert(
        "structure",
        vec!["multiscale", "lppls_tc_proximity", "velocity_spike"],
    );
    cats.insert(
        "risk_premium",
        vec!["vrp_signal", "rv_spike", "jump_risk_signal"],
    );
    cats.insert(
        "liquidity",
        vec!["sofr_ois_z", "ted_z", "amihud_pct", "xccy_basis_z"],
    );
    cats.insert(
        "vol_regime",
        vec!["vix_slope_z", "skew_z", "move_z", "vvix_z"],
    );
    cats.insert(
        "credit_macro",
        vec!["ofr_fsi_z", "credit_spread_z", "ebp_z", "yield_curve_z"],
    );
    cats.insert(
        "structure_flows",
        vec!["eigenvalue_z", "cot_z", "etf_flows_z"],
    );
    cats.insert("contagion", vec!["covar_z", "mes_z", "srisk_z"]);
    cats.insert("sentiment", vec!["fomc_tone_z", "news_uncertainty_z"]);
    cats
}

/// Combine individual indicator signals into a crash probability.
///
/// Architecture: LPPLS-first.
///   1. LPPLS (+ GSADF) is the primary bubble detector.
///   2. If no bubble detected → LOW, regardless of other signals.
///   3. If bubble detected → other signals confirm or filter it down.
///
/// This focuses on endogenous bubble-driven crashes. Other signals
/// act as a jury: if they agree, probability stays high; if they
/// disagree, the bubble signal gets discounted.
pub fn aggregate_signals(components: &HashMap<String, f64>) -> CrashSignal {
    let get = |key: &str| -> f64 {
        components.get(key).copied().unwrap_or(0.0).max(0.0)
    };

    // Primary: bubble detectors
    let lppls = get("lppls_confidence");
    let gsadf = get("gsadf_bubble");
    let bubble_score = lppls.max(gsadf);

    // Secondary: count other signals that confirm (>0.5) or deny (<0.2)
    let bubble_keys = ["lppls_confidence", "gsadf_bubble", "lppls_tc_proximity"];
    let w = default_weights();
    let mut n_confirm = 0_usize;
    let mut n_deny = 0_usize;
    let mut n_active = 0_usize;

    for (name, &weight) in &w {
        if weight <= 0.0 || bubble_keys.contains(name) {
            continue;
        }
        if let Some(&value) = components.get(*name) {
            if !value.is_nan() {
                n_active += 1;
                if value > 0.5 {
                    n_confirm += 1;
                } else if value < 0.2 {
                    n_deny += 1;
                }
            }
        }
    }

    // Confirmation score: -1.0 (all deny) to +1.0 (all confirm)
    let confirmation = if n_active > 0 {
        (n_confirm as f64 - n_deny as f64) / n_active as f64
    } else {
        0.0
    };

    // Count agreeing categories
    let cats = categories();
    let mut n_agreeing = 0_usize;
    for (_cat_name, keys) in &cats {
        let cat_signals: Vec<f64> = keys
            .iter()
            .filter_map(|k| components.get(*k).copied())
            .filter(|v| !v.is_nan())
            .collect();
        if !cat_signals.is_empty() {
            let max_sig = cat_signals
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if max_sig > 0.5 {
                n_agreeing += 1;
            }
        }
    }

    // Probability:
    // - No bubble (score < 0.1): essentially 0
    // - Bubble detected: score * filter_multiplier
    //   filter_multiplier = 0.3 (all deny) to 1.0 (all confirm)
    let probability = if bubble_score < 0.1 {
        0.0
    } else {
        // Map confirmation from [-1, +1] to multiplier [0.3, 1.0]
        // -1 → 0.3 (most signals disagree, heavily discount bubble)
        //  0 → 0.65 (neutral)
        // +1 → 1.0 (all confirm)
        let multiplier = 0.65 + 0.35 * confirmation;
        bubble_score * multiplier
    };

    let probability = probability.clamp(0.0, 1.0);

    let horizon = components
        .get("lppls_tc_days")
        .copied()
        .unwrap_or(f64::INFINITY);

    CrashSignal {
        probability,
        horizon_days: horizon,
        components: components.clone(),
        n_agreeing,
    }
}

// ── Signal converters ──────────────────────────────────────

/// Convert LPPLS confidence [0,1] to signal [0,1].
pub fn lppls_confidence_signal(confidence: f64) -> f64 {
    confidence.clamp(0.0, 1.0)
}

/// Convert days-to-tc to urgency signal. Closer = higher.
pub fn tc_proximity_signal(days_to_tc: f64, max_days: f64) -> f64 {
    if days_to_tc <= 0.0 || days_to_tc.is_nan() {
        return 0.0;
    }
    (1.0 - days_to_tc / max_days).clamp(0.0, 1.0)
}

/// Signal based on how much the current return exceeds VaR.
pub fn var_exceedance_signal(current_return: f64, var: f64) -> f64 {
    if var.is_nan() || var == 0.0 {
        return 0.0;
    }
    let loss = -current_return;
    if loss <= 0.0 {
        return 0.0;
    }
    let ratio = loss / var;
    ratio.clamp(0.0, 1.0)
}

/// Signal from kappa deviation below Gaussian benchmark.
pub fn kappa_regime_signal(kappa: f64, benchmark: f64) -> f64 {
    if kappa.is_nan() || benchmark.is_nan() || benchmark == 0.0 {
        return 0.0;
    }
    let ratio = kappa / benchmark;
    (1.0 - ratio).clamp(0.0, 1.0)
}

/// Signal from Taleb kappa exceeding Gaussian benchmark.
/// Higher kappa = fatter tails = higher signal.
pub fn taleb_kappa_signal(kappa: f64, benchmark: f64) -> f64 {
    if kappa.is_nan() || benchmark.is_nan() {
        return 0.0;
    }
    let excess = kappa - benchmark;
    if excess <= 0.0 {
        return 0.0;
    }
    // Scale: 0.3 excess -> full signal
    (excess / 0.3).clamp(0.0, 1.0)
}

/// Signal from declining Hill alpha (thickening tails).
pub fn hill_thinning_signal(alpha: f64, alpha_prev: f64) -> f64 {
    if alpha.is_nan() || alpha_prev.is_nan() {
        return 0.0;
    }
    if alpha_prev <= 0.0 {
        return 0.0;
    }
    let change = (alpha_prev - alpha) / alpha_prev;
    change.clamp(0.0, 1.0)
}

/// Signal from increasing Pickands gamma (thickening tails).
pub fn pickands_signal(gamma: f64, gamma_prev: f64) -> f64 {
    if gamma.is_nan() || gamma_prev.is_nan() {
        return 0.0;
    }
    if gamma_prev == 0.0 {
        return 0.0;
    }
    // Increasing gamma = heavier tails
    let change = (gamma - gamma_prev) / gamma_prev.abs();
    change.clamp(0.0, 1.0)
}

/// Signal from GSADF test. Above critical value = explosive bubble.
pub fn gsadf_signal(gsadf_stat: f64, cv95: f64) -> f64 {
    if gsadf_stat.is_nan() || cv95.is_nan() || cv95 == 0.0 {
        return 0.0;
    }
    let ratio = gsadf_stat / cv95;
    if ratio < 0.5 {
        return 0.0;
    }
    ((ratio - 0.5) / 1.5).clamp(0.0, 1.0)
}

/// Signal from Hurst exponent. H > 0.5 = trending = potential bubble buildup.
pub fn hurst_signal(h: f64) -> f64 {
    if h.is_nan() {
        return 0.0;
    }
    if h <= 0.55 {
        return 0.0;
    }
    ((h - 0.55) / 0.3).clamp(0.0, 1.0)
}

/// Signal from DFA exponent. alpha > 0.5 = persistent dynamics.
pub fn dfa_signal(alpha: f64) -> f64 {
    if alpha.is_nan() {
        return 0.0;
    }
    if alpha <= 0.55 {
        return 0.0;
    }
    ((alpha - 0.55) / 0.3).clamp(0.0, 1.0)
}

/// Signal from increasing DEH gamma (thickening tails).
pub fn deh_signal(gamma: f64, gamma_prev: f64) -> f64 {
    if gamma.is_nan() || gamma_prev.is_nan() {
        return 0.0;
    }
    if gamma_prev == 0.0 {
        return 0.0;
    }
    let change = (gamma - gamma_prev) / gamma_prev.abs();
    change.clamp(0.0, 1.0)
}

/// Signal from declining QQ alpha (thickening tails).
pub fn qq_signal(alpha: f64, alpha_prev: f64) -> f64 {
    if alpha.is_nan() || alpha_prev.is_nan() {
        return 0.0;
    }
    if alpha_prev <= 0.0 {
        return 0.0;
    }
    let change = (alpha_prev - alpha) / alpha_prev;
    change.clamp(0.0, 1.0)
}

/// Signal from max-to-sum ratio. High ratio = infinite variance.
pub fn maxsum_signal_fn(ratio: f64) -> f64 {
    if ratio.is_nan() {
        return 0.0;
    }
    // Scale: 0.02 baseline, 0.10 = full signal
    ((ratio - 0.02) / 0.08).clamp(0.0, 1.0)
}

/// Signal from spectral exponent. d > 0 = long memory.
pub fn spectral_signal(d: f64) -> f64 {
    if d.is_nan() {
        return 0.0;
    }
    if d <= 0.05 {
        return 0.0;
    }
    ((d - 0.05) / 0.4).clamp(0.0, 1.0)
}

/// Signal from momentum reversal (long momentum exceeds short).
pub fn momentum_reversal_signal(reversal: f64) -> f64 {
    if reversal.is_nan() {
        return 0.0;
    }
    if reversal <= 0.0 {
        return 0.0;
    }
    // Scale: 0.1 = mild divergence, 0.3 = full signal
    (reversal / 0.3).clamp(0.0, 1.0)
}

/// Signal from price velocity (volatility acceleration).
pub fn velocity_signal(velocity: f64) -> f64 {
    if velocity.is_nan() {
        return 0.0;
    }
    if velocity <= 0.0 {
        return 0.0;
    }
    // Scale: 0.5 = vol increased 50%, 2.0 = doubled (full signal)
    (velocity / 2.0).clamp(0.0, 1.0)
}

/// Signal from realized variance spike: short-term RV vs long-term baseline.
pub fn rv_spike_signal(rv_short: f64, rv_long: f64) -> f64 {
    if rv_short.is_nan() || rv_long.is_nan() || rv_long <= 0.0 {
        return 0.0;
    }
    let ratio = rv_short / rv_long;
    if ratio <= 1.0 {
        return 0.0;
    }
    // Scale: 1x = no spike, 2x = moderate (0.5), 3x+ = full signal (1.0)
    ((ratio - 1.0) / 2.0).clamp(0.0, 1.0)
}

/// Signal from Critical Slowing Down applied to volatility series.
/// Dual increase in AR(1) of vol + variance of vol = approaching tipping point.
/// Uses magnitude of increases, not just direction.
pub fn csd_warning_signal(ar1_roc: f64, var_roc: f64) -> f64 {
    if ar1_roc.is_nan() || var_roc.is_nan() {
        return 0.0;
    }
    let ar1_up = ar1_roc > 0.0;
    let var_up = var_roc > 0.0;
    if ar1_up && var_up {
        // Both increasing: scale by magnitude. Need substantial increases.
        // ar1_roc > 0.5 and var_roc > 0.5 for full signal
        let ar1_strength = (ar1_roc / 0.5).clamp(0.0, 1.0);
        let var_strength = (var_roc / 0.5).clamp(0.0, 1.0);
        (ar1_strength * var_strength).sqrt()
    } else if ar1_up || var_up {
        // Only one increasing: weak signal, proportional to magnitude
        let mag = if ar1_up { ar1_roc } else { var_roc };
        (mag / 1.0).clamp(0.0, 0.3)
    } else {
        0.0
    }
}

/// Signal from Hamilton filter P(stressed).
pub fn hamilton_stress_signal(prob_stressed: f64) -> f64 {
    if prob_stressed.is_nan() {
        return 0.0;
    }
    prob_stressed.clamp(0.0, 1.0)
}

/// Signal from realized skewness. Negative skew = left-tail risk.
pub fn realized_skewness_signal(skew: f64) -> f64 {
    if skew.is_nan() {
        return 0.0;
    }
    if skew >= 0.0 {
        return 0.0;
    }
    // Scale: -0.5 = moderate (0.5), -1.0+ = full signal (1.0)
    (-skew).clamp(0.0, 1.0)
}

/// Signal from Amihud illiquidity spike: current vs baseline.
pub fn amihud_spike_signal(amihud_current: f64, amihud_baseline: f64) -> f64 {
    if amihud_current.is_nan() || amihud_baseline.is_nan() || amihud_baseline <= 0.0 {
        return 0.0;
    }
    let ratio = amihud_current / amihud_baseline;
    if ratio <= 1.0 {
        return 0.0;
    }
    // Scale: 1x = no spike, 2x = moderate (0.5), 3x+ = full signal (1.0)
    ((ratio - 1.0) / 2.0).clamp(0.0, 1.0)
}

/// Signal from absorption ratio. High AR = systemic coupling = crisis.
pub fn absorption_ratio_signal(ar: f64, n_assets: usize) -> f64 {
    if ar.is_nan() || n_assets < 2 {
        return 0.0;
    }
    let baseline = 1.0 / n_assets as f64;
    let span = 0.9 - baseline;
    if span <= 0.0 {
        return 0.0;
    }
    ((ar - baseline) / span).clamp(0.0, 1.0)
}

/// Signal from jump variance fraction. High JV/RV = jump-driven stress.
pub fn jump_risk_signal_converter(jv: f64, rv: f64) -> f64 {
    if jv.is_nan() || rv.is_nan() || rv <= 0.0 {
        return 0.0;
    }
    let fraction = jv / rv;
    // Scale: 0.1 = some jumps, 0.5 = jump-dominated -> full signal
    (fraction / 0.5).clamp(0.0, 1.0)
}

// ── NN signal converters ──────────────────────────────────

/// Convert M-LNN result to signal [0,1].
/// Uses confidence directly if bubble filter passes, halved otherwise.
pub fn mlnn_signal(confidence: f64, is_bubble: bool) -> f64 {
    if confidence.is_nan() {
        return 0.0;
    }
    if is_bubble {
        confidence.clamp(0.0, 1.0)
    } else {
        (confidence * 0.5).clamp(0.0, 1.0)
    }
}

/// Convert P-LNN result to signal [0,1].
pub fn plnn_signal(confidence: f64, is_bubble: bool) -> f64 {
    if confidence.is_nan() {
        return 0.0;
    }
    if is_bubble {
        confidence.clamp(0.0, 1.0)
    } else {
        (confidence * 0.5).clamp(0.0, 1.0)
    }
}

// ── Market regime signal converters ───────────────────────

/// Convert z-score to [0,1] stress signal. Higher z = more stress.
/// Maps z in [0, 4] to [0, 1]; below 0 returns 0.
pub fn zscore_stress_signal(z: f64) -> f64 {
    if z.is_nan() {
        return 0.0;
    }
    (z / 4.0).clamp(0.0, 1.0)
}

/// Convert percentile rank [0,1] to stress signal [0,1].
pub fn percentile_stress_signal(pct: f64) -> f64 {
    if pct.is_nan() {
        return 0.0;
    }
    pct.clamp(0.0, 1.0)
}

/// Signal from variance risk premium.
/// High VRP = fear; negative VRP = acute stress.
pub fn vrp_signal_converter(implied_var: f64, realized_var: f64) -> f64 {
    if implied_var.is_nan() || realized_var.is_nan() {
        return 0.0;
    }
    let vrp = implied_var - realized_var;
    if vrp < 0.0 {
        // Negative VRP: realized exceeds implied -> acute stress
        (-vrp / 0.05).clamp(0.0, 1.0)
    } else {
        // High positive VRP: fear premium
        (vrp / 0.10).clamp(0.0, 1.0)
    }
}

/// Signal from VIX term structure slope. Negative slope (backwardation) = stress.
pub fn vix_slope_signal(slope: f64) -> f64 {
    if slope.is_nan() {
        return 0.0;
    }
    if slope >= 0.0 {
        return 0.0;
    }
    (-slope / 10.0).clamp(0.0, 1.0)
}

/// Signal from credit spread z-score. Widening spreads = stress.
pub fn credit_spread_signal(spread_z: f64) -> f64 {
    if spread_z.is_nan() {
        return 0.0;
    }
    (spread_z / 3.0).clamp(0.0, 1.0)
}

/// Signal from largest eigenvalue fraction of cross-asset correlation matrix.
/// lambda_frac approaching 1.0 = all assets moving together = crisis regime.
pub fn eigenvalue_signal(lambda_frac: f64) -> f64 {
    if lambda_frac.is_nan() {
        return 0.0;
    }
    // Scale from 0.3 (normal) to 0.8 (crisis)
    ((lambda_frac - 0.3) / 0.5).clamp(0.0, 1.0)
}

/// Return the list of categories that have signals > 0.6 in the given components.
pub fn agreeing_categories(components: &HashMap<String, f64>) -> Vec<String> {
    let cats = categories();
    let mut result = Vec::new();
    for (cat_name, keys) in &cats {
        let cat_signals: Vec<f64> = keys
            .iter()
            .filter_map(|k| components.get(*k).copied())
            .filter(|v| !v.is_nan())
            .collect();
        if !cat_signals.is_empty() {
            let max_sig = cat_signals
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if max_sig > 0.6 {
                result.push(cat_name.to_string());
            }
        }
    }
    result.sort();
    result
}
