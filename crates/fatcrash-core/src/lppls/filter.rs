use super::model::LpplsParams;

/// Sornette qualifying filter for LPPLS fits.
/// Based on Sornette et al. (2015) "Real-time prediction and post-mortem analysis
/// of the Shanghai 2015 stock market bubble and crash", Table 1, Filter Condition 2.
///
/// A fit must pass ALL constraints to be considered a valid bubble signal.
pub struct FilterConfig {
    /// Power law exponent range. Sornette: [0.01, 0.99].
    pub m_min: f64,
    pub m_max: f64,
    /// Log-periodic frequency range. Sornette: [6, 13].
    pub omega_min: f64,
    pub omega_max: f64,
    /// Minimum damping D = m|B|/(ω|C|). Sornette Condition 2: D ≥ 1.0.
    /// Higher D means the super-exponential trend dominates over oscillations.
    /// A bubble must have strong growth, not just wiggles.
    pub min_damping: f64,
    /// Minimum number of log-periodic oscillations in the fitting window.
    /// Sornette: O ≥ 2.5.
    pub min_oscillations: f64,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            m_min: 0.01,
            m_max: 0.99,
            omega_min: 6.0,
            omega_max: 13.0,
            min_damping: 1.0,
            min_oscillations: 2.5,
        }
    }
}

/// Check if LPPLS parameters pass Sornette's qualifying constraints.
///
/// Five conditions (all must pass):
/// 1. m ∈ [0.01, 0.99] — power law exponent in valid range
/// 2. ω ∈ [6, 13] — log-periodic frequency in valid range
/// 3. B < 0 — super-exponential growth (price accelerates toward tc)
/// 4. D = m|B|/(ω|C|) ≥ min_damping — trend dominates oscillations
/// 5. O = ω/(2π) * |ln(tc-t1) - ln(tc-t2)| ≥ 2.5 — enough oscillation cycles
pub fn passes_filter(
    params: &LpplsParams,
    config: &FilterConfig,
    t_start: f64,
    t_end: f64,
) -> bool {
    // 1. m in valid range
    if params.m < config.m_min || params.m > config.m_max {
        return false;
    }

    // 2. omega in valid range
    if params.omega < config.omega_min || params.omega > config.omega_max {
        return false;
    }

    // 3. B < 0 — crash/correction expected (price decreases at tc)
    if params.b >= 0.0 {
        return false;
    }

    // 4. Damping: D = m|B| / (ω|C|) must be ≥ min_damping.
    //    High D = trend dominates, oscillations are decorations (good).
    //    Low D = oscillations dominate the trend (noise, reject).
    let c_amp = params.c_amplitude();
    if c_amp > 1e-15 {
        let damping = (params.m * params.b.abs()) / (params.omega * c_amp);
        if damping < config.min_damping {
            return false;
        }
    }
    // If c_amp ≈ 0, no oscillations at all — pure power law, which is fine.

    // 5. Minimum oscillations in the fitting window
    let n_osc = count_oscillations(params, t_start, t_end);
    if n_osc < config.min_oscillations {
        return false;
    }

    true
}

/// Count number of log-periodic oscillations in the fit window.
/// O = ω/(2π) * |ln(tc - t_start) - ln(tc - t_end)|
pub fn count_oscillations(params: &LpplsParams, t_start: f64, t_end: f64) -> f64 {
    let dt_start = params.tc - t_start;
    let dt_end = params.tc - t_end;
    if dt_start <= 0.0 || dt_end <= 0.0 {
        return 0.0;
    }
    params.omega / (2.0 * std::f64::consts::PI) * (dt_start.ln() - dt_end.ln()).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_bubble_passes() {
        let params = LpplsParams {
            tc: 100.0,
            m: 0.5,
            omega: 10.0,
            a: 10.0,
            b: -1.0,
            c1: 0.02,
            c2: 0.02,
        };
        let config = FilterConfig::default();
        // D = 0.5 * 1.0 / (10.0 * 0.0283) = 0.5/0.283 ≈ 1.77 — passes D ≥ 1.0
        // O = 10/(2π) * |ln(100) - ln(10)| = 1.59 * 2.30 ≈ 3.66 — passes O ≥ 2.5
        assert!(passes_filter(&params, &config, 0.0, 90.0));
    }

    #[test]
    fn test_positive_b_rejected() {
        let params = LpplsParams {
            tc: 100.0,
            m: 0.5,
            omega: 8.0,
            a: 10.0,
            b: 0.5, // positive B = no bubble
            c1: 0.02,
            c2: 0.02,
        };
        assert!(!passes_filter(&params, &FilterConfig::default(), 0.0, 80.0));
    }

    #[test]
    fn test_low_damping_rejected() {
        // Large oscillations relative to trend
        let params = LpplsParams {
            tc: 100.0,
            m: 0.1,
            omega: 12.0,
            a: 10.0,
            b: -0.01,
            c1: 0.5,
            c2: 0.5,
        };
        let config = FilterConfig::default();
        // D = 0.1 * 0.01 / (12.0 * 0.707) = 0.001 / 8.49 ≈ 0.0001 — fails D ≥ 1.0
        assert!(!passes_filter(&params, &config, 0.0, 80.0));
    }
}
