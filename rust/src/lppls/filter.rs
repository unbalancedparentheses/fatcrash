use super::model::LpplsParams;

/// Sornette qualifying filter for LPPLS fits.
/// A fit must pass all constraints to be considered valid.
pub struct FilterConfig {
    pub m_min: f64,
    pub m_max: f64,
    pub omega_min: f64,
    pub omega_max: f64,
    pub max_damping: f64,
    pub min_oscillations: f64,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            m_min: 0.1,
            m_max: 0.9,
            omega_min: 6.0,
            omega_max: 13.0,
            max_damping: 0.7,
            min_oscillations: 2.5,
        }
    }
}

/// Check if LPPLS parameters pass Sornette's qualifying constraints.
/// `t_start` and `t_end` are the first and last time values of the fitting window,
/// used for the oscillation count check.
pub fn passes_filter(
    params: &LpplsParams,
    config: &FilterConfig,
    t_start: f64,
    t_end: f64,
) -> bool {
    // 1. m in [0.1, 0.9]
    if params.m < config.m_min || params.m > config.m_max {
        return false;
    }

    // 2. omega in [6, 13] (Nielsen et al. 2024 filter)
    if params.omega < config.omega_min || params.omega > config.omega_max {
        return false;
    }

    // 3. B < 0 for bubble (price must be super-exponential)
    if params.b >= 0.0 {
        return false;
    }

    // 4. Damping ratio: |m*B| / (omega * |C|) should indicate oscillations are damped
    let c_amp = params.c_amplitude();
    if c_amp > 0.0 {
        let damping = (params.m * params.b).abs() / (params.omega * c_amp);
        if damping > config.max_damping {
            return false;
        }
    }

    // 5. Minimum oscillations in the fitting window
    let n_osc = count_oscillations(params, t_start, t_end);
    if n_osc < config.min_oscillations {
        return false;
    }

    true
}

/// Count number of oscillations in the fit window.
pub fn count_oscillations(params: &LpplsParams, t_start: f64, t_end: f64) -> f64 {
    let dt_start = params.tc - t_start;
    let dt_end = params.tc - t_end;
    if dt_start <= 0.0 || dt_end <= 0.0 {
        return 0.0;
    }
    params.omega / (2.0 * std::f64::consts::PI) * (dt_start.ln() - dt_end.ln()).abs()
}
