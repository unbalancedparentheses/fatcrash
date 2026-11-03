//! Expectile-based tail risk (Newey & Powell, 1987; Daouia, Girard & Stupfler, 2018).
//!
//! The tau-expectile e_tau solves `tau E(X - e)+ = (1 - tau) E(e - X)+`.
//! Expectiles are the only risk measure that is both coherent and
//! elicitable. Unlike VaR, they depend on the size of tail losses, not only
//! their frequency.
//!
//! At extreme levels the empirical expectile has too few observations, so
//! we extrapolate. For a heavy tail with 0 < gamma < 1,
//! `e_tau / q_tau -> (1/gamma - 1)^(-gamma)`. That gives the indirect
//! estimator `e_tau = (1/gamma - 1)^(-gamma) q_tau`, with Hill gamma and the
//! Weissman quantile `q_tau = X_(k+1) (k / (n (1 - tau)))^gamma`.
//! The expected loss beyond the expectile is
//! `XES_tau = E[X | X > e_tau] ~ e_tau / (1 - gamma)`, the same 1/(1 - gamma)
//! factor as asymptotic tail persistence.

use crate::tail::hill::hill_estimate;
use crate::utils::sorted_desc_positive;

/// Empirical tau-expectile by bisection (the defining equation is monotone in e).
pub fn empirical_expectile(data: &[f64], tau: f64) -> f64 {
    let x: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if x.is_empty() || !(tau > 0.0 && tau < 1.0) {
        return f64::NAN;
    }
    let score = |e: f64| -> f64 {
        x.iter()
            .map(|&v| {
                if v > e {
                    tau * (v - e)
                } else {
                    -(1.0 - tau) * (e - v)
                }
            })
            .sum()
    };
    let mut lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let mut hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if score(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Extreme expectile and expectile-based expected shortfall of the losses
/// (negated returns, same convention as `gpd_var_es_slice`) at level `tau`
/// (e.g. 0.99). `k` defaults to sqrt(n) of the positive losses.
///
/// Returns `(expectile, xes, gamma)`. Both are infinite when gamma >= 1.
pub fn expectile_var_es_slice(returns: &[f64], tau: f64, k: Option<usize>) -> (f64, f64, f64) {
    let losses: Vec<f64> = returns.iter().map(|r| -r).collect();
    let v = sorted_desc_positive(&losses, false);
    let n_pos = v.len();
    if n_pos < 10 || !(tau > 0.0 && tau < 1.0) {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    // Tail probabilities are relative to the full sample, not only the losses.
    let n = returns.iter().filter(|r| r.is_finite()).count() as f64;
    let k = k
        .unwrap_or((n_pos as f64).sqrt() as usize)
        .clamp(1, n_pos - 1);
    let gamma = 1.0 / hill_estimate(&v, k);
    if !gamma.is_finite() || gamma <= 0.0 {
        return (f64::NAN, f64::NAN, gamma);
    }
    if gamma >= 1.0 {
        return (f64::INFINITY, f64::INFINITY, gamma);
    }
    let q = v[k] * (k as f64 / (n * (1.0 - tau))).powf(gamma);
    let e = (1.0 / gamma - 1.0).powf(-gamma) * q;
    (e, e / (1.0 - gamma), gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn expectile_half_is_mean() {
        let x = [1.0, 2.0, 3.0, 10.0];
        assert!((empirical_expectile(&x, 0.5) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn extreme_expectile_matches_pareto_closed_form() {
        // Pareto(alpha=3) losses: q_tau = (1-tau)^(-1/3); e_tau/q_tau -> (1/gamma-1)^(-gamma) = 2^(-1/3).
        let mut rng = StdRng::seed_from_u64(51);
        let returns: Vec<f64> = (0..200_000)
            .map(|_| -rng.gen::<f64>().powf(-1.0 / 3.0))
            .collect();
        let tau = 0.999;
        let (e, xes, gamma) = expectile_var_es_slice(&returns, tau, Some(2000));
        let truth = 2f64.powf(-1.0 / 3.0) * (1.0f64 - tau).powf(-1.0 / 3.0);
        assert!((gamma - 1.0 / 3.0).abs() < 0.03, "gamma={gamma}");
        assert!((e / truth - 1.0).abs() < 0.1, "e={e} truth={truth}");
        assert!((xes / e - 1.5).abs() < 0.1);
    }
}
