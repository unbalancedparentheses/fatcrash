//! Power-mean, Lehmer-mean, and reduced-bias (RB) estimators of the extreme
//! value index xi (Henriques-Rodrigues, Caeiro & Gomes, 2026).
//!
//! With log-excesses `V_i = log X_(i) - log X_(k+1)`, i = 1..k, and moments
//! `M(p) = (1/k) sum V_i^p`:
//!
//! - Power mean: `xi_PM(p) = (M(p) / Gamma(p+1))^(1/p)`, with the p = 0 limit
//!   `e^gamma * (prod V_i)^(1/k)`. `xi_PM(1)` is Hill.
//! - Lehmer mean: `xi_L(p) = M(p) / (p M(p-1))`, p > 0. `xi_L(1)` is Hill.
//!
//! Both carry a leading bias of order `A(n/k) = xi beta (n/k)^rho`. The RB
//! versions remove it with second-order parameters (rho, beta) estimated at a
//! high level `k1 = floor(n^0.999)`: rho by Fraga Alves, Gomes & de Haan
//! (2003), beta by Gomes & Martins (2002). The nonlinear version replaces
//! `1 - c A` by `2 - exp(c A)`, which in the paper's simulations flattens
//! the estimate as a function of k. That makes the choice of k less critical.
//!
//! All estimators return xi (alpha = 1/xi) and target heavy tails (xi > 0).

use crate::special::{ln_gamma, EULER_GAMMA};
use crate::utils::sorted_desc_positive;

/// Which estimator `reduced_bias_evi_slice` computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbKind {
    /// Bias-corrected Hill (Caeiro, Gomes & Pestana, 2005).
    Hill,
    /// RB power mean, linear correction (eq. 28/30).
    PowerMean,
    /// RB power mean, nonlinear correction `2 - exp(.)` (eq. 32).
    NonlinearPowerMean,
    /// RB Lehmer mean (eq. 29).
    Lehmer,
}

fn log_excesses(sorted_desc: &[f64], k: usize) -> Option<Vec<f64>> {
    if k == 0 || k >= sorted_desc.len() || sorted_desc[k] <= 0.0 {
        return None;
    }
    let base = sorted_desc[k].ln();
    Some(sorted_desc[..k].iter().map(|x| x.ln() - base).collect())
}

fn moment(v: &[f64], p: f64) -> f64 {
    v.iter().map(|&x| x.powf(p)).sum::<f64>() / v.len() as f64
}

fn finite(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        f64::NAN
    }
}

/// Power-mean EVI estimator on descending-sorted data, `p > -0.5`.
pub fn power_mean_estimate(sorted_desc: &[f64], k: usize, p: f64) -> f64 {
    let Some(v) = log_excesses(sorted_desc, k) else {
        return f64::NAN;
    };
    if p.abs() < 1e-12 {
        let mean_log = v.iter().map(|x| x.ln()).sum::<f64>() / k as f64;
        return finite(EULER_GAMMA.exp() * mean_log.exp());
    }
    finite((moment(&v, p).ln() - ln_gamma(p + 1.0)).exp().powf(1.0 / p))
}

/// Lehmer-mean (Lp) EVI estimator on descending-sorted data, `p > 0`.
pub fn lehmer_estimate(sorted_desc: &[f64], k: usize, p: f64) -> f64 {
    let Some(v) = log_excesses(sorted_desc, k) else {
        return f64::NAN;
    };
    if p <= 0.0 {
        return f64::NAN;
    }
    finite(moment(&v, p) / (p * moment(&v, p - 1.0)))
}

/// Second-order parameters `(rho, beta)` estimated at `k1 = floor(n^0.999)`.
///
/// `tau` is the Fraga Alves et al. tuning parameter: 0 suits `-1 <= rho < 0`
/// (typical for financial returns), 1 suits `rho < -1`.
pub fn second_order_params(sorted_desc: &[f64], tau: f64) -> (f64, f64) {
    let n = sorted_desc.len();
    if n < 20 {
        return (f64::NAN, f64::NAN);
    }
    let k1 = ((n as f64).powf(0.999) as usize).min(n - 1);
    let Some(v) = log_excesses(sorted_desc, k1) else {
        return (f64::NAN, f64::NAN);
    };
    let (m1, m2, m3) = (
        moment(&v, 1.0),
        moment(&v, 2.0) / 2.0,
        moment(&v, 3.0) / 6.0,
    );
    let t = if tau.abs() < 1e-12 {
        (m1.ln() - 0.5 * m2.ln()) / (0.5 * m2.ln() - m3.ln() / 3.0)
    } else {
        (m1.powf(tau) - m2.powf(tau / 2.0)) / (m2.powf(tau / 2.0) - m3.powf(tau / 3.0))
    };
    let rho = -(3.0 * (t - 1.0) / (t - 3.0)).abs();
    if !rho.is_finite() || rho >= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    // Gomes–Martins beta from scaled log-spacings U_i = i (log X_(i) - log X_(i+1)).
    let kf = k1 as f64;
    let (mut d_rho, mut d0, mut dd_rho, mut dd_2rho) = (0.0, 0.0, 0.0, 0.0);
    for i in 1..=k1 {
        let u = i as f64 * (sorted_desc[i - 1].ln() - sorted_desc[i].ln());
        let w = (i as f64 / kf).powf(-rho);
        d_rho += w;
        d0 += u;
        dd_rho += w * u;
        dd_2rho += w * w * u;
    }
    let (d_rho, d0, dd_rho, dd_2rho) = (d_rho / kf, d0 / kf, dd_rho / kf, dd_2rho / kf);
    let beta = (kf / n as f64).powf(rho) * (d_rho * d0 - dd_rho) / (d_rho * dd_rho - dd_2rho);
    (rho, finite(beta))
}

/// Correction constant c with `xi_RB = xi (1 - c beta (n/k)^rho)`.
fn correction_constant(kind: RbKind, p: f64, rho: f64) -> f64 {
    match kind {
        RbKind::Hill => 1.0 / (1.0 - rho),
        RbKind::PowerMean | RbKind::NonlinearPowerMean => {
            if p.abs() < 1e-12 {
                -(1.0 - rho).ln() / rho
            } else {
                (1.0 - (1.0 - rho).powf(p)) / (p * rho * (1.0 - rho).powf(p))
            }
        }
        RbKind::Lehmer => 1.0 / (1.0 - rho).powf(p),
    }
}

/// Reduced-bias EVI estimate on descending-sorted data with known `(rho, beta)`.
pub fn reduced_bias_estimate(
    sorted_desc: &[f64],
    k: usize,
    kind: RbKind,
    p: f64,
    rho: f64,
    beta: f64,
) -> f64 {
    let n = sorted_desc.len() as f64;
    let base = match kind {
        RbKind::Hill => power_mean_estimate(sorted_desc, k, 1.0),
        RbKind::PowerMean | RbKind::NonlinearPowerMean => power_mean_estimate(sorted_desc, k, p),
        RbKind::Lehmer => lehmer_estimate(sorted_desc, k, p),
    };
    let a = correction_constant(kind, p, rho) * beta * (n / k as f64).powf(rho);
    let factor = match kind {
        RbKind::NonlinearPowerMean => 2.0 - a.exp(),
        _ => 1.0 - a,
    };
    finite(base * factor)
}

fn default_k(n: usize, k: Option<usize>) -> usize {
    k.unwrap_or((n as f64).sqrt() as usize).clamp(1, n - 1)
}

/// Power-mean EVI estimator. `k` defaults to sqrt(n).
pub fn power_mean_evi_slice(data: &[f64], k: Option<usize>, p: f64, use_abs: bool) -> f64 {
    let v = sorted_desc_positive(data, use_abs);
    if v.len() < 3 {
        return f64::NAN;
    }
    power_mean_estimate(&v, default_k(v.len(), k), p)
}

/// Lehmer-mean EVI estimator. `k` defaults to sqrt(n).
pub fn lehmer_evi_slice(data: &[f64], k: Option<usize>, p: f64, use_abs: bool) -> f64 {
    let v = sorted_desc_positive(data, use_abs);
    if v.len() < 3 {
        return f64::NAN;
    }
    lehmer_estimate(&v, default_k(v.len(), k), p)
}

/// Second-order parameters `(rho, beta)` from raw data.
pub fn second_order_params_slice(data: &[f64], tau: f64, use_abs: bool) -> (f64, f64) {
    second_order_params(&sorted_desc_positive(data, use_abs), tau)
}

/// Reduced-bias EVI estimator with (rho, beta) estimated from the data.
/// Needs at least 20 positive observations. `k` defaults to sqrt(n).
pub fn reduced_bias_evi_slice(
    data: &[f64],
    k: Option<usize>,
    kind: RbKind,
    p: f64,
    tau: f64,
    use_abs: bool,
) -> f64 {
    let v = sorted_desc_positive(data, use_abs);
    let (rho, beta) = second_order_params(&v, tau);
    if rho.is_nan() || beta.is_nan() {
        return f64::NAN;
    }
    reduced_bias_estimate(&v, default_k(v.len(), k), kind, p, rho, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tail::hill::hill_estimate;
    use rand::prelude::*;

    fn sorted(mut xs: Vec<f64>) -> Vec<f64> {
        xs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        xs
    }

    /// Student-t-like tail via Burr: P(X > x) = (1 + x^(-rho/xi))^(1/rho),
    /// a standard test model with second-order parameter rho.
    fn burr(rng: &mut StdRng, n: usize, xi: f64, rho: f64) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                (u.powf(rho) - 1.0).powf(-xi / rho)
            })
            .collect()
    }

    #[test]
    fn p_one_reduces_to_hill() {
        let mut rng = StdRng::seed_from_u64(1);
        let v = sorted(burr(&mut rng, 2000, 0.5, -1.0));
        let h = 1.0 / hill_estimate(&v, 100);
        assert!((power_mean_estimate(&v, 100, 1.0) - h).abs() < 1e-12);
        assert!((lehmer_estimate(&v, 100, 1.0) - h).abs() < 1e-12);
    }

    #[test]
    fn rb_estimators_cut_bias_on_burr() {
        let (xi, rho, n, k, reps) = (0.5, -0.5, 2000usize, 400usize, 60);
        let mut rng = StdRng::seed_from_u64(11);
        let mut err_hill = 0.0;
        let mut err_rb = 0.0;
        let mut err_nrb = 0.0;
        for _ in 0..reps {
            let v = sorted(burr(&mut rng, n, xi, rho));
            let (r, b) = second_order_params(&v, 0.0);
            err_hill += power_mean_estimate(&v, k, 1.0) - xi;
            err_rb += reduced_bias_estimate(&v, k, RbKind::Hill, 1.0, r, b) - xi;
            err_nrb += reduced_bias_estimate(&v, k, RbKind::NonlinearPowerMean, 0.5, r, b) - xi;
        }
        let (bh, brb, bnrb) = (
            err_hill / reps as f64,
            err_rb / reps as f64,
            err_nrb / reps as f64,
        );
        assert!(bh > 0.05, "Hill should be biased upward at large k: {bh}");
        assert!(
            brb.abs() < bh.abs() / 2.0,
            "RB-Hill bias {brb} vs Hill {bh}"
        );
        assert!(
            bnrb.abs() < bh.abs() / 2.0,
            "NRB-PM bias {bnrb} vs Hill {bh}"
        );
    }

    #[test]
    fn second_order_rho_negative() {
        let mut rng = StdRng::seed_from_u64(3);
        let v = sorted(burr(&mut rng, 5000, 0.5, -1.0));
        let (rho, beta) = second_order_params(&v, 0.0);
        assert!(rho < 0.0 && rho > -3.0, "rho={rho}");
        assert!(beta.is_finite() && beta > 0.0, "beta={beta}");
    }

    #[test]
    fn small_samples_are_nan() {
        assert!(
            reduced_bias_evi_slice(&[1.0, 2.0, 3.0], None, RbKind::Hill, 1.0, 0.0, true).is_nan()
        );
    }
}
