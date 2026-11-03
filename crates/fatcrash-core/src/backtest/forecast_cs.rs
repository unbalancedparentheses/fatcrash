//! Anytime-valid comparison of two tail quantile forecasts (Coelho, 2026).
//!
//! For lower-tail quantile forecasts `q_A`, `q_B` at level `tau`, the
//! quantile-score difference is
//!
//! ```text
//! d_t = tau (q_B - q_A) + [(q_A - y)+ - (q_B - y)+]
//! ```
//!
//! The first term is known before y_t arrives. The bracket is zero unless
//! y_t falls below one of the forecasts. So the information is in the
//! exceedances: the effective sample size is the number of exceedances,
//! not the number of days. A Diebold–Mariano t-test ignores this, and with
//! rare exceedances it tends to favour the less conservative forecaster.
//!
//! This module implements the paper's empirical-Bernstein confidence
//! sequence `(L_t, U_t)` for the running mean score difference. It needs no
//! distributional assumption and holds simultaneously for all t:
//! `P(exists t: mean_t not in (L_t, U_t)) <= delta`. L_t > 0 means B is better.
//! U_t < 0 means A is better.

/// Confidence-sequence output (NaN during warm-up).
#[derive(Clone, Debug)]
pub struct ForecastCs {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    /// Running mean of d_t over the evaluated rounds.
    pub mean: Vec<f64>,
    /// Number of rounds where y fell below at least one forecast.
    pub n_exceedances: usize,
    /// First index where the sequence excludes zero, with the sign of the
    /// winner: +1 = B better, -1 = A better.
    pub decision: Option<(usize, i8)>,
}

fn g(u: f64) -> f64 {
    if u >= 1.0 {
        f64::INFINITY
    } else if u < 1e-4 {
        0.5 + u / 3.0 + u * u / 4.0
    } else {
        (-(1.0 - u).ln() - u) / (u * u)
    }
}

/// Running lower bounds for the mean of `d` with predictable centres `c` and
/// lower ranges `r_minus` (so `d - c >= -r_minus`).
fn lower_bounds(d: &[f64], c: &[f64], r_minus: &[f64], lambdas: &[f64], delta: f64) -> Vec<f64> {
    let kf = lambdas.len() as f64;
    let target = (2.0 / delta).ln();
    let mut alive = vec![true; lambdas.len()];
    let mut v = vec![0.0; lambdas.len()];
    let mut sum = 0.0;
    let mut out = Vec::with_capacity(d.len());
    for t in 0..d.len() {
        let dev = d[t] - c[t];
        for (k, &lam) in lambdas.iter().enumerate() {
            if alive[k] && lam * r_minus[t] >= 1.0 {
                alive[k] = false;
            }
            if alive[k] {
                v[k] += lam * lam * g(lam * r_minus[t]) * dev * dev;
            }
        }
        sum += d[t];
        let tf = (t + 1) as f64;
        // log Phi(m) = logsumexp_k [lam_k (sum - t m) - V_k] - log K, decreasing in m.
        let log_phi = |m: f64| -> f64 {
            let terms: Vec<f64> = lambdas
                .iter()
                .zip(&v)
                .zip(&alive)
                .filter(|(_, &a)| a)
                .map(|((&lam, &vk), _)| lam * (sum - tf * m) - vk)
                .collect();
            let mx = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            mx + terms.iter().map(|x| (x - mx).exp()).sum::<f64>().ln() - kf.ln()
        };
        // Bracket: max term alone gives Phi between max/K and max.
        let roots = |extra: f64| {
            lambdas
                .iter()
                .zip(&v)
                .zip(&alive)
                .filter(|(_, &a)| a)
                .map(|((&lam, &vk), _)| (sum - (vk + extra) / lam) / tf)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let (mut lo, mut hi) = (roots(target + kf.ln()), roots(target));
        if !lo.is_finite() {
            out.push(f64::NEG_INFINITY);
            continue;
        }
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if log_phi(mid) > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        out.push(0.5 * (lo + hi));
    }
    out
}

/// Compare lower-tail quantile forecasts `q_a`, `q_b` (level `tau`, e.g. 0.025)
/// against outcomes `y`.
///
/// The first `warmup` rounds only set the bet scale
/// `lambda_0 = 0.5 / max range` and are excluded from the sequence.
/// `n_lambdas` geometric bet sizes `lambda_0 2^-k` are mixed uniformly.
pub fn compare_quantile_forecasts(
    y: &[f64],
    q_a: &[f64],
    q_b: &[f64],
    tau: f64,
    delta: f64,
    warmup: usize,
    n_lambdas: usize,
) -> Result<ForecastCs, String> {
    let n = y.len();
    if q_a.len() != n || q_b.len() != n {
        return Err("y, q_a, q_b must have the same length".into());
    }
    if !(tau > 0.0 && tau < 0.5) || !(delta > 0.0 && delta < 1.0) || n_lambdas == 0 {
        return Err("require 0<tau<0.5, 0<delta<1, n_lambdas>=1".into());
    }
    if warmup == 0 || warmup >= n {
        return Err("need 0 < warmup < len(y)".into());
    }
    let mut d = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let mut r_minus = Vec::with_capacity(n);
    let mut r_plus = Vec::with_capacity(n);
    let mut n_exc = 0;
    for t in 0..n {
        let (a, b, yt) = (q_a[t], q_b[t], y[t]);
        let ct = tau * (b - a);
        d.push(ct + ((a - yt).max(0.0) - (b - yt).max(0.0)));
        c.push(ct);
        r_minus.push((b - a).max(0.0));
        r_plus.push((a - b).max(0.0));
        if t >= warmup && yt < a.max(b) {
            n_exc += 1;
        }
    }
    let max_range = r_minus[..warmup]
        .iter()
        .chain(&r_plus[..warmup])
        .copied()
        .fold(0.0, f64::max);
    if max_range <= 0.0 {
        return Err("forecasts are identical during warm-up; cannot set the bet scale".into());
    }
    let lambdas: Vec<f64> = (0..n_lambdas)
        .map(|k| 0.5 / max_range * 0.5f64.powi(k as i32))
        .collect();

    let (d_e, c_e) = (&d[warmup..], &c[warmup..]);
    let lower = lower_bounds(d_e, c_e, &r_minus[warmup..], &lambdas, delta);
    let neg_d: Vec<f64> = d_e.iter().map(|x| -x).collect();
    let neg_c: Vec<f64> = c_e.iter().map(|x| -x).collect();
    let upper: Vec<f64> = lower_bounds(&neg_d, &neg_c, &r_plus[warmup..], &lambdas, delta)
        .into_iter()
        .map(|x| -x)
        .collect();

    let mut mean = Vec::with_capacity(n);
    let mut s = 0.0;
    for (i, x) in d_e.iter().enumerate() {
        s += x;
        mean.push(s / (i + 1) as f64);
    }
    let decision = lower
        .iter()
        .zip(&upper)
        .position(|(l, u)| *l > 0.0 || *u < 0.0)
        .map(|i| (warmup + i, if lower[i] > 0.0 { 1 } else { -1 }));

    let pad = |v: Vec<f64>| -> Vec<f64> {
        let mut out = vec![f64::NAN; warmup];
        out.extend(v);
        out
    };
    Ok(ForecastCs {
        lower: pad(lower),
        upper: pad(upper),
        mean: pad(mean),
        n_exceedances: n_exc,
        decision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    #[test]
    fn covers_mean_and_detects_better_forecast() {
        let mut rng = StdRng::seed_from_u64(41);
        let n = 20_000;
        let tau = 0.025;
        let y: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        // True 2.5% quantile of N(0,1) is -1.96; A is too conservative.
        let q_true = vec![-1.959_964; n];
        let q_bad = vec![-3.0; n];
        let cs = compare_quantile_forecasts(&y, &q_bad, &q_true, tau, 0.05, 200, 20).unwrap();
        let t = n - 1;
        assert!(cs.lower[t] <= cs.mean[t] && cs.mean[t] <= cs.upper[t]);
        assert!(cs.n_exceedances > 0);
        assert_eq!(
            cs.decision.map(|d| d.1),
            Some(1),
            "B (true quantile) should win"
        );
    }

    #[test]
    fn equal_forecasts_do_not_decide() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 5000;
        let y: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        // Same skill, different by noise only in the warm-up.
        let mut qa = vec![-1.96; n];
        qa[0] = -2.0;
        let qb = vec![-1.96; n];
        let cs = compare_quantile_forecasts(&y, &qa, &qb, 0.025, 0.05, 100, 20).unwrap();
        assert!(cs.decision.is_none());
    }
}
