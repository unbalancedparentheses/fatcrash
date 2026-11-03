//! Anytime-valid ES backtest without VaR forecasts (Wang & Yu, 2026, Section 2.2).
//!
//! Tests H0: ES_p(L) <= r for iid losses L_t and a constant ES forecast r.
//! The ES e-statistic `(x - z)+ / ((1 - p)(r - z))` needs the true VaR z, which
//! is not reported. So the first `n_cal` losses give a one-sided binomial
//! lower confidence bound `z_delta` for VaR, and the monitoring process takes
//! the infimum of the betting product over `z in [z_delta, r]`:
//!
//! ```text
//! M_t = min{ inf_z prod_{s > n_cal} (1 - lambda + lambda e(L_s, r, z)), Gamma } / (1 + Gamma delta)
//! ```
//!
//! `M_t` is an e-process: `P(sup_t M_t >= 1/alpha) <= alpha` under H0, so the
//! ES forecast can be monitored every day and rejected the first time `M_t`
//! crosses `1/alpha`, with no multiple-testing penalty. Losses must be iid,
//! which daily returns are not. Apply it to standardized residuals or
//! declustered losses, or read it as a benchmark.

use crate::special::binom_cdf_lt;

/// Output of the e-backtest.
#[derive(Clone, Debug)]
pub struct EBacktest {
    /// E-process path for monitoring rounds `n_cal + 1 ..= T` (first value is
    /// after one monitored loss).
    pub e_process: Vec<f64>,
    /// Index into the input losses of the first rejection, if any.
    pub reject_at: Option<usize>,
    /// VaR lower confidence bound from the calibration window.
    pub var_lower_bound: f64,
    /// Constant betting fraction used.
    pub lambda: f64,
}

fn e_stat(x: f64, r: f64, z: f64, p: f64) -> f64 {
    if z >= r {
        // Boundary: 0/0 = 1, a/0 = infinity.
        return if x > r { f64::INFINITY } else { 1.0 };
    }
    (x - z).max(0.0) / ((1.0 - p) * (r - z))
}

/// Choose a constant lambda in `[0, 1 - p]` maximizing the average log factor
/// on the calibration losses at the VaR lower bound `z` (which is below r).
fn choose_lambda(cal: &[f64], r: f64, z: f64, p: f64) -> f64 {
    let grid = 50;
    (0..=grid)
        .map(|i| (1.0 - p) * i as f64 / grid as f64)
        .map(|lam| {
            let g: f64 = cal
                .iter()
                .map(|&x| (1.0 - lam + lam * e_stat(x, r, z, p)).ln())
                .sum();
            (lam, g)
        })
        .filter(|(_, g)| g.is_finite())
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(lam, _)| lam)
        .unwrap_or(0.0)
}

/// Run the ES-only e-backtest.
///
/// - `losses`: loss series (positive = loss).
/// - `r`: constant ES forecast at level `p` (e.g. p = 0.975).
/// - `n_cal`: calibration window length; monitoring starts after it.
/// - `delta`: VaR confidence-bound error level (paper uses small values like 0.01).
/// - `gamma`: cap Γ ≥ 1 on the infimum.
/// - `alpha`: test level; reject when the e-process reaches 1/alpha.
/// - `lambda`: betting fraction; chosen from the calibration window if None.
#[allow(clippy::too_many_arguments)]
pub fn es_e_backtest_iid(
    losses: &[f64],
    r: f64,
    p: f64,
    n_cal: usize,
    delta: f64,
    gamma: f64,
    alpha: f64,
    lambda: Option<f64>,
) -> Result<EBacktest, String> {
    if !(p > 0.0 && p < 1.0)
        || !(delta > 0.0 && delta < 1.0)
        || gamma < 1.0
        || !(alpha > 0.0 && alpha < 1.0)
    {
        return Err("require 0<p<1, 0<delta<1, gamma>=1, 0<alpha<1".into());
    }
    if n_cal < 10 || n_cal >= losses.len() {
        return Err("need 10 <= n_cal < len(losses)".into());
    }
    let cal = &losses[..n_cal];
    let mut sorted = cal.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // j_delta = max{ j in 0..=n_cal : P(Bin(n_cal, p) < j) <= delta }; z = L_(j).
    let j = (0..=n_cal)
        .rev()
        .find(|&j| binom_cdf_lt(n_cal, p, j) <= delta)
        .unwrap_or(0);
    if j == 0 {
        return Err("calibration window too short for this delta (VaR bound is -infinity)".into());
    }
    let z_lo = sorted[j - 1];
    let scale = 1.0 + gamma * delta;
    let threshold = 1.0 / alpha;
    let monitor = &losses[n_cal..];

    if z_lo > r {
        let lambda = lambda.unwrap_or(0.0);
        // VaR lower bound already exceeds the reported ES: maximal evidence.
        let v = gamma / scale;
        let reject_at = (v >= threshold).then_some(n_cal);
        return Ok(EBacktest {
            e_process: vec![v; monitor.len()],
            reject_at,
            var_lower_bound: z_lo,
            lambda,
        });
    }

    let lambda = lambda
        .unwrap_or_else(|| choose_lambda(cal, r, z_lo, p))
        .clamp(0.0, 1.0 - p);

    // Grid over [z_lo, r): log-spaced toward r where the e-statistic varies
    // fastest, plus the boundary z = r evaluated explicitly.
    let n_grid = 256;
    let mut zs: Vec<f64> = (0..n_grid)
        .map(|i| {
            let f = i as f64 / n_grid as f64;
            r - (r - z_lo) * (1.0 - f).powi(3)
        })
        .collect();
    zs.push(r);
    let mut log_m = vec![0.0f64; zs.len()];

    let mut e_process = Vec::with_capacity(monitor.len());
    let mut reject_at = None;
    for (t, &x) in monitor.iter().enumerate() {
        for (lm, &z) in log_m.iter_mut().zip(&zs) {
            if lambda > 0.0 {
                *lm += (1.0 - lambda + lambda * e_stat(x, r, z, p)).ln();
            }
        }
        let inf = log_m.iter().copied().fold(f64::INFINITY, f64::min).exp();
        let m = inf.min(gamma) / scale;
        if reject_at.is_none() && m >= threshold {
            reject_at = Some(n_cal + t);
        }
        e_process.push(m);
    }
    Ok(EBacktest {
        e_process,
        reject_at,
        var_lower_bound: z_lo,
        lambda,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// Pareto(alpha=3) losses: VaR_p = (1-p)^(-1/3), ES_p = 1.5 VaR_p.
    fn pareto(rng: &mut StdRng, n: usize) -> Vec<f64> {
        (0..n).map(|_| rng.gen::<f64>().powf(-1.0 / 3.0)).collect()
    }

    #[test]
    fn correct_forecast_rarely_rejected() {
        let p = 0.975;
        let es = 1.5 * (1.0f64 - p).powf(-1.0 / 3.0);
        let mut rng = StdRng::seed_from_u64(31);
        let mut rejections = 0;
        for _ in 0..100 {
            let l = pareto(&mut rng, 3000);
            let bt = es_e_backtest_iid(&l, es, p, 1000, 0.01, 50.0, 0.05, None).unwrap();
            rejections += bt.reject_at.is_some() as usize;
        }
        assert!(rejections <= 10, "rejections={rejections}");
    }

    #[test]
    fn underestimated_es_rejected() {
        let p = 0.975;
        let es = 1.5 * (1.0f64 - p).powf(-1.0 / 3.0);
        let mut rng = StdRng::seed_from_u64(32);
        let l = pareto(&mut rng, 5000);
        let bt = es_e_backtest_iid(&l, 0.7 * es, p, 1000, 0.01, 50.0, 0.05, None).unwrap();
        assert!(
            bt.reject_at.is_some(),
            "max e = {:?}",
            bt.e_process.iter().cloned().fold(0.0, f64::max)
        );
    }
}
