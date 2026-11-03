//! Conservative tail-index estimation (Gösgens, van Parys & Zwart, 2026).
//!
//! The Hill estimator overestimates alpha with non-negligible probability,
//! and overestimating alpha underestimates rare-event probabilities. The
//! rescaled Hill estimator
//!
//! ```text
//! alpha_hat(theta) = (1 - theta) / H_k,   H_k = (1/k) sum log(X_(i) / X_(k+1))
//! ```
//!
//! satisfies `-log P(alpha_hat(theta) > alpha) ~ k f(theta)` with rate
//! `f(theta) = -theta - log(1 - theta)`. Among scale-invariant estimators that
//! use the same k order statistics, no estimator with that decay rate has
//! smaller bias.
//!
//! Instead of picking theta directly, the caller sets a target probability
//! `delta` of overestimating alpha. Then `lambda = log(1/delta) / k` and
//! `theta = 1 + W0(-exp(-(1 + lambda)))` (Remark 1 of the paper).
//!
//! The guarantee is a large-deviations rate, not an exact finite-sample bound,
//! and assumes iid data. Overlapping or volatility-clustered returns make the
//! effective k smaller.

use crate::special::lambert_w0;
use crate::tail::hill::hill_estimate;
use crate::utils::sorted_desc_positive;

/// Rate function `f(theta) = -theta - log(1 - theta)`.
pub fn rate_function(theta: f64) -> f64 {
    -theta - (1.0 - theta).ln()
}

/// Shrinkage `theta` in `[0, 1)` whose decay rate `k f(theta)` equals
/// `log(1/delta)`.
pub fn theta_for_delta(k: usize, delta: f64) -> f64 {
    if k == 0 || !(delta > 0.0 && delta < 1.0) {
        return f64::NAN;
    }
    let lambda = (1.0 / delta).ln() / k as f64;
    (1.0 + lambert_w0(-(-(1.0 + lambda)).exp())).clamp(0.0, 1.0 - 1e-12)
}

/// Rescaled Hill estimate on data already sorted in descending order.
pub fn conservative_hill_estimate(sorted_desc: &[f64], k: usize, theta: f64) -> f64 {
    (1.0 - theta) * hill_estimate(sorted_desc, k)
}

/// Conservative Hill estimator.
///
/// Returns `(alpha, theta)`. `k` defaults to sqrt(n). `delta` is the target
/// probability of overestimating alpha (e.g. 0.05).
pub fn conservative_hill_slice(
    data: &[f64],
    k: Option<usize>,
    delta: f64,
    use_abs: bool,
) -> (f64, f64) {
    let values = sorted_desc_positive(data, use_abs);
    let n = values.len();
    if n < 3 {
        return (f64::NAN, f64::NAN);
    }
    let k = k.unwrap_or((n as f64).sqrt() as usize).clamp(1, n - 1);
    let theta = theta_for_delta(k, delta);
    (conservative_hill_estimate(&values, k, theta), theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn theta_solves_rate_equation() {
        for &(k, delta) in &[(10usize, 0.05), (50, 0.01), (500, 0.1)] {
            let theta = theta_for_delta(k, delta);
            let lambda = (1.0 / delta).ln() / k as f64;
            assert!((rate_function(theta) - lambda).abs() < 1e-10);
            assert!(theta > 0.0 && theta < 1.0);
        }
    }

    #[test]
    fn overestimates_rarely_on_pareto() {
        let mut rng = StdRng::seed_from_u64(7);
        let alpha = 3.0;
        let (n, k, delta, reps) = (2000usize, 100usize, 0.05, 400);
        let mut over_hill = 0;
        let mut over_cons = 0;
        for _ in 0..reps {
            let xs: Vec<f64> = (0..n)
                .map(|_| rng.gen::<f64>().powf(-1.0 / alpha))
                .collect();
            let (a_cons, theta) = conservative_hill_slice(&xs, Some(k), delta, false);
            let a_hill = a_cons / (1.0 - theta);
            over_hill += (a_hill > alpha) as usize;
            over_cons += (a_cons > alpha) as usize;
        }
        // Hill overestimates roughly half the time; the rescaled one rarely.
        assert!(over_hill > reps / 4, "hill over={over_hill}");
        assert!(
            (over_cons as f64) / (reps as f64) <= 2.0 * delta,
            "cons over={over_cons}"
        );
    }
}
