//! Tail persistence (Sankaran et al., 2026; Rawat, Yadav, Sharma & Yadav, 2026).
//!
//! ```text
//! P(u) = E[X | X > Q(u)] / Q(u)
//! ```
//!
//! This is the ES/VaR ratio at level u: how far, in units of the threshold,
//! observations overshoot once they pass it. Under regular variation with
//! extreme value index 0 < xi < 1, Karamata's theorem gives
//! `P(u) -> 1 / (1 - xi)` as u -> 1. For xi <= 0 the limit is 1.
//!
//! The asymptotic value depends on xi alone, so it carries no information
//! that Hill or DEH do not already have. The empirical finite-level
//! `P(u)` does depend on location and scale, which makes it the more
//! useful diagnostic.

use crate::utils::sorted_desc_positive;

/// Asymptotic persistence `1 / (1 - xi)`. Returns 1 for `xi <= 0` and
/// infinity for `xi >= 1` (the mean excess is infinite).
pub fn asymptotic_persistence(xi: f64) -> f64 {
    if xi.is_nan() {
        f64::NAN
    } else if xi <= 0.0 {
        1.0
    } else if xi >= 1.0 {
        f64::INFINITY
    } else {
        1.0 / (1.0 - xi)
    }
}

/// Empirical persistence at quantile level `u` on descending-sorted positive
/// data: the mean of the `m = ceil((1 - u) n)` largest values divided by the
/// next order statistic (the empirical u-quantile).
pub fn empirical_persistence_sorted(sorted_desc: &[f64], u: f64) -> f64 {
    let n = sorted_desc.len();
    if n < 2 || !(u > 0.0 && u < 1.0) {
        return f64::NAN;
    }
    let m = (((1.0 - u) * n as f64).ceil() as usize).clamp(1, n - 1);
    let q = sorted_desc[m];
    if q <= 0.0 {
        return f64::NAN;
    }
    sorted_desc[..m].iter().sum::<f64>() / m as f64 / q
}

/// Empirical tail persistence of `data` at level `u` (e.g. 0.95).
pub fn tail_persistence_slice(data: &[f64], u: f64, use_abs: bool) -> f64 {
    empirical_persistence_sorted(&sorted_desc_positive(data, use_abs), u)
}

/// Rolling empirical tail persistence.
pub fn tail_persistence_rolling_slice(
    data: &[f64],
    window: usize,
    u: f64,
    use_abs: bool,
) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window < 10 || window > n {
        return out;
    }
    for i in (window - 1)..n {
        out[i] = tail_persistence_slice(&data[(i + 1 - window)..=i], u, use_abs);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn asymptotic_limits() {
        assert_eq!(asymptotic_persistence(-0.2), 1.0);
        assert!((asymptotic_persistence(0.5) - 2.0).abs() < 1e-12);
        assert!(asymptotic_persistence(1.0).is_infinite());
    }

    #[test]
    fn pareto_matches_limit() {
        // Exact Pareto: P(u) = alpha / (alpha - 1) = 1 / (1 - xi) at every u.
        let mut rng = StdRng::seed_from_u64(5);
        let alpha = 3.0;
        let xs: Vec<f64> = (0..200_000)
            .map(|_| rng.gen::<f64>().powf(-1.0 / alpha))
            .collect();
        let p = tail_persistence_slice(&xs, 0.99, false);
        assert!((p - 1.5).abs() < 0.05, "p={p}");
    }

    #[test]
    fn gaussian_is_near_one() {
        use rand_distr::StandardNormal;
        let mut rng = StdRng::seed_from_u64(9);
        let xs: Vec<f64> = (0..100_000)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        let p = tail_persistence_slice(&xs, 0.999, true);
        assert!(p > 1.0 && p < 1.15, "p={p}");
    }
}
