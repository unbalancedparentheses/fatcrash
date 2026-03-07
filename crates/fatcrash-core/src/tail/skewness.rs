/// Realized skewness from intraperiod returns.
///
/// RS = sqrt(W) * sum(r_i^3) / RV^(3/2)
///
/// where RV = sum(r_i^2) is realized variance (unannualized).
///
/// Negative skewness indicates asymmetric downside risk —
/// a precursor to crash events (Neuberger, 2012).
pub fn compute_realized_skewness(returns: &[f64], window: usize) -> f64 {
    let n = returns.len();
    if window < 3 || window > n {
        return f64::NAN;
    }

    let slice = &returns[(n - window)..];
    let w = slice.len() as f64;

    let rv: f64 = slice.iter().map(|r| r * r).sum();
    if rv <= 0.0 {
        return f64::NAN;
    }

    let m3: f64 = slice.iter().map(|r| r * r * r).sum();

    w.sqrt() * m3 / rv.powf(1.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realized_skewness_symmetric() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 2000;
        let returns: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.01)
            .collect();

        let skew = compute_realized_skewness(&returns, n);
        assert!(
            skew.abs() < 1.0,
            "Gaussian returns should have skewness near 0, got {}",
            skew
        );
    }

    #[test]
    fn test_realized_skewness_negative() {
        // Returns with large negative outliers should give negative skewness
        let mut returns = vec![0.01; 100];
        returns[50] = -0.10;
        returns[70] = -0.08;

        let skew = compute_realized_skewness(&returns, returns.len());
        assert!(
            skew < 0.0,
            "Returns with negative outliers should have negative skewness, got {}",
            skew
        );
    }

    #[test]
    fn test_realized_skewness_insufficient_data() {
        let returns = vec![0.01, 0.02];
        assert!(compute_realized_skewness(&returns, 3).is_nan());
    }
}
