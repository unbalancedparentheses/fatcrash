/// Pickands tail index estimator.
/// More robust than Hill — works for all tail types, not just heavy tails.
///
/// gamma = (1/ln(2)) * ln((X_{k} - X_{2k}) / (X_{2k} - X_{4k}))
///
/// where X_{k} is the k-th largest order statistic (1-indexed).
/// gamma > 0 indicates heavy tails (Frechet domain).
/// gamma = 0 indicates light tails (Gumbel domain).
/// gamma < 0 indicates bounded tails (Weibull domain).
pub fn pickands_estimate(sorted_desc: &[f64], k: usize) -> f64 {
    // Need at least 4k observations (indices 0..4k-1 in 0-indexed)
    if k == 0 || 4 * k > sorted_desc.len() {
        return f64::NAN;
    }

    // sorted_desc is descending, so index k-1 is X_{k} (the k-th largest)
    let x_k = sorted_desc[k - 1];
    let x_2k = sorted_desc[2 * k - 1];
    let x_4k = sorted_desc[4 * k - 1];

    let numerator = x_k - x_2k;
    let denominator = x_2k - x_4k;

    if denominator <= 0.0 || numerator <= 0.0 {
        return f64::NAN;
    }

    (1.0 / 2.0_f64.ln()) * (numerator / denominator).ln()
}

/// Compute Pickands estimator for a given array of data.
/// `k` is the order statistic parameter. If None, uses floor(sqrt(n) / 4).
/// `use_abs` controls whether to use absolute values (for two-sided tail).
pub fn pickands_estimator_slice(data: &[f64], k: Option<usize>, use_abs: bool) -> f64 {
    let mut values: Vec<f64> = if use_abs {
        data.iter()
            .map(|x| x.abs())
            .filter(|x| x.is_finite() && *x > 0.0)
            .collect()
    } else {
        data.iter()
            .copied()
            .filter(|x| x.is_finite() && *x > 0.0)
            .collect()
    };

    let n = values.len();
    if n < 4 {
        return f64::NAN;
    }

    values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Default k: ensure 4k <= n, so k <= n/4
    let k = k.unwrap_or_else(|| ((n as f64).sqrt() / 4.0).max(1.0) as usize);
    let k = k.min(n / 4).max(1);

    pickands_estimate(&values, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pickands_pareto() {
        // Pareto with alpha=2 has gamma = 1/alpha = 0.5
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let alpha_true = 2.0;
        let gamma_true = 1.0 / alpha_true; // 0.5
        let n = 20000;
        let samples: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                u.powf(-1.0 / alpha_true)
            })
            .collect();

        let mut sorted: Vec<f64> = samples;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let k = ((n as f64).sqrt() / 4.0) as usize;
        let gamma_hat = pickands_estimate(&sorted, k);

        assert!(
            (gamma_hat - gamma_true).abs() < 0.3,
            "Pickands estimate {} too far from true gamma {} (k={})",
            gamma_hat,
            gamma_true,
            k,
        );
    }

    #[test]
    fn test_pickands_agrees_with_hill_direction() {
        // Both Hill and Pickands should indicate heavy tails for Pareto data.
        // Hill gives alpha (tail index) > 0, Pickands gives gamma > 0 for heavy tails.
        // The Pickands estimator has higher variance than Hill, so we only check
        // that both agree on the qualitative conclusion: heavy tails (gamma > 0, alpha finite).
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(123);
        let alpha_true = 2.0;
        let n = 10000;
        let mut samples: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                u.powf(-1.0 / alpha_true)
            })
            .collect();

        samples.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let k_hill = (n as f64).sqrt() as usize;
        let hill_alpha = crate::tail::hill::hill_estimate(&samples, k_hill);

        let k_pick = ((n as f64).sqrt() / 4.0) as usize;
        let pickands_gamma = pickands_estimate(&samples, k_pick);

        // Both should be finite and positive (indicating heavy tails)
        assert!(
            hill_alpha.is_finite() && hill_alpha > 0.0,
            "Hill alpha should be positive, got {}",
            hill_alpha,
        );
        assert!(
            pickands_gamma.is_finite() && pickands_gamma > 0.0,
            "Pickands gamma should be positive for heavy-tailed data, got {}",
            pickands_gamma,
        );
    }

    #[test]
    fn test_pickands_insufficient_data() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut sorted = data;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let result = pickands_estimate(&sorted, 1);
        assert!(
            result.is_nan(),
            "Should be NaN with only 3 data points and k=1 (need 4k=4)"
        );
    }
}
