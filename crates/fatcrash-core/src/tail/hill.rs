/// Hill (1975) estimator for tail index alpha.
/// Given sorted descending absolute exceedances, alpha = k / sum(ln(x_i / x_{k+1}))
/// Returns alpha (tail index). Lower alpha = fatter tails.
pub fn hill_estimate(sorted_desc: &[f64], k: usize) -> f64 {
    if k == 0 || k >= sorted_desc.len() {
        return f64::NAN;
    }
    let x_k = sorted_desc[k];
    if x_k <= 0.0 {
        return f64::NAN;
    }

    let sum_log: f64 = sorted_desc[..k].iter().map(|&x| (x / x_k).ln()).sum();

    if sum_log <= 0.0 {
        return f64::NAN;
    }

    k as f64 / sum_log
}

/// Compute Hill estimator for a given array of returns.
/// `k` is the number of order statistics to use. If None, uses sqrt(n).
/// `use_abs` controls whether to use absolute values (for two-sided tail).
pub fn hill_estimator_slice(data: &[f64], k: Option<usize>, use_abs: bool) -> f64 {
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
    if n < 2 {
        return f64::NAN;
    }

    values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let k = k.unwrap_or_else(|| (n as f64).sqrt() as usize);
    let k = k.min(n.saturating_sub(1)).max(1);

    hill_estimate(&values, k)
}

/// Rolling Hill estimator over a window.
pub fn hill_rolling_slice(data: &[f64], window: usize, k: Option<usize>, use_abs: bool) -> Vec<f64> {
    let n = data.len();

    if window > n || window < 3 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        let mut values: Vec<f64> = if use_abs {
            slice
                .iter()
                .map(|x| x.abs())
                .filter(|x| x.is_finite() && *x > 0.0)
                .collect()
        } else {
            slice
                .iter()
                .copied()
                .filter(|x| x.is_finite() && *x > 0.0)
                .collect()
        };

        let vn = values.len();
        if vn < 2 {
            continue;
        }

        values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let ki = k.unwrap_or_else(|| (vn as f64).sqrt() as usize);
        let ki = ki.min(vn.saturating_sub(1)).max(1);

        result[i] = hill_estimate(&values, ki);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hill_pareto() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let alpha_true = 2.0;
        let n = 10000;
        let samples: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                u.powf(-1.0 / alpha_true)
            })
            .collect();

        let mut sorted: Vec<f64> = samples;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let k = (n as f64).sqrt() as usize;
        let alpha_hat = hill_estimate(&sorted, k);

        assert!(
            (alpha_hat - alpha_true).abs() < 0.5,
            "Hill estimate {} too far from true alpha {}",
            alpha_hat,
            alpha_true
        );
    }

    #[test]
    fn test_hill_empty_input() {
        let empty: Vec<f64> = vec![];
        let result = hill_estimate(&empty, 0);
        assert!(result.is_nan());
    }
}
