use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hill estimator for tail index alpha.
/// Given sorted descending absolute exceedances, alpha = k / sum(ln(x_i / x_k))
/// Returns alpha (tail index). Lower alpha = fatter tails.
///
/// Includes Taleb's small-sample bias correction:
///   alpha_corrected = alpha * (1 - 2/k) for k > 2
fn hill_estimate(sorted_desc: &[f64], k: usize) -> f64 {
    if k == 0 || k >= sorted_desc.len() {
        return f64::NAN;
    }
    let x_k = sorted_desc[k];
    if x_k <= 0.0 {
        return f64::NAN;
    }

    let sum_log: f64 = sorted_desc[..k]
        .iter()
        .map(|&x| (x / x_k).ln())
        .sum();

    if sum_log <= 0.0 {
        return f64::NAN;
    }

    let alpha = k as f64 / sum_log;

    // Taleb small-sample bias correction
    if k > 2 {
        alpha * (1.0 - 2.0 / k as f64)
    } else {
        alpha
    }
}

/// Compute Hill estimator for a given array of returns.
/// `k` is the number of order statistics to use. If None, uses sqrt(n).
/// `use_abs` controls whether to use absolute values (for two-sided tail).
#[pyfunction]
#[pyo3(signature = (data, k=None, use_abs=true))]
pub fn hill_estimator(
    data: PyReadonlyArray1<'_, f64>,
    k: Option<usize>,
    use_abs: Option<bool>,
) -> PyResult<f64> {
    let data = data.as_slice()?;
    let use_abs = use_abs.unwrap_or(true);

    let mut values: Vec<f64> = if use_abs {
        data.iter().map(|x| x.abs()).filter(|x| *x > 0.0).collect()
    } else {
        data.iter().copied().filter(|x| *x > 0.0).collect()
    };

    values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let n = values.len();
    let k = k.unwrap_or_else(|| (n as f64).sqrt() as usize);
    let k = k.min(n - 1).max(1);

    Ok(hill_estimate(&values, k))
}

/// Rolling Hill estimator over a window.
#[pyfunction]
#[pyo3(signature = (data, window, k=None, use_abs=true))]
pub fn hill_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    k: Option<usize>,
    use_abs: Option<bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    let use_abs = use_abs.unwrap_or(true);

    if window > n || window < 3 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        let mut values: Vec<f64> = if use_abs {
            slice.iter().map(|x| x.abs()).filter(|x| *x > 0.0).collect()
        } else {
            slice.iter().copied().filter(|x| *x > 0.0).collect()
        };

        values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let vn = values.len();
        let ki = k.unwrap_or_else(|| (vn as f64).sqrt() as usize);
        let ki = ki.min(vn.saturating_sub(1)).max(1);

        result[i] = hill_estimate(&values, ki);
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hill_pareto() {
        // Generate Pareto(alpha=2) samples: x = u^(-1/alpha) where u ~ Uniform(0,1)
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

        let mut sorted: Vec<f64> = samples.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let k = (n as f64).sqrt() as usize;
        let alpha_hat = hill_estimate(&sorted, k);

        // Should be close to 2.0 (within ~20% for this sample size)
        assert!(
            (alpha_hat - alpha_true).abs() < 0.5,
            "Hill estimate {} too far from true alpha {}",
            alpha_hat,
            alpha_true
        );
    }
}
