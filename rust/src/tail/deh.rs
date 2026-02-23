use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Dekkers-Einmahl-de Haan (DEH) moment estimator for the tail index.
///
/// Valid for all three domains of attraction (unlike Hill which requires heavy tails).
/// Uses first and second moments of log-spacings:
///   M1 = (1/k) * sum_{i=1}^{k} log(X_(i) / X_(k+1))
///   M2 = (1/k) * sum_{i=1}^{k} [log(X_(i) / X_(k+1))]^2
///   gamma = M1 + 1 - (1/2) * (1 - M1^2/M2)^(-1)
///
/// Returns gamma (extreme value index):
///   gamma > 0: heavy tails (Frechet)
///   gamma = 0: light tails (Gumbel)
///   gamma < 0: bounded tails (Weibull)
fn deh_estimate(sorted_desc: &[f64], k: usize) -> f64 {
    if k == 0 || k >= sorted_desc.len() {
        return f64::NAN;
    }

    let x_kp1 = sorted_desc[k]; // X_(k+1) in 0-indexed
    if x_kp1 <= 0.0 {
        return f64::NAN;
    }

    let mut m1 = 0.0;
    let mut m2 = 0.0;

    for &x in sorted_desc.iter().take(k) {
        let log_spacing = (x / x_kp1).ln();
        m1 += log_spacing;
        m2 += log_spacing * log_spacing;
    }

    m1 /= k as f64;
    m2 /= k as f64;

    if m2 < 1e-15 {
        return f64::NAN;
    }

    let ratio = m1 * m1 / m2;
    let denom = 1.0 - ratio;

    if denom.abs() < 1e-15 {
        return f64::NAN;
    }

    m1 + 1.0 - 0.5 / denom
}

/// Compute DEH moment estimator for a given array of data.
/// `k` is the number of order statistics. If None, uses sqrt(n).
/// `use_abs` controls whether to use absolute values.
#[pyfunction]
#[pyo3(signature = (data, k=None, use_abs=true))]
pub fn deh_estimator(
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

    let n = values.len();
    if n < 4 {
        return Ok(f64::NAN);
    }

    values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let k = k.unwrap_or_else(|| (n as f64).sqrt() as usize);
    let k = k.min(n.saturating_sub(1)).max(2);

    Ok(deh_estimate(&values, k))
}

/// Rolling DEH estimator over a window.
#[pyfunction]
#[pyo3(signature = (data, window, k=None, use_abs=true))]
pub fn deh_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    k: Option<usize>,
    use_abs: Option<bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    let use_abs = use_abs.unwrap_or(true);

    if window > n || window < 4 {
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

        let vn = values.len();
        if vn < 4 {
            continue;
        }

        values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let ki = k.unwrap_or_else(|| (vn as f64).sqrt() as usize);
        let ki = ki.min(vn.saturating_sub(1)).max(2);

        result[i] = deh_estimate(&values, ki);
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deh_pareto() {
        // Pareto with alpha=2 → gamma = 1/alpha = 0.5
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let alpha_true = 2.0;
        let gamma_true = 1.0 / alpha_true;
        let n = 10000;
        let samples: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                u.powf(-1.0 / alpha_true)
            })
            .collect();

        let mut sorted = samples;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let k = (n as f64).sqrt() as usize;
        let gamma_hat = deh_estimate(&sorted, k);

        assert!(
            gamma_hat.is_finite(),
            "DEH gamma should be finite, got {}",
            gamma_hat
        );
        assert!(
            (gamma_hat - gamma_true).abs() < 0.3,
            "DEH estimate {} too far from true gamma {}",
            gamma_hat,
            gamma_true,
        );
    }

    #[test]
    fn test_deh_gaussian() {
        // Gaussian → gamma = 0 (Gumbel domain)
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 10000;
        let samples: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal).abs())
            .filter(|x| *x > 0.0)
            .collect();

        let mut sorted = samples;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let k = (sorted.len() as f64).sqrt() as usize;
        let gamma_hat = deh_estimate(&sorted, k);

        assert!(
            gamma_hat.is_finite(),
            "DEH gamma should be finite, got {}",
            gamma_hat
        );
        // For Gaussian, gamma should be near 0 (within tolerance)
        assert!(
            gamma_hat < 0.3,
            "DEH gamma for Gaussian should be near 0, got {}",
            gamma_hat
        );
    }

    #[test]
    fn test_deh_insufficient_data() {
        let data = vec![3.0, 2.0, 1.0];
        let result = deh_estimate(&data, 1);
        // k=1 with 3 points — M2 might be degenerate
        // Just check it doesn't panic
        assert!(result.is_finite() || result.is_nan());
    }
}
