use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils::ols_slope;

/// QQ estimator for the tail index.
///
/// Estimates tail index from slope of log-log QQ plot against exponential quantiles.
/// For the k largest observations (sorted descending), the i-th order statistic
/// (i=1,...,k) has survival probability ~ i/(k+1). Under a Pareto tail,
/// log(X_(i)) is linear in log(-log(i/(k+1))).
///
/// We regress log(X_(i)) vs the exponential quantile -log(i/(k+1)).
/// Slope = 1/alpha → alpha = 1/slope.
///
/// Returns alpha (tail index): lower = fatter tails.
fn qq_estimate(sorted_desc: &[f64], k: usize) -> f64 {
    let n = sorted_desc.len();
    if k < 3 || k > n {
        return f64::NAN;
    }

    let mut log_x: Vec<f64> = Vec::with_capacity(k);
    let mut exp_quantile: Vec<f64> = Vec::with_capacity(k);

    for (i, &x) in sorted_desc.iter().enumerate().take(k) {
        if x <= 0.0 {
            continue;
        }
        // Plotting position: the (i+1)-th largest has survival probability (i+1)/(k+1)
        let p = (i + 1) as f64 / (k + 1) as f64;
        let eq = -p.ln(); // Exponential quantile: -log(p) decreases as i increases
        if eq.is_finite() && eq > 0.0 {
            log_x.push(x.ln());
            exp_quantile.push(eq);
        }
    }

    if log_x.len() < 3 {
        return f64::NAN;
    }

    // OLS: log_x = a + slope * exp_quantile
    // For Pareto: log(X) ~ (1/alpha) * (-log(p)) + const
    // So slope = 1/alpha
    let slope = ols_slope(&exp_quantile, &log_x);

    if !slope.is_finite() || slope <= 0.0 {
        return f64::NAN;
    }

    1.0 / slope
}

/// Compute QQ tail index estimator for a given array of data.
/// `k` is the number of order statistics. If None, uses sqrt(n).
/// `use_abs` controls whether to use absolute values.
#[pyfunction]
#[pyo3(signature = (data, k=None, use_abs=true))]
pub fn qq_estimator(
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
    let k = k.min(n).max(3);

    Ok(qq_estimate(&values, k))
}

/// Rolling QQ estimator over a window.
#[pyfunction]
#[pyo3(signature = (data, window, k=None, use_abs=true))]
pub fn qq_rolling<'py>(
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
        let ki = ki.min(vn).max(3);

        result[i] = qq_estimate(&values, ki);
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qq_pareto() {
        // Pareto with alpha=2
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

        let mut sorted = samples;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let k = (n as f64).sqrt() as usize;
        let alpha_hat = qq_estimate(&sorted, k);

        assert!(
            alpha_hat.is_finite(),
            "QQ alpha should be finite, got {}",
            alpha_hat
        );
        assert!(
            (alpha_hat - alpha_true).abs() < 1.0,
            "QQ estimate {} too far from true alpha {}",
            alpha_hat,
            alpha_true,
        );
    }

    #[test]
    fn test_qq_gaussian_high_alpha() {
        // Gaussian should give high alpha (thin tails)
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
        let alpha_hat = qq_estimate(&sorted, k);

        assert!(
            alpha_hat.is_finite(),
            "QQ alpha should be finite, got {}",
            alpha_hat
        );
        assert!(
            alpha_hat > 2.0,
            "QQ alpha for Gaussian should be high, got {}",
            alpha_hat
        );
    }

    #[test]
    fn test_qq_insufficient_data() {
        let data = vec![3.0, 2.0];
        let result = qq_estimate(&data, 2);
        assert!(result.is_nan(), "Should be NaN with only 2 data points");
    }
}
