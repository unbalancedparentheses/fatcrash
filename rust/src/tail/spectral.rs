use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils::ols_slope;

/// Spectral exponent estimator via GPH (Geweke-Porter-Hudak) log-periodogram regression.
///
/// Estimates the long-memory parameter d from the spectral density near frequency zero:
///   f(lambda) ~ |lambda|^(1-2d) as lambda → 0
///
/// The relationship to Hurst exponent: d = H - 0.5
///   d > 0: long memory (persistent)
///   d = 0: short memory (white noise)
///   d < 0: anti-persistent
///
/// Uses the GPH estimator: regress log(I(lambda_j)) on log|2*sin(lambda_j/2)|^2
/// for the lowest m frequencies, where m = n^bandwidth_exp (typically 0.5-0.8).
fn compute_spectral(data: &[f64], bandwidth_exp: f64) -> f64 {
    let n = data.len();
    if n < 32 {
        return f64::NAN;
    }

    // Demean
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Number of frequencies to use
    let m = ((n as f64).powf(bandwidth_exp)).floor() as usize;
    let m = m.max(3).min(n / 2);

    let n_f = n as f64;

    // Compute periodogram at Fourier frequencies lambda_j = 2*pi*j/n
    let mut log_freq: Vec<f64> = Vec::with_capacity(m);
    let mut log_periodo: Vec<f64> = Vec::with_capacity(m);

    for j in 1..=m {
        let lambda = 2.0 * std::f64::consts::PI * j as f64 / n_f;

        // Periodogram: I(lambda) = (1/n) |sum x_t * exp(-i*lambda*t)|^2
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;
        for (t, &x) in centered.iter().enumerate() {
            let angle = lambda * t as f64;
            cos_sum += x * angle.cos();
            sin_sum += x * angle.sin();
        }

        let periodogram = (cos_sum * cos_sum + sin_sum * sin_sum) / n_f;

        if periodogram <= 0.0 {
            continue;
        }

        // GPH regressor: log|2*sin(lambda/2)|^2
        let regressor = 2.0 * (lambda / 2.0).sin();
        if regressor.abs() < 1e-15 {
            continue;
        }

        log_freq.push((regressor * regressor).ln());
        log_periodo.push(periodogram.ln());
    }

    if log_freq.len() < 3 {
        return f64::NAN;
    }

    // OLS: log(I(lambda_j)) = c - d * log|2*sin(lambda_j/2)|^2
    // So d = -slope
    let slope = ols_slope(&log_freq, &log_periodo);

    if !slope.is_finite() {
        return f64::NAN;
    }

    -slope
}

/// Compute spectral exponent (long-memory parameter d) via GPH estimator.
/// d > 0: long memory, d = 0: short memory, d < 0: anti-persistent.
/// Relationship to Hurst: d = H - 0.5.
/// `bandwidth_exp` controls the number of frequencies: m = n^bandwidth_exp. Default 0.65.
#[pyfunction]
#[pyo3(signature = (data, bandwidth_exp=0.65))]
pub fn spectral_exponent(
    data: PyReadonlyArray1<'_, f64>,
    bandwidth_exp: Option<f64>,
) -> PyResult<f64> {
    let data = data.as_slice()?;
    let bw = bandwidth_exp.unwrap_or(0.65);
    Ok(compute_spectral(data, bw))
}

/// Rolling spectral exponent over a window.
#[pyfunction]
#[pyo3(signature = (data, window, bandwidth_exp=0.65))]
pub fn spectral_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    bandwidth_exp: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    let bw = bandwidth_exp.unwrap_or(0.65);

    if window > n || window < 32 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        result[i] = compute_spectral(slice, bw);
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_white_noise() {
        // White noise: d should be near 0
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 4096;
        let data: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        let d = compute_spectral(&data, 0.65);

        assert!(d.is_finite(), "d should be finite, got {}", d);
        assert!(
            d.abs() < 0.25,
            "Spectral d for white noise should be near 0, got {}",
            d
        );
    }

    #[test]
    fn test_spectral_persistent() {
        // Cumulative sum of noise → long memory (d > 0)
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(99);
        let n = 4096;
        let increments: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        let cumulative: Vec<f64> = increments
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let d = compute_spectral(&cumulative, 0.65);

        assert!(d.is_finite(), "d should be finite, got {}", d);
        assert!(
            d > 0.3,
            "Spectral d for integrated noise should be > 0.3, got {}",
            d
        );
    }

    #[test]
    fn test_spectral_insufficient_data() {
        let data = vec![1.0; 10];
        let d = compute_spectral(&data, 0.65);
        assert!(d.is_nan(), "Should be NaN with only 10 points");
    }
}
