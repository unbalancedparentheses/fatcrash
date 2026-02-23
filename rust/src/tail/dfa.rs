use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils::ols_slope;

/// Detrended Fluctuation Analysis (DFA).
///
/// Alternative to Hurst R/S that handles non-stationarity better.
/// Algorithm: divide into windows, fit linear trend per window,
/// compute RMS of residuals, regress log(RMS) vs log(window_size).
///
/// Returns alpha exponent:
/// - 0.5 = white noise (uncorrelated)
/// - above 0.5 = persistent (long memory)
/// - below 0.5 = anti-persistent
fn compute_dfa(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 32 {
        return f64::NAN;
    }

    // Compute cumulative sum (profile)
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut profile = Vec::with_capacity(n);
    let mut running = 0.0;
    for &x in data {
        running += x - mean;
        profile.push(running);
    }

    // Generate window sizes: powers of 2 from 4 up to n/4
    let mut sizes: Vec<usize> = Vec::new();
    let mut s = 4;
    while s <= n / 4 {
        sizes.push(s);
        s = (s as f64 * 1.5).ceil() as usize;
    }
    sizes.dedup();

    if sizes.len() < 3 {
        return f64::NAN;
    }

    let mut log_s: Vec<f64> = Vec::new();
    let mut log_f: Vec<f64> = Vec::new();

    for &size in &sizes {
        let n_windows = n / size;
        if n_windows == 0 {
            continue;
        }

        let mut rms_sum = 0.0;
        let mut valid = 0;

        for w in 0..n_windows {
            let start = w * size;
            let end = start + size;
            let window = &profile[start..end];

            // Fit linear trend via OLS: y = a + b*x
            let s_f = size as f64;
            let x_mean = (s_f - 1.0) / 2.0;
            let mut cov = 0.0;
            let mut var_x = 0.0;
            let y_mean: f64 = window.iter().sum::<f64>() / s_f;

            for (j, &y) in window.iter().enumerate() {
                let xj = j as f64;
                cov += (xj - x_mean) * (y - y_mean);
                var_x += (xj - x_mean).powi(2);
            }

            if var_x < 1e-15 {
                continue;
            }

            let slope = cov / var_x;
            let intercept = y_mean - slope * x_mean;

            // RMS of detrended residuals
            let mut sq_sum = 0.0;
            for (j, &y) in window.iter().enumerate() {
                let trend = intercept + slope * j as f64;
                sq_sum += (y - trend).powi(2);
            }
            let rms = (sq_sum / s_f).sqrt();
            if rms > 0.0 {
                rms_sum += rms;
                valid += 1;
            }
        }

        if valid > 0 {
            let avg_rms = rms_sum / valid as f64;
            if avg_rms > 0.0 {
                log_s.push((size as f64).ln());
                log_f.push(avg_rms.ln());
            }
        }
    }

    if log_s.len() < 3 {
        return f64::NAN;
    }

    ols_slope(&log_s, &log_f)
}

/// Compute DFA exponent for the given data.
/// Returns alpha: 0.5 = white noise, >0.5 persistent, <0.5 anti-persistent.
#[pyfunction]
#[pyo3(signature = (data,))]
pub fn dfa_exponent(data: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(compute_dfa(data))
}

/// Rolling DFA exponent over a window.
#[pyfunction]
#[pyo3(signature = (data, window))]
pub fn dfa_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();

    if window > n || window < 32 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        result[i] = compute_dfa(slice);
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfa_white_noise() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 4096;
        let data: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        let alpha = compute_dfa(&data);
        assert!(
            alpha.is_finite(),
            "DFA alpha should be finite, got {}",
            alpha
        );
        assert!(
            (alpha - 0.5).abs() < 0.15,
            "DFA alpha for white noise should be near 0.5, got {}",
            alpha
        );
    }

    #[test]
    fn test_dfa_persistent() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(99);
        let n = 4096;
        let increments: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        // Cumulative sum → persistent / trending series
        let cumulative: Vec<f64> = increments
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let alpha = compute_dfa(&cumulative);
        assert!(
            alpha.is_finite(),
            "DFA alpha should be finite, got {}",
            alpha
        );
        assert!(
            alpha > 0.8,
            "DFA alpha for random walk (integrated noise) should be > 0.8, got {}",
            alpha
        );
    }

    #[test]
    fn test_dfa_insufficient_data() {
        let data = vec![1.0; 10];
        let alpha = compute_dfa(&data);
        assert!(alpha.is_nan(), "Should be NaN with only 10 points");
    }
}
