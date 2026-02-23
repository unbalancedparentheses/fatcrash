use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hurst exponent via rescaled range (R/S) analysis.
///
/// H > 0.5: trending / persistent (long memory)
/// H = 0.5: random walk / no memory
/// H < 0.5: mean-reverting / anti-persistent
///
/// For each sub-window size n, compute R(n)/S(n):
///   - mean of the window
///   - cumulative deviation from mean: Y_t = sum_{i=1}^{t} (x_i - mean)
///   - R(n) = max(Y) - min(Y)
///   - S(n) = std dev of x in window
///   - average R/S over all non-overlapping sub-windows
///
/// Fit log(R/S) vs log(n) via OLS -- the slope is H.
fn compute_hurst(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 20 {
        return f64::NAN;
    }

    // Generate window sizes: powers of 2 from 8 up to n/2
    let mut sizes: Vec<usize> = Vec::new();
    let mut s = 8;
    while s <= n / 2 {
        sizes.push(s);
        s *= 2;
    }

    if sizes.len() < 2 {
        return f64::NAN;
    }

    let mut log_n: Vec<f64> = Vec::new();
    let mut log_rs: Vec<f64> = Vec::new();

    for &size in &sizes {
        let n_windows = n / size;
        if n_windows == 0 {
            continue;
        }

        let mut rs_sum = 0.0;
        let mut valid_windows = 0;

        for w in 0..n_windows {
            let start = w * size;
            let end = start + size;
            let window = &data[start..end];

            let mean: f64 = window.iter().sum::<f64>() / size as f64;

            // Standard deviation
            let variance: f64 =
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / size as f64;
            let std_dev = variance.sqrt();

            if std_dev < 1e-15 {
                continue;
            }

            // Cumulative deviation from mean
            let mut cum_dev = Vec::with_capacity(size);
            let mut running = 0.0;
            for &x in window {
                running += x - mean;
                cum_dev.push(running);
            }

            let r = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);

            rs_sum += r / std_dev;
            valid_windows += 1;
        }

        if valid_windows > 0 {
            let avg_rs = rs_sum / valid_windows as f64;
            if avg_rs > 0.0 {
                log_n.push((size as f64).ln());
                log_rs.push(avg_rs.ln());
            }
        }
    }

    if log_n.len() < 2 {
        return f64::NAN;
    }

    // OLS: slope of log_rs vs log_n
    ols_slope(&log_n, &log_rs)
}

/// Simple OLS slope: beta = cov(x,y) / var(x)
fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let cov: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - x_mean) * (yi - y_mean)).sum();
    let var_x: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();

    if var_x < 1e-15 {
        return f64::NAN;
    }

    cov / var_x
}

/// Compute Hurst exponent for the given data via R/S analysis.
/// Returns H in [0, 1] approximately.
#[pyfunction]
#[pyo3(signature = (data,))]
pub fn hurst_exponent(data: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(compute_hurst(data))
}

/// Rolling Hurst exponent over a window.
#[pyfunction]
#[pyo3(signature = (data, window))]
pub fn hurst_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();

    if window > n || window < 20 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        result[i] = compute_hurst(slice);
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hurst_random_walk() {
        // A Gaussian random walk (cumulative sum of iid normal) should have H ~ 0.5
        // But we feed increments (iid normal), which are also H ~ 0.5
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 8192;
        let increments: Vec<f64> = (0..n).map(|_| rng.sample::<f64, _>(StandardNormal)).collect();

        let h = compute_hurst(&increments);

        assert!(
            h.is_finite(),
            "Hurst exponent should be finite, got {}",
            h
        );
        assert!(
            (h - 0.5).abs() < 0.15,
            "Hurst exponent for iid Gaussian should be near 0.5, got {}",
            h
        );
    }

    #[test]
    fn test_hurst_trending() {
        // A trending series (cumulative sum of positively autocorrelated noise)
        // should give H > 0.5
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(99);
        let n = 8192;

        // Generate fractional-like trending series by cumulating
        let increments: Vec<f64> = (0..n).map(|_| rng.sample::<f64, _>(StandardNormal)).collect();
        let cumulative: Vec<f64> = increments
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let h = compute_hurst(&cumulative);

        assert!(
            h.is_finite(),
            "Hurst exponent should be finite, got {}",
            h
        );
        // Cumulative sum of iid should give H > 0.5 (more persistent)
        assert!(
            h > 0.5,
            "Hurst exponent for cumulative random walk should be > 0.5, got {}",
            h
        );
    }

    #[test]
    fn test_hurst_insufficient_data() {
        let data = vec![1.0; 10];
        let h = compute_hurst(&data);
        assert!(h.is_nan(), "Should be NaN with only 10 points");
    }
}
