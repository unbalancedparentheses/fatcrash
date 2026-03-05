use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils::ols_slope;

/// Rolling AR(1) coefficient: fit x_t = α + β*x_{t-1} via OLS.
/// Returns β for each window position.
fn compute_rolling_ar1(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    if window < 3 || window > n {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; n];

    for i in window..n {
        let x: Vec<f64> = data[(i - window)..(i - 1)].to_vec();
        let y: Vec<f64> = data[(i - window + 1)..i].to_vec();
        result[i] = ols_slope(&x, &y);
    }

    result
}

/// Rolling variance using Welford's online algorithm.
fn compute_rolling_variance(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    if window < 2 || window > n {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; n];

    // Initialize with first window
    let w = window as f64;
    let mut sum: f64 = data[..window].iter().sum();
    let mut sum_sq: f64 = data[..window].iter().map(|x| x * x).sum();
    let mean = sum / w;
    result[window - 1] = (sum_sq / w) - mean * mean;

    // Slide window
    for i in window..n {
        let old = data[i - window];
        let new = data[i];
        sum += new - old;
        sum_sq += new * new - old * old;
        let mean = sum / w;
        let var = (sum_sq / w) - mean * mean;
        result[i] = var.max(0.0); // Clamp numerical noise
    }

    result
}

/// Rate of change: (x[t] - x[t-lag]) / |x[t-lag]|
fn rate_of_change(series: &[f64], lag: usize) -> Vec<f64> {
    let n = series.len();
    let mut result = vec![f64::NAN; n];
    for i in lag..n {
        let prev = series[i - lag];
        if prev.is_nan() || prev.abs() < 1e-15 {
            continue;
        }
        if series[i].is_nan() {
            continue;
        }
        result[i] = (series[i] - prev) / prev.abs();
    }
    result
}

/// Rolling AR(1) coefficient over a window.
///
/// β trending toward 1 = system losing resilience (critical slowing down).
///
/// Args:
///     data: Time series (e.g., credit spreads, VIX).
///     window: Rolling OLS window (default 252 = 1 year).
#[pyfunction]
#[pyo3(signature = (data, window=252))]
pub fn rolling_ar1<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let result = compute_rolling_ar1(data, window);
    Ok(PyArray1::from_vec(py, result))
}

/// Rolling variance of a time series.
///
/// Rising variance = increasing fluctuations (early warning signal).
///
/// Args:
///     data: Time series.
///     window: Rolling window (default 252).
#[pyfunction]
#[pyo3(signature = (data, window=252))]
pub fn rolling_variance<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let result = compute_rolling_variance(data, window);
    Ok(PyArray1::from_vec(py, result))
}

/// Combined Critical Slowing Down indicator.
///
/// Returns a tuple of three arrays: (ar1_roc, var_roc, csd_signal).
/// - ar1_roc: Rate of change of AR(1) coefficient
/// - var_roc: Rate of change of rolling variance
/// - csd_signal: 1.0 where both ar1_roc > 0 AND var_roc > 0 (dual increase), else 0.0
///
/// When both rise simultaneously, the system is approaching a tipping point.
///
/// Args:
///     data: Time series.
///     window: AR(1) and variance window (default 252).
///     roc_window: Rate of change lookback (default 63 = quarterly).
#[pyfunction]
#[pyo3(signature = (data, window=252, roc_window=63))]
#[allow(clippy::type_complexity)]
pub fn csd_indicator<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    roc_window: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_slice()?;
    let n = data.len();

    let ar1 = compute_rolling_ar1(data, window);
    let var = compute_rolling_variance(data, window);
    let ar1_roc = rate_of_change(&ar1, roc_window);
    let var_roc = rate_of_change(&var, roc_window);

    let mut csd_signal = vec![0.0_f64; n];
    for i in 0..n {
        if !ar1_roc[i].is_nan() && !var_roc[i].is_nan() && ar1_roc[i] > 0.0 && var_roc[i] > 0.0
        {
            csd_signal[i] = 1.0;
        }
    }

    Ok((
        PyArray1::from_vec(py, ar1_roc),
        PyArray1::from_vec(py, var_roc),
        PyArray1::from_vec(py, csd_signal),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_ar1_persistent() {
        // AR(1) with β ≈ 0.9 should give estimates near 0.9
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, 0.1).unwrap();
        let beta = 0.9;
        let n = 600;
        let mut data = vec![0.0; n];
        for i in 1..n {
            data[i] = beta * data[i - 1] + rng.sample(dist);
        }

        let ar1 = compute_rolling_ar1(&data, 252);
        // Check a late value — should be close to 0.9
        let estimate = ar1[500];
        assert!(
            estimate.is_finite(),
            "AR(1) estimate should be finite, got {}",
            estimate
        );
        assert!(
            (estimate - beta).abs() < 0.15,
            "AR(1) estimate {} too far from true beta {}",
            estimate,
            beta
        );
    }

    #[test]
    fn test_rolling_variance_constant() {
        let data = vec![5.0; 100];
        let var = compute_rolling_variance(&data, 20);
        // Constant series → variance ≈ 0
        assert!(var[99] < 1e-10, "Constant series should have ~0 variance, got {}", var[99]);
    }

    #[test]
    fn test_rolling_variance_known() {
        // Alternating ±1 → variance = 1
        let data: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let var = compute_rolling_variance(&data, 100);
        let v = var[199];
        assert!(
            (v - 1.0).abs() < 0.01,
            "Alternating ±1 should have variance ~1, got {}",
            v
        );
    }

    #[test]
    fn test_rate_of_change() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let roc = rate_of_change(&series, 2);
        // roc[2] = (3 - 1) / 1 = 2.0
        assert!((roc[2] - 2.0).abs() < 1e-10);
        // roc[4] = (5 - 3) / 3 ≈ 0.667
        assert!((roc[4] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_csd_short_data() {
        let data = vec![1.0; 10];
        let ar1 = compute_rolling_ar1(&data, 252);
        assert!(ar1.iter().all(|x| x.is_nan()), "Should all be NaN for short data");
    }
}
