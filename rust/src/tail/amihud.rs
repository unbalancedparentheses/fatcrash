use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Amihud illiquidity ratio: mean(|return| / volume) over a window.
///
/// Higher values = less liquid = harder to trade without moving price.
/// Illiquidity spikes precede and accompany crashes because market makers
/// widen spreads and pull quotes when risk rises.
///
/// Reference:
/// - Amihud (2002). "Illiquidity and Stock Returns: Cross-Section and
///   Time-Series Effects." Expected return is increasing in illiquidity.
///   The ratio captures Kyle's lambda (price impact per unit of trading).
fn compute_amihud(returns: &[f64], volume: &[f64], window: usize) -> f64 {
    let n = returns.len();
    if n < window || volume.len() < window || window == 0 {
        return f64::NAN;
    }

    let start = n - window;
    let mut sum = 0.0;
    let mut count = 0usize;

    for i in start..n {
        let v = volume[i];
        if v > 0.0 && v.is_finite() && returns[i].is_finite() {
            sum += returns[i].abs() / v;
            count += 1;
        }
    }

    if count == 0 {
        return f64::NAN;
    }

    sum / count as f64
}

/// Compute Amihud illiquidity ratio over the last `window` observations.
///
/// Args:
///     returns: Array of log returns.
///     volume: Array of trading volume (same length as returns).
///     window: Window size (default 21 = 1 month).
#[pyfunction]
#[pyo3(signature = (returns, volume, window=21))]
pub fn amihud_illiquidity(
    returns: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    let volume = volume.as_slice()?;
    if returns.len() != volume.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "returns and volume must have the same length",
        ));
    }
    Ok(compute_amihud(returns, volume, window))
}

/// Rolling Amihud illiquidity.
#[pyfunction]
#[pyo3(signature = (returns, volume, window=21, step=1))]
pub fn amihud_illiquidity_rolling<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    window: usize,
    step: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let volume = volume.as_slice()?;
    let n = returns.len();

    if returns.len() != volume.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "returns and volume must have the same length",
        ));
    }

    if window == 0 || window > n || step == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    let mut i = window - 1;
    while i < n {
        let r_slice = &returns[(i + 1 - window)..=i];
        let v_slice = &volume[(i + 1 - window)..=i];
        result[i] = compute_amihud(r_slice, v_slice, window);
        i += step;
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amihud_basic() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.005];
        let volume = vec![1e6, 1e6, 1e6, 1e6, 1e6];
        let ratio = compute_amihud(&returns, &volume, 5);
        assert!(ratio.is_finite());
        assert!(ratio > 0.0);
        // |0.01|/1e6 + |0.02|/1e6 + ... = 0.06/1e6 / 5 = 1.2e-8
        let expected = (0.01 + 0.02 + 0.015 + 0.01 + 0.005) / 5.0 / 1e6;
        assert!(
            (ratio - expected).abs() < 1e-15,
            "Expected {}, got {}",
            expected,
            ratio
        );
    }

    #[test]
    fn test_amihud_low_volume_means_illiquid() {
        let returns = vec![0.01; 10];
        let high_vol = vec![1e8; 10];
        let low_vol = vec![1e4; 10];

        let liquid = compute_amihud(&returns, &high_vol, 10);
        let illiquid = compute_amihud(&returns, &low_vol, 10);
        assert!(
            illiquid > liquid * 100.0,
            "Low volume should give much higher illiquidity: {} vs {}",
            illiquid,
            liquid
        );
    }

    #[test]
    fn test_amihud_zero_volume_skipped() {
        let returns = vec![0.01, 0.02, 0.03];
        let volume = vec![0.0, 1e6, 0.0]; // Only middle has valid volume
        let ratio = compute_amihud(&returns, &volume, 3);
        assert!(ratio.is_finite());
        // Should only count the middle observation
        let expected = 0.02 / 1e6;
        assert!(
            (ratio - expected).abs() < 1e-15,
            "Expected {}, got {}",
            expected,
            ratio
        );
    }

    #[test]
    fn test_amihud_insufficient_data() {
        let returns = vec![0.01; 3];
        let volume = vec![1e6; 3];
        let ratio = compute_amihud(&returns, &volume, 10);
        assert!(ratio.is_nan());
    }
}
