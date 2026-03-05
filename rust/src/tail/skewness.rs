use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Realized skewness: third central moment / std^3 over a rolling window.
///
/// Captures distributional asymmetry — which tail is fatter. Negative skew
/// means left tail is heavier (crash risk), positive means right tail
/// (bubble/melt-up). This is orthogonal to Hill/kappa which measure how
/// fat *both* tails are without distinguishing direction.
///
/// Reference:
/// - Amaya et al. (2015). "Does realized skewness predict the cross-section
///   of equity returns?" Most negative realized skewness quintile earns 24 bps/week
///   less than most positive — skewness predicts future returns.
fn compute_realized_skewness(returns: &[f64], window: usize) -> f64 {
    let n = returns.len();
    if n < window || window < 3 {
        return f64::NAN;
    }

    let slice = &returns[(n - window)..n];
    let w = window as f64;

    let mean = slice.iter().sum::<f64>() / w;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &r in slice {
        let d = r - mean;
        m2 += d * d;
        m3 += d * d * d;
    }

    m2 /= w;
    m3 /= w;

    let std = m2.sqrt();
    if std < 1e-15 {
        return f64::NAN;
    }

    m3 / (std * std * std)
}

/// Compute realized skewness over the last `window` returns.
///
/// Returns the third standardized moment: negative = left-skewed (crash risk),
/// positive = right-skewed (melt-up risk).
///
/// Args:
///     returns: Array of log returns.
///     window: Window size for skewness calculation (default 63 = 3 months).
#[pyfunction]
#[pyo3(signature = (returns, window=63))]
pub fn realized_skewness(
    returns: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    Ok(compute_realized_skewness(returns, window))
}

/// Rolling realized skewness.
#[pyfunction]
#[pyo3(signature = (returns, window=63, step=1))]
pub fn realized_skewness_rolling<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    window: usize,
    step: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();

    if window < 3 || window > n || step == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    let mut i = window - 1;
    while i < n {
        let slice = &returns[(i + 1 - window)..=i];
        result[i] = compute_realized_skewness(slice, window);
        i += step;
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric returns: skewness should be near zero
        let returns: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let skew = compute_realized_skewness(&returns, 200);
        assert!(skew.is_finite());
        assert!(
            skew.abs() < 0.1,
            "Symmetric returns should have ~zero skewness, got {}",
            skew
        );
    }

    #[test]
    fn test_skewness_left_tail() {
        // One large negative return among small positives: should be negative skew
        let mut returns = vec![0.005; 99];
        returns.push(-0.20); // big crash
        let skew = compute_realized_skewness(&returns, 100);
        assert!(skew.is_finite());
        assert!(
            skew < -1.0,
            "Left-tail event should give negative skewness, got {}",
            skew
        );
    }

    #[test]
    fn test_skewness_right_tail() {
        // One large positive return: should be positive skew
        let mut returns = vec![-0.005; 99];
        returns.push(0.20); // big spike up
        let skew = compute_realized_skewness(&returns, 100);
        assert!(skew.is_finite());
        assert!(
            skew > 1.0,
            "Right-tail event should give positive skewness, got {}",
            skew
        );
    }

    #[test]
    fn test_skewness_insufficient_data() {
        let returns = vec![0.01; 2];
        let skew = compute_realized_skewness(&returns, 5);
        assert!(skew.is_nan());
    }

    #[test]
    fn test_skewness_rolling() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let returns: Vec<f64> = (0..300)
            .map(|_| 0.01 * rng.sample::<f64, _>(StandardNormal))
            .collect();

        // Use compute function directly for rolling test
        let window = 63;
        let n = returns.len();
        let mut results = vec![f64::NAN; n];
        for i in (window - 1)..n {
            let slice = &returns[(i + 1 - window)..=i];
            results[i] = compute_realized_skewness(slice, window);
        }

        // First (window-1) should be NaN
        assert!(results[0].is_nan());
        // After window, should be finite
        assert!(results[window - 1].is_finite());
        assert!(results[n - 1].is_finite());
    }
}
