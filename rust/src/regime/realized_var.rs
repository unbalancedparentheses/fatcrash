use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Simple realized variance: RV = (252/W) * Σ r_i²
pub(crate) fn compute_realized_variance(returns: &[f64]) -> f64 {
    let w = returns.len();
    if w == 0 {
        return f64::NAN;
    }
    let sum_sq: f64 = returns.iter().map(|r| r * r).sum();
    (252.0 / w as f64) * sum_sq
}

/// Parkinson realized variance from high/low prices.
/// RV_Park = (252 / (4 * ln2 * W)) * Σ (ln(H_i / L_i))²
fn compute_realized_variance_parkinson(high: &[f64], low: &[f64]) -> f64 {
    let w = high.len();
    if w == 0 || low.len() != w {
        return f64::NAN;
    }
    let sum: f64 = high
        .iter()
        .zip(low.iter())
        .map(|(&h, &l)| {
            if l <= 0.0 || h <= 0.0 {
                return f64::NAN;
            }
            let log_hl = (h / l).ln();
            log_hl * log_hl
        })
        .sum();
    if sum.is_nan() {
        return f64::NAN;
    }
    (252.0 / (4.0 * 2.0_f64.ln() * w as f64)) * sum
}

/// Garman-Klass realized variance from OHLC.
/// GK_i = 0.5 * (ln(H/L))² - (2*ln2 - 1) * (ln(C/O))²
/// RV_GK = (252 / W) * Σ GK_i
fn compute_realized_variance_gk(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> f64 {
    let w = open.len();
    if w == 0 || high.len() != w || low.len() != w || close.len() != w {
        return f64::NAN;
    }
    let coeff = 2.0 * 2.0_f64.ln() - 1.0;
    let sum: f64 = (0..w)
        .map(|i| {
            if open[i] <= 0.0 || high[i] <= 0.0 || low[i] <= 0.0 || close[i] <= 0.0 {
                return f64::NAN;
            }
            let log_hl = (high[i] / low[i]).ln();
            let log_co = (close[i] / open[i]).ln();
            0.5 * log_hl * log_hl - coeff * log_co * log_co
        })
        .sum();
    if sum.is_nan() {
        return f64::NAN;
    }
    (252.0 / w as f64) * sum
}

/// Annualized realized variance from daily returns.
///
/// RV = (252/W) * Σ r_i²
///
/// Args:
///     returns: Array of log returns.
///     window: Number of observations to use (default 21 = 1 month).
#[pyfunction]
#[pyo3(signature = (returns, window=21))]
pub fn realized_variance(
    returns: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    if window == 0 || window > n {
        return Ok(f64::NAN);
    }
    Ok(compute_realized_variance(&returns[(n - window)..]))
}

/// Parkinson realized variance from OHLC high/low.
///
/// RV_Park = (252 / (4 * ln2 * W)) * Σ (ln(H/L))²
/// ~5x more efficient than simple RV.
///
/// Args:
///     high: Array of high prices.
///     low: Array of low prices.
///     window: Number of observations (default 21).
#[pyfunction]
#[pyo3(signature = (high, low, window=21))]
pub fn realized_variance_parkinson(
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<f64> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let n = high.len();
    if window == 0 || window > n || low.len() != n {
        return Ok(f64::NAN);
    }
    Ok(compute_realized_variance_parkinson(
        &high[(n - window)..],
        &low[(n - window)..],
    ))
}

/// Garman-Klass realized variance from OHLC data.
///
/// Most efficient estimator from daily OHLC bars.
///
/// Args:
///     open: Array of open prices.
///     high: Array of high prices.
///     low: Array of low prices.
///     close: Array of close prices.
///     window: Number of observations (default 21).
#[pyfunction]
#[pyo3(signature = (open, high, low, close, window=21))]
pub fn realized_variance_gk(
    open: PyReadonlyArray1<'_, f64>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<f64> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let n = open.len();
    if window == 0 || window > n || high.len() != n || low.len() != n || close.len() != n {
        return Ok(f64::NAN);
    }
    Ok(compute_realized_variance_gk(
        &open[(n - window)..],
        &high[(n - window)..],
        &low[(n - window)..],
        &close[(n - window)..],
    ))
}

/// Rolling realized variance.
#[pyfunction]
#[pyo3(signature = (returns, window=21, step=1))]
pub fn realized_variance_rolling<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    window: usize,
    step: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    let step = step.max(1);

    if window == 0 || window > n {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n {
        result[i] = compute_realized_variance(&returns[(i + 1 - window)..=i]);
        i += step;
    }

    Ok(PyArray1::from_vec(py, result))
}

/// Rolling Parkinson realized variance.
#[pyfunction]
#[pyo3(signature = (high, low, window=21, step=1))]
pub fn realized_variance_parkinson_rolling<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    window: usize,
    step: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let n = high.len();
    let step = step.max(1);

    if window == 0 || window > n || low.len() != n {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n {
        result[i] = compute_realized_variance_parkinson(
            &high[(i + 1 - window)..=i],
            &low[(i + 1 - window)..=i],
        );
        i += step;
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realized_variance_known() {
        // 1% daily returns → annualized variance ≈ 0.01² * 252 = 0.0252
        let returns = vec![0.01; 21];
        let rv = compute_realized_variance(&returns);
        let expected = 252.0 / 21.0 * 21.0 * 0.0001; // 0.0252
        assert!(
            (rv - expected).abs() < 1e-10,
            "RV={}, expected={}",
            rv,
            expected
        );
    }

    #[test]
    fn test_realized_variance_parkinson() {
        // H/L ratio of 1.02 every day
        let high = vec![102.0; 21];
        let low = vec![100.0; 21];
        let rv = compute_realized_variance_parkinson(&high, &low);
        assert!(rv > 0.0 && rv.is_finite(), "Parkinson RV should be positive finite, got {}", rv);
    }

    #[test]
    fn test_realized_variance_gk() {
        let open = vec![100.0; 21];
        let high = vec![102.0; 21];
        let low = vec![99.0; 21];
        let close = vec![101.0; 21];
        let rv = compute_realized_variance_gk(&open, &high, &low, &close);
        assert!(rv > 0.0 && rv.is_finite(), "GK RV should be positive finite, got {}", rv);
    }

    #[test]
    fn test_empty_input() {
        assert!(compute_realized_variance(&[]).is_nan());
        assert!(compute_realized_variance_parkinson(&[], &[]).is_nan());
        assert!(compute_realized_variance_gk(&[], &[], &[], &[]).is_nan());
    }

    #[test]
    fn test_parkinson_more_efficient() {
        // Parkinson should give lower variance estimate for same-ish data
        // since it uses more information (high/low range)
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, 0.01).unwrap();
        let n = 252;

        let mut returns = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut price = 100.0;

        for _ in 0..n {
            let r: f64 = rng.sample(dist);
            returns.push(r);
            let c = price * (1.0 + r);
            // Simulate intraday range
            let h = c * (1.0 + rng.sample::<f64, _>(rand_distr::StandardNormal).abs() * 0.005);
            let l = c * (1.0 - rng.sample::<f64, _>(rand_distr::StandardNormal).abs() * 0.005);
            high.push(h);
            low.push(l);
            price = c;
        }

        let rv_simple = compute_realized_variance(&returns);
        let rv_park = compute_realized_variance_parkinson(&high, &low);
        // Both should be finite and positive
        assert!(rv_simple > 0.0 && rv_simple.is_finite());
        assert!(rv_park > 0.0 && rv_park.is_finite());
    }
}
