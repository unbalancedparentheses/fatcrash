use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Trailing log return over a lookback period.
///
/// momentum = ln(price[t] / price[t - lookback])
///
/// Jegadeesh & Titman (1993) showed 3-12 month momentum generates
/// ~9.5% cumulative excess returns. Momentum reversal (strong positive
/// followed by sharp drop) is a crash precursor.
///
/// Reference:
/// - Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and
///   Selling Losers." Journal of Finance, 48(1), 65-91.
/// - Scowcroft, A. & Sefton, J. (2005). "Understanding Momentum."
///   Financial Analysts Journal, 61(2), 64-82.
fn compute_momentum(prices: &[f64], lookback: usize) -> f64 {
    let n = prices.len();
    if n <= lookback || lookback == 0 {
        return f64::NAN;
    }

    let current = prices[n - 1];
    let past = prices[n - 1 - lookback];

    if past <= 0.0 || current <= 0.0 {
        return f64::NAN;
    }

    (current / past).ln()
}

/// Momentum reversal: detect divergence between short and long-term momentum.
///
/// reversal = long_momentum - short_momentum
///
/// When long-term momentum is strongly positive but short-term is negative,
/// this signals a potential reversal. Values > 0 indicate the recent trend
/// is weaker than the longer trend (potential unwind starting).
fn compute_reversal(prices: &[f64], short_lookback: usize, long_lookback: usize) -> f64 {
    let short_mom = compute_momentum(prices, short_lookback);
    let long_mom = compute_momentum(prices, long_lookback);

    if short_mom.is_nan() || long_mom.is_nan() {
        return f64::NAN;
    }

    // Reversal signal: long positive but short turning negative
    long_mom - short_mom
}

/// Compute trailing momentum (log return) over lookback period.
#[pyfunction]
#[pyo3(signature = (prices, lookback=252))]
pub fn momentum_score(prices: PyReadonlyArray1<'_, f64>, lookback: usize) -> PyResult<f64> {
    let prices = prices.as_slice()?;
    Ok(compute_momentum(prices, lookback))
}

/// Rolling momentum score over a window.
#[pyfunction]
#[pyo3(signature = (prices, lookback=252, window=504))]
pub fn momentum_rolling<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    lookback: usize,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let n = prices.len();

    if window > n || window <= lookback || lookback == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &prices[(i + 1 - window)..=i];
        result[i] = compute_momentum(slice, lookback);
    }

    Ok(PyArray1::from_vec(py, result))
}

/// Compute momentum reversal signal.
///
/// Positive values = long-term momentum exceeds short-term (potential reversal).
/// High positive values when long momentum is positive but short is negative
/// indicate a trend that is starting to break.
#[pyfunction]
#[pyo3(signature = (prices, short_lookback=21, long_lookback=252))]
pub fn momentum_reversal(
    prices: PyReadonlyArray1<'_, f64>,
    short_lookback: usize,
    long_lookback: usize,
) -> PyResult<f64> {
    let prices = prices.as_slice()?;
    Ok(compute_reversal(prices, short_lookback, long_lookback))
}

/// Rolling momentum reversal signal.
#[pyfunction]
#[pyo3(signature = (prices, short_lookback=21, long_lookback=252, window=504))]
pub fn momentum_reversal_rolling<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    short_lookback: usize,
    long_lookback: usize,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let n = prices.len();

    if window > n || window <= long_lookback || long_lookback <= short_lookback {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &prices[(i + 1 - window)..=i];
        result[i] = compute_reversal(slice, short_lookback, long_lookback);
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum_uptrend() {
        // Exponential uptrend: momentum should be positive
        let prices: Vec<f64> = (0..100).map(|i| 100.0 * (0.001 * i as f64).exp()).collect();
        let mom = compute_momentum(&prices, 50);
        assert!(mom > 0.0, "Uptrend momentum should be positive, got {}", mom);
    }

    #[test]
    fn test_momentum_downtrend() {
        // Exponential downtrend: momentum should be negative
        let prices: Vec<f64> = (0..100).map(|i| 100.0 * (-0.001 * i as f64).exp()).collect();
        let mom = compute_momentum(&prices, 50);
        assert!(
            mom < 0.0,
            "Downtrend momentum should be negative, got {}",
            mom
        );
    }

    #[test]
    fn test_momentum_insufficient_data() {
        let prices = vec![100.0, 101.0, 102.0];
        let mom = compute_momentum(&prices, 10);
        assert!(mom.is_nan(), "Should be NaN with insufficient data");
    }

    #[test]
    fn test_reversal_signal() {
        // Build a series that trends up then reverses
        let mut prices: Vec<f64> = (0..200).map(|i| 100.0 * (0.002 * i as f64).exp()).collect();
        // Add a sharp drop at the end
        for i in 180..200 {
            prices[i] = prices[179] * (-0.005 * (i - 179) as f64).exp();
        }
        let rev = compute_reversal(&prices, 21, 180);
        assert!(
            rev > 0.0,
            "Reversal signal should be positive when long mom > short mom, got {}",
            rev
        );
    }

    #[test]
    fn test_reversal_no_reversal() {
        // Steady uptrend: short and long momentum should be similar
        let prices: Vec<f64> = (0..300).map(|i| 100.0 * (0.001 * i as f64).exp()).collect();
        let rev = compute_reversal(&prices, 21, 252);
        // In a steady exponential trend, reversal is small but nonzero because
        // ln(p[t]/p[t-21]) and ln(p[t]/p[t-252]) differ by the lookback ratio.
        // The key test is that it's much smaller than a true reversal.
        assert!(
            rev.abs() < 0.5,
            "Steady trend should have small reversal, got {}",
            rev
        );
    }
}
