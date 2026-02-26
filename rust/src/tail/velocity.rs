use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Price velocity: rate of change of rolling realized volatility.
///
/// Detects cascade dynamics where volatility itself accelerates — the
/// signature of forced liquidation events like Volmageddon (Feb 5, 2018)
/// where XIV lost 97% in one day, or the Sep 2019 repo market blowup.
///
/// velocity = (vol[t] - vol[t-lag]) / vol[t-lag]
///
/// where vol[t] = realized volatility over a short window.
///
/// High positive velocity = volatility is spiking (cascade forming).
///
/// Reference:
/// - Six Figure Investing (2018). "What Happened on Feb 5, 2018?"
///   (XIV termination, 116% VIX spike, forced rebalancing cascade)
/// - Monday Morning Macro (2019). "The Impossible has Happened. Again."
///   (10 four-sigma events in one month in Treasuries)
fn compute_velocity(returns: &[f64], vol_window: usize, lag: usize) -> f64 {
    let n = returns.len();
    if n < vol_window + lag || vol_window < 2 || lag == 0 {
        return f64::NAN;
    }

    // Current realized vol (end of series)
    let current_slice = &returns[(n - vol_window)..n];
    let current_vol = realized_vol(current_slice);

    // Prior realized vol (lagged)
    let prior_end = n - lag;
    if prior_end < vol_window {
        return f64::NAN;
    }
    let prior_slice = &returns[(prior_end - vol_window)..prior_end];
    let prior_vol = realized_vol(prior_slice);

    if prior_vol < 1e-15 {
        return f64::NAN;
    }

    (current_vol - prior_vol) / prior_vol
}

/// Simple realized volatility (std dev of returns).
fn realized_vol(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / n as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    variance.sqrt()
}

/// Compute price velocity (rate of change of realized volatility).
///
/// Returns the fractional change in realized volatility over a lag period.
/// Positive = volatility increasing (potential cascade).
///
/// Args:
///     returns: Array of log returns.
///     vol_window: Window for realized volatility calculation (default 21 = 1 month).
///     lag: Periods over which to measure velocity (default 5 = 1 week).
#[pyfunction]
#[pyo3(signature = (returns, vol_window=21, lag=5))]
pub fn price_velocity(
    returns: PyReadonlyArray1<'_, f64>,
    vol_window: usize,
    lag: usize,
) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    Ok(compute_velocity(returns, vol_window, lag))
}

/// Rolling price velocity.
#[pyfunction]
#[pyo3(signature = (returns, vol_window=21, lag=5, window=252))]
pub fn price_velocity_rolling<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    vol_window: usize,
    lag: usize,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();

    let min_required = vol_window + lag;
    if window > n || window < min_required || vol_window < 2 || lag == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &returns[(i + 1 - window)..=i];
        result[i] = compute_velocity(slice, vol_window, lag);
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_stable() {
        // Stable volatility: velocity should be near zero
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let returns: Vec<f64> = (0..500)
            .map(|_| 0.01 * rng.sample::<f64, _>(StandardNormal))
            .collect();

        let v = compute_velocity(&returns, 21, 5);
        assert!(v.is_finite(), "Velocity should be finite, got {}", v);
        // Stable vol: velocity should be moderate
        assert!(
            v.abs() < 2.0,
            "Stable vol velocity should be moderate, got {}",
            v
        );
    }

    #[test]
    fn test_velocity_spike() {
        // Regime shift: low vol then high vol
        let mut returns = vec![0.001; 100]; // low vol
        // Add high-vol spike at end
        returns.extend(vec![0.05, -0.04, 0.06, -0.05, 0.07, -0.06, 0.08, -0.07,
                           0.09, -0.08, 0.10, -0.09, 0.11, -0.10, 0.05, -0.04,
                           0.06, -0.05, 0.07, -0.06, 0.08]);

        let v = compute_velocity(&returns, 21, 5);
        assert!(
            v > 0.0,
            "Velocity should be positive during vol spike, got {}",
            v
        );
    }

    #[test]
    fn test_velocity_insufficient_data() {
        let returns = vec![0.01; 10];
        let v = compute_velocity(&returns, 21, 5);
        assert!(v.is_nan(), "Should be NaN with insufficient data");
    }

    #[test]
    fn test_realized_vol() {
        let data = vec![0.01, -0.01, 0.01, -0.01, 0.01];
        let vol = realized_vol(&data);
        assert!(vol > 0.0, "Realized vol should be positive");
        assert!(
            (vol - 0.01).abs() < 0.005,
            "Vol of +-1% returns should be ~0.01, got {}",
            vol
        );
    }
}
