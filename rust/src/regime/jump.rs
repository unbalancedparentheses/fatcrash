use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::realized_var::compute_realized_variance;

/// Bipower variation: BV = (π/2) * (1/(W-1)) * Σ |r_i| * |r_{i-1}|
/// Robust to jumps — converges to integrated variance even with jumps.
pub(crate) fn compute_bipower_variation(returns: &[f64]) -> f64 {
    let w = returns.len();
    if w < 2 {
        return f64::NAN;
    }
    let mu1 = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
    // μ₁ = sqrt(2/π) ≈ 0.7979
    // BV = (252 / (μ₁² * (W-1))) * Σ |r_i| * |r_{i-1}|
    let sum: f64 = returns
        .windows(2)
        .map(|pair| pair[0].abs() * pair[1].abs())
        .sum();
    (252.0 / (mu1 * mu1 * (w - 1) as f64)) * sum
}

/// Jump variance: JV = max(RV - BV, 0)
fn compute_jump_variance(returns: &[f64]) -> f64 {
    let rv = compute_realized_variance(returns);
    let bv = compute_bipower_variation(returns);
    if rv.is_nan() || bv.is_nan() {
        return f64::NAN;
    }
    (rv - bv).max(0.0)
}

/// Tri-power quarticity for BNS test denominator.
/// TPQ = W * (252 / ((W-2) * μ_{4/3}³)) * Σ |r_i|^{4/3} * |r_{i-1}|^{4/3} * |r_{i-2}|^{4/3}
fn compute_tripower_quarticity(returns: &[f64]) -> f64 {
    let w = returns.len();
    if w < 3 {
        return f64::NAN;
    }
    // μ_{4/3} = 2^{2/3} * Γ(7/6) / Γ(1/2)
    // For standard normal: E[|Z|^{4/3}] = 2^{2/3} * Γ(7/6) / sqrt(π)
    let mu_43: f64 = 2.0_f64.powf(2.0 / 3.0) * gamma_ratio_7_6();
    let mu_43_cubed = mu_43 * mu_43 * mu_43;

    let sum: f64 = (2..w)
        .map(|i| {
            returns[i].abs().powf(4.0 / 3.0)
                * returns[i - 1].abs().powf(4.0 / 3.0)
                * returns[i - 2].abs().powf(4.0 / 3.0)
        })
        .sum();

    let n = w as f64;
    n * (252.0 / ((w - 2) as f64 * mu_43_cubed)) * sum
}

/// Γ(7/6) / Γ(1/2) ≈ Γ(7/6) / sqrt(π)
/// Γ(7/6) ≈ 0.9407 (from tables)
fn gamma_ratio_7_6() -> f64 {
    0.9407_f64 / std::f64::consts::PI.sqrt()
}

/// BNS jump test z-statistic.
fn compute_jump_test(returns: &[f64]) -> (f64, f64) {
    let rv = compute_realized_variance(returns);
    let bv = compute_bipower_variation(returns);
    let tpq = compute_tripower_quarticity(returns);

    if rv.is_nan() || bv.is_nan() || tpq.is_nan() || bv < 1e-15 {
        return (f64::NAN, f64::NAN);
    }

    // Relative jump: (RV - BV) / RV
    let rj = (rv - bv) / rv;

    // Variance of the relative jump statistic under H0 (no jumps):
    // Var = (π²/4 + π - 5) * (1/n) * max(1, TPQ/BV²)
    let vq = (std::f64::consts::PI.powi(2) / 4.0 + std::f64::consts::PI - 5.0)
        * (1.0 / returns.len() as f64)
        * (tpq / (bv * bv)).max(1.0);

    if vq <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let z = rj / vq.sqrt();
    let jv = (rv - bv).max(0.0);
    (z, jv)
}

/// Bipower variation (robust to jumps).
///
/// Converges to integrated variance even in the presence of jumps.
///
/// Args:
///     returns: Array of log returns.
///     window: Window size (default 21).
#[pyfunction]
#[pyo3(signature = (returns, window=21))]
pub fn bipower_variation(
    returns: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    if window < 2 || window > n {
        return Ok(f64::NAN);
    }
    Ok(compute_bipower_variation(&returns[(n - window)..]))
}

/// Jump variance: JV = max(RV - BV, 0).
///
/// Isolates the discontinuous (jump) component of total variance.
/// Spikes around Flash Crash, Lehman, COVID.
///
/// Args:
///     returns: Array of log returns.
///     window: Window size (default 21).
#[pyfunction]
#[pyo3(signature = (returns, window=21))]
pub fn jump_variance(
    returns: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    if window < 2 || window > n {
        return Ok(f64::NAN);
    }
    Ok(compute_jump_variance(&returns[(n - window)..]))
}

/// BNS jump test statistic.
///
/// Returns (z_statistic, jump_variance). Large positive z rejects H0 of
/// no jumps. Uses tri-power quarticity for the denominator.
///
/// Args:
///     returns: Array of log returns.
///     window: Window size (default 63).
#[pyfunction]
#[pyo3(signature = (returns, window=63))]
pub fn jump_test(
    returns: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<(f64, f64)> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    if window < 3 || window > n {
        return Ok((f64::NAN, f64::NAN));
    }
    Ok(compute_jump_test(&returns[(n - window)..]))
}

/// Rolling bipower variation.
#[pyfunction]
#[pyo3(signature = (returns, window=21, step=1))]
pub fn bipower_variation_rolling<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    window: usize,
    step: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    let step = step.max(1);

    if window < 2 || window > n {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n {
        result[i] = compute_bipower_variation(&returns[(i + 1 - window)..=i]);
        i += step;
    }

    Ok(PyArray1::from_vec(py, result))
}

/// Rolling jump variance.
#[pyfunction]
#[pyo3(signature = (returns, window=21, step=1))]
pub fn jump_variance_rolling<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    window: usize,
    step: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    let step = step.max(1);

    if window < 2 || window > n {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n {
        result[i] = compute_jump_variance(&returns[(i + 1 - window)..=i]);
        i += step;
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bipower_variation_constant_returns() {
        let returns = vec![0.01; 50];
        let bv = compute_bipower_variation(&returns);
        assert!(bv > 0.0 && bv.is_finite(), "BV should be positive finite, got {}", bv);
    }

    #[test]
    fn test_jump_variance_no_jumps() {
        // With constant returns, RV ≈ BV, so JV ≈ 0
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, 0.01).unwrap();
        let returns: Vec<f64> = (0..500).map(|_| rng.sample(dist)).collect();

        let jv = compute_jump_variance(&returns);
        let rv = compute_realized_variance(&returns);
        // Jump variance should be small relative to total RV
        assert!(jv >= 0.0, "Jump variance must be non-negative");
        assert!(
            jv < rv * 0.5,
            "JV={} should be small fraction of RV={} for Gaussian returns",
            jv,
            rv
        );
    }

    #[test]
    fn test_jump_variance_with_jumps() {
        // Gaussian returns + large jumps → JV should be significant
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, 0.005).unwrap();
        let mut returns: Vec<f64> = (0..200).map(|_| rng.sample(dist)).collect();
        // Insert jumps
        returns[50] = 0.10;
        returns[100] = -0.08;
        returns[150] = 0.12;

        let jv = compute_jump_variance(&returns);
        assert!(jv > 0.0, "JV should be positive with jumps, got {}", jv);
    }

    #[test]
    fn test_jump_test_gaussian() {
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, 0.01).unwrap();
        let returns: Vec<f64> = (0..500).map(|_| rng.sample(dist)).collect();

        let (z, _jv) = compute_jump_test(&returns);
        assert!(z.is_finite(), "z-stat should be finite, got {}", z);
    }

    #[test]
    fn test_insufficient_data() {
        assert!(compute_bipower_variation(&[0.01]).is_nan());
        assert!(compute_jump_variance(&[0.01]).is_nan());
        let (z, jv) = compute_jump_test(&[0.01, 0.02]);
        assert!(z.is_nan());
        assert!(jv.is_nan());
    }
}
