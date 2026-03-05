use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Absorption ratio: fraction of total variance explained by the top k
/// eigenvectors of the cross-asset correlation matrix.
///
/// When AR → 1, all assets are moving together (systemic risk / herding).
/// When AR is low, assets are diverse and idiosyncratic. AR spikes ~1 week
/// before major crashes.
///
/// Reference:
/// - Kritzman et al. (2011). "Principal Components as a Measure of Systemic
///   Risk." AR spikes preceded the 1998 LTCM crisis, 2007 quant meltdown,
///   and 2008 financial crisis by days to weeks.
fn compute_absorption_ratio(
    asset_returns: &[&[f64]],
    window: usize,
    n_components: usize,
) -> f64 {
    let n_assets = asset_returns.len();
    if n_assets < 2 || window < n_assets || n_components == 0 || n_components > n_assets {
        return f64::NAN;
    }

    // Check all assets have enough data
    for returns in asset_returns.iter() {
        if returns.len() < window {
            return f64::NAN;
        }
    }

    // Extract last `window` returns for each asset
    let slices: Vec<&[f64]> = asset_returns
        .iter()
        .map(|r| &r[(r.len() - window)..])
        .collect();

    // Compute means
    let w = window as f64;
    let means: Vec<f64> = slices
        .iter()
        .map(|s| s.iter().sum::<f64>() / w)
        .collect();

    // Build correlation matrix (using nalgebra)
    // First compute std devs
    let stds: Vec<f64> = slices
        .iter()
        .zip(means.iter())
        .map(|(s, &m)| {
            let var = s.iter().map(|&r| (r - m).powi(2)).sum::<f64>() / (w - 1.0);
            var.sqrt()
        })
        .collect();

    // Check for zero variance
    for &s in &stds {
        if s < 1e-15 {
            return f64::NAN;
        }
    }

    // Build symmetric correlation matrix
    let mut corr = nalgebra::DMatrix::zeros(n_assets, n_assets);
    for i in 0..n_assets {
        corr[(i, i)] = 1.0;
        for j in (i + 1)..n_assets {
            let cov: f64 = slices[i]
                .iter()
                .zip(slices[j].iter())
                .map(|(&ri, &rj)| (ri - means[i]) * (rj - means[j]))
                .sum::<f64>()
                / (w - 1.0);
            let c = cov / (stds[i] * stds[j]);
            let c = c.clamp(-1.0, 1.0);
            corr[(i, j)] = c;
            corr[(j, i)] = c;
        }
    }

    // Eigendecomposition
    let eig = corr.symmetric_eigen();
    let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();

    // Sort descending
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let total: f64 = eigenvalues.iter().filter(|&&v| v > 0.0).sum();
    if total < 1e-15 {
        return f64::NAN;
    }

    let top_k: f64 = eigenvalues.iter().take(n_components).filter(|&&v| v > 0.0).sum();
    top_k / total
}

/// Compute absorption ratio: fraction of variance absorbed by top k eigenvectors.
///
/// Args:
///     returns_list: List of return arrays (one per asset). All must be the same length.
///     window: Rolling window for correlation matrix (default 63 = 3 months).
///     n_components: Number of top eigenvectors (default 1).
///
/// Returns:
///     Absorption ratio in [0, 1]. Higher = more systemic coupling.
#[pyfunction]
#[pyo3(signature = (returns_list, window=63, n_components=1))]
pub fn absorption_ratio(
    returns_list: Vec<PyReadonlyArray1<'_, f64>>,
    window: usize,
    n_components: usize,
) -> PyResult<f64> {
    let slices: Vec<Vec<f64>> = returns_list
        .iter()
        .map(|r| r.as_slice().map(|s| s.to_vec()))
        .collect::<Result<Vec<_>, _>>()?;

    // Verify all same length
    if slices.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 2 assets for absorption ratio",
        ));
    }
    let n = slices[0].len();
    for s in &slices {
        if s.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All return arrays must have the same length",
            ));
        }
    }

    let refs: Vec<&[f64]> = slices.iter().map(|s| s.as_slice()).collect();
    Ok(compute_absorption_ratio(&refs, window, n_components))
}

/// Rolling absorption ratio.
#[pyfunction]
#[pyo3(signature = (returns_list, window=63, n_components=1, step=1))]
pub fn absorption_ratio_rolling<'py>(
    py: Python<'py>,
    returns_list: Vec<PyReadonlyArray1<'py, f64>>,
    window: usize,
    n_components: usize,
    step: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slices: Vec<Vec<f64>> = returns_list
        .iter()
        .map(|r| r.as_slice().map(|s| s.to_vec()))
        .collect::<Result<Vec<_>, _>>()?;

    if slices.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 2 assets for absorption ratio",
        ));
    }

    let n = slices[0].len();
    for s in &slices {
        if s.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All return arrays must have the same length",
            ));
        }
    }

    if window == 0 || window > n || step == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    let mut i = window - 1;
    while i < n {
        let sub_slices: Vec<&[f64]> = slices
            .iter()
            .map(|s| &s[(i + 1 - window)..=i])
            .collect();
        result[i] = compute_absorption_ratio(&sub_slices, window, n_components);
        i += step;
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorption_perfectly_correlated() {
        // Two identical series: AR should be 1.0 (one component explains everything)
        let r1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let r2 = r1.clone();
        let ar = compute_absorption_ratio(&[&r1, &r2], 100, 1);
        assert!(ar.is_finite());
        assert!(
            (ar - 1.0).abs() < 0.01,
            "Perfectly correlated should give AR ~1.0, got {}",
            ar
        );
    }

    #[test]
    fn test_absorption_uncorrelated() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let r1: Vec<f64> = (0..500)
            .map(|_| 0.01 * rng.sample::<f64, _>(StandardNormal))
            .collect();
        let r2: Vec<f64> = (0..500)
            .map(|_| 0.01 * rng.sample::<f64, _>(StandardNormal))
            .collect();

        let ar = compute_absorption_ratio(&[&r1, &r2], 500, 1);
        assert!(ar.is_finite());
        // For 2 uncorrelated assets, top eigenvalue ~ 0.5 of total
        assert!(
            ar < 0.7,
            "Uncorrelated assets should have AR < 0.7, got {}",
            ar
        );
        assert!(
            ar > 0.3,
            "Uncorrelated 2 assets should have AR > 0.3, got {}",
            ar
        );
    }

    #[test]
    fn test_absorption_three_assets() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(123);

        // Common factor + noise
        let factor: Vec<f64> = (0..200)
            .map(|_| 0.01 * rng.sample::<f64, _>(StandardNormal))
            .collect();

        let r1: Vec<f64> = factor
            .iter()
            .map(|&f| f + 0.002 * rng.sample::<f64, _>(StandardNormal))
            .collect();
        let r2: Vec<f64> = factor
            .iter()
            .map(|&f| f + 0.002 * rng.sample::<f64, _>(StandardNormal))
            .collect();
        let r3: Vec<f64> = factor
            .iter()
            .map(|&f| f + 0.002 * rng.sample::<f64, _>(StandardNormal))
            .collect();

        let ar = compute_absorption_ratio(&[&r1, &r2, &r3], 200, 1);
        assert!(ar.is_finite());
        // Strong common factor: top eigenvalue should dominate
        assert!(
            ar > 0.7,
            "Common factor should give high AR, got {}",
            ar
        );
    }

    #[test]
    fn test_absorption_insufficient_data() {
        let r1 = vec![0.01; 5];
        let r2 = vec![0.02; 5];
        let ar = compute_absorption_ratio(&[&r1, &r2], 20, 1);
        assert!(ar.is_nan());
    }
}
