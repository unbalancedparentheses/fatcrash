/// Absorption ratio (Kritzman, Li, Page & Rigobon, 2011).
///
/// AR = (sum of first n_components eigenvalues) / (sum of all eigenvalues)
///
/// Measures the fraction of total variance in a cross-section of assets
/// explained by the first few principal components. Higher AR indicates
/// tighter coupling (systemic risk). Spikes precede market crises.
///
/// Args:
///   assets: slice of per-asset return slices (all same length).
///   window: number of trailing observations to use.
///   n_components: number of principal components in numerator.
pub fn compute_absorption_ratio(assets: &[&[f64]], window: usize, n_components: usize) -> f64 {
    let n_assets = assets.len();
    if n_assets < 2 || n_components == 0 || n_components > n_assets {
        return f64::NAN;
    }

    // Use trailing `window` observations from each asset
    let n_obs = assets[0].len();
    if window < 2 || window > n_obs {
        return f64::NAN;
    }
    // Validate all assets have same length
    if assets.iter().any(|a| a.len() != n_obs) {
        return f64::NAN;
    }

    let start = n_obs - window;
    let w = window as f64;

    // Compute correlation matrix (standardized covariance)
    // First: means and stds
    let mut means = vec![0.0; n_assets];
    let mut stds = vec![0.0; n_assets];

    for (i, asset) in assets.iter().enumerate() {
        let slice = &asset[start..];
        let mean = slice.iter().sum::<f64>() / w;
        means[i] = mean;
        let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (w - 1.0);
        stds[i] = var.sqrt();
        if stds[i] < 1e-15 {
            return f64::NAN;
        }
    }

    // Build correlation matrix (symmetric, n_assets x n_assets)
    let mut corr = vec![vec![0.0; n_assets]; n_assets];
    for i in 0..n_assets {
        corr[i][i] = 1.0;
        for j in (i + 1)..n_assets {
            let si = &assets[i][start..];
            let sj = &assets[j][start..];
            let cov: f64 = si
                .iter()
                .zip(sj.iter())
                .map(|(&a, &b)| (a - means[i]) * (b - means[j]))
                .sum::<f64>()
                / (w - 1.0);
            let r = cov / (stds[i] * stds[j]);
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }

    // Eigenvalues via power iteration with deflation
    let mut eigenvalues = compute_eigenvalues(&corr, n_assets);
    // Sort descending to ensure top components are first
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Trace of correlation matrix is analytically n_assets
    let total = n_assets as f64;

    let top: f64 = eigenvalues.iter().take(n_components).sum();
    top / total
}

/// Compute eigenvalues of a symmetric matrix via power iteration with deflation.
/// Returns eigenvalues sorted in descending order.
fn compute_eigenvalues(matrix: &[Vec<f64>], n: usize) -> Vec<f64> {
    let mut mat = matrix.to_vec();
    let mut eigenvalues = Vec::with_capacity(n);

    for _ in 0..n {
        let (eigenvalue, eigenvector) = power_iteration(&mat, n, 200);
        eigenvalues.push(eigenvalue);

        // Deflate: A = A - lambda * v * v^T
        for i in 0..n {
            for j in 0..n {
                mat[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    eigenvalues
}

/// Power iteration to find dominant eigenvalue and eigenvector.
fn power_iteration(matrix: &[Vec<f64>], n: usize, max_iter: usize) -> (f64, Vec<f64>) {
    // Deterministic non-uniform initial vector to avoid orthogonality with eigenvectors
    let mut v: Vec<f64> = (0..n).map(|i| ((i * 7 + 13) % 101) as f64 + 1.0).collect();
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        // w = A * v
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i][j] * v[j];
            }
        }

        // Rayleigh quotient: lambda = v^T * w
        let new_eigenvalue: f64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();

        // Normalize w
        let w_norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if w_norm < 1e-15 {
            break;
        }
        for x in &mut w {
            *x /= w_norm;
        }

        let converged = (new_eigenvalue - eigenvalue).abs() < 1e-10;
        eigenvalue = new_eigenvalue;
        v = w;

        if converged {
            break;
        }
    }

    (eigenvalue, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorption_ratio_identity() {
        // Uncorrelated assets: AR with 1 component ≈ 1/n_assets
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 500;
        let n_assets = 4;
        let assets_data: Vec<Vec<f64>> = (0..n_assets)
            .map(|_| {
                (0..n)
                    .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.01)
                    .collect()
            })
            .collect();
        let refs: Vec<&[f64]> = assets_data.iter().map(|a| a.as_slice()).collect();

        let ar = compute_absorption_ratio(&refs, n, 1);
        assert!(ar.is_finite(), "AR should be finite, got {}", ar);
        // For 4 uncorrelated assets, first PC explains ~1/4
        assert!(
            ar < 0.5,
            "AR for uncorrelated assets should be low, got {}",
            ar
        );
    }

    #[test]
    fn test_absorption_ratio_correlated() {
        // Highly correlated assets: AR should be high
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 500;
        let common: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.01)
            .collect();
        // Each asset = common factor + small noise
        let assets_data: Vec<Vec<f64>> = (0..4)
            .map(|_| {
                common
                    .iter()
                    .map(|&c| c + rng.sample::<f64, _>(StandardNormal) * 0.001)
                    .collect()
            })
            .collect();
        let refs: Vec<&[f64]> = assets_data.iter().map(|a| a.as_slice()).collect();

        let ar = compute_absorption_ratio(&refs, n, 1);
        assert!(ar.is_finite(), "AR should be finite, got {}", ar);
        assert!(
            ar > 0.8,
            "AR for highly correlated assets should be high, got {}",
            ar
        );
    }

    #[test]
    fn test_absorption_ratio_insufficient_assets() {
        let asset = vec![0.01; 100];
        let refs = vec![asset.as_slice()];
        assert!(compute_absorption_ratio(&refs, 100, 1).is_nan());
    }
}
