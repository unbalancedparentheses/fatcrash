use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// Maximum lag order for ADF regression, following PSY (2015):
/// p_max = floor(12 * (T/100)^(1/4))
fn max_lag(n: usize) -> usize {
    (12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize
}

/// Run an Augmented Dickey-Fuller regression on a data slice.
///
/// Regression: Δy_t = μ + δ*y_{t-1} + Σ_{j=1}^{p} φ_j*Δy_{t-j} + ε_t
///
/// Lag order p is selected by BIC up to p_max = floor(12*(T/100)^{1/4}),
/// following Phillips, Shi & Yu (2015).
///
/// Returns the t-statistic for δ (ADF statistic).
pub fn adf_stat(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 6 {
        return f64::NEG_INFINITY;
    }

    let p_max = max_lag(n).min(n / 3).max(1);

    // Try each lag order 0..=p_max and pick by BIC
    let mut best_bic = f64::INFINITY;
    let mut best_t_stat = f64::NEG_INFINITY;

    for p in 0..=p_max {
        if let Some((t_stat, bic)) = adf_stat_with_lag(data, p) {
            if bic < best_bic {
                best_bic = bic;
                best_t_stat = t_stat;
            }
        }
    }

    best_t_stat
}

/// ADF regression with a specific lag order p.
/// Returns (t-statistic for delta, BIC).
fn adf_stat_with_lag(data: &[f64], p: usize) -> Option<(f64, f64)> {
    let n = data.len();
    // Need at least p+1 differences, so n >= p+2 observations
    let start = p + 1; // first usable index for Δy_t
    if start >= n {
        return None;
    }
    let m = n - start; // number of regression observations
    let k = 2 + p; // number of regressors: intercept + y_{t-1} + p lagged diffs
    if m <= k {
        return None;
    }

    // Build OLS system: y = X*beta + e
    // y[i] = Δy_{start+i}
    // X[i] = [1, y_{start+i-1}, Δy_{start+i-1}, ..., Δy_{start+i-p}]
    //
    // Use normal equations: (X'X) beta = X'y
    // For small k (typically 2-15), direct solve via Cholesky/inverse is fine.

    let mut xty = vec![0.0; k];
    let mut xtx = vec![0.0; k * k];
    let mut yty = 0.0;

    for i in 0..m {
        let t = start + i;
        let dy = data[t] - data[t - 1];

        // Build row of X
        let mut row = Vec::with_capacity(k);
        row.push(1.0); // intercept
        row.push(data[t - 1]); // y_{t-1}
        for j in 1..=p {
            row.push(data[t - j] - data[t - j - 1]); // Δy_{t-j}
        }

        // Accumulate X'X and X'y
        for a in 0..k {
            xty[a] += row[a] * dy;
            for b in 0..k {
                xtx[a * k + b] += row[a] * row[b];
            }
        }
        yty += dy * dy;
    }

    // Solve (X'X) beta = X'y via Gauss elimination
    let beta = solve_symmetric(k, &xtx, &xty)?;

    // SSE = y'y - beta'X'y
    let sse: f64 = yty - beta.iter().zip(xty.iter()).map(|(b, xy)| b * xy).sum::<f64>();
    let sse = sse.max(0.0);

    let df = m - k;
    let sigma2 = sse / df as f64;

    // Variance of delta (beta[1]) = sigma^2 * [(X'X)^{-1}]_{1,1}
    let xtx_inv = invert_symmetric(k, &xtx)?;
    let var_delta = sigma2 * xtx_inv[1 * k + 1];

    if var_delta <= 0.0 {
        return None;
    }

    let t_stat = beta[1] / var_delta.sqrt();

    // BIC = m * ln(SSE/m) + k * ln(m)
    let bic = m as f64 * (sse / m as f64).ln() + k as f64 * (m as f64).ln();

    Some((t_stat, bic))
}

/// Solve A*x = b for symmetric positive-definite A via Gaussian elimination with pivoting.
fn solve_symmetric(n: usize, a_flat: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a_flat[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // singular
        }
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            s -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = s / aug[i * (n + 1) + i];
    }

    Some(x)
}

/// Invert symmetric matrix via Gaussian elimination (for small n).
fn invert_symmetric(n: usize, a_flat: &[f64]) -> Option<Vec<f64>> {
    // Augment with identity
    let mut aug = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a_flat[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
        }
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[col * 2 * n + j];
                aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                aug[max_row * 2 * n + j] = tmp;
            }
        }
        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }

    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    Some(inv)
}

/// BSADF sequence: for each r2, compute sup over r1 of ADF(data[r1..r2]).
pub fn bsadf_sequence(data: &[f64], min_window: usize) -> Vec<f64> {
    let n = data.len();
    let mut bsadf = vec![f64::NAN; n];

    for r2 in min_window..n {
        let mut sup_adf = f64::NEG_INFINITY;
        let max_r1 = if r2 >= min_window {
            r2 - min_window + 1
        } else {
            0
        };
        for r1 in 0..=max_r1 {
            let window_size = r2 - r1 + 1;
            if window_size < min_window {
                continue;
            }
            let stat = adf_stat(&data[r1..=r2]);
            if stat > sup_adf {
                sup_adf = stat;
            }
        }
        bsadf[r2] = sup_adf;
    }

    bsadf
}

/// Monte Carlo critical values under the null (random walk).
/// Parallelized with rayon for speed.
pub fn monte_carlo_critical_values(
    n: usize,
    min_window: usize,
    n_sims: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let gsadf_dist: Vec<f64> = (0..n_sims)
        .into_par_iter()
        .filter_map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            let mut rw = Vec::with_capacity(n);
            rw.push(0.0);
            for _ in 1..n {
                let eps: f64 = rng.sample(StandardNormal);
                rw.push(rw.last().unwrap() + eps);
            }

            let bsadf = bsadf_sequence(&rw, min_window);
            let gsadf = bsadf
                .iter()
                .filter(|x| x.is_finite())
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            if gsadf.is_finite() {
                Some(gsadf)
            } else {
                None
            }
        })
        .collect();

    let mut sorted = gsadf_dist;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = sorted.len();
    if len == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let cv_90 = sorted[((0.90 * len as f64) as usize).min(len - 1)];
    let cv_95 = sorted[((0.95 * len as f64) as usize).min(len - 1)];
    let cv_99 = sorted[((0.99 * len as f64) as usize).min(len - 1)];

    (cv_90, cv_95, cv_99)
}

/// Generalized Sup ADF test (Phillips, Shi, and Yu, 2015).
///
/// Tests for explosive unit root behavior (speculative bubbles).
/// Returns (gsadf_statistic, bsadf_sequence, (cv_90, cv_95, cv_99)).
///
/// `min_window`: minimum window size for ADF regressions (default auto-calculated).
/// `n_sims`: number of Monte Carlo simulations for critical values (default 1000).
/// `seed`: RNG seed (default 42).
pub fn gsadf_test_slice(
    data: &[f64],
    min_window: Option<usize>,
    n_sims: Option<usize>,
    seed: Option<u64>,
) -> (f64, Vec<f64>, (f64, f64, f64)) {
    let n = data.len();
    let n_sims = n_sims.unwrap_or(1000);
    let seed = seed.unwrap_or(42);

    let min_win = min_window
        .unwrap_or_else(|| (0.01 * n as f64 + 1.8 * (n as f64).sqrt()).floor() as usize)
        .max(4);

    if n < min_win + 1 {
        return (
            f64::NAN,
            vec![f64::NAN; n],
            (f64::NAN, f64::NAN, f64::NAN),
        );
    }

    let bsadf = bsadf_sequence(data, min_win);
    let gsadf = bsadf
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let cvs = monte_carlo_critical_values(n, min_win, n_sims, seed);

    (gsadf, bsadf, cvs)
}

/// Rolling GSADF for bubble detection.
/// Parallelized with rayon across time steps.
pub fn gsadf_rolling_slice(
    data: &[f64],
    window: usize,
    min_window: Option<usize>,
) -> Vec<f64> {
    let n = data.len();

    let min_win = min_window
        .unwrap_or_else(|| (0.01 * window as f64 + 1.8 * (window as f64).sqrt()).floor() as usize)
        .max(4);

    if window > n || window < min_win + 1 {
        return vec![f64::NAN; n];
    }

    let data_vec = data.to_vec();
    (0..n)
        .into_par_iter()
        .map(|i| {
            if i < window - 1 {
                return f64::NAN;
            }
            let slice = &data_vec[(i + 1 - window)..=i];
            let bsadf = bsadf_sequence(slice, min_win);
            let gsadf = bsadf
                .iter()
                .filter(|x| x.is_finite())
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            if gsadf.is_finite() {
                gsadf
            } else {
                f64::NAN
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adf_stat_random_walk() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 200;
        let mut rw = vec![0.0; n];
        for i in 1..n {
            rw[i] = rw[i - 1] + rng.sample::<f64, _>(StandardNormal);
        }
        let stat = adf_stat(&rw);
        assert!(stat.is_finite(), "ADF stat should be finite, got {}", stat);
    }

    #[test]
    fn test_adf_stat_explosive() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 200;
        let mut explosive = vec![1.0; n];
        for i in 1..n {
            explosive[i] = 1.05 * explosive[i - 1] + 0.1 * rng.sample::<f64, _>(StandardNormal);
        }
        let stat = adf_stat(&explosive);
        assert!(
            stat > 0.0,
            "ADF stat for explosive process should be positive, got {}",
            stat
        );
    }

    #[test]
    fn test_bsadf_length() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let bsadf = bsadf_sequence(&data, 20);
        assert_eq!(bsadf.len(), 100);
        for i in 0..20 {
            assert!(bsadf[i].is_nan(), "bsadf[{}] should be NaN", i);
        }
    }

    #[test]
    fn test_gsadf_bubble_detection() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 200;
        let mut series = vec![100.0; n];
        for i in 1..100 {
            series[i] = series[i - 1] + rng.sample::<f64, _>(StandardNormal);
        }
        for i in 100..n {
            series[i] = series[i - 1] * 1.02 + 0.5 * rng.sample::<f64, _>(StandardNormal);
        }

        let min_win = 20;
        let bsadf = bsadf_sequence(&series, min_win);

        let max_first: f64 = bsadf[min_win..100]
            .iter()
            .filter(|x| x.is_finite())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let max_second: f64 = bsadf[100..]
            .iter()
            .filter(|x| x.is_finite())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            max_second > max_first,
            "BSADF should be higher during bubble: first={}, second={}",
            max_first,
            max_second
        );
    }

    #[test]
    fn test_monte_carlo_critical_values_ordering() {
        let (cv90, cv95, cv99) = monte_carlo_critical_values(100, 20, 100, 42);
        assert!(cv90.is_finite());
        assert!(cv95.is_finite());
        assert!(cv99.is_finite());
        assert!(
            cv90 <= cv95 && cv95 <= cv99,
            "CVs should be ordered: {} <= {} <= {}",
            cv90,
            cv95,
            cv99
        );
    }
}
