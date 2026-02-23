use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

// ── Taleb's kappa ────────────────────────────────────────────────
//
// From "How Much Data Do You Need?" (Taleb, 2018):
//
//   M(n) = E[|S_n - E[S_n]|]      (MAD of partial sums)
//   kappa(n0, n) = 2 - log(n/n0) / log(M(n)/M(n0))
//
// kappa = 0  → Gaussian (CLT converges at normal sqrt(n) rate)
// kappa → 1  → Cauchy-like (no convergence)
//
// Empirical estimation: partition data into blocks, compute MAD of
// block sums at two scales, plug into the formula.

/// Estimate M(n) = E[|S_n - E[S_n]|] from data by partitioning into
/// non-overlapping blocks of size `block_size` and averaging |sum - mean_sum|.
fn estimate_mad_of_sums(data: &[f64], block_size: usize) -> f64 {
    let n = data.len();
    if block_size == 0 || block_size > n {
        return f64::NAN;
    }

    let n_blocks = n / block_size;
    if n_blocks < 2 {
        return f64::NAN;
    }

    let block_sums: Vec<f64> = (0..n_blocks)
        .map(|i| {
            let start = i * block_size;
            data[start..start + block_size].iter().sum::<f64>()
        })
        .collect();

    let mean_sum: f64 = block_sums.iter().sum::<f64>() / n_blocks as f64;

    block_sums.iter().map(|s| (s - mean_sum).abs()).sum::<f64>() / n_blocks as f64
}

/// Compute Taleb's kappa from data.
///
/// Uses two block sizes (n0, n) to measure how the MAD of partial sums
/// scales. Returns kappa in [0, 1] (clamped).
fn compute_taleb_kappa(data: &[f64], n0: usize, n: usize) -> f64 {
    if n <= n0 || n0 == 0 {
        return f64::NAN;
    }

    let m_n0 = estimate_mad_of_sums(data, n0);
    let m_n = estimate_mad_of_sums(data, n);

    if m_n0.is_nan() || m_n.is_nan() || m_n0 <= 0.0 || m_n <= 0.0 {
        return f64::NAN;
    }

    let log_ratio_n = (n as f64).ln() - (n0 as f64).ln();
    let log_ratio_m = (m_n).ln() - (m_n0).ln();

    if log_ratio_m == 0.0 {
        return f64::NAN;
    }

    let kappa = 2.0 - log_ratio_n / log_ratio_m;
    kappa.clamp(0.0, 1.0)
}

/// Gaussian benchmark for Taleb kappa via Monte Carlo.
fn gaussian_taleb_kappa_benchmark(data_len: usize, n0: usize, n: usize, n_sims: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(12345);
    let mut sum = 0.0;
    let mut count = 0;

    for _ in 0..n_sims {
        let samples: Vec<f64> = (0..data_len)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        let k = compute_taleb_kappa(&samples, n0, n);
        if !k.is_nan() {
            sum += k;
            count += 1;
        }
    }

    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Taleb's kappa metric: rate of MAD convergence.
///
/// kappa = 0 → Gaussian, kappa → 1 → Cauchy.
/// Returns (kappa, gaussian_benchmark).
#[pyfunction]
#[pyo3(signature = (data, n0=30, n1=100, n_sims=500))]
pub fn taleb_kappa(
    data: PyReadonlyArray1<'_, f64>,
    n0: Option<usize>,
    n1: Option<usize>,
    n_sims: Option<usize>,
) -> PyResult<(f64, f64)> {
    let data = data.as_slice()?;
    let n0 = n0.unwrap_or(30);
    let n1_val = n1.unwrap_or(100).min(data.len() / 2);
    let n_sims = n_sims.unwrap_or(500);

    let kappa = compute_taleb_kappa(data, n0, n1_val);
    let benchmark = gaussian_taleb_kappa_benchmark(data.len(), n0, n1_val, n_sims);

    Ok((kappa, benchmark))
}

/// Rolling Taleb kappa.
#[pyfunction]
#[pyo3(signature = (data, window, n0=30, n1=100, n_sims=100))]
pub fn taleb_kappa_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    n0: Option<usize>,
    n1: Option<usize>,
    n_sims: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let data = data.as_slice()?;
    let len = data.len();
    let n0 = n0.unwrap_or(30);
    let n1_val = n1.unwrap_or(100).min(window / 2);
    let n_sims = n_sims.unwrap_or(100);

    let benchmark = gaussian_taleb_kappa_benchmark(window, n0, n1_val, n_sims);

    let mut result = vec![f64::NAN; len];
    if window == 0 || n1_val <= n0 {
        return Ok((PyArray1::from_vec(py, result), benchmark));
    }
    for i in (window - 1)..len {
        let slice = &data[(i + 1 - window)..=i];
        result[i] = compute_taleb_kappa(slice, n0, n1_val);
    }

    Ok((PyArray1::from_vec(py, result), benchmark))
}

// ── Max-stability kappa (original metric) ────────────────────────
//
// kappa = (mean of maxima over subsamples) / (overall max)
// Compared to Gaussian benchmark via Monte Carlo.
// kappa < benchmark → fatter tails than Gaussian.

fn compute_kappa(data: &[f64], n_subsamples: usize) -> f64 {
    let n = data.len();
    if n < n_subsamples || n_subsamples == 0 {
        return f64::NAN;
    }

    let overall_max = data.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if overall_max == 0.0 {
        return f64::NAN;
    }

    let chunk_size = n / n_subsamples;
    if chunk_size == 0 {
        return f64::NAN;
    }

    let sum_maxima: f64 = (0..n_subsamples)
        .map(|i| {
            let start = i * chunk_size;
            let end = start + chunk_size;
            data[start..end]
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max)
        })
        .sum();

    let mean_maxima = sum_maxima / n_subsamples as f64;
    mean_maxima / overall_max
}

/// Gaussian benchmark for max-stability kappa via Monte Carlo.
fn gaussian_kappa_benchmark(n: usize, n_subsamples: usize, n_sims: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(12345);
    let mut sum = 0.0;

    for _ in 0..n_sims {
        let samples: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        sum += compute_kappa(&samples, n_subsamples);
    }

    sum / n_sims as f64
}

/// Max-stability kappa: subsample-max ratio.
/// Returns (kappa, gaussian_benchmark).
#[pyfunction]
#[pyo3(signature = (data, n_subsamples=10, n_sims=1000))]
pub fn kappa_metric(
    data: PyReadonlyArray1<'_, f64>,
    n_subsamples: Option<usize>,
    n_sims: Option<usize>,
) -> PyResult<(f64, f64)> {
    let data = data.as_slice()?;
    let n_sub = n_subsamples.unwrap_or(10);
    let n_sims = n_sims.unwrap_or(1000);

    let kappa = compute_kappa(data, n_sub);
    let benchmark = gaussian_kappa_benchmark(data.len(), n_sub, n_sims);

    Ok((kappa, benchmark))
}

/// Rolling max-stability kappa.
#[pyfunction]
#[pyo3(signature = (data, window, n_subsamples=10, n_sims=200))]
pub fn kappa_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    n_subsamples: Option<usize>,
    n_sims: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let data = data.as_slice()?;
    let n = data.len();
    let n_sub = n_subsamples.unwrap_or(10);
    let n_sims = n_sims.unwrap_or(200);

    let benchmark = gaussian_kappa_benchmark(window, n_sub, n_sims);

    let mut result = vec![f64::NAN; n];
    if window == 0 {
        return Ok((PyArray1::from_vec(py, result), benchmark));
    }
    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        result[i] = compute_kappa(slice, n_sub);
    }

    Ok((PyArray1::from_vec(py, result), benchmark))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kappa_near_benchmark() {
        let mut rng = StdRng::seed_from_u64(99);
        let n = 1000;
        let samples: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        let kappa = compute_kappa(&samples, 10);
        let benchmark = gaussian_kappa_benchmark(n, 10, 500);

        assert!(
            (kappa - benchmark).abs() < 0.15,
            "kappa {} too far from Gaussian benchmark {}",
            kappa,
            benchmark
        );
    }

    #[test]
    fn test_taleb_kappa_gaussian_near_zero() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 2000;
        let samples: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        let kappa = compute_taleb_kappa(&samples, 30, 200);
        // Gaussian should have kappa near 0
        assert!(
            kappa < 0.4,
            "Gaussian taleb_kappa {} should be near 0",
            kappa
        );
    }

    #[test]
    fn test_taleb_kappa_fat_tails_higher() {
        // Cauchy-like: generate heavy-tailed data
        let mut rng = StdRng::seed_from_u64(77);
        let n = 2000;

        // Gaussian
        let gaussian: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        // Cauchy-ish: ratio of two normals
        let cauchy: Vec<f64> = (0..n)
            .map(|_| {
                let a: f64 = rng.sample(StandardNormal);
                let b: f64 = rng.sample(StandardNormal);
                if b.abs() < 1e-10 { a } else { a / b }
            })
            .collect();

        let k_gauss = compute_taleb_kappa(&gaussian, 30, 200);
        let k_cauchy = compute_taleb_kappa(&cauchy, 30, 200);

        assert!(
            k_cauchy > k_gauss,
            "Cauchy kappa {} should exceed Gaussian kappa {}",
            k_cauchy,
            k_gauss
        );
    }
}
