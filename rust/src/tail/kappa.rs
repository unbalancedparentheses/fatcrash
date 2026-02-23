use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Taleb's kappa metric.
/// Measures how far a distribution's tail behavior is from Gaussian.
/// kappa = (mean of maxima over subsamples) / (overall max)
/// For Gaussian, kappa → 1/sqrt(n_subsamples). Deviation indicates fat tails.
///
/// kappa < gaussian_benchmark → fatter tails than Gaussian
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

/// Compute Gaussian benchmark for kappa via Monte Carlo.
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

/// Compute kappa metric for the given data.
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

/// Rolling kappa metric.
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

        // Gaussian data should have kappa close to benchmark
        assert!(
            (kappa - benchmark).abs() < 0.15,
            "kappa {} too far from Gaussian benchmark {}",
            kappa,
            benchmark
        );
    }
}
