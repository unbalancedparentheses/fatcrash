use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Maximum-to-Sum ratio diagnostic.
///
/// R_n = max(|X_i|) / sum(|X_i|)
///
/// For thin-tailed data (alpha > 2), R_n → 0 as n → infinity.
/// For heavy-tailed data (alpha < 2), R_n stays positive — a single
/// observation dominates the sum.
///
/// Direct test of the infinite variance hypothesis.
fn compute_maxsum(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;

    for &x in data {
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
        sum_abs += a;
    }

    if sum_abs < 1e-15 {
        return f64::NAN;
    }

    max_abs / sum_abs
}

/// Compute max-to-sum ratio for the given data.
/// Returns R_n in [0, 1]: higher = more concentrated (fatter tails).
#[pyfunction]
#[pyo3(signature = (data,))]
pub fn maxsum_ratio(data: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(compute_maxsum(data))
}

/// Rolling max-to-sum ratio over a window.
#[pyfunction]
#[pyo3(signature = (data, window))]
pub fn maxsum_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();

    if window > n || window < 2 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        result[i] = compute_maxsum(slice);
    }

    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsum_gaussian() {
        // Gaussian: R_n should be relatively small for large n
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 10000;
        let data: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        let r = compute_maxsum(&data);
        assert!(r.is_finite(), "R_n should be finite, got {}", r);
        // For 10k Gaussian samples, max/sum should be small
        assert!(
            r < 0.05,
            "Gaussian max/sum ratio should be small, got {}",
            r
        );
    }

    #[test]
    fn test_maxsum_cauchy() {
        // Cauchy (alpha=1): R_n should be larger than Gaussian
        use rand::prelude::*;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 10000;
        let data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();

        // Generate Cauchy via tan transform
        let cauchy: Vec<f64> = data
            .iter()
            .map(|&u| (std::f64::consts::PI * (u - 0.5)).tan())
            .collect();
        let gauss: Vec<f64> = {
            use rand_distr::StandardNormal;
            let mut rng2 = StdRng::seed_from_u64(42);
            (0..n)
                .map(|_| rng2.sample::<f64, _>(StandardNormal))
                .collect()
        };

        let r_cauchy = compute_maxsum(&cauchy);
        let r_gauss = compute_maxsum(&gauss);

        assert!(
            r_cauchy > r_gauss,
            "Cauchy R_n ({}) should > Gaussian R_n ({})",
            r_cauchy,
            r_gauss
        );
    }

    #[test]
    fn test_maxsum_empty() {
        let data: Vec<f64> = vec![];
        let r = compute_maxsum(&data);
        assert!(r.is_nan());
    }
}
