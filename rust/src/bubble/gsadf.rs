use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// (gsadf_statistic, bsadf_sequence, (cv_90, cv_95, cv_99))
type GsadfResult<'py> = (f64, Bound<'py, PyArray1<f64>>, (f64, f64, f64));

/// Run an ADF regression on a data slice.
/// Regression: delta_y_t = mu + delta * y_{t-1} + epsilon_t
/// Returns the t-statistic for delta (ADF statistic).
fn adf_stat(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 4 {
        return f64::NEG_INFINITY;
    }

    let m = n - 1;

    let mut sum_x = 0.0_f64;
    let mut sum_x2 = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xy = 0.0_f64;

    for t in 1..n {
        let dy = data[t] - data[t - 1];
        let y_lag = data[t - 1];
        sum_x += y_lag;
        sum_x2 += y_lag * y_lag;
        sum_y += dy;
        sum_xy += y_lag * dy;
    }

    let m_f = m as f64;

    let det = m_f * sum_x2 - sum_x * sum_x;
    if det.abs() < 1e-15 {
        return f64::NEG_INFINITY;
    }

    let mu = (sum_x2 * sum_y - sum_x * sum_xy) / det;
    let delta = (m_f * sum_xy - sum_x * sum_y) / det;

    let mut sse = 0.0;
    for t in 1..n {
        let dy = data[t] - data[t - 1];
        let y_lag = data[t - 1];
        let residual = dy - mu - delta * y_lag;
        sse += residual * residual;
    }

    if m <= 2 {
        return f64::NEG_INFINITY;
    }

    let sigma2 = sse / (m - 2) as f64;
    let var_delta = sigma2 * m_f / det;

    if var_delta <= 0.0 {
        return f64::NEG_INFINITY;
    }

    delta / var_delta.sqrt()
}

/// BSADF sequence: for each r2, compute sup over r1 of ADF(data[r1..r2]).
fn bsadf_sequence(data: &[f64], min_window: usize) -> Vec<f64> {
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
fn monte_carlo_critical_values(
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
#[pyfunction]
#[pyo3(signature = (data, min_window=None, n_sims=200, seed=42))]
pub fn gsadf_test<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    min_window: Option<usize>,
    n_sims: Option<usize>,
    seed: Option<u64>,
) -> PyResult<GsadfResult<'py>> {
    let data_slice = data.as_slice()?;
    let n = data_slice.len();
    let n_sims = n_sims.unwrap_or(200);
    let seed = seed.unwrap_or(42);

    let min_win = min_window
        .unwrap_or_else(|| (0.01 * n as f64 + 1.8 * (n as f64).sqrt()).floor() as usize)
        .max(4);

    if n < min_win + 1 {
        return Ok((
            f64::NAN,
            PyArray1::from_vec(py, vec![f64::NAN; n]),
            (f64::NAN, f64::NAN, f64::NAN),
        ));
    }

    // Release GIL for heavy computation
    let (gsadf, bsadf, cvs) = py.allow_threads(|| {
        let bsadf = bsadf_sequence(data_slice, min_win);
        let gsadf = bsadf
            .iter()
            .filter(|x| x.is_finite())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let cvs = monte_carlo_critical_values(n, min_win, n_sims, seed);
        (gsadf, bsadf, cvs)
    });

    Ok((gsadf, PyArray1::from_vec(py, bsadf), cvs))
}

/// Rolling GSADF for bubble detection.
/// Parallelized with rayon across time steps.
#[pyfunction]
#[pyo3(signature = (data, window, min_window=None))]
pub fn gsadf_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    min_window: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let n = data_slice.len();

    let min_win = min_window
        .unwrap_or_else(|| (0.01 * window as f64 + 1.8 * (window as f64).sqrt()).floor() as usize)
        .max(4);

    if window > n || window < min_win + 1 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }

    let result: Vec<f64> = py.allow_threads(|| {
        let data_vec = data_slice.to_vec();
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
    });

    Ok(PyArray1::from_vec(py, result))
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
