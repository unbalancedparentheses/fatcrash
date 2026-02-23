use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::filter::{passes_filter, FilterConfig};
use super::model::solve_linear;

/// DS LPPLS confidence indicator.
/// For each anchor date t2, fit LPPLS over many (t1, t2) windows.
/// Confidence = fraction of fits that pass the Sornette filter.

fn fit_single_window(
    times: &[f64],
    log_prices: &[f64],
    t1_idx: usize,
    t2_idx: usize,
    filter_config: &FilterConfig,
    n_candidates: usize,
    seed: u64,
) -> bool {
    let slice_t = &times[t1_idx..=t2_idx];
    let slice_p = &log_prices[t1_idx..=t2_idx];

    if slice_t.len() < 20 {
        return false;
    }

    let t_end = *slice_t.last().unwrap();
    let t_range = t_end - slice_t[0];

    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_rss = f64::INFINITY;
    let mut best_passes = false;

    for _ in 0..n_candidates {
        let tc = t_end + rng.gen_range(0.0..t_range * 0.3);
        let m = rng.gen_range(0.1..0.9);
        let omega = rng.gen_range(2.0..25.0);

        if let Some((a, b, c1, c2, rss)) = solve_linear(slice_t, slice_p, tc, m, omega) {
            if rss < best_rss {
                best_rss = rss;
                let params = super::model::LpplsParams {
                    tc,
                    m,
                    omega,
                    a,
                    b,
                    c1,
                    c2,
                };
                best_passes = passes_filter(&params, filter_config);
            }
        }
    }

    best_passes
}

/// Compute LPPLS confidence for each time step.
/// Uses rayon for parallel computation across windows.
///
/// Returns array of confidence values [0, 1] for each time step.
/// confidence[t] = fraction of (t1, t2=t) windows that produce qualifying fits.
#[pyfunction]
#[pyo3(signature = (times, log_prices, min_window=60, max_window=750, n_windows=50, n_candidates=30))]
pub fn lppls_confidence<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    log_prices: PyReadonlyArray1<'py, f64>,
    min_window: Option<usize>,
    max_window: Option<usize>,
    n_windows: Option<usize>,
    n_candidates: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let times_slice = times.as_slice()?;
    let log_prices_slice = log_prices.as_slice()?;
    let n = times_slice.len();

    let min_win = min_window.unwrap_or(60);
    let max_win = max_window.unwrap_or(750).min(n);
    let n_win = n_windows.unwrap_or(50);
    let n_cand = n_candidates.unwrap_or(30);

    let filter_config = FilterConfig::default();

    // Allow Python threads during heavy computation
    let confidence: Vec<f64> = py.allow_threads(|| {
        let times_vec = times_slice.to_vec();
        let lp_vec = log_prices_slice.to_vec();

        (0..n)
            .into_par_iter()
            .map(|t2_idx| {
                if t2_idx < min_win {
                    return f64::NAN;
                }

                let actual_max = max_win.min(t2_idx);
                if min_win > actual_max {
                    return f64::NAN;
                }

                let step = ((actual_max - min_win) as f64 / n_win as f64).max(1.0) as usize;
                let mut pass_count = 0;
                let mut total_count = 0;

                let mut win_size = min_win;
                while win_size <= actual_max {
                    let t1_idx = t2_idx - win_size;
                    let seed = (t2_idx * 1000 + win_size) as u64;

                    if fit_single_window(
                        &times_vec,
                        &lp_vec,
                        t1_idx,
                        t2_idx,
                        &filter_config,
                        n_cand,
                        seed,
                    ) {
                        pass_count += 1;
                    }
                    total_count += 1;
                    win_size += step;
                }

                if total_count == 0 {
                    f64::NAN
                } else {
                    pass_count as f64 / total_count as f64
                }
            })
            .collect()
    });

    Ok(PyArray1::from_vec(py, confidence))
}
