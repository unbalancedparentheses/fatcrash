use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::filter::{passes_filter, FilterConfig};
use super::model::{r_squared, solve_linear};

const MIN_R_SQUARED: f64 = 0.80;

/// DS LPPLS confidence indicator.
/// For each anchor date t2, fit LPPLS over many (t1, t2) windows.
/// Confidence = fraction of fits that pass the Sornette filter.
/// Returns Some(tc) if the best fit passes, None otherwise.
fn fit_single_window(
    times: &[f64],
    log_prices: &[f64],
    t1_idx: usize,
    t2_idx: usize,
    filter_config: &FilterConfig,
    n_candidates: usize,
    seed: u64,
) -> Option<f64> {
    let slice_t = &times[t1_idx..=t2_idx];
    let slice_p = &log_prices[t1_idx..=t2_idx];

    if slice_t.len() < 20 {
        return None;
    }

    let t_end = *slice_t.last().unwrap();
    let t_range = t_end - slice_t[0];

    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_rss = f64::INFINITY;
    let mut best_result: Option<f64> = None;

    for _ in 0..n_candidates {
        let tc = t_end + rng.gen_range(0.0..t_range * 0.2);
        let m = rng.gen_range(0.1..0.9);
        let omega = rng.gen_range(6.0..13.0);

        if let Some((a, b, c1, c2, rss)) = solve_linear(slice_t, slice_p, tc, m, omega) {
            if rss < best_rss {
                best_rss = rss;
                let r2 = r_squared(slice_p, rss);
                let params = super::model::LpplsParams {
                    tc,
                    m,
                    omega,
                    a,
                    b,
                    c1,
                    c2,
                };
                if r2 >= MIN_R_SQUARED
                    && passes_filter(&params, filter_config, slice_t[0], *slice_t.last().unwrap())
                {
                    best_result = Some(tc);
                } else {
                    best_result = None;
                }
            }
        }
    }

    best_result
}

/// Compute LPPLS confidence for each time step.
/// Uses rayon for parallel computation across windows.
///
/// Returns three arrays (confidence, tc_mean, tc_std) for each time step.
/// confidence[t] = fraction of (t1, t2=t) windows that produce qualifying fits.
/// tc_mean[t] = mean tc from passing fits (NaN if none pass).
/// tc_std[t] = std of tc from passing fits (NaN if < 2 pass).
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
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let times_slice = times.as_slice()?;
    let log_prices_slice = log_prices.as_slice()?;
    let n = times_slice.len();

    let min_win = min_window.unwrap_or(60);
    let max_win = max_window.unwrap_or(750).min(n);
    let n_win = n_windows.unwrap_or(50);
    let n_cand = n_candidates.unwrap_or(30);

    let filter_config = FilterConfig::default();

    // Allow Python threads during heavy computation
    let results: Vec<(f64, f64, f64)> = py.allow_threads(|| {
        let times_vec = times_slice.to_vec();
        let lp_vec = log_prices_slice.to_vec();

        (0..n)
            .into_par_iter()
            .map(|t2_idx| {
                if t2_idx < min_win {
                    return (f64::NAN, f64::NAN, f64::NAN);
                }

                let actual_max = max_win.min(t2_idx);
                if min_win > actual_max {
                    return (f64::NAN, f64::NAN, f64::NAN);
                }

                let step = ((actual_max - min_win) as f64 / n_win as f64).max(1.0) as usize;
                let mut tc_values: Vec<f64> = Vec::new();
                let mut total_count = 0;

                let mut win_size = min_win;
                while win_size <= actual_max {
                    let t1_idx = t2_idx - win_size;
                    let seed = (t2_idx * 1000 + win_size) as u64;

                    if let Some(tc) = fit_single_window(
                        &times_vec,
                        &lp_vec,
                        t1_idx,
                        t2_idx,
                        &filter_config,
                        n_cand,
                        seed,
                    ) {
                        tc_values.push(tc);
                    }
                    total_count += 1;
                    win_size += step;
                }

                if total_count == 0 {
                    return (f64::NAN, f64::NAN, f64::NAN);
                }

                let confidence = tc_values.len() as f64 / total_count as f64;

                if tc_values.is_empty() {
                    return (confidence, f64::NAN, f64::NAN);
                }

                let tc_mean =
                    tc_values.iter().sum::<f64>() / tc_values.len() as f64;

                let tc_std = if tc_values.len() >= 2 {
                    let var = tc_values
                        .iter()
                        .map(|&v| (v - tc_mean).powi(2))
                        .sum::<f64>()
                        / (tc_values.len() - 1) as f64;
                    var.sqrt()
                } else {
                    f64::NAN
                };

                (confidence, tc_mean, tc_std)
            })
            .collect()
    });

    let mut conf_vec = Vec::with_capacity(n);
    let mut mean_vec = Vec::with_capacity(n);
    let mut std_vec = Vec::with_capacity(n);

    for (c, m, s) in results {
        conf_vec.push(c);
        mean_vec.push(m);
        std_vec.push(s);
    }

    Ok((
        PyArray1::from_vec(py, conf_vec),
        PyArray1::from_vec(py, mean_vec),
        PyArray1::from_vec(py, std_vec),
    ))
}
