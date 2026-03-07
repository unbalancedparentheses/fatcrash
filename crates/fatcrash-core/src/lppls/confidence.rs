use rayon::prelude::*;

use super::filter::{passes_filter, FilterConfig};
use super::fitter::{search_lppls, SearchBounds};

/// DS LPPLS confidence indicator for a single anchor date.
///
/// Fits LPPLS across many (t1, t2) windows ending at t2_idx.
/// Confidence = fraction of windows that produce a qualifying fit
/// (passing Sornette filter with R² ≥ 0.80).
///
/// Uses CMA-ES for each window instead of random search.
///
/// Returns (confidence, tc_mean, tc_std).
pub fn lppls_confidence_at(
    times: &[f64],
    log_prices: &[f64],
    t2_idx: usize,
    min_window: usize,
    max_window: usize,
    n_windows: usize,
    pop_size: usize,
    n_generations: usize,
) -> (f64, f64, f64) {
    if t2_idx < min_window {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let actual_max = max_window.min(t2_idx);
    if min_window > actual_max {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let step = ((actual_max - min_window) as f64 / n_windows as f64).max(1.0) as usize;
    let mut tc_values: Vec<f64> = Vec::new();
    let mut total_count = 0;

    let mut win_size = min_window;
    while win_size <= actual_max {
        let t1_idx = t2_idx - win_size;
        let slice_t = &times[t1_idx..=t2_idx];
        let slice_p = &log_prices[t1_idx..=t2_idx];

        if slice_t.len() >= 20 {
            let t_end = *slice_t.last().unwrap();
            let t_range = t_end - slice_t[0];

            let bounds = SearchBounds {
                tc_min: t_end,
                tc_max: t_end + t_range * 0.15,
                m_min: 0.01,
                m_max: 0.99,
                omega_min: 6.0,
                omega_max: 13.0,
            };

            // Deterministic seed per window
            let seed = (t2_idx as u64)
                .wrapping_mul(10007)
                .wrapping_add(win_size as u64);

            if let Some((params, _rss, _r2)) =
                search_lppls(slice_t, slice_p, &bounds, pop_size, n_generations, seed)
            {
                // Re-check filter since search_lppls falls back to unfiltered
                let filter_config = FilterConfig::default();
                let t_start = slice_t[0];
                if passes_filter(&params, &filter_config, t_start, t_end) {
                    tc_values.push(params.tc);
                }
            }
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

    let tc_mean = tc_values.iter().sum::<f64>() / tc_values.len() as f64;

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
}

/// Compute LPPLS confidence for each time step.
/// Uses rayon for parallel computation across time points.
///
/// Returns three vectors (confidence, tc_mean, tc_std) for each time step.
/// confidence[t] = fraction of (t1, t2=t) windows producing qualifying fits.
/// tc_mean[t] = mean tc from passing fits (NaN if none pass).
/// tc_std[t] = std of tc from passing fits (NaN if < 2 pass).
pub fn lppls_confidence_slice(
    times: &[f64],
    log_prices: &[f64],
    min_window: Option<usize>,
    max_window: Option<usize>,
    n_windows: Option<usize>,
    n_candidates: Option<usize>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = times.len();

    let min_win = min_window.unwrap_or(60);
    let max_win = max_window.unwrap_or(750).min(n);
    let n_win = n_windows.unwrap_or(50);
    // n_candidates now controls CMA-ES population size
    let pop_size = n_candidates.unwrap_or(20);
    let n_gen = 20; // Lighter CMA-ES for confidence (many windows to test)

    let times_vec = times.to_vec();
    let lp_vec = log_prices.to_vec();

    let results: Vec<(f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|t2_idx| {
            lppls_confidence_at(
                &times_vec, &lp_vec, t2_idx, min_win, max_win, n_win, pop_size, n_gen,
            )
        })
        .collect();

    let mut conf_vec = Vec::with_capacity(n);
    let mut mean_vec = Vec::with_capacity(n);
    let mut std_vec = Vec::with_capacity(n);

    for (c, m, s) in results {
        conf_vec.push(c);
        mean_vec.push(m);
        std_vec.push(s);
    }

    (conf_vec, mean_vec, std_vec)
}
