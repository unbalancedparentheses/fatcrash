use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rand::prelude::*;

use super::filter::{passes_filter, FilterConfig};
use super::model::{solve_linear, LpplsParams};

/// Population-based stochastic search for LPPLS nonlinear parameters.
/// Searches over (tc, m, omega) space, solving linear params via OLS at each point.
/// Tracks both the best overall fit and the best filter-passing fit.

struct SearchBounds {
    tc_min: f64,
    tc_max: f64,
    m_min: f64,
    m_max: f64,
    omega_min: f64,
    omega_max: f64,
}

fn search_lppls(
    times: &[f64],
    log_prices: &[f64],
    bounds: &SearchBounds,
    pop_size: usize,
    n_generations: usize,
    seed: u64,
) -> Option<(LpplsParams, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let filter_config = FilterConfig::default();

    let mut best_filtered: Option<(LpplsParams, f64)> = None;
    let mut best_unfiltered: Option<(LpplsParams, f64)> = None;

    let mut mean = [
        (bounds.tc_min + bounds.tc_max) / 2.0,
        (bounds.m_min + bounds.m_max) / 2.0,
        (bounds.omega_min + bounds.omega_max) / 2.0,
    ];
    let mut sigma = [
        (bounds.tc_max - bounds.tc_min) / 4.0,
        (bounds.m_max - bounds.m_min) / 4.0,
        (bounds.omega_max - bounds.omega_min) / 4.0,
    ];

    for _gen in 0..n_generations {
        let mut candidates: Vec<([f64; 3], f64)> = Vec::with_capacity(pop_size);

        for _ in 0..pop_size {
            let tc =
                (mean[0] + sigma[0] * rng.gen_range(-1.0..1.0)).clamp(bounds.tc_min, bounds.tc_max);
            let m =
                (mean[1] + sigma[1] * rng.gen_range(-1.0..1.0)).clamp(bounds.m_min, bounds.m_max);
            let omega = (mean[2] + sigma[2] * rng.gen_range(-1.0..1.0))
                .clamp(bounds.omega_min, bounds.omega_max);

            if let Some((a, b, c1, c2, rss)) = solve_linear(times, log_prices, tc, m, omega) {
                let params = LpplsParams {
                    tc,
                    m,
                    omega,
                    a,
                    b,
                    c1,
                    c2,
                };
                candidates.push(([tc, m, omega], rss));

                // Track best filter-passing fit
                if passes_filter(&params, &filter_config) {
                    match &best_filtered {
                        Some((_, prev_rss)) if rss < *prev_rss => {
                            best_filtered = Some((params.clone(), rss));
                        }
                        None => {
                            best_filtered = Some((params.clone(), rss));
                        }
                        _ => {}
                    }
                }

                // Track best overall fit (fallback)
                match &best_unfiltered {
                    Some((_, prev_rss)) if rss < *prev_rss => {
                        best_unfiltered = Some((params, rss));
                    }
                    None => {
                        best_unfiltered = Some((params, rss));
                    }
                    _ => {}
                }
            }
        }

        // Sort by RSS for elite selection
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update distribution from top 25%
        let n_valid = candidates.len();
        if n_valid == 0 {
            continue;
        }
        let elite_count = (n_valid / 4).max(1);

        for dim in 0..3 {
            let elite_mean: f64 = candidates
                .iter()
                .take(elite_count)
                .map(|c| c.0[dim])
                .sum::<f64>()
                / elite_count as f64;
            let elite_var: f64 = candidates
                .iter()
                .take(elite_count)
                .map(|c| (c.0[dim] - elite_mean).powi(2))
                .sum::<f64>()
                / elite_count as f64;
            mean[dim] = elite_mean;
            sigma[dim] = elite_var.sqrt().max(1e-6);
        }
    }

    // Prefer filter-passing fit; fall back to best unfiltered
    best_filtered.or(best_unfiltered)
}

/// Fit LPPLS model to log-price time series.
/// Returns (tc, m, omega, a, b, c1, c2, rss) or raises error.
#[pyfunction]
#[pyo3(signature = (times, log_prices, tc_range=None, pop_size=50, n_generations=40, seed=42))]
pub fn lppls_fit(
    _py: Python<'_>,
    times: PyReadonlyArray1<'_, f64>,
    log_prices: PyReadonlyArray1<'_, f64>,
    tc_range: Option<(f64, f64)>,
    pop_size: Option<usize>,
    n_generations: Option<usize>,
    seed: Option<u64>,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64)> {
    let times = times.as_slice()?;
    let log_prices = log_prices.as_slice()?;

    if times.len() != log_prices.len() || times.len() < 20 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 20 data points and matching lengths",
        ));
    }

    let t_end = *times.last().unwrap();
    let t_range = times.last().unwrap() - times.first().unwrap();

    let (tc_min, tc_max) = tc_range.unwrap_or((t_end, t_end + t_range * 0.5));

    let bounds = SearchBounds {
        tc_min,
        tc_max,
        m_min: 0.1,
        m_max: 0.9,
        omega_min: 2.0,
        omega_max: 25.0,
    };

    let pop_size = pop_size.unwrap_or(50);
    let n_generations = n_generations.unwrap_or(40);
    let seed = seed.unwrap_or(42);

    match search_lppls(times, log_prices, &bounds, pop_size, n_generations, seed) {
        Some((p, rss)) => Ok((p.tc, p.m, p.omega, p.a, p.b, p.c1, p.c2, rss)),
        None => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "LPPLS fitting failed to converge",
        )),
    }
}
