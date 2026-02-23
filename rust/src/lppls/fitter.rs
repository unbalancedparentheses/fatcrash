use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;

use super::filter::{passes_filter, FilterConfig};
use super::model::{solve_linear, LpplsParams};

/// CMA-ES inspired stochastic search for LPPLS nonlinear parameters.
/// Searches over (tc, m, omega) space, solving linear params via OLS at each point.

struct SearchBounds {
    tc_min: f64,
    tc_max: f64,
    m_min: f64,
    m_max: f64,
    omega_min: f64,
    omega_max: f64,
}

/// Simple population-based search (simplified CMA-ES).
/// For each candidate (tc, m, omega), solve OLS for (A, B, C1, C2).
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

    let mut best_params: Option<LpplsParams> = None;
    let mut best_rss = f64::INFINITY;

    // Mean and std for each parameter
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
        let mut candidates: Vec<([f64; 3], f64, Option<LpplsParams>)> =
            Vec::with_capacity(pop_size);

        for _ in 0..pop_size {
            // Sample candidate
            let tc = (mean[0] + sigma[0] * rng.gen_range(-1.0..1.0))
                .clamp(bounds.tc_min, bounds.tc_max);
            let m = (mean[1] + sigma[1] * rng.gen_range(-1.0..1.0))
                .clamp(bounds.m_min, bounds.m_max);
            let omega = (mean[2] + sigma[2] * rng.gen_range(-1.0..1.0))
                .clamp(bounds.omega_min, bounds.omega_max);

            if let Some((a, b, c1, c2, rss)) = solve_linear(times, log_prices, tc, m, omega) {
                let params = LpplsParams {
                    tc, m, omega, a, b, c1, c2,
                };
                candidates.push(([tc, m, omega], rss, Some(params)));
            }
        }

        // Sort by RSS
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update best
        if let Some(ref top) = candidates.first() {
            if top.1 < best_rss {
                if let Some(ref p) = top.2 {
                    best_rss = top.1;
                    best_params = Some(p.clone());
                }
            }
        }

        // Update mean/sigma from top 25%
        let elite_count = (pop_size / 4).max(2);
        let elite: Vec<&([f64; 3], f64, Option<LpplsParams>)> =
            candidates.iter().take(elite_count).collect();

        for dim in 0..3 {
            let elite_mean =
                elite.iter().map(|c| c.0[dim]).sum::<f64>() / elite_count as f64;
            let elite_var = elite
                .iter()
                .map(|c| (c.0[dim] - elite_mean).powi(2))
                .sum::<f64>()
                / elite_count as f64;
            mean[dim] = elite_mean;
            sigma[dim] = elite_var.sqrt().max(1e-6);
        }
    }

    best_params.map(|p| (p, best_rss))
}

/// Fit LPPLS model to log-price time series.
/// Returns (tc, m, omega, a, b, c1, c2, rss) or raises error.
#[pyfunction]
#[pyo3(signature = (times, log_prices, tc_range=None, pop_size=50, n_generations=40, seed=42))]
pub fn lppls_fit<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    log_prices: PyReadonlyArray1<'py, f64>,
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
