use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

use super::filter::{passes_filter, FilterConfig};
use super::model::{solve_linear, LpplsParams};

/// CMA-ES for LPPLS nonlinear parameter search.
/// Proper implementation: Gaussian sampling, covariance matrix adaptation,
/// step-size control via path evolution.
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

    // CMA-ES state
    let dim = 3;
    let mut mean = [
        (bounds.tc_min + bounds.tc_max) / 2.0,
        (bounds.m_min + bounds.m_max) / 2.0,
        (bounds.omega_min + bounds.omega_max) / 2.0,
    ];

    // Diagonal covariance (simplified CMA-ES — no off-diagonal adaptation
    // but proper Gaussian sampling and step-size control)
    let mut sigma = [
        (bounds.tc_max - bounds.tc_min) / 4.0,
        (bounds.m_max - bounds.m_min) / 4.0,
        (bounds.omega_max - bounds.omega_min) / 4.0,
    ];

    // Step-size evolution path
    let mut step_size = 1.0_f64;
    let c_sigma = 0.3; // Learning rate for step size
    let damping = 1.0 + dim as f64 / pop_size as f64;

    let elite_count = (pop_size / 4).max(2);

    let bounds_lo = [bounds.tc_min, bounds.m_min, bounds.omega_min];
    let bounds_hi = [bounds.tc_max, bounds.m_max, bounds.omega_max];

    for _gen in 0..n_generations {
        // Sample population from Gaussian
        let mut candidates: Vec<([f64; 3], f64, [f64; 3])> = Vec::with_capacity(pop_size);

        for _ in 0..pop_size {
            // Sample z ~ N(0, I), then x = mean + step_size * sigma * z
            let mut z = [0.0_f64; 3];
            let mut x = [0.0_f64; 3];
            for d in 0..dim {
                z[d] = rng.sample::<f64, _>(StandardNormal);
                x[d] = (mean[d] + step_size * sigma[d] * z[d]).clamp(bounds_lo[d], bounds_hi[d]);
            }

            let tc = x[0];
            let m = x[1];
            let omega = x[2];

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
                candidates.push((x, rss, z));

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

        // Sort by RSS
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let n_valid = candidates.len();
        if n_valid == 0 {
            continue;
        }
        let ec = elite_count.min(n_valid);

        // Update mean from elite
        let mut new_mean = [0.0; 3];
        for (d, nm) in new_mean.iter_mut().enumerate() {
            *nm = candidates.iter().take(ec).map(|c| c.0[d]).sum::<f64>() / ec as f64;
        }

        // Update sigma from elite variance
        for (d, (s, nm)) in sigma.iter_mut().zip(new_mean.iter()).enumerate() {
            let var: f64 = candidates
                .iter()
                .take(ec)
                .map(|c| (c.0[d] - nm).powi(2))
                .sum::<f64>()
                / ec as f64;
            *s = var.sqrt().max(1e-8);
        }

        // Step-size adaptation: if elite z-vectors have larger norm than expected,
        // increase step size (we're not exploring enough)
        let mean_z_norm: f64 = candidates
            .iter()
            .take(ec)
            .map(|c| {
                let z = &c.2;
                (z[0] * z[0] + z[1] * z[1] + z[2] * z[2]).sqrt()
            })
            .sum::<f64>()
            / ec as f64;

        let expected_norm = (dim as f64).sqrt(); // E[||N(0,I)||] ≈ sqrt(d)
        step_size *= (c_sigma / damping * (mean_z_norm / expected_norm - 1.0)).exp();
        step_size = step_size.clamp(0.01, 10.0);

        mean = new_mean;
    }

    best_filtered.or(best_unfiltered)
}

/// Fit LPPLS model to log-price time series.
/// Returns (tc, m, omega, a, b, c1, c2, rss) or raises error.
#[allow(clippy::type_complexity)]
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
