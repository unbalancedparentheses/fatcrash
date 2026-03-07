use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::StandardNormal;

use super::filter::{passes_filter, FilterConfig};
use super::model::{r_squared, solve_linear, LpplsParams};

const MIN_R_SQUARED: f64 = 0.80;

/// Search bounds for the 3 nonlinear LPPLS parameters.
pub struct SearchBounds {
    pub tc_min: f64,
    pub tc_max: f64,
    pub m_min: f64,
    pub m_max: f64,
    pub omega_min: f64,
    pub omega_max: f64,
}

/// Full CMA-ES (Hansen 2001) for LPPLS nonlinear parameter optimization.
///
/// Implements rank-1 and rank-μ covariance matrix adaptation with cumulative
/// step-size adaptation (CSA). The 3 nonlinear parameters (tc, m, ω) are
/// optimized while linear parameters (A, B, C1, C2) are solved via OLS.
///
/// Returns the best fit that passes the Sornette filter, or the best overall
/// fit if no filtered fit is found.
pub fn search_lppls(
    times: &[f64],
    log_prices: &[f64],
    bounds: &SearchBounds,
    pop_size: usize,
    max_generations: usize,
    seed: u64,
) -> Option<(LpplsParams, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let filter_config = FilterConfig::default();
    let t_start = times[0];
    let t_end = *times.last().unwrap();

    let n = 3_usize; // dimension: tc, m, omega
    let lambda = pop_size;
    let mu = lambda / 2;

    // Recombination weights (log-linear, normalized)
    let raw_weights: Vec<f64> = (0..mu)
        .map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0))
        .collect();
    let w_sum: f64 = raw_weights.iter().sum();
    let weights: Vec<f64> = raw_weights.iter().map(|w| w / w_sum).collect();
    let mu_eff: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

    // Step-size control (CSA) parameters
    let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
    let d_sigma = 1.0
        + 2.0 * (((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0).max(0.0)
        + c_sigma;
    let chi_n = (n as f64).sqrt()
        * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * n as f64 * n as f64));

    // Covariance matrix adaptation parameters
    let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
    let c_1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
    let c_mu = (1.0 - c_1)
        .min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f64 + 2.0).powi(2) + mu_eff));

    // Bounds as arrays
    let lo = [bounds.tc_min, bounds.m_min, bounds.omega_min];
    let hi = [bounds.tc_max, bounds.m_max, bounds.omega_max];

    // Initial state
    let mut mean = DVector::from_vec(vec![
        (lo[0] + hi[0]) / 2.0,
        (lo[1] + hi[1]) / 2.0,
        (lo[2] + hi[2]) / 2.0,
    ]);

    // Initial covariance: diagonal scaled to 1/4 of each dimension's range
    let init_std: Vec<f64> = (0..n).map(|i| (hi[i] - lo[i]) / 4.0).collect();
    let mut cov = DMatrix::from_diagonal(&DVector::from_vec(
        init_std.iter().map(|s| s * s).collect(),
    ));

    let mut sigma = 1.0_f64; // global step size
    let mut p_sigma = DVector::zeros(n); // evolution path for step-size
    let mut p_c = DVector::zeros(n); // evolution path for covariance

    let mut best_filtered: Option<(LpplsParams, f64, f64)> = None;
    let mut best_unfiltered: Option<(LpplsParams, f64, f64)> = None;
    let mut stagnation = 0_usize;

    for gen in 0..max_generations {
        // Eigendecomposition of C for sampling: C = B * D^2 * B^T
        let eigen = cov.clone().symmetric_eigen();
        let eigenvalues = &eigen.eigenvalues;
        let eigenvectors = &eigen.eigenvectors;

        // D_sqrt = diag(sqrt(eigenvalues)), clamping negatives
        let d_sqrt = DMatrix::from_diagonal(
            &eigenvalues.map(|v| v.max(1e-20).sqrt()),
        );
        // B * D for sampling y = B*D*z
        let bd = eigenvectors * &d_sqrt;
        // B * D^(-1) for p_sigma update: C^(-1/2) = B * D^(-1) * B^T
        let d_inv = DMatrix::from_diagonal(
            &eigenvalues.map(|v| 1.0 / v.max(1e-20).sqrt()),
        );
        let c_inv_sqrt = eigenvectors * &d_inv * eigenvectors.transpose();

        // Sample population
        struct Candidate {
            x: DVector<f64>,
            y: DVector<f64>, // mutation vector (y = B*D*z)
            rss: f64,
        }

        let mut candidates: Vec<Candidate> = Vec::with_capacity(lambda);

        for _ in 0..lambda {
            let z = DVector::from_fn(n, |_, _| rng.sample::<f64, _>(StandardNormal));
            let y_raw = &bd * &z;
            let mut x = &mean + sigma * &y_raw;

            // Clip to bounds
            for i in 0..n {
                x[i] = x[i].clamp(lo[i], hi[i]);
            }

            // Recompute y after clipping so covariance update reflects actual steps
            let y = (&x - &mean) / sigma;

            let tc = x[0];
            let m_val = x[1];
            let omega = x[2];

            if let Some((a, b, c1, c2, rss)) =
                solve_linear(times, log_prices, tc, m_val, omega)
            {
                let r2 = r_squared(log_prices, rss);
                let params = LpplsParams {
                    tc,
                    m: m_val,
                    omega,
                    a,
                    b,
                    c1,
                    c2,
                };

                // Track best filtered fit
                if r2 >= MIN_R_SQUARED && passes_filter(&params, &filter_config, t_start, t_end)
                {
                    let improved = match &best_filtered {
                        Some((_, prev_rss, _)) => rss < *prev_rss,
                        None => true,
                    };
                    if improved {
                        best_filtered = Some((params.clone(), rss, r2));
                        stagnation = 0;
                    }
                }

                // Track best unfiltered fit
                let improved = match &best_unfiltered {
                    Some((_, prev_rss, _)) => rss < *prev_rss,
                    None => true,
                };
                if improved {
                    best_unfiltered = Some((params.clone(), rss, r2));
                    stagnation = 0;
                }

                candidates.push(Candidate { x, y, rss });
            }
        }

        // Sort by RSS (lower is better)
        candidates.sort_by(|a, b| a.rss.partial_cmp(&b.rss).unwrap_or(std::cmp::Ordering::Equal));

        let n_valid = candidates.len();
        if n_valid < mu {
            continue;
        }

        // Weighted mean of elite
        let old_mean = mean.clone();
        mean = DVector::zeros(n);
        for i in 0..mu {
            mean += weights[i] * &candidates[i].x;
        }

        let mean_diff = (&mean - &old_mean) / sigma;

        // Weighted mean of elite z-vectors (for p_sigma)
        // C^(-1/2) * mean_diff = sum(w_i * z_i) for the elite
        let invsqrt_diff = &c_inv_sqrt * &mean_diff;

        // Update evolution path for step-size (CSA)
        p_sigma = (1.0 - c_sigma) * &p_sigma
            + (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt() * &invsqrt_diff;

        // Update step-size
        sigma *= (c_sigma / d_sigma * (p_sigma.norm() / chi_n - 1.0)).exp();
        sigma = sigma.clamp(1e-12, 1e6);

        // Heaviside function h_sigma for rank-one update
        let gen_factor = (1.0 - (1.0 - c_sigma).powi(2 * (gen as i32 + 1))).sqrt();
        let h_sigma = if gen_factor > 0.0
            && p_sigma.norm() / gen_factor < (1.4 + 2.0 / (n as f64 + 1.0)) * chi_n
        {
            1.0
        } else {
            0.0
        };

        // Update evolution path for covariance
        p_c = (1.0 - c_c) * &p_c + h_sigma * (c_c * (2.0 - c_c) * mu_eff).sqrt() * &mean_diff;

        // Rank-1 update
        let rank_one = &p_c * p_c.transpose();

        // Rank-μ update: weighted sum of y_i * y_i^T from elite
        let mut rank_mu = DMatrix::zeros(n, n);
        for i in 0..mu {
            let yi = &candidates[i].y;
            rank_mu += weights[i] * (yi * yi.transpose());
        }

        // Update covariance matrix
        cov = (1.0 - c_1 - c_mu) * &cov
            + c_1 * (&rank_one + (1.0 - h_sigma) * c_c * (2.0 - c_c) * &cov)
            + c_mu * &rank_mu;

        // Convergence checks
        stagnation += 1;
        if stagnation > 10 + 30 / lambda {
            // No improvement for a while
            if best_filtered.is_some() {
                break;
            }
        }
        if sigma < 1e-10 {
            break; // Step size collapsed — converged
        }
        // Condition number check
        let max_eig = eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
        let min_eig = eigenvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .max(1e-20);
        if max_eig / min_eig > 1e14 {
            break; // Ill-conditioned
        }
    }

    best_filtered.or(best_unfiltered)
}

/// Fit LPPLS model to log-price time series.
///
/// Uses CMA-ES (Hansen 2001) with full covariance matrix adaptation to find
/// the nonlinear parameters (tc, m, ω), then OLS for linear parameters.
///
/// Returns (tc, m, omega, a, b, c1, c2, rss, r2) or an error string.
#[allow(clippy::type_complexity)]
pub fn lppls_fit_slice(
    times: &[f64],
    log_prices: &[f64],
    tc_range: Option<(f64, f64)>,
    pop_size: Option<usize>,
    n_generations: Option<usize>,
    seed: Option<u64>,
) -> Result<(f64, f64, f64, f64, f64, f64, f64, f64, f64), String> {
    if times.len() != log_prices.len() || times.len() < 20 {
        return Err("Need at least 20 data points and matching lengths".to_string());
    }

    let t_end = *times.last().unwrap();
    let t_range = times.last().unwrap() - times.first().unwrap();

    let (tc_min, tc_max) = tc_range.unwrap_or((t_end, t_end + t_range * 0.15));

    let bounds = SearchBounds {
        tc_min,
        tc_max,
        m_min: 0.01,
        m_max: 0.99,
        omega_min: 6.0,
        omega_max: 13.0,
    };

    let pop_size = pop_size.unwrap_or(50);
    let n_generations = n_generations.unwrap_or(60);
    let seed = seed.unwrap_or(42);

    match search_lppls(times, log_prices, &bounds, pop_size, n_generations, seed) {
        Some((p, rss, r2)) => Ok((p.tc, p.m, p.omega, p.a, p.b, p.c1, p.c2, rss, r2)),
        None => Err("LPPLS fitting failed to converge".to_string()),
    }
}
