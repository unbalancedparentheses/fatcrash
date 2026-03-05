use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

const CLIP_LO: f64 = 1e-10;
const CLIP_HI: f64 = 1.0 - 1e-10;
const EM_TOL: f64 = 1e-8;
const EM_MAX_ITER: usize = 500;

/// Clip probability to [CLIP_LO, CLIP_HI].
fn clip(p: f64) -> f64 {
    p.clamp(CLIP_LO, CLIP_HI)
}

/// Log of Gaussian PDF.
fn log_normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Hamilton forward filter.
///
/// Given parameters (mu, sigma for each state, transition matrix),
/// compute P(s_t = 1 | y_1..y_t) at each time step.
///
/// Returns filtered probabilities of state 1 (stressed).
fn forward_filter(
    data: &[f64],
    mu: [f64; 2],
    sigma: [f64; 2],
    p00: f64,
    p11: f64,
) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let mut filtered = vec![0.0; n];
    // Ergodic (stationary) starting probability
    let p01 = 1.0 - p00;
    let p10 = 1.0 - p11;
    let mut prob_s1 = clip(p10 / (p01 + p10));

    for t in 0..n {
        // Predict: P(s_t=j | y_{1..t-1})
        let pred_s0 = (1.0 - prob_s1) * p00 + prob_s1 * (1.0 - p11);
        let pred_s1 = (1.0 - prob_s1) * (1.0 - p00) + prob_s1 * p11;

        // Observation likelihood in log-space
        let ll0 = log_normal_pdf(data[t], mu[0], sigma[0]);
        let ll1 = log_normal_pdf(data[t], mu[1], sigma[1]);

        // Log of joint: log P(y_t, s_t=j | y_{1..t-1})
        let log_joint_0 = ll0 + pred_s0.ln();
        let log_joint_1 = ll1 + pred_s1.ln();

        // Log-sum-exp for numerical stability
        let max_lj = log_joint_0.max(log_joint_1);
        let log_marginal = max_lj + ((-max_lj + log_joint_0).exp() + (-max_lj + log_joint_1).exp()).ln();

        // Updated (filtered) probability
        prob_s1 = clip((log_joint_1 - log_marginal).exp());
        filtered[t] = prob_s1;
    }

    filtered
}

/// Kim smoother: backward pass to get smoothed probabilities.
fn smooth(
    data: &[f64],
    mu: [f64; 2],
    sigma: [f64; 2],
    p00: f64,
    p11: f64,
) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    // Forward pass: collect filtered and predicted probabilities
    let p01 = 1.0 - p00;
    let p10 = 1.0 - p11;
    let mut filtered_s1 = vec![0.0; n];
    let mut predicted_s1 = vec![0.0; n];

    let mut prob_s1 = clip(p10 / (p01 + p10));

    for t in 0..n {
        let pred_s0 = (1.0 - prob_s1) * p00 + prob_s1 * (1.0 - p11);
        let pred_s1_t = (1.0 - prob_s1) * (1.0 - p00) + prob_s1 * p11;
        predicted_s1[t] = pred_s1_t;

        let ll0 = log_normal_pdf(data[t], mu[0], sigma[0]);
        let ll1 = log_normal_pdf(data[t], mu[1], sigma[1]);

        let log_joint_0 = ll0 + pred_s0.ln();
        let log_joint_1 = ll1 + pred_s1_t.ln();
        let max_lj = log_joint_0.max(log_joint_1);
        let log_marginal = max_lj + ((-max_lj + log_joint_0).exp() + (-max_lj + log_joint_1).exp()).ln();

        prob_s1 = clip((log_joint_1 - log_marginal).exp());
        filtered_s1[t] = prob_s1;
    }

    // Backward (Kim smoother)
    let mut smoothed = vec![0.0; n];
    smoothed[n - 1] = filtered_s1[n - 1];

    for t in (0..(n - 1)).rev() {
        let filt_s1 = filtered_s1[t];
        let pred_s0_next = 1.0 - predicted_s1[t + 1];
        let pred_s1_next = predicted_s1[t + 1];
        let smooth_s1_next = smoothed[t + 1];
        let smooth_s0_next = 1.0 - smooth_s1_next;

        // P(s_t=1 | all data) = Σ_j P(s_t=1, s_{t+1}=j | all data)
        // = Σ_j P(s_t=1 | y_{1..t}) * P(s_{t+1}=j | s_t=1) * P(s_{t+1}=j | all data) / P(s_{t+1}=j | y_{1..t})
        let s1_to_s0 = if pred_s0_next > CLIP_LO {
            filt_s1 * (1.0 - p11) * smooth_s0_next / pred_s0_next
        } else {
            0.0
        };
        let s1_to_s1 = if pred_s1_next > CLIP_LO {
            filt_s1 * p11 * smooth_s1_next / pred_s1_next
        } else {
            0.0
        };

        smoothed[t] = clip(s1_to_s0 + s1_to_s1);
    }

    smoothed
}

/// EM estimation of 2-state HMM with a single random start.
fn em_single(data: &[f64], seed: u64) -> (f64, [f64; 2], [f64; 2], f64, f64, f64) {
    let n = data.len();

    // Initialize from seed
    let mut state = seed;
    let next_rand = |s: &mut u64| -> f64 {
        // xorshift64
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s as f64) / (u64::MAX as f64)
    };

    let data_mean = data.iter().sum::<f64>() / n as f64;
    let data_std = (data.iter().map(|x| (x - data_mean).powi(2)).sum::<f64>() / n as f64).sqrt();

    // Random initialization around data statistics
    let r1 = next_rand(&mut state);
    let r2 = next_rand(&mut state);
    let mut mu = [
        data_mean - data_std * (0.5 + r1),
        data_mean + data_std * (0.5 + r2),
    ];
    let mut sigma = [
        (data_std * (0.5 + next_rand(&mut state))).max(1e-6),
        (data_std * (1.0 + next_rand(&mut state))).max(1e-6),
    ];
    let mut p00 = 0.9 + 0.09 * next_rand(&mut state);
    let mut p11 = 0.8 + 0.15 * next_rand(&mut state);

    let mut prev_ll = f64::NEG_INFINITY;

    for _ in 0..EM_MAX_ITER {
        // E-step: smoothed probabilities
        let smoothed = smooth(data, mu, sigma, p00, p11);

        // Compute log-likelihood
        let ll: f64 = data
            .iter()
            .enumerate()
            .map(|(t, &y)| {
                let p1 = smoothed[t];
                let p0 = 1.0 - p1;
                let l0 = log_normal_pdf(y, mu[0], sigma[0]);
                let l1 = log_normal_pdf(y, mu[1], sigma[1]);
                let max_l = l0.max(l1);
                max_l + (p0 * (l0 - max_l).exp() + p1 * (l1 - max_l).exp()).ln()
            })
            .sum();

        if (ll - prev_ll).abs() < EM_TOL {
            return (ll, mu, sigma, p00, p11, smoothed.last().copied().unwrap_or(f64::NAN));
        }
        prev_ll = ll;

        // M-step: update parameters from smoothed probabilities
        let w0_sum: f64 = smoothed.iter().map(|&p| 1.0 - p).sum();
        let w1_sum: f64 = smoothed.iter().sum();

        if w0_sum > 1e-10 {
            mu[0] = data
                .iter()
                .zip(smoothed.iter())
                .map(|(&y, &p)| (1.0 - p) * y)
                .sum::<f64>()
                / w0_sum;
            sigma[0] = (data
                .iter()
                .zip(smoothed.iter())
                .map(|(&y, &p)| (1.0 - p) * (y - mu[0]).powi(2))
                .sum::<f64>()
                / w0_sum)
                .sqrt()
                .max(1e-6);
        }

        if w1_sum > 1e-10 {
            mu[1] = data
                .iter()
                .zip(smoothed.iter())
                .map(|(&y, &p)| p * y)
                .sum::<f64>()
                / w1_sum;
            sigma[1] = (data
                .iter()
                .zip(smoothed.iter())
                .map(|(&y, &p)| p * (y - mu[1]).powi(2))
                .sum::<f64>()
                / w1_sum)
                .sqrt()
                .max(1e-6);
        }

        // Transition probabilities from consecutive smoothed probs
        if n > 1 {
            let mut n00 = 0.0;
            let mut n0 = 0.0;
            let mut n11 = 0.0;
            let mut n1 = 0.0;

            for t in 0..(n - 1) {
                let p0_t = 1.0 - smoothed[t];
                let p1_t = smoothed[t];
                let p0_next = 1.0 - smoothed[t + 1];
                let p1_next = smoothed[t + 1];

                n00 += p0_t * p0_next;
                n0 += p0_t;
                n11 += p1_t * p1_next;
                n1 += p1_t;
            }

            if n0 > 1e-10 {
                p00 = clip(n00 / n0);
            }
            if n1 > 1e-10 {
                p11 = clip(n11 / n1);
            }
        }
    }

    let smoothed = smooth(data, mu, sigma, p00, p11);
    (prev_ll, mu, sigma, p00, p11, smoothed.last().copied().unwrap_or(f64::NAN))
}

/// Hamilton forward filter with known parameters.
///
/// Returns P(stressed) at each time step given known model parameters.
///
/// Args:
///     data: Observed time series (e.g., returns).
///     mu: Mean for each state [normal, stressed].
///     sigma: Std dev for each state [normal, stressed].
///     p00: Probability of staying in state 0 (normal).
///     p11: Probability of staying in state 1 (stressed).
#[pyfunction]
#[pyo3(signature = (data, mu, sigma, p00, p11))]
pub fn hamilton_filter<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    mu: [f64; 2],
    sigma: [f64; 2],
    p00: f64,
    p11: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let result = forward_filter(data, mu, sigma, p00, p11);
    Ok(PyArray1::from_vec(py, result))
}

/// Hamilton filter with EM parameter estimation.
///
/// Fits a 2-state Gaussian HMM via Expectation-Maximization with
/// multiple random restarts to avoid local optima.
///
/// Returns: (mu_0, sigma_0, mu_1, sigma_1, p00, p11, filtered_probs)
///
/// Args:
///     data: Observed time series.
///     n_restarts: Number of random EM restarts (default 10).
#[pyfunction]
#[pyo3(signature = (data, n_restarts=10))]
#[allow(clippy::type_complexity)]
pub fn hamilton_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    n_restarts: usize,
) -> PyResult<(f64, f64, f64, f64, f64, f64, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let n = data.len();
    if n < 10 {
        return Ok((
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            PyArray1::from_vec(py, vec![f64::NAN; n]),
        ));
    }

    let n_restarts = n_restarts.max(1);

    // Parallel random restarts
    let results: Vec<_> = (0..n_restarts)
        .into_par_iter()
        .map(|i| {
            let seed = 12345u64.wrapping_mul(i as u64 + 1).wrapping_add(67890);
            em_single(data, seed)
        })
        .collect();

    // Pick best by log-likelihood
    let (_, mu, sigma, p00, p11, _) = results
        .into_iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // Ensure state 1 is the "stressed" state (higher volatility)
    let (mu, sigma, p00, p11) = if sigma[1] >= sigma[0] {
        (mu, sigma, p00, p11)
    } else {
        ([mu[1], mu[0]], [sigma[1], sigma[0]], p11, p00)
    };

    let filtered = forward_filter(data, mu, sigma, p00, p11);

    Ok((
        mu[0],
        sigma[0],
        mu[1],
        sigma[1],
        p00,
        p11,
        PyArray1::from_vec(py, filtered),
    ))
}

/// Hamilton smoother with known parameters.
///
/// Returns smoothed P(stressed | all data) using Kim smoother.
///
/// Args:
///     data: Observed time series.
///     mu: Mean for each state [normal, stressed].
///     sigma: Std dev for each state [normal, stressed].
///     p00: Probability of staying in state 0.
///     p11: Probability of staying in state 1.
#[pyfunction]
#[pyo3(signature = (data, mu, sigma, p00, p11))]
pub fn hamilton_smooth<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    mu: [f64; 2],
    sigma: [f64; 2],
    p00: f64,
    p11: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let result = smooth(data, mu, sigma, p00, p11);
    Ok(PyArray1::from_vec(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_regime_data(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let next_rand = |s: &mut u64| -> f64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s as f64) / (u64::MAX as f64)
        };
        let next_normal = |s: &mut u64| -> f64 {
            // Box-Muller
            let u1 = next_rand(s).max(1e-15);
            let u2 = next_rand(s);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let mu = [0.0005, -0.002];
        let sigma = [0.01, 0.03];
        let p00 = 0.98;
        let p11 = 0.95;

        let mut regime = 0;
        let mut data = Vec::with_capacity(n);

        for _ in 0..n {
            let r = next_rand(&mut state);
            if regime == 0 {
                if r > p00 {
                    regime = 1;
                }
            } else if r > p11 {
                regime = 0;
            }
            data.push(mu[regime] + sigma[regime] * next_normal(&mut state));
        }
        data
    }

    #[test]
    fn test_forward_filter_known_params() {
        let data = generate_regime_data(500, 42);
        let filtered = forward_filter(&data, [0.0005, -0.002], [0.01, 0.03], 0.98, 0.95);

        assert_eq!(filtered.len(), 500);
        for &p in &filtered {
            assert!(p >= 0.0 && p <= 1.0, "Probability out of range: {}", p);
        }
    }

    #[test]
    fn test_smoother_bounded() {
        let data = generate_regime_data(500, 42);
        let smoothed = smooth(&data, [0.0005, -0.002], [0.01, 0.03], 0.98, 0.95);

        assert_eq!(smoothed.len(), 500);
        for &p in &smoothed {
            assert!(p >= 0.0 && p <= 1.0, "Smoothed probability out of range: {}", p);
        }
    }

    #[test]
    fn test_em_converges() {
        let data = generate_regime_data(1000, 123);
        let (ll, mu, sigma, p00, p11, _) = em_single(&data, 42);

        assert!(ll.is_finite(), "Log-likelihood should be finite: {}", ll);
        // Sigma should be ordered (state 0 = low vol, state 1 = high vol or vice versa)
        assert!(sigma[0] > 0.0 && sigma[1] > 0.0);
        assert!(p00 > 0.5 && p11 > 0.5, "Transition probs should show persistence");
        assert!(
            sigma[0] != sigma[1] || mu[0] != mu[1],
            "States should differ"
        );
    }

    #[test]
    fn test_em_empty_data() {
        let data: Vec<f64> = vec![];
        let (ll, _, _, _, _, _) = em_single(&data, 42);
        // Should not panic, ll may be -inf or NaN
        let _ = ll;
    }

    #[test]
    fn test_label_swap() {
        // After EM, state 1 should have higher sigma (stressed)
        let data = generate_regime_data(2000, 999);

        let results: Vec<_> = (0..5)
            .map(|i| em_single(&data, i * 1000 + 1))
            .collect();

        let (_, mu, sigma, p00, p11, _) = results
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let (_, sigma, _, _) = if sigma[1] >= sigma[0] {
            (mu, sigma, p00, p11)
        } else {
            ([mu[1], mu[0]], [sigma[1], sigma[0]], p11, p00)
        };

        assert!(
            sigma[1] > sigma[0],
            "State 1 (stressed) should have higher vol: sigma={:?}",
            sigma
        );
    }
}
