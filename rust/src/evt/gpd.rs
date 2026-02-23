use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Generalized Pareto Distribution.
/// F(x) = 1 - (1 + xi*x/sigma)^(-1/xi)  for xi != 0
/// F(x) = 1 - exp(-x/sigma)               for xi = 0
///
/// Used for modeling exceedances over a threshold.

/// GPD log-likelihood for a single exceedance.
fn gpd_log_pdf(x: f64, sigma: f64, xi: f64) -> f64 {
    if sigma <= 0.0 || x < 0.0 {
        return f64::NEG_INFINITY;
    }
    if xi.abs() < 1e-8 {
        -sigma.ln() - x / sigma
    } else {
        let t = 1.0 + xi * x / sigma;
        if t <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -sigma.ln() - (1.0 + 1.0 / xi) * t.ln()
    }
}

/// Grimshaw's method for GPD parameter estimation.
/// More robust than direct MLE for heavy tails.
fn gpd_grimshaw(exceedances: &[f64]) -> (f64, f64) {
    let n = exceedances.len() as f64;
    let mean = exceedances.iter().sum::<f64>() / n;
    let var = exceedances.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    // Method of moments initial estimate
    let xi_mom = 0.5 * (mean * mean / var - 1.0);
    let sigma_mom = mean * (1.0 - xi_mom);

    // Refine with MLE via coordinate descent
    let mut sigma = if sigma_mom > 0.0 { sigma_mom } else { mean };
    let mut xi = xi_mom.max(-0.49).min(2.0);

    let neg_ll = |sigma: f64, xi: f64| -> f64 {
        -exceedances
            .iter()
            .map(|&x| gpd_log_pdf(x, sigma, xi))
            .sum::<f64>()
    };

    let step_sizes = [0.5, 0.1, 0.01, 0.001];
    for &step in &step_sizes {
        for _ in 0..100 {
            let current = neg_ll(sigma, xi);

            for ds in [-step, step] {
                let new_sigma = sigma + ds;
                if new_sigma > 0.01 && neg_ll(new_sigma, xi) < current {
                    sigma = new_sigma;
                }
            }

            for dx in [-step * 0.1, step * 0.1] {
                let new_xi = xi + dx;
                if new_xi > -0.49 && new_xi < 2.0 && neg_ll(sigma, new_xi) < neg_ll(sigma, xi) {
                    xi = new_xi;
                }
            }
        }
    }

    (sigma, xi)
}

/// Select threshold as the `quantile`-th quantile (e.g. 0.9 = 90th percentile).
fn select_threshold(data: &[f64], quantile: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64 - 1.0) * quantile) as usize;
    sorted[idx]
}

/// Fit GPD to exceedances over threshold.
/// Returns (sigma, xi, threshold, n_exceedances).
#[pyfunction]
#[pyo3(signature = (data, quantile=0.95))]
pub fn gpd_fit(
    data: PyReadonlyArray1<'_, f64>,
    quantile: Option<f64>,
) -> PyResult<(f64, f64, f64, usize)> {
    let data = data.as_slice()?;
    let quantile = quantile.unwrap_or(0.95);

    // Use absolute values of negative returns (losses)
    let losses: Vec<f64> = data.iter().map(|x| -x).filter(|x| *x > 0.0).collect();

    let threshold = select_threshold(&losses, quantile);
    let exceedances: Vec<f64> = losses
        .iter()
        .filter(|&&x| x > threshold)
        .map(|&x| x - threshold)
        .collect();

    if exceedances.len() < 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Too few exceedances for GPD fitting",
        ));
    }

    let (sigma, xi) = gpd_grimshaw(&exceedances);
    Ok((sigma, xi, threshold, exceedances.len()))
}

/// Compute VaR and Expected Shortfall from fitted GPD.
/// `p` is the probability level (e.g. 0.99 for 99% VaR).
/// Returns (VaR, ES).
#[pyfunction]
#[pyo3(signature = (data, p=0.99, quantile=0.95))]
pub fn gpd_var_es(
    data: PyReadonlyArray1<'_, f64>,
    p: Option<f64>,
    quantile: Option<f64>,
) -> PyResult<(f64, f64)> {
    let data = data.as_slice()?;
    let p = p.unwrap_or(0.99);
    let quantile = quantile.unwrap_or(0.95);

    let losses: Vec<f64> = data.iter().map(|x| -x).filter(|x| *x > 0.0).collect();
    let n = losses.len() as f64;

    let threshold = select_threshold(&losses, quantile);
    let exceedances: Vec<f64> = losses
        .iter()
        .filter(|&&x| x > threshold)
        .map(|&x| x - threshold)
        .collect();
    let n_exc = exceedances.len() as f64;

    if exceedances.len() < 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Too few exceedances",
        ));
    }

    let (sigma, xi) = gpd_grimshaw(&exceedances);
    let rate = n_exc / n;

    // VaR
    let var = if xi.abs() < 1e-8 {
        threshold - sigma * (rate / (1.0 - p)).ln()
    } else {
        threshold + sigma / xi * (((1.0 - p) / rate).powf(-xi) - 1.0)
    };

    // Expected Shortfall
    let es = if xi.abs() < 1e-8 {
        var + sigma
    } else if xi < 1.0 {
        (var + sigma - xi * threshold) / (1.0 - xi)
    } else {
        f64::INFINITY
    };

    Ok((var, es))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpd_exponential() {
        // Exponential distribution is GPD with xi=0
        use rand::prelude::*;
        use rand_distr::Exp;

        let mut rng = StdRng::seed_from_u64(42);
        let rate = 1.0;
        let dist = Exp::new(rate).unwrap();
        let n = 5000;

        let samples: Vec<f64> = (0..n).map(|_| rng.sample(dist)).collect();
        let threshold = select_threshold(&samples, 0.9);
        let exceedances: Vec<f64> = samples
            .iter()
            .filter(|&&x| x > threshold)
            .map(|&x| x - threshold)
            .collect();

        let (sigma, xi) = gpd_grimshaw(&exceedances);

        // For Exp(1), exceedances are also Exp(1), so sigma≈1, xi≈0
        assert!(xi.abs() < 0.2, "xi should be near 0: {}", xi);
        assert!(
            (sigma - 1.0 / rate).abs() < 0.3,
            "sigma: {} vs {}",
            sigma,
            1.0 / rate
        );
    }
}
