use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// GEV distribution: F(x) = exp(-(1 + xi*(x-mu)/sigma)^(-1/xi))
/// Parameters: mu (location), sigma (scale), xi (shape)
///
/// xi > 0: Frechet (fat tail)
/// xi = 0: Gumbel (exponential tail)
/// xi < 0: Weibull (bounded tail)

/// GEV log-likelihood for a single observation.
fn gev_log_pdf(x: f64, mu: f64, sigma: f64, xi: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (x - mu) / sigma;

    if xi.abs() < 1e-8 {
        // Gumbel case: f(x) = (1/sigma) * exp(-(z + exp(-z)))
        -sigma.ln() - z - (-z).exp()
    } else {
        let t = 1.0 + xi * z;
        if t <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let inv_xi = 1.0 / xi;
        -sigma.ln() - (1.0 + inv_xi) * t.ln() - t.powf(-inv_xi)
    }
}

/// GEV MLE fitting via coordinate descent with adaptive step sizes.
/// Returns (mu, sigma, xi).
fn gev_fit_mle(data: &[f64]) -> (f64, f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();

    // Initial guesses (method of moments for Gumbel)
    let euler = 0.5772156649;
    let mut sigma = std * (6.0_f64).sqrt() / std::f64::consts::PI;
    let mut mu = mean - euler * sigma;
    let mut xi = 0.0_f64;

    let neg_ll = |mu: f64, sigma: f64, xi: f64| -> f64 {
        let ll: f64 = data.iter().map(|&x| gev_log_pdf(x, mu, sigma, xi)).sum();
        if ll.is_nan() || ll.is_infinite() {
            f64::INFINITY
        } else {
            -ll
        }
    };

    // Adaptive coordinate descent: step sizes proportional to parameter scale
    let base_steps = [0.5, 0.2, 0.05, 0.01, 0.002];
    for &base in &base_steps {
        let mu_step = base * std;
        let sigma_step = base * sigma.max(0.1);
        let xi_step = base * 0.5;

        for _ in 0..200 {
            let mut improved = false;

            // mu
            for &delta in &[-mu_step, mu_step] {
                if neg_ll(mu + delta, sigma, xi) < neg_ll(mu, sigma, xi) {
                    mu += delta;
                    improved = true;
                }
            }
            // sigma
            for &delta in &[-sigma_step, sigma_step] {
                let new_sigma = sigma + delta;
                if new_sigma > 1e-6 && neg_ll(mu, new_sigma, xi) < neg_ll(mu, sigma, xi) {
                    sigma = new_sigma;
                    improved = true;
                }
            }
            // xi
            for &delta in &[-xi_step, xi_step] {
                let new_xi = xi + delta;
                if new_xi > -0.5
                    && new_xi < 1.0
                    && neg_ll(mu, sigma, new_xi) < neg_ll(mu, sigma, xi)
                {
                    xi = new_xi;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }
    }

    (mu, sigma, xi)
}

/// Fit GEV distribution to block maxima.
/// Returns (mu, sigma, xi) — location, scale, shape.
#[pyfunction]
pub fn gev_fit(data: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64, f64)> {
    let data = data.as_slice()?;
    if data.len() < 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 10 observations for GEV fitting",
        ));
    }
    Ok(gev_fit_mle(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gev_gumbel() {
        // Generate Gumbel (GEV with xi=0) samples
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let mu_true = 1.0;
        let sigma_true = 2.0;
        let n = 5000;

        // Gumbel quantile: Q(p) = mu - sigma * ln(-ln(p))
        let samples: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                mu_true - sigma_true * (-u.ln()).ln()
            })
            .collect();

        let (mu, sigma, xi) = gev_fit_mle(&samples);

        assert!((mu - mu_true).abs() < 0.3, "mu: {} vs {}", mu, mu_true);
        assert!(
            (sigma - sigma_true).abs() < 0.3,
            "sigma: {} vs {}",
            sigma,
            sigma_true
        );
        assert!(xi.abs() < 0.15, "xi should be near 0 for Gumbel: {}", xi);
    }
}
