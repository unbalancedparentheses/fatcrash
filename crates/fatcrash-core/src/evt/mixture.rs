//! Threshold-free lognormal–GPD mixture (Bee & Santi, 2026).
//!
//! ```text
//! f(x) = p * LN(x; mu, s) + (1 - p) * GPD(x; beta, xi),   x > 0, xi > 0
//! ```
//!
//! The GPD component has location zero, so the model needs no threshold.
//! The lognormal body is in the Gumbel domain and the GPD (xi > 0) is in the
//! Fréchet domain, so the mixture's tail is the GPD's, and xi is the tail
//! index of the whole loss distribution.
//!
//! This is the univariate model. The multivariate paper builds on it by
//! joining GPD marginals with a Gumbel copula. Fitted by EM: the E-step gives
//! posterior body probabilities, the M-step uses closed-form weighted
//! lognormal MLEs and a weighted GPD MLE. Initialization follows the paper:
//! full-sample lognormal and GPD MLEs, p = 0.5. EM finds a local maximum, so
//! compare log-likelihoods across starts when the fit matters.

use crate::evt::gpd::{gpd_log_pdf, gpd_mle_weighted};
use crate::special::norm_cdf;

const XI_BOUNDS: (f64, f64) = (1e-4, 2.0);
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// Fitted mixture parameters.
#[derive(Clone, Debug)]
pub struct MixtureFit {
    /// Weight of the lognormal body.
    pub p: f64,
    /// Lognormal location (mean of log x).
    pub mu: f64,
    /// Lognormal scale (sd of log x).
    pub s: f64,
    /// GPD shape (tail index of the mixture).
    pub xi: f64,
    /// GPD scale.
    pub beta: f64,
    pub loglik: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn lognormal_log_pdf(x: f64, mu: f64, s: f64) -> f64 {
    let z = (x.ln() - mu) / s;
    -x.ln() - s.ln() - LN_SQRT_2PI - 0.5 * z * z
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        m
    } else {
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
}

/// Fit the mixture to positive losses.
pub fn fit_lognormal_gpd(losses: &[f64], max_iter: usize, tol: f64) -> Result<MixtureFit, String> {
    let x: Vec<f64> = losses
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    let n = x.len();
    if n < 30 {
        return Err("need at least 30 positive losses".into());
    }
    let nf = n as f64;
    let logs: Vec<f64> = x.iter().map(|v| v.ln()).collect();
    let mut mu = logs.iter().sum::<f64>() / nf;
    let mut s = (logs.iter().map(|l| (l - mu).powi(2)).sum::<f64>() / nf)
        .sqrt()
        .max(1e-6);
    let ones = vec![1.0; n];
    let (mut beta, mut xi) = gpd_mle_weighted(&x, &ones, XI_BOUNDS, None);
    let mut p: f64 = 0.5;

    let mut tau = vec![0.0; n];
    let mut prev_ll = f64::NEG_INFINITY;
    let mut loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    for it in 1..=max_iter {
        iterations = it;
        // E-step.
        loglik = 0.0;
        for i in 0..n {
            let a = p.ln() + lognormal_log_pdf(x[i], mu, s);
            let b = (1.0 - p).ln() + gpd_log_pdf(x[i], beta, xi);
            let total = log_add_exp(a, b);
            tau[i] = (a - total).exp();
            loglik += total;
        }
        if (loglik - prev_ll).abs() <= tol * (1.0 + loglik.abs()) {
            converged = true;
            break;
        }
        prev_ll = loglik;

        // M-step.
        let w_body: f64 = tau.iter().sum();
        p = (w_body / nf).clamp(1e-6, 1.0 - 1e-6);
        mu = tau.iter().zip(&logs).map(|(t, l)| t * l).sum::<f64>() / w_body;
        s = (tau
            .iter()
            .zip(&logs)
            .map(|(t, l)| t * (l - mu).powi(2))
            .sum::<f64>()
            / w_body)
            .sqrt()
            .max(1e-6);
        let w_tail: Vec<f64> = tau.iter().map(|t| 1.0 - t).collect();
        (beta, xi) = gpd_mle_weighted(&x, &w_tail, XI_BOUNDS, Some((beta, xi)));
    }

    Ok(MixtureFit {
        p,
        mu,
        s,
        xi,
        beta,
        loglik,
        iterations,
        converged,
    })
}

impl MixtureFit {
    /// Mixture CDF.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let f1 = norm_cdf((x.ln() - self.mu) / self.s);
        let f2 = 1.0 - (1.0 + self.xi * x / self.beta).powf(-1.0 / self.xi);
        self.p * f1 + (1.0 - self.p) * f2
    }

    /// VaR at confidence level `q` (e.g. 0.99), by bisection on the CDF.
    pub fn var(&self, q: f64) -> f64 {
        if !(q > 0.0 && q < 1.0) {
            return f64::NAN;
        }
        let mut hi = (self.mu + 3.0 * self.s).exp().max(self.beta);
        while self.cdf(hi) < q {
            hi *= 2.0;
            if !hi.is_finite() {
                return f64::INFINITY;
            }
        }
        let mut lo = 0.0;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if self.cdf(mid) < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Expected shortfall at level `q`: `VaR + E[(X - VaR)+] / (1 - q)`, with
    /// closed-form stop-loss transforms for both components. Infinite for xi >= 1.
    pub fn es(&self, q: f64) -> f64 {
        if self.xi >= 1.0 {
            return f64::INFINITY;
        }
        let v = self.var(q);
        let d2 = (self.mu - v.ln()) / self.s;
        let d1 = d2 + self.s;
        let excess_body = (self.mu + 0.5 * self.s * self.s).exp() * norm_cdf(d1) - v * norm_cdf(d2);
        let surv_tail = (1.0 + self.xi * v / self.beta).powf(-1.0 / self.xi);
        let excess_tail = (self.beta + self.xi * v) / (1.0 - self.xi) * surv_tail;
        v + (self.p * excess_body + (1.0 - self.p) * excess_tail) / (1.0 - q)
    }
}

/// Fit the mixture to the losses (negated negative returns) of a return series,
/// the same convention as `gpd_fit_slice`.
pub fn lognormal_gpd_mixture_slice(
    returns: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<MixtureFit, String> {
    let losses: Vec<f64> = returns.iter().map(|r| -r).collect();
    fit_lognormal_gpd(&losses, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::{Distribution, LogNormal};

    fn sample(rng: &mut StdRng, n: usize, p: f64, mu: f64, s: f64, xi: f64, beta: f64) -> Vec<f64> {
        let ln = LogNormal::new(mu, s).unwrap();
        (0..n)
            .map(|_| {
                if rng.gen::<f64>() < p {
                    ln.sample(rng)
                } else {
                    let u: f64 = rng.gen();
                    beta / xi * (u.powf(-xi) - 1.0)
                }
            })
            .collect()
    }

    #[test]
    fn recovers_tail_index() {
        let mut rng = StdRng::seed_from_u64(21);
        let x = sample(&mut rng, 8000, 0.8, 0.0, 0.5, 0.4, 1.0);
        let fit = fit_lognormal_gpd(&x, 2000, 1e-9).unwrap();
        assert!((fit.xi - 0.4).abs() < 0.1, "xi={}", fit.xi);
        assert!((fit.p - 0.8).abs() < 0.1, "p={}", fit.p);
    }

    #[test]
    fn var_es_consistent_with_empirical() {
        let mut rng = StdRng::seed_from_u64(22);
        let mut x = sample(&mut rng, 20_000, 0.8, 0.0, 0.5, 0.3, 1.0);
        let fit = fit_lognormal_gpd(&x, 2000, 1e-9).unwrap();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let emp = x[(0.99 * x.len() as f64) as usize];
        let v = fit.var(0.99);
        assert!((v / emp - 1.0).abs() < 0.15, "var={v} emp={emp}");
        assert!((fit.cdf(v) - 0.99).abs() < 1e-9);
        assert!(fit.es(0.99) > v);
    }
}
