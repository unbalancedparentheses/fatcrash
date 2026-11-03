//! CUSUM volatility-shift monitor.
//!
//! Standardizes log squared returns of a recent window by the mean and
//! standard deviation of log squared returns in an earlier baseline window,
//! then runs a one-sided Page CUSUM:
//!
//! ```text
//! u_t = (log(r_t^2 + c) - mean_base) / sd_base
//! S_t = max(0, S_{t-1} + u_t - k),   statistic = max_t S_t
//! ```
//!
//! Unlike a ratio of two averages, it accumulates evidence as days pass, so it
//! reacts to a sustained rise in volatility and ignores isolated spikes. Log
//! squared returns keep single extreme days from dominating. In the spirit of
//! sequential monitoring for volatility regimes (Horváth, Trapani & Wang,
//! arXiv:2404.17885), but without a GARCH fit.
//!
//! In the crash-window backtest (`analysis/ensemble_variants_check.py`) it is
//! the strongest single volatility signal (AUC 0.61 / 0.65 / 0.69 at 30 / 60 /
//! 120 days). Replacing the RV-spike signal with it raised ensemble AUC by
//! +0.039 at 30d and +0.022 at 120d (quarter-clustered CIs exclude zero).

const LOG_EPS: f64 = 1e-12;

/// Maximum of the Page CUSUM of standardized log squared returns of `recent`
/// against `baseline`. `allowance` is the drift k (0.5 by default). Returns NaN
/// if either window has fewer than 10 returns or the baseline is degenerate.
pub fn vol_cusum(recent: &[f64], baseline: &[f64], allowance: f64) -> f64 {
    let finite = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .filter(|x| x.is_finite())
            .map(|x| (x * x + LOG_EPS).ln())
            .collect()
    };
    let (lr, lb) = (finite(recent), finite(baseline));
    if lr.len() < 10 || lb.len() < 10 {
        return f64::NAN;
    }
    let n = lb.len() as f64;
    let mean = lb.iter().sum::<f64>() / n;
    let sd = (lb.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    if sd <= 0.0 || !sd.is_finite() {
        return f64::NAN;
    }
    let mut s = 0.0f64;
    let mut s_max = 0.0f64;
    for x in lr {
        s = (s + (x - mean) / sd - allowance).max(0.0);
        s_max = s_max.max(s);
    }
    s_max
}

/// Map the CUSUM statistic to [0, 1]. The threshold grows with the recent
/// window length so long windows do not saturate: `s_max / (2 + 0.2 n)`.
pub fn vol_cusum_signal(s_max: f64, n_recent: usize) -> f64 {
    if !s_max.is_finite() {
        return 0.0;
    }
    (s_max / (2.0 + 0.2 * n_recent as f64)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn normal(rng: &mut StdRng, n: usize, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|_| scale * rng.sample::<f64, _>(StandardNormal))
            .collect()
    }

    #[test]
    fn rises_with_volatility_shift() {
        let mut rng = StdRng::seed_from_u64(1);
        let base = normal(&mut rng, 120, 0.01);
        let calm = normal(&mut rng, 120, 0.01);
        let stressed = normal(&mut rng, 120, 0.02);
        let s_calm = vol_cusum(&calm, &base, 0.5);
        let s_stress = vol_cusum(&stressed, &base, 0.5);
        assert!(
            s_stress > 3.0 * s_calm.max(1.0),
            "calm={s_calm} stress={s_stress}"
        );
        assert!(vol_cusum_signal(s_stress, 120) > vol_cusum_signal(s_calm, 120));
    }

    #[test]
    fn short_inputs_are_nan() {
        assert!(vol_cusum(&[0.01; 5], &[0.01; 50], 0.5).is_nan());
        assert_eq!(vol_cusum_signal(f64::NAN, 30), 0.0);
    }
}
