/// Detrended Fluctuation Analysis (DFA).
///
/// Alternative to Hurst R/S that handles non-stationarity better.
/// Algorithm: divide into windows, fit linear trend per window,
/// compute RMS of residuals, regress log(RMS) vs log(window_size).
///
/// Returns alpha exponent:
/// - 0.5 = white noise (uncorrelated)
/// - above 0.5 = persistent (long memory)
/// - below 0.5 = anti-persistent
pub fn compute_dfa(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 32 {
        return f64::NAN;
    }

    // Generate window sizes from 4 up to n/4 (factor 1.5 spacing)
    let mut sizes: Vec<usize> = Vec::new();
    let mut s = 4;
    while s <= n / 4 {
        sizes.push(s);
        s = (s as f64 * 1.5).ceil() as usize;
    }
    sizes.dedup();

    if sizes.len() < 3 {
        return f64::NAN;
    }

    // Segments are taken from both ends of the profile (Kantelhardt et al. 2002),
    // so the n % size points at the end are not dropped.
    mfdfa::dfa_hurst(data, &sizes, 1).unwrap_or(f64::NAN)
}

/// Multifractal DFA-1 spectrum: h(q), tau(q), alpha and f(alpha) over q in [-5, 5].
///
/// `spectrum.width()` near 0 means monofractal; a wide spectrum means the
/// small and large fluctuations scale differently (intermittent, crash-prone).
/// Returns `None` when the series is too short or degenerate.
pub fn compute_mfdfa(data: &[f64]) -> Option<mfdfa::MfdfaSpectrum> {
    let scales = mfdfa::default_scales(data.len(), 1);
    mfdfa::mfdfa(data, &mfdfa::default_q(), &scales, 1).ok()
}

/// Compute DFA exponent for the given data.
/// Returns alpha: 0.5 = white noise, >0.5 persistent, <0.5 anti-persistent.
pub fn dfa_exponent_slice(data: &[f64]) -> f64 {
    compute_dfa(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfa_white_noise() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 4096;
        let data: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        let alpha = compute_dfa(&data);
        assert!(
            alpha.is_finite(),
            "DFA alpha should be finite, got {}",
            alpha
        );
        assert!(
            (alpha - 0.5).abs() < 0.15,
            "DFA alpha for white noise should be near 0.5, got {}",
            alpha
        );
    }

    #[test]
    fn test_dfa_persistent() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(99);
        let n = 4096;
        let increments: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        // Cumulative sum → persistent / trending series
        let cumulative: Vec<f64> = increments
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let alpha = compute_dfa(&cumulative);
        assert!(
            alpha.is_finite(),
            "DFA alpha should be finite, got {}",
            alpha
        );
        assert!(
            alpha > 0.8,
            "DFA alpha for random walk (integrated noise) should be > 0.8, got {}",
            alpha
        );
    }

    #[test]
    fn test_dfa_insufficient_data() {
        let data = vec![1.0; 10];
        let alpha = compute_dfa(&data);
        assert!(alpha.is_nan(), "Should be NaN with only 10 points");
    }

    #[test]
    fn test_mfdfa_spectrum_width() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = StdRng::seed_from_u64(7);
        let n = 4096;
        let noise: Vec<f64> = (0..n)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        // Binomial multiplicative cascade (p = 0.3): a textbook multifractal.
        let levels = 12;
        let mut cascade = vec![1.0f64];
        for _ in 0..levels {
            cascade = cascade.iter().flat_map(|&m| [m * 0.3, m * 0.7]).collect();
        }

        let mono = compute_mfdfa(&noise).expect("noise spectrum");
        let multi = compute_mfdfa(&cascade).expect("cascade spectrum");
        assert!((mono.hurst() - 0.5).abs() < 0.1, "h(2) = {}", mono.hurst());
        assert!(mono.width() < 0.4, "noise width = {}", mono.width());
        assert!(
            multi.width() > 2.0 * mono.width(),
            "cascade width {} vs noise {}",
            multi.width(),
            mono.width()
        );
    }

    #[test]
    fn test_dfa_uses_tail_points() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        // 4096 + 7 points: the last 7 fall outside every forward-only window.
        let mut rng = StdRng::seed_from_u64(3);
        let mut data: Vec<f64> = (0..4096)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        let base = compute_dfa(&data);
        data.extend([50.0, -50.0, 50.0, -50.0, 50.0, -50.0, 50.0]);
        assert_ne!(compute_dfa(&data), base);
    }
}
