//! Special functions used by the estimators: Lambert W, log-gamma,
//! the standard normal CDF, and the binomial CDF.

/// Euler–Mascheroni constant.
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Principal branch W0 of the Lambert W function for `x >= -1/e`.
///
/// Solves `w * exp(w) = x` with Halley iteration. Returns NaN below the
/// branch point.
pub fn lambert_w0(x: f64) -> f64 {
    let branch = -(-1.0f64).exp();
    if x.is_nan() || x < branch {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if (x - branch).abs() < 1e-15 {
        return -1.0;
    }
    // Initial guess: branch-point series near -1/e, log asymptote for large x.
    let mut w = if x < -0.25 {
        let p = (2.0 * (std::f64::consts::E * x + 1.0)).sqrt();
        -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
    } else if x < 3.0 {
        x / (1.0 + x)
    } else {
        let l = x.ln();
        l - l.ln()
    };
    for _ in 0..64 {
        let ew = w.exp();
        let f = w * ew - x;
        let wp1 = w + 1.0;
        if wp1.abs() < 1e-300 {
            break;
        }
        let step = f / (ew * wp1 - (w + 2.0) * f / (2.0 * wp1));
        w -= step;
        if step.abs() <= 1e-15 * (1.0 + w.abs()) {
            break;
        }
    }
    w
}

/// Natural log of the gamma function for `x > 0` (Lanczos, g = 7).
pub fn ln_gamma(x: f64) -> f64 {
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x <= 0.0 {
        return f64::NAN;
    }
    if x < 0.5 {
        // Reflection formula.
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEF[0];
    let t = x + 7.5;
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Complementary error function (Numerical Recipes `erfcc`, |error| < 1.2e-7).
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        r
    } else {
        2.0 - r
    }
}

/// Standard normal CDF.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// `P{Bin(n, p) < j}` (strictly less than `j`).
pub fn binom_cdf_lt(n: usize, p: f64, j: usize) -> f64 {
    if j == 0 {
        return 0.0;
    }
    if j > n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let nf = n as f64;
    let ln_p = p.ln();
    let ln_q = (1.0 - p).ln();
    let ln_n_fact = ln_gamma(nf + 1.0);
    (0..j)
        .map(|i| {
            let i = i as f64;
            (ln_n_fact - ln_gamma(i + 1.0) - ln_gamma(nf - i + 1.0) + i * ln_p + (nf - i) * ln_q)
                .exp()
        })
        .sum::<f64>()
        .min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambert_w0_inverts() {
        for &x in &[-0.367, -0.3, -0.1, -1e-6, 0.5, 1.0, 10.0, 1e4] {
            let w = lambert_w0(x);
            assert!(
                (w * w.exp() - x).abs() < 1e-10 * (1.0 + x.abs()),
                "x={x} w={w}"
            );
        }
        assert!(lambert_w0(-0.5).is_nan());
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        assert!((ln_gamma(1.0)).abs() < 1e-12);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-12);
    }

    #[test]
    fn norm_cdf_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((norm_cdf(1.959_964) - 0.975).abs() < 1e-6);
    }

    #[test]
    fn binom_cdf_sums_to_one() {
        assert!((binom_cdf_lt(10, 0.3, 11) - 1.0).abs() < 1e-12);
        // P(Bin(10, .5) < 5) = 386/1024
        assert!((binom_cdf_lt(10, 0.5, 5) - 386.0 / 1024.0).abs() < 1e-10);
    }
}
