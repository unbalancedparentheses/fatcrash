use nalgebra::{DMatrix, DVector};

/// LPPLS equation:
///   ln(p(t)) = A + B*(tc - t)^m + C*(tc - t)^m * cos(w*ln(tc - t) + phi)
///
/// Decompose C*cos(...+phi) = C1*cos(...) + C2*sin(...) to make it linear in (A, B, C1, C2).
///
/// Nonlinear params: (tc, m, omega)
/// Linear params solved via OLS: (A, B, C1, C2)

#[derive(Debug, Clone)]
pub struct LpplsParams {
    pub tc: f64,
    pub m: f64,
    pub omega: f64,
    pub a: f64,
    pub b: f64,
    pub c1: f64,
    pub c2: f64,
}

impl LpplsParams {
    /// Amplitude of oscillation
    pub fn c_amplitude(&self) -> f64 {
        (self.c1 * self.c1 + self.c2 * self.c2).sqrt()
    }

    /// Phase of oscillation
    pub fn phi(&self) -> f64 {
        self.c2.atan2(self.c1)
    }
}

/// Compute LPPLS value at time t given parameters.
pub fn lppls_value(t: f64, p: &LpplsParams) -> f64 {
    let dt = p.tc - t;
    if dt <= 0.0 {
        return f64::NAN;
    }
    let dt_m = dt.powf(p.m);
    let log_dt = dt.ln();
    p.a + p.b * dt_m + dt_m * (p.c1 * (p.omega * log_dt).cos() + p.c2 * (p.omega * log_dt).sin())
}

/// Given nonlinear params (tc, m, omega) and time series data,
/// solve for linear params (A, B, C1, C2) via OLS.
/// Returns (A, B, C1, C2, residual_sum_of_squares).
pub fn solve_linear(
    times: &[f64],
    log_prices: &[f64],
    tc: f64,
    m: f64,
    omega: f64,
) -> Option<(f64, f64, f64, f64, f64)> {
    let n = times.len();
    if n < 5 {
        return None;
    }

    // Build design matrix X = [1, f, g, h] where:
    //   f = (tc-t)^m
    //   g = (tc-t)^m * cos(w*ln(tc-t))
    //   h = (tc-t)^m * sin(w*ln(tc-t))
    let mut x_data = Vec::with_capacity(n * 4);
    let mut y_data = Vec::with_capacity(n);

    for i in 0..n {
        let dt = tc - times[i];
        if dt <= 0.0 {
            return None;
        }
        let dt_m = dt.powf(m);
        let log_dt = dt.ln();

        x_data.push(1.0);
        x_data.push(dt_m);
        x_data.push(dt_m * (omega * log_dt).cos());
        x_data.push(dt_m * (omega * log_dt).sin());

        y_data.push(log_prices[i]);
    }

    let x = DMatrix::from_row_slice(n, 4, &x_data);
    let y = DVector::from_vec(y_data);

    // OLS: beta = (X'X)^(-1) X'y
    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;

    let beta = xtx.try_inverse()?.clone() * xty;

    let a = beta[0];
    let b = beta[1];
    let c1 = beta[2];
    let c2 = beta[3];

    // Compute RSS
    let predicted = x * beta;
    let residuals = &y - &predicted;
    let rss = residuals.dot(&residuals);

    Some((a, b, c1, c2, rss))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lppls_roundtrip() {
        let params = LpplsParams {
            tc: 100.0,
            m: 0.5,
            omega: 6.0,
            a: 10.0,
            b: -0.5,
            c1: 0.01,
            c2: 0.01,
        };

        // Generate synthetic LPPLS data
        let times: Vec<f64> = (0..80).map(|t| t as f64).collect();
        let log_prices: Vec<f64> = times.iter().map(|&t| lppls_value(t, &params)).collect();

        // Recover linear params given true nonlinear params
        let result = solve_linear(&times, &log_prices, params.tc, params.m, params.omega);
        assert!(result.is_some());

        let (a, b, c1, c2, rss) = result.unwrap();
        assert!((a - params.a).abs() < 1e-6, "a: {}", a);
        assert!((b - params.b).abs() < 1e-6, "b: {}", b);
        assert!(rss < 1e-10, "rss should be near zero: {}", rss);
    }
}
