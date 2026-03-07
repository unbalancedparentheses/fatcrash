/// Price velocity: fractional change in realized volatility.
///
/// Compares current realized vol (last `vol_window` returns) to past realized vol
/// (`lag` periods earlier). Positive values indicate accelerating volatility.
///
/// Args:
///     returns: Log return series.
///     vol_window: Window for realized volatility computation.
///     lag: How many periods back to compare.
///
/// Returns: (current_vol - past_vol) / past_vol
pub fn compute_velocity(returns: &[f64], vol_window: usize, lag: usize) -> f64 {
    let n = returns.len();
    if n < vol_window + lag || vol_window < 2 || lag == 0 {
        return f64::NAN;
    }
    let current_slice = &returns[(n - vol_window)..];
    let current_vol =
        (current_slice.iter().map(|r| r * r).sum::<f64>() / vol_window as f64).sqrt();

    let past_end = n - lag;
    if past_end < vol_window {
        return f64::NAN;
    }
    let past_slice = &returns[(past_end - vol_window)..past_end];
    let past_vol = (past_slice.iter().map(|r| r * r).sum::<f64>() / vol_window as f64).sqrt();

    if past_vol < 1e-15 {
        return f64::NAN;
    }
    (current_vol - past_vol) / past_vol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_increasing_vol() {
        // First half calm, second half volatile
        let mut returns = vec![0.001; 50];
        returns.extend(vec![0.05; 50]);
        let v = compute_velocity(&returns, 20, 60);
        assert!(v > 0.0, "Velocity should be positive when vol increases, got {}", v);
    }

    #[test]
    fn test_velocity_insufficient_data() {
        let returns = vec![0.01; 5];
        assert!(compute_velocity(&returns, 10, 5).is_nan());
    }
}
