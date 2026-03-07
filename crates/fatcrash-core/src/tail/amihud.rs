/// Amihud illiquidity ratio: mean(|return| / volume) over a window.
///
/// Higher values indicate less liquid markets. A spike in Amihud illiquidity
/// often precedes or accompanies market stress.
///
/// Args:
///     returns: Log return series.
///     volumes: Trading volume series (same length as returns).
///     window: Lookback window (from the end of the series).
///
/// Returns: Average |return_i| / volume_i over the last `window` observations.
pub fn compute_amihud(returns: &[f64], volumes: &[f64], window: usize) -> f64 {
    let n = returns.len().min(volumes.len());
    if n < window || window == 0 {
        return f64::NAN;
    }
    let start = n - window;
    let mut sum = 0.0;
    let mut count = 0;
    for i in start..n {
        if volumes[i] > 0.0 && returns[i].is_finite() {
            sum += returns[i].abs() / volumes[i];
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amihud_basic() {
        let returns = vec![0.01, -0.02, 0.03];
        let volumes = vec![1000.0, 2000.0, 500.0];
        let a = compute_amihud(&returns, &volumes, 3);
        let expected = (0.01 / 1000.0 + 0.02 / 2000.0 + 0.03 / 500.0) / 3.0;
        assert!((a - expected).abs() < 1e-15);
    }

    #[test]
    fn test_amihud_zero_volume_skipped() {
        let returns = vec![0.01, 0.02];
        let volumes = vec![0.0, 1000.0];
        let a = compute_amihud(&returns, &volumes, 2);
        assert!((a - 0.02 / 1000.0).abs() < 1e-15);
    }

    #[test]
    fn test_amihud_insufficient_data() {
        let returns = vec![0.01];
        let volumes = vec![1000.0];
        assert!(compute_amihud(&returns, &volumes, 5).is_nan());
    }
}
