/// Multi-timeframe signal aggregation.
/// Resamples data at different frequencies (e.g. 1D, 3D, 7D)
/// and checks if signals agree across scales.
/// Compute agreement score across timeframes.
/// Returns a value in [0, 1] where 1 = all scales agree.
pub fn multiscale_signals_slice(s1: &[f64], s3: &[f64], s7: &[f64]) -> Vec<f64> {
    let n = s1.len();
    let mut agreement = vec![f64::NAN; n];

    for i in 0..n {
        let i3 = i / 3;
        let i7 = i / 7;

        if i3 < s3.len() && i7 < s7.len() {
            let v1 = s1[i];
            let v3 = s3[i3];
            let v7 = s7[i7];

            if v1.is_nan() || v3.is_nan() || v7.is_nan() {
                continue;
            }

            // Agreement = geometric mean of signals across scales
            // Penalize disagreement (one high, others low)
            let signals = [v1.clamp(0.0, 1.0), v3.clamp(0.0, 1.0), v7.clamp(0.0, 1.0)];
            let mean = (signals[0] * signals[1] * signals[2]).cbrt();
            agreement[i] = mean;
        }
    }

    agreement
}
