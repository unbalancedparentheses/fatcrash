/// Multi-timeframe signal aggregation.
///
/// Checks whether crash signals agree across different observation
/// frequencies (1-day, 3-day, 7-day). Agreement is measured as the
/// geometric mean of clamped \[0, 1\] signals, penalizing disagreement
/// (e.g., one scale high while others are low).
///
/// Returns a vector of agreement scores in \[0, 1\] where 1 = all
/// three scales agree on a high signal.
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
