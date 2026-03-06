use std::collections::HashMap;

use chrono::Utc;
use rayon::prelude::*;

use crate::cache;
use crate::config::WatchlistEntry;
use crate::data;
use crate::signals::{self, CrashSignal};

#[derive(Debug, Clone)]
pub struct AssetScan {
    pub asset: String,
    pub signal: CrashSignal,
    pub results: HashMap<String, Option<bool>>,
    pub components: HashMap<String, f64>,
    /// Raw intermediate values per method for drill-down explanations.
    /// Each key is a component name, value is a vec of (label, value) pairs.
    pub raw_values: HashMap<String, Vec<(String, f64)>>,
    pub timestamp: chrono::DateTime<Utc>,
    pub data_points: usize,
    pub prices: Vec<f64>,
    pub error: Option<String>,
}

/// Safely run a computation, catching panics. Returns NAN on failure.
fn safe_call<F: FnOnce() -> f64>(f: F) -> f64 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) if v.is_finite() => v,
        _ => f64::NAN,
    }
}

/// Compute realized variance (sum of squared returns) over a window.
fn realized_variance(returns: &[f64], window: usize) -> f64 {
    if returns.len() < window || window == 0 {
        return f64::NAN;
    }
    let start = returns.len() - window;
    returns[start..].iter().map(|r| r * r).sum::<f64>()
}

/// Compute bipower variation (for jump decomposition).
fn bipower_variation(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 2 {
        return f64::NAN;
    }
    let bv: f64 = (1..n)
        .map(|i| returns[i].abs() * returns[i - 1].abs())
        .sum();
    (std::f64::consts::PI / 2.0) * bv / (n - 1) as f64
}

/// Compute price velocity: fractional change in realized volatility.
fn compute_velocity(returns: &[f64], vol_window: usize, lag: usize) -> f64 {
    let n = returns.len();
    if n < vol_window + lag || vol_window < 2 || lag == 0 {
        return f64::NAN;
    }
    let current_slice = &returns[(n - vol_window)..];
    let current_vol = (current_slice.iter().map(|r| r * r).sum::<f64>() / vol_window as f64).sqrt();

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

/// Compute Amihud illiquidity: mean(|return| / volume).
fn compute_amihud(returns: &[f64], volumes: &[f64], window: usize) -> f64 {
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

/// Compute AR(1) coefficient.
fn ar1_coefficient(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 3 {
        return f64::NAN;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 1..n {
        num += (data[i] - mean) * (data[i - 1] - mean);
        den += (data[i - 1] - mean).powi(2);
    }
    if den < 1e-15 {
        f64::NAN
    } else {
        num / den
    }
}

/// Hamilton 2-state HMM via EM. Returns P(stressed) at last time step.
fn compute_hamilton_prob(returns: &[f64], n_restarts: usize) -> f64 {
    let n = returns.len();
    if n < 30 {
        return f64::NAN;
    }

    let mean = returns.iter().sum::<f64>() / n as f64;
    let std_dev = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    if std_dev < 1e-15 {
        return f64::NAN;
    }

    let gaussian_pdf = |x: f64, mu: f64, sigma: f64| -> f64 {
        if sigma <= 0.0 {
            return 1e-300;
        }
        let z = (x - mu) / sigma;
        ((-0.5 * z * z).exp()) / (sigma * (2.0 * std::f64::consts::PI).sqrt())
    };

    let mut best_ll = f64::NEG_INFINITY;
    let mut best_stressed_prob = 0.5;

    for restart in 0..n_restarts {
        let offset = restart as f64 * 0.1;
        let mut mu = [mean + std_dev * (0.5 + offset * 0.1), mean - std_dev * (1.0 + offset * 0.1)];
        let mut sigma = [std_dev * (0.8 + offset * 0.05), std_dev * (1.5 + offset * 0.05)];
        let mut p00 = 0.95;
        let mut p11 = 0.90;
        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..80 {
            let trans = [[p00, 1.0 - p00], [1.0 - p11, p11]];
            let mut filtered = vec![[0.5_f64; 2]; n];
            let mut prev = [0.5, 0.5];
            let mut ll = 0.0_f64;

            for t in 0..n {
                let pred = [
                    trans[0][0] * prev[0] + trans[1][0] * prev[1],
                    trans[0][1] * prev[0] + trans[1][1] * prev[1],
                ];
                let l0 = gaussian_pdf(returns[t], mu[0], sigma[0]);
                let l1 = gaussian_pdf(returns[t], mu[1], sigma[1]);
                let marginal = pred[0] * l0 + pred[1] * l1;
                if marginal < 1e-300 {
                    filtered[t] = pred;
                    prev = pred;
                    continue;
                }
                ll += marginal.ln();
                filtered[t] = [pred[0] * l0 / marginal, pred[1] * l1 / marginal];
                prev = filtered[t];
            }

            if (ll - prev_ll).abs() < 1e-6 {
                if ll > best_ll {
                    best_ll = ll;
                    let last = filtered[n - 1];
                    best_stressed_prob = if sigma[1] > sigma[0] { last[1] } else { last[0] };
                }
                break;
            }
            prev_ll = ll;

            // M-step
            let (mut w0, mut w1) = (0.0, 0.0);
            let (mut mu0_s, mut mu1_s) = (0.0, 0.0);
            for t in 0..n {
                w0 += filtered[t][0];
                w1 += filtered[t][1];
                mu0_s += filtered[t][0] * returns[t];
                mu1_s += filtered[t][1] * returns[t];
            }
            if w0 > 1e-10 { mu[0] = mu0_s / w0; }
            if w1 > 1e-10 { mu[1] = mu1_s / w1; }

            let (mut v0, mut v1) = (0.0, 0.0);
            for t in 0..n {
                v0 += filtered[t][0] * (returns[t] - mu[0]).powi(2);
                v1 += filtered[t][1] * (returns[t] - mu[1]).powi(2);
            }
            if w0 > 1e-10 { sigma[0] = (v0 / w0).sqrt().max(1e-10); }
            if w1 > 1e-10 { sigma[1] = (v1 / w1).sqrt().max(1e-10); }

            let (mut n00, mut n11, mut f0, mut f1) = (0.0, 0.0, 0.0, 0.0);
            for t in 1..n {
                let xi00 = filtered[t - 1][0] * trans[0][0] * gaussian_pdf(returns[t], mu[0], sigma[0]);
                let xi01 = filtered[t - 1][0] * trans[0][1] * gaussian_pdf(returns[t], mu[1], sigma[1]);
                let xi10 = filtered[t - 1][1] * trans[1][0] * gaussian_pdf(returns[t], mu[0], sigma[0]);
                let xi11 = filtered[t - 1][1] * trans[1][1] * gaussian_pdf(returns[t], mu[1], sigma[1]);
                let total = xi00 + xi01 + xi10 + xi11;
                if total > 1e-300 {
                    n00 += xi00 / total;
                    n11 += xi11 / total;
                    f0 += (xi00 + xi01) / total;
                    f1 += (xi10 + xi11) / total;
                }
            }
            if f0 > 1e-10 { p00 = (n00 / f0).clamp(0.01, 0.99); }
            if f1 > 1e-10 { p11 = (n11 / f1).clamp(0.01, 0.99); }

            if ll > best_ll {
                best_ll = ll;
                let last = filtered[n - 1];
                best_stressed_prob = if sigma[1] > sigma[0] { last[1] } else { last[0] };
            }
        }
    }

    best_stressed_prob
}

/// GSADF test: compute max ADF stat across subsamples.
fn compute_gsadf(log_prices: &[f64]) -> (f64, f64) {
    let n = log_prices.len();
    let min_window = ((0.01 * n as f64 + 1.8 * (n as f64).sqrt()).floor() as usize).max(4);
    if n < min_window + 2 {
        return (f64::NAN, f64::NAN);
    }

    let mut max_adf = f64::NEG_INFINITY;
    for r2 in min_window..n {
        let max_r1 = r2.saturating_sub(min_window);
        for r1 in 0..=max_r1 {
            if r2 - r1 < min_window {
                continue;
            }
            let t = adf_t_stat(&log_prices[r1..=r2]);
            if t.is_finite() && t > max_adf {
                max_adf = t;
            }
        }
    }

    let cv95 = 0.25 * (n as f64).ln().ln() + 1.27;
    if max_adf.is_finite() { (max_adf, cv95) } else { (f64::NAN, cv95) }
}

/// ADF t-statistic for unit root test.
fn adf_t_stat(prices: &[f64]) -> f64 {
    let n = prices.len();
    if n < 4 {
        return f64::NAN;
    }
    let dy: Vec<f64> = (1..n).map(|i| prices[i] - prices[i - 1]).collect();
    let y_lag: Vec<f64> = (0..n - 1).map(|i| prices[i]).collect();
    let m = dy.len() as f64;

    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..dy.len() {
        sx += y_lag[i]; sy += dy[i]; sxx += y_lag[i] * y_lag[i]; sxy += y_lag[i] * dy[i];
    }
    let denom = m * sxx - sx * sx;
    if denom.abs() < 1e-15 { return f64::NAN; }

    let beta = (m * sxy - sx * sy) / denom;
    let alpha = (sy - beta * sx) / m;

    let mut ssr = 0.0;
    for i in 0..dy.len() {
        let r = dy[i] - alpha - beta * y_lag[i];
        ssr += r * r;
    }
    if m - 2.0 <= 0.0 { return f64::NAN; }
    let s2 = ssr / (m - 2.0);
    let se = (s2 * m / denom).sqrt();
    if se < 1e-15 || !se.is_finite() { return f64::NAN; }
    beta / se
}

/// LPPLS simplified fit via grid search.
/// Multi-window LPPLS confidence following Sornette's methodology.
///
/// Runs LPPLS fits across multiple time windows and counts what fraction
/// pass ALL Sornette filter conditions:
///   - 0.1 ≤ m ≤ 0.9
///   - 6 ≤ ω ≤ 13
///   - B < 0 (crash = price decrease at tc)
///   - |C|/|B| < 0.5 (oscillations strongly damped by trend)
///   - Fit improvement over linear > 50%
///   - tc estimates must cluster (majority within ±15 days)
///
/// Returns (confidence, best_days_to_tc).
fn compute_lppls_signal(log_prices: &[f64], times: &[f64]) -> (f64, f64) {
    let n = log_prices.len();
    if n < 80 {
        return (0.0, f64::INFINITY);
    }

    let t_end = times[n - 1];

    // Multi-window: test windows from 60 to n, stepping by 15
    let min_win = 60_usize;
    let mut n_windows = 0_usize;
    let mut passing_tcs: Vec<f64> = Vec::new(); // tc values from passing windows

    let mut win = min_win;
    while win <= n {
        let start = n - win;
        let lp = &log_prices[start..];
        let t: Vec<f64> = (0..win).map(|i| i as f64).collect();
        let t_end_win = t[win - 1];
        let linear_sse = linear_fit_sse(lp, &t);

        n_windows += 1;

        // Grid search with Sornette-constrained parameters
        let mut win_best_sse = f64::INFINITY;
        let mut win_best_tc = f64::NAN;

        for d in 1..=60 {
            let tc = t_end_win + d as f64;
            for &m in &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
                for &omega in &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0] {
                    if let Some((sse, coeffs)) = lppls_linear_solve_full(lp, &t, tc, m, omega) {
                        let b = coeffs[1];
                        let c_amp = (coeffs[2] * coeffs[2] + coeffs[3] * coeffs[3]).sqrt();
                        // Strict damping: oscillations must be < 50% of trend
                        if b.abs() < 1e-15 || c_amp / b.abs() >= 0.5 {
                            continue;
                        }
                        if sse < win_best_sse {
                            win_best_sse = sse;
                            win_best_tc = tc;
                        }
                    }
                }
            }
        }

        if !win_best_tc.is_nan() && linear_sse > 0.0 {
            let improvement = 1.0 - (win_best_sse / linear_sse);
            // Require > 50% improvement (strict: 7 params vs 2 must really earn it)
            if improvement > 0.50 {
                let tc_days = t_end + (win_best_tc - t_end_win);
                passing_tcs.push(tc_days);
            }
        }

        win += 15;
    }

    if n_windows == 0 {
        return (0.0, f64::INFINITY);
    }

    // tc clustering: find the tc value with the most neighbors within ±15 days.
    // If tc estimates are scattered, the LPPLS fit is noise.
    let mut best_cluster_count = 0_usize;
    let mut best_cluster_tc = f64::NAN;
    for &tc in &passing_tcs {
        let nearby = passing_tcs.iter().filter(|&&other| (other - tc).abs() <= 15.0).count();
        if nearby > best_cluster_count {
            best_cluster_count = nearby;
            best_cluster_tc = tc;
        }
    }

    // Confidence = fraction of windows in the best tc cluster / total windows
    // Must have at least 2 windows agreeing on tc for any signal
    let confidence = if best_cluster_count >= 2 {
        best_cluster_count as f64 / n_windows as f64
    } else {
        0.0
    };

    let days_to_tc = if best_cluster_tc.is_finite() { best_cluster_tc - t_end } else { f64::INFINITY };
    (confidence, days_to_tc)
}

/// LPPLS linear solve returning both SSE and coefficients [A, B, C1, C2].
fn lppls_linear_solve_full(lp: &[f64], t: &[f64], tc: f64, m: f64, omega: f64) -> Option<(f64, [f64; 4])> {
    let n = lp.len();
    if n < 4 { return None; }

    let mut ata = [[0.0_f64; 4]; 4];
    let mut atb = [0.0_f64; 4];

    for i in 0..n {
        let dt = tc - t[i];
        if dt <= 0.0 { return None; }
        let f = dt.powf(m);
        let log_dt = dt.ln();
        let g = f * (omega * log_dt).cos();
        let h = f * (omega * log_dt).sin();
        let x = [1.0, f, g, h];
        for j in 0..4 {
            for k in 0..4 { ata[j][k] += x[j] * x[k]; }
            atb[j] += x[j] * lp[i];
        }
    }

    let coeffs = solve_4x4(&ata, &atb)?;
    if coeffs[1] > 0.0 { return None; } // B must be negative

    let mut sse = 0.0;
    for i in 0..n {
        let dt = tc - t[i];
        if dt <= 0.0 { return None; }
        let f = dt.powf(m);
        let log_dt = dt.ln();
        let pred = coeffs[0] + coeffs[1] * f + coeffs[2] * f * (omega * log_dt).cos() + coeffs[3] * f * (omega * log_dt).sin();
        sse += (lp[i] - pred).powi(2);
    }
    Some((sse, coeffs))
}

fn solve_4x4(a: &[[f64; 4]; 4], b: &[f64; 4]) -> Option<[f64; 4]> {
    let mut aug = [[0.0_f64; 5]; 4];
    for i in 0..4 {
        for j in 0..4 { aug[i][j] = a[i][j]; }
        aug[i][4] = b[i];
    }
    for col in 0..4 {
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..4 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 { return None; }
        if max_row != col { aug.swap(col, max_row); }
        let pivot = aug[col][col];
        for j in col..5 { aug[col][j] /= pivot; }
        for row in 0..4 {
            if row != col {
                let factor = aug[row][col];
                for j in col..5 { aug[row][j] -= factor * aug[col][j]; }
            }
        }
    }
    let result = [aug[0][4], aug[1][4], aug[2][4], aug[3][4]];
    if result.iter().all(|v| v.is_finite()) { Some(result) } else { None }
}

fn linear_fit_sse(lp: &[f64], t: &[f64]) -> f64 {
    let n = lp.len();
    if n < 2 { return f64::INFINITY; }
    let xm = t.iter().sum::<f64>() / n as f64;
    let ym = lp.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut vx = 0.0;
    for i in 0..n {
        cov += (t[i] - xm) * (lp[i] - ym);
        vx += (t[i] - xm).powi(2);
    }
    if vx < 1e-15 { return f64::INFINITY; }
    let slope = cov / vx;
    let intercept = ym - slope * xm;
    let mut sse = 0.0;
    for i in 0..n { sse += (lp[i] - intercept - slope * t[i]).powi(2); }
    sse
}

fn error_scan(asset: &str, now: chrono::DateTime<Utc>, err: &str) -> AssetScan {
    AssetScan {
        asset: asset.to_string(),
        signal: CrashSignal {
            probability: 0.0,
            horizon_days: f64::INFINITY,
            components: HashMap::new(),
            n_agreeing: 0,
        },
        results: HashMap::new(),
        components: HashMap::new(),
        raw_values: HashMap::new(),
        timestamp: now,
        data_points: 0,
        prices: vec![],
        error: Some(err.to_string()),
    }
}

/// Scan a single asset: fetch data, run all methods, produce aggregated signal.
pub fn scan_asset(
    entry: &WatchlistEntry,
    window: usize,
    days: usize,
    use_cache: bool,
) -> AssetScan {
    let now = Utc::now();
    let source_str = entry.source.to_string();
    let scope = format!("{}d", days);

    // Fetch data with optional cache
    let bars = if use_cache {
        if let Some(cached) = cache::load_cached(&source_str, &entry.symbol, &scope) {
            cached
        } else {
            match data::fetch(&entry.source, days) {
                Ok(b) => {
                    cache::save_cache(&source_str, &entry.symbol, &scope, &b);
                    b
                }
                Err(e) => return error_scan(&entry.symbol, now, &e),
            }
        }
    } else {
        match data::fetch(&entry.source, days) {
            Ok(b) => b,
            Err(e) => return error_scan(&entry.symbol, now, &e),
        }
    };

    let close_prices: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let data_points = close_prices.len();

    if data_points < 270 {
        return error_scan(
            &entry.symbol,
            now,
            &format!("Insufficient data: {} bars (need 270+)", data_points),
        );
    }

    // Compute log returns via fatcrash_core
    let returns = match fatcrash_core::utils::log_returns_slice(&close_prices) {
        Ok(r) => r,
        Err(e) => return error_scan(&entry.symbol, now, &format!("Log returns: {}", e)),
    };

    let n = returns.len();
    let pre_end = n;
    let pre_start = n.saturating_sub(window);
    let base_end = pre_start.saturating_sub(30);
    let base_start = base_end.saturating_sub(window);

    if pre_start >= pre_end || base_start >= base_end {
        return error_scan(&entry.symbol, now, "Window yields empty slices");
    }

    let pre = &returns[pre_start..pre_end];
    let base = &returns[base_start..base_end];
    let pre_prices = &close_prices[pre_start..];

    // Sparkline prices: last 120
    let spark: Vec<f64> = if close_prices.len() > 120 {
        close_prices[close_prices.len() - 120..].to_vec()
    } else {
        close_prices.clone()
    };

    let mut comp = HashMap::new();
    let mut res: HashMap<String, Option<bool>> = HashMap::new();
    let mut raw: HashMap<String, Vec<(String, f64)>> = HashMap::new();

    // Helper: insert both component and result using the same key (the weight/component key)
    macro_rules! record {
        ($key:expr, $sig:expr) => {{
            let s = $sig;
            let detected = if s.is_nan() { None } else { Some(s > 0.5) };
            comp.insert($key.to_string(), s);
            res.insert($key.to_string(), detected);
        }};
        ($key:expr, $sig:expr, $raw_vals:expr) => {{
            let s = $sig;
            let detected = if s.is_nan() { None } else { Some(s > 0.5) };
            comp.insert($key.to_string(), s);
            res.insert($key.to_string(), detected);
            raw.insert($key.to_string(), $raw_vals);
        }};
    }

    // 1. Hill (pre vs base — tail thinning)
    let hill_pre = safe_call(|| fatcrash_core::tail::hill::hill_estimator_slice(pre, None, true));
    let hill_base = safe_call(|| fatcrash_core::tail::hill::hill_estimator_slice(base, None, true));
    record!("hill_thinning", signals::hill_thinning_signal(hill_pre, hill_base), vec![
        ("Alpha (pre-window)".into(), hill_pre),
        ("Alpha (baseline)".into(), hill_base),
        ("Ratio pre/base".into(), if hill_base > 0.0 { hill_pre / hill_base } else { f64::NAN }),
    ]);

    // 2. Kappa (max-stability, multi-subsample)
    let mut kappa_vals = Vec::new();
    let kappa_sig = safe_call(|| {
        let mut best = 0.0_f64;
        for &ns in &[5_usize, 10, 20] {
            let (k, b) = fatcrash_core::tail::kappa::kappa_metric_slice(pre, ns, 200);
            kappa_vals.push((format!("Kappa (n={})", ns), k));
            kappa_vals.push((format!("Gaussian bench (n={})", ns), b));
            let s = signals::kappa_regime_signal(k, b);
            if s > best { best = s; }
        }
        best
    });
    record!("kappa_regime", kappa_sig, kappa_vals);

    // 3. Taleb kappa
    let taleb_sig = safe_call(|| {
        let n0 = 30_usize.min(pre.len() / 4).max(1);
        let n1 = 100_usize.min(pre.len() / 2);
        if n1 <= n0 { return f64::NAN; }
        let (k, b) = fatcrash_core::tail::kappa::taleb_kappa_slice(pre, n0, n1, 200);
        raw.insert("taleb_kappa".into(), vec![
            ("Taleb kappa".into(), k), ("Gaussian bench".into(), b),
        ]);
        signals::taleb_kappa_signal(k, b)
    });
    record!("taleb_kappa", taleb_sig);

    // 4. Pickands (pre vs base)
    let pick_pre = safe_call(|| fatcrash_core::tail::pickands::pickands_estimator_slice(pre, None, true));
    let pick_base = safe_call(|| fatcrash_core::tail::pickands::pickands_estimator_slice(base, None, true));
    record!("pickands_thinning", signals::pickands_signal(pick_pre, pick_base), vec![
        ("Gamma (pre-window)".into(), pick_pre),
        ("Gamma (baseline)".into(), pick_base),
    ]);

    // 5. Hurst exponent
    let hurst = safe_call(|| fatcrash_core::tail::hurst::hurst_exponent_slice(pre));
    record!("hurst_trending", signals::hurst_signal(hurst), vec![
        ("Hurst exponent".into(), hurst),
        ("Random walk = 0.5".into(), 0.5),
    ]);

    // 6. DFA exponent
    let dfa = safe_call(|| fatcrash_core::tail::dfa::dfa_exponent_slice(pre));
    record!("dfa_trending", signals::dfa_signal(dfa), vec![
        ("DFA exponent".into(), dfa),
        ("Random walk = 0.5".into(), 0.5),
    ]);

    // 7. DEH (pre vs base)
    let deh_pre = safe_call(|| fatcrash_core::tail::deh::deh_estimator_slice(pre, None, true));
    let deh_base = safe_call(|| fatcrash_core::tail::deh::deh_estimator_slice(base, None, true));
    record!("deh_thinning", signals::deh_signal(deh_pre, deh_base), vec![
        ("DEH gamma (pre)".into(), deh_pre),
        ("DEH gamma (base)".into(), deh_base),
    ]);

    // 8. QQ (pre vs base)
    let qq_pre = safe_call(|| fatcrash_core::tail::qq::qq_estimator_slice(pre, None, true));
    let qq_base = safe_call(|| fatcrash_core::tail::qq::qq_estimator_slice(base, None, true));
    record!("qq_thinning", signals::qq_signal(qq_pre, qq_base), vec![
        ("QQ alpha (pre)".into(), qq_pre),
        ("QQ alpha (base)".into(), qq_base),
    ]);

    // 9. MaxSum ratio
    let maxsum = safe_call(|| fatcrash_core::tail::maxsum::maxsum_ratio_slice(pre));
    record!("maxsum_signal", signals::maxsum_signal_fn(maxsum), vec![
        ("Max/sum ratio".into(), maxsum),
    ]);

    // 10. Spectral exponent
    let spectral = safe_call(|| fatcrash_core::tail::spectral::spectral_exponent_slice(pre, 0.65));
    record!("spectral_memory", signals::spectral_signal(spectral), vec![
        ("GPH exponent (d)".into(), spectral),
        ("White noise = 0".into(), 0.0),
    ]);

    // 11. Momentum reversal
    let reversal = safe_call(|| {
        let long_lb = 90_usize.min(pre_prices.len().saturating_sub(1));
        fatcrash_core::tail::momentum::momentum_reversal_slice(pre_prices, 21, long_lb)
    });
    record!("momentum_reversal", signals::momentum_reversal_signal(reversal), vec![
        ("Reversal score".into(), reversal),
    ]);

    // 12. RV spike (short vs long)
    let rv_short = realized_variance(&returns, 21.min(n));
    let rv_long = realized_variance(&returns, 126.min(n));
    record!("rv_spike", signals::rv_spike_signal(rv_short, rv_long), vec![
        ("RV 21d".into(), rv_short),
        ("RV 126d".into(), rv_long),
        ("Ratio short/long".into(), if rv_long > 0.0 { rv_short / rv_long } else { f64::NAN }),
    ]);

    // 13. Velocity (vol acceleration)
    let velocity = safe_call(|| compute_velocity(pre, 21, 5));
    record!("velocity_spike", signals::velocity_signal(velocity), vec![
        ("Vol velocity".into(), velocity),
    ]);

    // 14. Hamilton stress (2-state HMM)
    let hamilton = safe_call(|| compute_hamilton_prob(pre, 5));
    record!("hamilton_stress", signals::hamilton_stress_signal(hamilton), vec![
        ("P(stressed state)".into(), hamilton),
    ]);

    // 15. CSD warning (Critical Slowing Down on vol)
    let mut csd_ar1_roc = f64::NAN;
    let mut csd_var_roc = f64::NAN;
    let csd_sig = safe_call(|| {
        let vol_window = 21_usize;
        if pre.len() < vol_window * 3 { return f64::NAN; }
        let rolling_rv: Vec<f64> = (vol_window..pre.len())
            .map(|i| pre[i - vol_window..i].iter().map(|r| r * r).sum::<f64>().sqrt())
            .collect();
        if rolling_rv.len() < 42 { return f64::NAN; }
        let half = rolling_rv.len() / 2;
        let (first, second) = (&rolling_rv[..half], &rolling_rv[half..]);

        let ar1_first = ar1_coefficient(first);
        let ar1_second = ar1_coefficient(second);
        csd_ar1_roc = if ar1_first.is_finite() && ar1_second.is_finite() && ar1_first.abs() > 1e-10 {
            (ar1_second - ar1_first) / ar1_first.abs()
        } else { 0.0 };

        let var_of = |s: &[f64]| -> f64 {
            let m = s.iter().sum::<f64>() / s.len() as f64;
            s.iter().map(|v| (v - m).powi(2)).sum::<f64>() / s.len() as f64
        };
        csd_var_roc = {
            let vf = var_of(first);
            let vs = var_of(second);
            if vf > 1e-15 { (vs - vf) / vf } else { 0.0 }
        };
        signals::csd_warning_signal(csd_ar1_roc, csd_var_roc)
    });
    record!("csd_warning", csd_sig, vec![
        ("AR(1) rate of change".into(), csd_ar1_roc),
        ("Variance rate of change".into(), csd_var_roc),
    ]);

    // 16. Amihud illiquidity spike
    let mut amihud_short = f64::NAN;
    let mut amihud_long = f64::NAN;
    let amihud_sig = safe_call(|| {
        let ret_start = returns.len().saturating_sub(window);
        let vol_start = ret_start + 1;
        if vol_start >= volumes.len() { return f64::NAN; }
        let ret_s = &returns[ret_start..];
        let vol_s = &volumes[vol_start..];
        let min_len = ret_s.len().min(vol_s.len());
        if min_len < 42 { return f64::NAN; }
        amihud_short = compute_amihud(&ret_s[..min_len], &vol_s[..min_len], 21.min(min_len));
        amihud_long = compute_amihud(&ret_s[..min_len], &vol_s[..min_len], min_len);
        signals::amihud_spike_signal(amihud_short, amihud_long)
    });
    record!("amihud_spike", amihud_sig, vec![
        ("Amihud 21d".into(), amihud_short),
        ("Amihud full window".into(), amihud_long),
        ("Ratio short/long".into(), if amihud_long > 0.0 { amihud_short / amihud_long } else { f64::NAN }),
    ]);

    // 17. LPPLS confidence & tc proximity
    let (lppls_conf, lppls_days) = {
        let log_p: Vec<f64> = pre_prices.iter().filter_map(|p| if *p > 0.0 { Some(p.ln()) } else { None }).collect();
        let times: Vec<f64> = (0..log_p.len()).map(|i| i as f64).collect();
        if log_p.len() >= 60 {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compute_lppls_signal(&log_p, &times))) {
                Ok(r) => r,
                _ => (0.0, f64::INFINITY),
            }
        } else {
            (0.0, f64::INFINITY)
        }
    };
    record!("lppls_confidence", signals::lppls_confidence_signal(lppls_conf), vec![
        ("Fit improvement".into(), lppls_conf),
    ]);
    let tc_sig = signals::tc_proximity_signal(lppls_days, 90.0);
    record!("lppls_tc_proximity", tc_sig, vec![
        ("Days to tc".into(), lppls_days),
    ]);

    // 18. GSADF bubble
    let mut gsadf_stat = f64::NAN;
    let mut gsadf_cv = f64::NAN;
    let gsadf_sig = safe_call(|| {
        let log_p: Vec<f64> = pre_prices.iter().filter_map(|p| if *p > 0.0 { Some(p.ln()) } else { None }).collect();
        let (stat, cv) = compute_gsadf(&log_p);
        gsadf_stat = stat;
        gsadf_cv = cv;
        signals::gsadf_signal(stat, cv)
    });
    record!("gsadf_bubble", gsadf_sig, vec![
        ("GSADF statistic".into(), gsadf_stat),
        ("95% critical value".into(), gsadf_cv),
        ("Excess over CV".into(), gsadf_stat - gsadf_cv),
    ]);

    // Jump risk signal
    let mut jump_var = f64::NAN;
    let mut jump_rv = f64::NAN;
    let jump_sig = safe_call(|| {
        let recent = &returns[n.saturating_sub(63)..];
        let rv = recent.iter().map(|r| r * r).sum::<f64>();
        let bv = bipower_variation(recent);
        if rv.is_nan() || bv.is_nan() { return f64::NAN; }
        jump_var = (rv - bv).max(0.0);
        jump_rv = rv;
        signals::jump_risk_signal_converter(jump_var, rv)
    });
    record!("jump_risk_signal", jump_sig, vec![
        ("Jump variance".into(), jump_var),
        ("Realized variance".into(), jump_rv),
        ("Jump fraction".into(), if jump_rv > 0.0 { jump_var / jump_rv } else { f64::NAN }),
    ]);

    // Multiscale agreement (Hill at 3 timeframes)
    let mut h1_val = f64::NAN;
    let mut h3_val = f64::NAN;
    let mut h7_val = f64::NAN;
    let multi_sig = safe_call(|| {
        h1_val = fatcrash_core::tail::hill::hill_estimator_slice(pre, None, true);
        let r3: Vec<f64> = pre.chunks(3).filter(|c| c.len() == 3).map(|c| c.iter().sum()).collect();
        let r7: Vec<f64> = pre.chunks(7).filter(|c| c.len() == 7).map(|c| c.iter().sum()).collect();
        if r3.len() < 10 || r7.len() < 5 { return f64::NAN; }
        h3_val = fatcrash_core::tail::hill::hill_estimator_slice(&r3, None, true);
        h7_val = fatcrash_core::tail::hill::hill_estimator_slice(&r7, None, true);
        if h1_val.is_nan() || h3_val.is_nan() || h7_val.is_nan() { return f64::NAN; }
        let to_sig = |a: f64| (1.0 - a / 5.0).clamp(0.0, 1.0);
        (to_sig(h1_val) * to_sig(h3_val) * to_sig(h7_val)).cbrt()
    });
    record!("multiscale", multi_sig, vec![
        ("Hill alpha 1d".into(), h1_val),
        ("Hill alpha 3d".into(), h3_val),
        ("Hill alpha 7d".into(), h7_val),
    ]);

    // Aggregate all signals
    let signal = signals::aggregate_signals(&comp);

    AssetScan {
        asset: entry.symbol.clone(),
        signal,
        results: res,
        components: comp,
        raw_values: raw,
        timestamp: now,
        data_points,
        prices: spark,
        error: None,
    }
}

/// Scan all watchlist entries in parallel (max 3 threads for API rate limiting).
pub fn scan_watchlist(
    entries: &[WatchlistEntry],
    window: usize,
    days: usize,
    use_cache: bool,
) -> Vec<AssetScan> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());
    pool.install(|| {
        entries
            .par_iter()
            .map(|e| scan_asset(e, window, days, use_cache))
            .collect()
    })
}
