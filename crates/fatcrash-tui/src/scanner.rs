use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};

use chrono::Utc;
use rayon::prelude::*;

use crate::cache;
use crate::config::WatchlistEntry;
use crate::data;
use crate::signals::{self, CrashSignal};

/// Messages sent from the scanner thread to the TUI.
#[derive(Debug, Clone)]
pub enum ScanMsg {
    /// One asset finished scanning.
    Progress {
        done: usize,
        total: usize,
        asset: String,
    },
    /// Phase 1 done: all methods except GSADF computed.
    PartialResults(Vec<AssetScan>),
    /// One asset's GSADF completed during Phase 2.
    GsadfUpdate(Box<AssetScan>),
    /// Phase 2 done: everything including GSADF.
    Done,
}

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

fn error_scan(asset: &str, now: chrono::DateTime<Utc>, err: &str) -> AssetScan {
    AssetScan {
        asset: asset.to_string(),
        signal: CrashSignal {
            probability: 0.0,
            horizon_days: f64::INFINITY,
            components: HashMap::new(),
            n_confirming: 0,
            confirming_categories: Vec::new(),
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

/// Compute GSADF bubble signal from log prices.
/// Returns `(signal_value, raw_values_vec)` for merging into an `AssetScan`.
pub fn compute_gsadf(log_prices: &[f64]) -> (f64, Vec<(String, f64)>) {
    let mut gsadf_stat = f64::NAN;
    let mut gsadf_cv = f64::NAN;
    let sig = safe_call(|| {
        let (stat, _, (_, cv95, _)) =
            fatcrash_core::bubble::gsadf::gsadf_test_slice(log_prices, None, Some(100), Some(42));
        gsadf_stat = stat;
        gsadf_cv = cv95;
        signals::gsadf_signal(stat, cv95)
    });
    let raw = vec![
        ("GSADF statistic".into(), gsadf_stat),
        ("95% critical value".into(), gsadf_cv),
        ("Excess over CV".into(), gsadf_stat - gsadf_cv),
    ];
    (sig, raw)
}

/// Scan a single asset: fetch data, run all methods, produce aggregated signal.
#[allow(dead_code)]
pub fn scan_asset(
    entry: &WatchlistEntry,
    window: usize,
    days: usize,
    use_cache: bool,
) -> AssetScan {
    let (scan, _log_prices) = scan_asset_inner(entry, window, days, use_cache, true);
    scan
}

/// Inner scan: runs all methods. When `run_gsadf` is false, skips GSADF and returns
/// the log-prices so the caller can run `compute_gsadf` later.
fn scan_asset_inner(
    entry: &WatchlistEntry,
    window: usize,
    days: usize,
    use_cache: bool,
    run_gsadf: bool,
) -> (AssetScan, Vec<f64>) {
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
                Err(e) => return (error_scan(&entry.symbol, now, &e), vec![]),
            }
        }
    } else {
        match data::fetch(&entry.source, days) {
            Ok(b) => b,
            Err(e) => return (error_scan(&entry.symbol, now, &e), vec![]),
        }
    };

    let open_prices: Vec<f64> = bars.iter().map(|b| b.open).collect();
    let high_prices: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low_prices: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let close_prices: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let data_points = close_prices.len();

    if data_points < 270 {
        return (error_scan(
            &entry.symbol,
            now,
            &format!("Insufficient data: {} bars (need 270+)", data_points),
        ), vec![]);
    }

    // Compute log returns via fatcrash_core
    let returns = match fatcrash_core::utils::log_returns_slice(&close_prices) {
        Ok(r) => r,
        Err(e) => return (error_scan(&entry.symbol, now, &format!("Log returns: {}", e)), vec![]),
    };

    let n = returns.len();
    let pre_end = n;
    let pre_start = n.saturating_sub(window);
    let base_end = pre_start.saturating_sub(30);
    let base_start = base_end.saturating_sub(window);

    if pre_start >= pre_end || base_start >= base_end {
        return (error_scan(&entry.symbol, now, "Window yields empty slices"), vec![]);
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

    // 19. Realized skewness (Neuberger 2012)
    let skew = safe_call(|| {
        fatcrash_core::tail::skewness::compute_realized_skewness(pre, pre.len())
    });
    record!("realized_skewness", signals::realized_skewness_signal(skew), vec![
        ("Realized skewness".into(), skew),
    ]);

    // 20. GPD VaR exceedance
    let gpd_sig = safe_call(|| {
        match fatcrash_core::evt::gpd::gpd_var_es_slice(pre, 0.99, 0.95) {
            Ok((var, es)) => {
                let current_loss = -pre.last().unwrap_or(&0.0);
                raw.insert("gpd_var_exceedance".into(), vec![
                    ("VaR 99%".into(), var),
                    ("ES 99%".into(), es),
                    ("Current loss".into(), current_loss),
                ]);
                signals::var_exceedance_signal(current_loss, var)
            }
            Err(_) => f64::NAN,
        }
    });
    record!("gpd_var_exceedance", gpd_sig);

    // 12. RV spike (short vs long) — use Garman-Klass when real OHLC available
    let has_ohlc = high_prices.iter().zip(low_prices.iter()).any(|(h, l)| (h - l).abs() > 1e-10);

    let rv_short;
    let rv_long;
    let rv_estimator: &str;
    if has_ohlc {
        let short_start = open_prices.len().saturating_sub(21);
        let long_start = open_prices.len().saturating_sub(126);
        rv_short = safe_call(|| fatcrash_core::regime::realized_var::compute_realized_variance_gk(
            &open_prices[short_start..], &high_prices[short_start..],
            &low_prices[short_start..], &close_prices[short_start..],
        ));
        rv_long = safe_call(|| fatcrash_core::regime::realized_var::compute_realized_variance_gk(
            &open_prices[long_start..], &high_prices[long_start..],
            &low_prices[long_start..], &close_prices[long_start..],
        ));
        rv_estimator = "Garman-Klass";
    } else {
        rv_short = safe_call(|| fatcrash_core::regime::realized_var::compute_realized_variance(&returns[n.saturating_sub(21)..]));
        rv_long = safe_call(|| fatcrash_core::regime::realized_var::compute_realized_variance(&returns[n.saturating_sub(126)..]));
        rv_estimator = "Simple RV";
    }

    record!("rv_spike", signals::rv_spike_signal(rv_short, rv_long), vec![
        (format!("RV 21d ({})", rv_estimator), rv_short),
        (format!("RV 126d ({})", rv_estimator), rv_long),
        ("Ratio short/long".into(), if rv_long > 0.0 { rv_short / rv_long } else { f64::NAN }),
    ]);

    // 13. Velocity (vol acceleration)
    let velocity = safe_call(|| fatcrash_core::tail::velocity::compute_velocity(pre, 21, 5));
    record!("velocity_spike", signals::velocity_signal(velocity), vec![
        ("Vol velocity".into(), velocity),
    ]);

    // 14. Hamilton stress (2-state HMM) — uses fatcrash-core with parallel restarts
    let hamilton = safe_call(|| {
        let (_, _, _, _, _, _, filtered) =
            fatcrash_core::regime::hamilton::hamilton_fit_slice(pre, None);
        // filtered contains P(stressed state) at each time step (state 1 = higher vol)
        filtered.last().copied().unwrap_or(f64::NAN)
    });
    record!("hamilton_stress", signals::hamilton_stress_signal(hamilton), vec![
        ("P(stressed state)".into(), hamilton),
    ]);

    // 15. CSD warning (Critical Slowing Down) — uses fatcrash-core rolling AR1 + variance
    let mut csd_ar1_roc = f64::NAN;
    let mut csd_var_roc = f64::NAN;
    let csd_sig = safe_call(|| {
        if returns.len() < 126 { return f64::NAN; }
        let (ar1_roc, var_roc, _) =
            fatcrash_core::regime::csd::csd_indicator_slice(&returns, 63, 21);
        // Take last valid values as the current CSD state
        csd_ar1_roc = ar1_roc.iter().rev().find(|v| v.is_finite()).copied().unwrap_or(0.0);
        csd_var_roc = var_roc.iter().rev().find(|v| v.is_finite()).copied().unwrap_or(0.0);
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
        amihud_short = fatcrash_core::tail::amihud::compute_amihud(&ret_s[..min_len], &vol_s[..min_len], 21.min(min_len));
        amihud_long = fatcrash_core::tail::amihud::compute_amihud(&ret_s[..min_len], &vol_s[..min_len], min_len);
        signals::amihud_spike_signal(amihud_short, amihud_long)
    });
    record!("amihud_spike", amihud_sig, vec![
        ("Amihud 21d".into(), amihud_short),
        ("Amihud full window".into(), amihud_long),
        ("Ratio short/long".into(), if amihud_long > 0.0 { amihud_short / amihud_long } else { f64::NAN }),
    ]);

    // 17. LPPLS confidence & tc proximity (using fatcrash-core proper implementation)
    let (lppls_conf, lppls_days, lppls_tc_std) = {
        let log_p: Vec<f64> = pre_prices.iter().filter_map(|p| if *p > 0.0 { Some(p.ln()) } else { None }).collect();
        let times: Vec<f64> = (0..log_p.len()).map(|i| i as f64).collect();
        if log_p.len() >= 60 {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Use the real multi-window LPPLS confidence from fatcrash-core.
                // This runs CMA-ES optimization with Sornette filter across multiple windows.
                let (conf_vec, tc_mean_vec, tc_std_vec) =
                    fatcrash_core::lppls::confidence::lppls_confidence_slice(
                        &times,
                        &log_p,
                        Some(60),   // min_window
                        None,       // max_window (auto)
                        Some(30),   // n_windows to sample
                        Some(20),   // n_candidates per window
                    );
                // Take the last value (current time step)
                let conf = conf_vec.last().copied().unwrap_or(0.0);
                let tc_mean = tc_mean_vec.last().copied().unwrap_or(f64::NAN);
                let tc_std = tc_std_vec.last().copied().unwrap_or(f64::NAN);
                let t_now = times.last().copied().unwrap_or(0.0);
                let days_to_tc = if tc_mean.is_finite() { tc_mean - t_now } else { f64::INFINITY };
                (conf, days_to_tc, tc_std)
            })) {
                Ok(r) => r,
                _ => (0.0, f64::INFINITY, f64::NAN),
            }
        } else {
            (0.0, f64::INFINITY, f64::NAN)
        }
    };
    record!("lppls_confidence", signals::lppls_confidence_signal(lppls_conf), vec![
        ("Confidence (frac windows)".into(), lppls_conf),
        ("tc std dev (days)".into(), lppls_tc_std),
    ]);
    let tc_sig = signals::tc_proximity_signal(lppls_days, 90.0);
    record!("lppls_tc_proximity", tc_sig, vec![
        ("Days to tc".into(), lppls_days),
    ]);

    // 18. GSADF bubble — uses fatcrash-core with Monte Carlo critical values
    // Compute log prices for GSADF (returned for deferred computation when run_gsadf=false)
    let log_p: Vec<f64> = pre_prices.iter().filter_map(|p| if *p > 0.0 { Some(p.ln()) } else { None }).collect();
    if run_gsadf {
        let (gsadf_sig, gsadf_raw) = compute_gsadf(&log_p);
        record!("gsadf_bubble", gsadf_sig, gsadf_raw);
    }

    // Jump risk signal — BNS z-test (Barndorff-Nielsen & Shephard)
    let mut jump_z = f64::NAN;
    let mut jump_var = f64::NAN;
    let mut jump_rv = f64::NAN;
    let jump_sig = safe_call(|| {
        let recent = &returns[n.saturating_sub(63)..];
        let (z_stat, jv) = fatcrash_core::regime::jump::compute_jump_test(recent);
        let rv = fatcrash_core::regime::realized_var::compute_realized_variance(recent);
        jump_z = z_stat;
        jump_var = jv;
        jump_rv = rv;
        signals::jump_risk_signal_converter(jv, rv)
    });
    record!("jump_risk_signal", jump_sig, vec![
        ("BNS z-statistic".into(), jump_z),
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

    (AssetScan {
        asset: entry.symbol.clone(),
        signal,
        results: res,
        components: comp,
        raw_values: raw,
        timestamp: now,
        data_points,
        prices: spark,
        error: None,
    }, log_p)
}

/// Scan all watchlist entries in parallel (max 3 threads for API rate limiting).
/// Phase 1: run everything except GSADF, send `PartialResults`.
/// Phase 2: run GSADF per asset, update components, re-aggregate, send `Done`.
pub fn scan_watchlist(
    entries: &[WatchlistEntry],
    window: usize,
    days: usize,
    use_cache: bool,
    tx: mpsc::Sender<ScanMsg>,
) {
    let total = entries.len();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

    // --- Phase 1: everything except GSADF ---
    let phase1_results = Mutex::new(Vec::with_capacity(total));
    let counter = AtomicUsize::new(0);

    let phase1_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.install(|| {
            entries.par_iter().for_each(|e| {
                let _ = tx.send(ScanMsg::Progress {
                    done: counter.load(Ordering::Relaxed),
                    total,
                    asset: e.symbol.clone(),
                });
                let (scan, log_prices) = scan_asset_inner(e, window, days, use_cache, false);
                let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = tx.send(ScanMsg::Progress {
                    done,
                    total,
                    asset: e.symbol.clone(),
                });
                phase1_results.lock().unwrap().push((scan, log_prices));
            });
        });
    }));

    let pairs: Vec<(AssetScan, Vec<f64>)> = match phase1_ok {
        Ok(()) => phase1_results.into_inner().unwrap_or_default(),
        Err(_) => phase1_results.into_inner().unwrap_or_default(),
    };

    // Sort pairs by display order (status tier, then LPPLS desc) so GSADF fills
    // in roughly top-to-bottom as the user sees the watchlist.
    let mut pairs = pairs;
    pairs.sort_by(|(a, _), (b, _)| {
        let tier = |s: &AssetScan| match s.signal.status() {
            "ALERT" => 0,
            "WATCH" => 1,
            _ => 2,
        };
        let ta = tier(a);
        let tb = tier(b);
        ta.cmp(&tb).then_with(|| {
            let la = a.signal.components.get("lppls_confidence").copied().unwrap_or(0.0);
            let lb = b.signal.components.get("lppls_confidence").copied().unwrap_or(0.0);
            lb.partial_cmp(&la).unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    // Send partial results (no GSADF yet) so the UI can display immediately.
    let partial_scans: Vec<AssetScan> = pairs.iter().map(|(s, _)| s.clone()).collect();
    let _ = tx.send(ScanMsg::PartialResults(partial_scans));

    // --- Phase 2: compute GSADF per asset, send each as it completes ---
    // Process sequentially to match display order (top assets first).
    let counter2 = AtomicUsize::new(0);

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.install(|| {
            pairs.into_par_iter().for_each(|(mut scan, log_prices)| {
                let _ = tx.send(ScanMsg::Progress {
                    done: counter2.load(Ordering::Relaxed),
                    total,
                    asset: scan.asset.clone(),
                });
                if !log_prices.is_empty() && scan.error.is_none() {
                    let (gsadf_sig, gsadf_raw) = compute_gsadf(&log_prices);
                    let detected = if gsadf_sig.is_nan() { None } else { Some(gsadf_sig > 0.5) };
                    scan.components.insert("gsadf_bubble".to_string(), gsadf_sig);
                    scan.results.insert("gsadf_bubble".to_string(), detected);
                    scan.raw_values.insert("gsadf_bubble".to_string(), gsadf_raw);
                    scan.signal = signals::aggregate_signals(&scan.components);
                }
                let done = counter2.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = tx.send(ScanMsg::Progress {
                    done,
                    total,
                    asset: scan.asset.clone(),
                });
                let _ = tx.send(ScanMsg::GsadfUpdate(Box::new(scan)));
            });
        });
    }));

    let _ = tx.send(ScanMsg::Done);
}
