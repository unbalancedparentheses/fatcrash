use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap};

use crate::signals;

use super::App;

/// Short description for each method.
fn method_description(key: &str) -> &'static str {
    match key {
        "hill_thinning" => concat!(
            "Hill tail index estimator (Hill 1975). Estimates alpha, the power-law exponent of the return distribution tail: P(X > x) ~ x^{-alpha}. ",
            "Uses the top k order statistics: alpha_hat = (1/k) * sum(ln(X_i/X_{k+1})). ",
            "We compute alpha on a recent pre-window and a baseline window, then compare. ",
            "A DROP in alpha means tails are getting fatter (more extreme events likely). ",
            "alpha < 2 => infinite variance (Levy regime). alpha < 4 => infinite kurtosis. ",
            "Gaussian has alpha -> infinity. BTC typically has alpha ~ 2.5-3.5. ",
            "Signal = max(0, 1 - alpha_pre/alpha_base), so it fires when pre-window alpha is lower than baseline.",
        ),
        "kappa_regime" => concat!(
            "Max-stability kappa metric (Taleb & Douady 2015). Tests whether the distribution belongs to the Gaussian ",
            "basin of attraction by comparing the ratio kappa = E[max(X_1..X_n)] / E[sum(|X_1..X_n|)] across subsample sizes n. ",
            "For Gaussian, kappa converges to a known benchmark. If observed kappa exceeds the benchmark, ",
            "the distribution has fatter tails than Gaussian. We test at n=5, 10, 20 and take the worst case. ",
            "The excess kappa/benchmark ratio maps to [0,1]: ratio > 1.5 is strong fat-tail signal. ",
            "This metric is particularly good at detecting regime changes because it responds to the SHAPE of the distribution, not just variance.",
        ),
        "taleb_kappa" => concat!(
            "Taleb's kappa (Taleb 2020, Statistical Consequences of Fat Tails). An alternative kappa formulation ",
            "that uses a range of subsample sizes [n0..n1] and fits the scaling behavior. ",
            "More robust than single-n kappa because it captures how the tail behavior changes with sample size. ",
            "Compared against a Gaussian bootstrap benchmark with matching mean/variance. ",
            "Taleb kappa >> benchmark indicates the distribution is in a fat-tailed regime where sample statistics ",
            "(mean, variance) converge slowly or not at all. Signal fires when ratio exceeds 1.3x benchmark.",
        ),
        "pickands_thinning" => concat!(
            "Pickands estimator (Pickands 1975). Estimates the extreme value index gamma using order statistics: ",
            "gamma = ln((X_{k} - X_{2k}) / (X_{2k} - X_{4k})) / ln(2). ",
            "gamma > 0 indicates heavy (Pareto-type) tails, gamma = 0 is exponential-type, gamma < 0 is bounded. ",
            "We compute on pre-window and baseline separately. Signal fires when pre-window gamma INCREASES ",
            "relative to baseline (tail getting heavier). Advantage over Hill: works for all tail types, not just Pareto. ",
            "Disadvantage: higher variance, needs larger samples.",
        ),
        "hurst_trending" => concat!(
            "Hurst exponent via Rescaled Range (R/S) analysis (Hurst 1951, Mandelbrot & Wallis 1969). ",
            "Measures long-range dependence: H = log(R/S) / log(n) across block sizes. ",
            "H = 0.5: independent (random walk). H > 0.5: persistent/trending (past trends continue). H < 0.5: anti-persistent/mean-reverting. ",
            "In returns, H >> 0.5 suggests herding behavior, momentum cascades, or reduced liquidity — ",
            "conditions where sell-offs can become self-reinforcing. H > 0.65 is considered a strong trending signal. ",
            "Empirically, H tends to rise before major market dislocations as volatility clusters.",
        ),
        "dfa_trending" => concat!(
            "Detrended Fluctuation Analysis (Peng et al. 1994). Measures long-range correlations more robustly than R/S. ",
            "Algorithm: divide series into blocks, fit local linear trend in each block, compute RMS of residuals F(n) vs block size n. ",
            "DFA exponent alpha = slope of log F(n) vs log n. alpha = 0.5: white noise. alpha > 0.5: persistent. alpha = 1.5: Brownian motion. ",
            "Advantage over Hurst: robust to non-stationarities, trends, and polynomial drifts. ",
            "Signal fires when alpha > 0.55, indicating returns have developed long-range dependence that ",
            "wasn't present in normal market conditions. Maps linearly: signal = (alpha - 0.5) / 0.3.",
        ),
        "deh_thinning" => concat!(
            "Dekkers-Einmahl-de Haan (DEH) moment estimator (1989). Estimates the extreme value index gamma ",
            "using the first two moments of log-spacings of upper order statistics: ",
            "gamma = M1 + 1 - (1/2)(1 - M1^2/M2)^{-1}, where M1, M2 are moment statistics. ",
            "Works for all tail types (heavy, light, bounded), unlike Hill which assumes Pareto. ",
            "We compare pre-window vs baseline gamma. An increase in gamma means the tail is getting heavier. ",
            "Gamma > 0.5 indicates very heavy tails (infinite variance territory). ",
            "This estimator is more robust to threshold choice than Hill or Pickands.",
        ),
        "qq_thinning" => concat!(
            "QQ-plot tail regression estimator. Fits a linear regression on the upper tail of the QQ plot ",
            "(empirical quantiles vs theoretical Pareto quantiles). The slope estimates 1/alpha (tail index). ",
            "We use -log(1 - F(x)) vs log(x) in the upper 15% of the distribution. ",
            "Pre-window vs baseline comparison: if the QQ slope INCREASES (alpha decreases), tails are thinning/fattening. ",
            "Advantage: visual interpretability — the QQ plot directly shows tail departure from the reference distribution. ",
            "Disadvantage: sensitive to the fraction of data used for the regression.",
        ),
        "maxsum_signal" => concat!(
            "Max-to-sum ratio (Embrechts et al. 1997). Computes R = max(|X_i|) / sum(|X_i|) for the return series. ",
            "For thin-tailed (Gaussian) distributions, R -> 0 as n -> infinity. ",
            "For fat-tailed distributions (alpha < 2), R converges to a positive constant. ",
            "A high ratio means a single extreme observation dominates the sum — classic fat-tail behavior. ",
            "For alpha-stable distributions, E[R] ~ n^{1/alpha - 1}. ",
            "Signal: R > 0.1 starts to indicate concentration. R > 0.3 is extreme. Cauchy (alpha=1) typically gives R ~ 0.3-0.5.",
        ),
        "spectral_memory" => concat!(
            "Geweke-Porter-Hudak (GPH) log-periodogram estimator (1983). Estimates the fractional differencing parameter d ",
            "by regressing log(periodogram) on log(frequency) at low frequencies. ",
            "d = 0: no long memory (iid). 0 < d < 0.5: long memory, stationary. d >= 0.5: non-stationary long memory. ",
            "Uses the lowest 65% of Fourier frequencies. Long memory in absolute/squared returns is well-documented ",
            "in financial time series (Ding, Granger & Engle 1993). An increase in d suggests volatility persistence is rising, ",
            "which precedes turbulent regimes. Signal maps d to [0,1] via (d/0.5), clamped.",
        ),
        "momentum_reversal" => concat!(
            "Momentum-reversal detector. Computes short-term (21-day) and long-term (90-day) log-price momentum, ",
            "then measures divergence: reversal = -(short_mom - long_mom) when short is decelerating. ",
            "A positive reversal score means recent prices have reversed against the longer trend — ",
            "for a prior uptrend, this means the rally is unwinding. ",
            "Signal fires when reversal > 0 (mapped via tanh). Strong reversals (> 0.1) indicate ",
            "potential trend exhaustion. This captures the DeLong et al. (1990) noise trader feedback dynamic.",
        ),
        "rv_spike" => concat!(
            "Realized variance spike detector. Computes RV = sum(r_i^2) over short (21-day) and long (126-day) windows. ",
            "The ratio RV_short / RV_long measures whether volatility has recently increased. ",
            "Ratio = 1: volatility is at its average level. Ratio > 2: vol has doubled vs recent history. ",
            "Signal = max(0, (ratio - 1) / 3), so it takes a 4x spike to saturate at 1.0. ",
            "This is the simplest volatility regime indicator — it doesn't model regimes explicitly, ",
            "just detects whether current vol is elevated. Often the first signal to fire in a sell-off.",
        ),
        "velocity_spike" => concat!(
            "Volatility velocity (vol-of-vol proxy). Computes the fractional change in realized volatility: ",
            "v = (vol_current - vol_past) / vol_past, where vol = sqrt(sum(r^2) / window). ",
            "Current window = last 21 days, past = 21 days ending 5 days ago. ",
            "v > 0: vol is accelerating. v < 0: vol is decelerating. ",
            "Rapid vol acceleration (v > 0.3) precedes turbulence because market makers widen spreads, ",
            "triggering stop-losses and margin calls (Brunnermeier & Pedersen 2009 liquidity spiral). ",
            "Signal = tanh(max(0, velocity)), so only positive acceleration counts.",
        ),
        "hamilton_stress" => concat!(
            "Hamilton 2-state regime-switching model (Hamilton 1989). Fits a Hidden Markov Model with two Gaussian states ",
            "(calm mu_0/sigma_0 and stressed mu_1/sigma_1) via Expectation-Maximization with multiple random restarts. ",
            "Transition probabilities p00, p11 capture regime persistence. ",
            "Output: P(S_t = stressed | data), the filtered probability of being in the high-vol state at the last observation. ",
            "P > 0.5: more likely stressed than calm. P > 0.8: strong stress signal. ",
            "EM runs 5 restarts x 80 iterations. The stressed state is identified as the one with higher sigma. ",
            "Key advantage: endogenous regime detection without arbitrary thresholds.",
        ),
        "csd_warning" => concat!(
            "Critical Slowing Down (CSD) indicator (Scheffer et al. 2009, 2012). Based on the theory that complex systems ",
            "approaching a tipping point exhibit two signatures: (1) rising autocorrelation (AR(1) coefficient increases) and ",
            "(2) rising variance in the state variable. We compute both on rolling 21-day realized volatility, split into two halves. ",
            "AR(1) rate of change = (ar1_second - ar1_first) / |ar1_first|. Variance rate of change = (var_second - var_first) / var_first. ",
            "If both are positive and significant, the system is slowing down — perturbations take longer to decay. ",
            "Signal = clamp((ar1_roc + var_roc) / 2, 0, 1). This is a leading indicator: it fires BEFORE the regime shift.",
        ),
        "amihud_spike" => concat!(
            "Amihud illiquidity ratio (Amihud 2002). ILLIQ = (1/N) * sum(|r_t| / volume_t). ",
            "Higher ILLIQ = less liquid (large price impact per unit of volume). ",
            "We compute ILLIQ over a short window (21d) and the full analysis window, then compare. ",
            "Ratio short/long > 1: liquidity is deteriorating. Ratio > 2: significant dry-up. ",
            "Liquidity withdrawal is one of the most reliable crash precursors — it causes bid-ask spreads to widen, ",
            "which triggers stop-losses, forced selling, and further liquidity withdrawal (Kyle & Obizhaeva 2016). ",
            "Signal = clamp((ratio - 1) / 2, 0, 1). Assets with zero volume (e.g., some CoinGecko data) return NaN.",
        ),
        "lppls_confidence" => concat!(
            "Log-Periodic Power Law Singularity model (Sornette et al. 1996, Johansen-Ledoit-Sornette 2000). ",
            "Fits ln(p(t)) = A + B*(tc-t)^m + C*(tc-t)^m*cos(omega*ln(tc-t) + phi) via grid search over (tc, m, omega) ",
            "with linear least-squares for (A, B, C, phi) at each grid point. ",
            "Fit improvement = 1 - SSE_lppls / SSE_linear. High improvement means the oscillatory power-law singularity ",
            "fits much better than a simple trend, suggesting bubble dynamics with log-periodic oscillations. ",
            "Requires B < 0 (finite-time singularity) and m in (0.1, 0.9). ",
            "Signal > 0.5 if the LPPLS explains > 50% more variance than a linear fit. Weight = 0.22 (highest in the system).",
        ),
        "lppls_tc_proximity" => concat!(
            "LPPLS critical time proximity. From the LPPLS fit, tc is the estimated time of the most probable regime change. ",
            "The model predicts that the bubble's growth rate diverges as t -> tc. ",
            "Days to tc = tc - t_now. If tc is in the past, the singularity already occurred. ",
            "Signal = max(0, 1 - days_to_tc / horizon) where horizon = 90 days. ",
            "tc < 30 days out is urgent. tc < 7 days is critical. tc in the past means the model thinks the crash window has arrived. ",
            "Note: tc is NOT a precise crash date prediction — it marks the end of the faster-than-exponential growth regime, ",
            "after which a crash becomes most probable but is not guaranteed (Sornette 2003).",
        ),
        "gsadf_bubble" => concat!(
            "Generalized Sup ADF (GSADF) test (Phillips, Shi & Yu 2015). Tests for explosive behavior in asset prices ",
            "by running Augmented Dickey-Fuller regressions on ALL possible subsamples [r1, r2] of the price series. ",
            "ADF regression: delta_p_t = alpha + beta * p_{t-1} + eps. Under H0 (unit root), beta = 0. ",
            "Under H1 (explosive), beta > 0 and the t-stat exceeds right-tail critical values. ",
            "GSADF = max over all subsamples of ADF t-statistics. Critical values from Monte Carlo simulation of random walks. ",
            "GSADF > CV_95: reject unit root in favor of explosive behavior (bubble detected at 5%). ",
            "The excess (GSADF - CV) measures bubble strength. This test can date-stamp bubble origination and collapse.",
        ),
        "jump_risk_signal" => concat!(
            "Jump risk decomposition via bipower variation (Barndorff-Nielsen & Shephard 2004, 2006). ",
            "Realized variance RV = sum(r_t^2) captures total variation. Bipower variation BV = (pi/2) * sum(|r_t| * |r_{t-1}|) / (n-1) ",
            "estimates the continuous (diffusive) component because it is robust to jumps. ",
            "Jump variance JV = max(RV - BV, 0) isolates the discontinuous component. ",
            "Jump fraction = JV / RV. High fraction (> 0.3) means returns are driven by discrete jumps, not smooth diffusion. ",
            "Jumps indicate Poisson-type tail risk that cannot be hedged with delta hedging (Merton 1976). ",
            "Signal = clamp(JV/RV, 0, 1). Recent 63-day window.",
        ),
        "multiscale" => concat!(
            "Multiscale tail agreement. Computes the Hill tail index alpha at three return frequencies: ",
            "daily (1d), 3-day aggregated, and 7-day aggregated returns. ",
            "If tails are genuinely fat (not just a few outliers), alpha should be consistently low across all time scales. ",
            "This exploits temporal aggregation: for iid alpha-stable, alpha is scale-invariant. ",
            "For Gaussian + occasional outlier, alpha appears low at 1d but normalizes at 7d (CLT kicks in). ",
            "Signal = geometric mean of per-scale signals, where each scale maps alpha to (1 - alpha/5). ",
            "Agreement across all three scales is strong evidence of genuine power-law tails, not artifacts.",
        ),
        "mlnn_signal" | "plnn_signal" => "Neural network bubble detector (M-LNN / P-LNN). These require trained PyTorch models and are only available via the Python interface. In TUI mode, they always return NaN.",
        _ => "No description available for this method.",
    }
}

/// Pretty-print a signal key.
fn pretty_name(key: &str) -> String {
    key.replace('_', " ")
        .split(' ')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(first) => {
                    let upper: String = first.to_uppercase().collect();
                    upper + c.as_str()
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Which category does this method belong to?
fn method_category(key: &str) -> &'static str {
    let cats = signals::categories();
    for (cat, members) in &cats {
        if members.contains(&key) {
            return cat;
        }
    }
    "other"
}

/// Render the method drill-down view.
pub fn render(f: &mut Frame, app: &App, scan_idx: usize, method_key: &str) {
    let scan = match app.scans.get(scan_idx) {
        Some(s) => s,
        None => {
            let msg = Paragraph::new("No scan data.")
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(msg, f.area());
            return;
        }
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Length(8),  // description
            Constraint::Min(8),    // raw values table
            Constraint::Length(3),  // footer
        ])
        .split(f.area());

    // Header
    let signal_val = scan.components.get(method_key).copied().unwrap_or(f64::NAN);
    let detected = scan.results.get(method_key).and_then(|v| *v);
    let weights = signals::default_weights();
    let weight = weights.get(method_key).copied().unwrap_or(0.0);
    let category = method_category(method_key);

    let det_str = match detected {
        Some(true) => "YES",
        Some(false) => "no",
        None => "-",
    };
    let det_color = match detected {
        Some(true) => Color::Red,
        Some(false) => Color::Green,
        None => Color::DarkGray,
    };

    let sig_str = if signal_val.is_nan() { "-".to_string() } else { format!("{:.4}", signal_val) };

    let header_text = Line::from(vec![
        Span::styled(
            format!(" {} ", pretty_name(method_key)),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | "),
        Span::raw(format!("Signal: {} ", sig_str)),
        Span::raw(" | "),
        Span::styled(format!("Detected: {} ", det_str), Style::default().fg(det_color)),
        Span::raw(" | "),
        Span::raw(format!("Weight: {:.2} ", weight)),
        Span::raw(" | "),
        Span::raw(format!("Category: {} ", category)),
        Span::raw(" | "),
        Span::raw(format!("Asset: {}", scan.asset)),
    ]);
    let header = Paragraph::new(header_text)
        .block(Block::default().borders(Borders::BOTTOM));
    f.render_widget(header, chunks[0]);

    // Description
    let desc = method_description(method_key);
    let desc_widget = Paragraph::new(desc)
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .title(" What is this? ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    f.render_widget(desc_widget, chunks[1]);

    // Raw values table
    let raw_vals = scan.raw_values.get(method_key);

    if let Some(vals) = raw_vals {
        let header_cells = ["Metric", "Value"]
            .iter()
            .map(|h| {
                Cell::from(*h).style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )
            });
        let table_header = Row::new(header_cells).height(1).bottom_margin(1);

        let rows: Vec<Row> = vals
            .iter()
            .map(|(label, val)| {
                let val_str = if val.is_nan() {
                    "-".to_string()
                } else if val.is_infinite() {
                    "\u{221e}".to_string()
                } else if val.abs() < 0.001 && *val != 0.0 {
                    format!("{:.2e}", val)
                } else {
                    format!("{:.6}", val)
                };

                let val_color = if val.is_nan() || val.is_infinite() {
                    Color::DarkGray
                } else {
                    Color::White
                };

                Row::new(vec![
                    Cell::from(label.as_str()),
                    Cell::from(val_str).style(Style::default().fg(val_color)),
                ])
            })
            .collect();

        let table = Table::new(
            rows,
            [Constraint::Length(30), Constraint::Length(20)],
        )
        .header(table_header)
        .block(
            Block::default()
                .title(" Raw Values ")
                .borders(Borders::ALL),
        );
        f.render_widget(table, chunks[2]);
    } else {
        let no_data = Paragraph::new(" No raw values recorded for this method.")
            .style(Style::default().fg(Color::DarkGray))
            .block(
                Block::default()
                    .title(" Raw Values ")
                    .borders(Borders::ALL),
            );
        f.render_widget(no_data, chunks[2]);
    }

    // Footer
    let footer = Paragraph::new(format!(
        " \u{2190}=back to {}  r=refresh  q=quit",
        scan.asset
    ))
    .style(Style::default().fg(Color::DarkGray))
    .block(Block::default().borders(Borders::TOP));
    f.render_widget(footer, chunks[3]);
}
