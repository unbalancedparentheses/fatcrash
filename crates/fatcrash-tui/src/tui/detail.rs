use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Sparkline, Table, Wrap};

use crate::scanner::AssetScan;
use crate::signals;

use super::App;

/// Pretty-print a signal key as a human-readable name.
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

/// Build sorted display rows for a scan: (key, signal_value, detected, weight).
/// Only includes methods with weight > 0 (or all if <= 30 total).
fn sorted_methods(scan: &AssetScan) -> Vec<(String, f64, Option<bool>, f64)> {
    let weights = signals::default_weights();
    let mut rows: Vec<(String, f64, Option<bool>, f64)> = Vec::new();

    for (name, &weight) in &weights {
        let signal_val = scan.components.get(*name).copied().unwrap_or(f64::NAN);
        let detected = scan.results.get(*name).and_then(|v| *v);
        rows.push((name.to_string(), signal_val, detected, weight));
    }

    for (name, detected) in &scan.results {
        if !weights.contains_key(name.as_str()) {
            let signal_val = scan.components.get(name).copied().unwrap_or(f64::NAN);
            rows.push((name.clone(), signal_val, *detected, 0.0));
        }
    }

    rows.sort_by(|a, b| {
        b.3.partial_cmp(&a.3)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let av = if a.1.is_nan() { -1.0 } else { a.1 };
                let bv = if b.1.is_nan() { -1.0 } else { b.1 };
                bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.0.cmp(&b.0))
    });

    // Always hide NN-based signals (not available in TUI mode)
    rows.retain(|(name, _, _, _)| name != "mlnn_signal" && name != "plnn_signal");

    // Filter zero-weight if there are many
    if rows.len() > 30 {
        rows.retain(|(_, _, _, w)| *w > 0.0);
    }

    rows
}

/// How many method rows are displayed for a given scan.
pub fn method_count(app: &App, scan_idx: usize) -> usize {
    app.scans
        .get(scan_idx)
        .map(|s| sorted_methods(s).len())
        .unwrap_or(0)
}

/// Get the key of the currently selected method in the detail view.
pub fn selected_method_key(app: &App, scan_idx: usize) -> Option<String> {
    let scan = app.scans.get(scan_idx)?;
    let methods = sorted_methods(scan);
    methods.get(app.method_selected).map(|(k, _, _, _)| k.clone())
}

/// Render the per-asset detail view.
pub fn render(f: &mut Frame, app: &mut App, scan_idx: usize) {
    let scan = match app.scans.get(scan_idx) {
        Some(s) => s,
        None => {
            let msg = Paragraph::new("No data for this asset.")
                .block(Block::default().title(" Detail ").borders(Borders::ALL));
            f.render_widget(msg, f.area());
            return;
        }
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Length(6),  // sparkline
            Constraint::Min(10),   // methods table
            Constraint::Length(4),  // footer
        ])
        .split(f.area());

    // Header
    let prob = scan.signal.probability;
    let level = scan.signal.level();
    let level_color = match level {
        "CRITICAL" | "HIGH" => Color::Red,
        "ELEVATED" => Color::Yellow,
        _ => Color::Green,
    };

    let header_text = format!(
        " {}  |  Prob: {:.1}%  |  Level: {}  |  Agreeing: {}  |  Data points: {}",
        scan.asset,
        prob * 100.0,
        level,
        scan.signal.n_agreeing,
        scan.data_points,
    );
    let header = Paragraph::new(header_text)
        .style(
            Style::default()
                .fg(level_color)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::default().borders(Borders::BOTTOM));
    f.render_widget(header, chunks[0]);

    // Sparkline
    let sparkline_area = chunks[1];
    if !scan.prices.is_empty() {
        let n = scan.prices.len().min(sparkline_area.width as usize);
        let tail = &scan.prices[scan.prices.len() - n..];
        let min = tail.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = tail.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let data: Vec<u64> = if range > 1e-10 {
            tail.iter()
                .map(|&p| ((p - min) / range * 100.0).round() as u64)
                .collect()
        } else {
            vec![50; n]
        };

        let last_price = scan.prices.last().copied().unwrap_or(0.0);
        let first_price = if scan.prices.len() > 1 { scan.prices[0] } else { last_price };
        let change_pct = if first_price > 0.0 {
            (last_price - first_price) / first_price * 100.0
        } else {
            0.0
        };
        let price_color = if change_pct >= 0.0 { Color::Green } else { Color::Red };

        let spark = Sparkline::default()
            .block(
                Block::default()
                    .title(format!(" Price: {:.2} ({:+.1}%) ", last_price, change_pct))
                    .borders(Borders::ALL),
            )
            .data(&data)
            .style(Style::default().fg(price_color));
        f.render_widget(spark, sparkline_area);
    } else {
        let no_data = Paragraph::new(" No price data available")
            .block(Block::default().title(" Price ").borders(Borders::ALL));
        f.render_widget(no_data, sparkline_area);
    }

    // Methods table with scroll
    let display_rows = sorted_methods(scan);
    let total_rows = display_rows.len();

    // Visible rows: table area height - borders(2) - header(1) - header margin(1)
    let visible = (chunks[2].height as usize).saturating_sub(4);

    // Adjust scroll offset to keep selection visible
    if app.method_selected < app.detail_offset {
        app.detail_offset = app.method_selected;
    } else if app.method_selected >= app.detail_offset + visible && visible > 0 {
        app.detail_offset = app.method_selected - visible + 1;
    }

    let offset = app.detail_offset;
    let end = (offset + visible).min(total_rows);

    let header_cells = ["Method", "Signal", "Detected", "Weight"]
        .iter()
        .map(|h| {
            Cell::from(*h).style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        });
    let table_header = Row::new(header_cells).height(1).bottom_margin(1);

    let rows: Vec<Row> = display_rows[offset..end]
        .iter()
        .enumerate()
        .map(|(vi, (name, sig, det, weight))| {
            let i = offset + vi;
            let sig_str = if sig.is_nan() {
                "-".to_string()
            } else {
                format!("{:.3}", sig)
            };

            let det_str = match det {
                Some(true) => "\u{2713}",
                Some(false) => "\u{2717}",
                None => "-",
            };

            let sig_color = if sig.is_nan() {
                Color::DarkGray
            } else if *sig > 0.7 {
                Color::Red
            } else if *sig > 0.5 {
                Color::Yellow
            } else {
                Color::Green
            };

            let det_color = match det {
                Some(true) => Color::Red,
                Some(false) => Color::Green,
                None => Color::DarkGray,
            };

            let is_selected = i == app.method_selected;
            let row_style = if is_selected {
                Style::default().bg(Color::Rgb(40, 40, 60)).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(pretty_name(name)),
                Cell::from(sig_str).style(Style::default().fg(sig_color)),
                Cell::from(det_str).style(Style::default().fg(det_color)),
                Cell::from(format!("{:.2}", weight)),
            ])
            .style(row_style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(22),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(8),
        ],
    )
    .header(table_header)
    .block(
        Block::default()
            .title(" Methods (sorted by weight) ")
            .borders(Borders::ALL),
    );
    f.render_widget(table, chunks[2]);

    // Footer
    let agreeing = signals::agreeing_categories(&scan.components);
    let cats_str = if agreeing.is_empty() {
        "none".to_string()
    } else {
        agreeing.join(", ")
    };

    let error_str = match &scan.error {
        Some(e) => format!("  |  Error: {}", e),
        None => String::new(),
    };

    let footer_text = format!(
        " Agreeing: {}{}  |  {}  |  \u{2190}=back  \u{2192}=inspect  \u{2191}\u{2193}=select  r=refresh  q=quit",
        cats_str,
        error_str,
        scan.timestamp.format("%H:%M:%S UTC"),
    );
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::DarkGray))
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::TOP));
    f.render_widget(footer, chunks[3]);
}
