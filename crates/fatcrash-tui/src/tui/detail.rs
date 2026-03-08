use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Sparkline, Table, Wrap};

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

/// Build the flat list of (category_name, method_key, signal_value) for display.
fn method_rows(app: &App, scan_idx: usize) -> Vec<(&'static str, &'static str, f64)> {
    let scan = match app.scans.get(scan_idx) {
        Some(s) => s,
        None => return vec![],
    };
    let cats = signals::confirmation_categories();
    let mut rows = Vec::new();
    for (cat_name, keys) in &cats {
        for key in keys {
            let val = scan.components.get(*key).copied().unwrap_or(0.0).max(0.0);
            rows.push((*cat_name, *key, val));
        }
    }
    rows
}

/// How many selectable rows in the detail view: all individual methods.
pub fn method_count(app: &App, scan_idx: usize) -> usize {
    method_rows(app, scan_idx).len()
}

/// Get the key of the currently selected method in the detail view.
pub fn selected_method_key(app: &App, scan_idx: usize) -> Option<String> {
    let rows = method_rows(app, scan_idx);
    rows.get(app.method_selected).map(|(_, key, _)| key.to_string())
}

/// Render the per-asset detail view with bubble panel + confirmation categories.
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
            Constraint::Length(7),  // bubble panel
            Constraint::Min(10),   // confirmation categories
            Constraint::Length(3),  // footer
        ])
        .split(f.area());

    // Header
    let status = scan.signal.status();
    let status_color = match status {
        "ALERT" => app.theme.signal_high,
        "WATCH" => app.theme.signal_mid,
        _ => app.theme.signal_low,
    };

    let header_text = format!(
        " {}  |  Status: {}  |  Confirms: {}/5  |  Data points: {}",
        scan.asset,
        status,
        scan.signal.n_confirming,
        scan.data_points,
    );
    let header = Paragraph::new(header_text)
        .style(
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(app.theme.border)));
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
        let price_color = if change_pct >= 0.0 { app.theme.spark_up } else { app.theme.spark_down };

        let spark = Sparkline::default()
            .block(
                Block::default()
                    .title(format!(" Price: {:.2} ({:+.1}%) ", last_price, change_pct))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(app.theme.border)),
            )
            .data(&data)
            .style(Style::default().fg(price_color));
        f.render_widget(spark, sparkline_area);
    } else {
        let no_data = Paragraph::new(" No price data available")
            .block(Block::default().title(" Price ").borders(Borders::ALL).border_style(Style::default().fg(app.theme.border)));
        f.render_widget(no_data, sparkline_area);
    }

    // Bubble panel
    let lppls = scan.signal.components.get("lppls_confidence").copied().unwrap_or(0.0).max(0.0);
    let tc_days = scan.signal.horizon_days;
    let tc_std = scan.raw_values.get("lppls_confidence")
        .and_then(|vals| vals.iter().find(|(k, _)| k == "tc std dev (days)").map(|(_, v)| *v))
        .unwrap_or(f64::NAN);
    let gsadf_stat = scan.raw_values.get("gsadf_bubble")
        .and_then(|vals| vals.iter().find(|(k, _)| k == "GSADF statistic").map(|(_, v)| *v))
        .unwrap_or(f64::NAN);
    let gsadf_cv = scan.raw_values.get("gsadf_bubble")
        .and_then(|vals| vals.iter().find(|(k, _)| k == "95% critical value").map(|(_, v)| *v))
        .unwrap_or(f64::NAN);
    let gsadf_excess = if gsadf_stat.is_finite() && gsadf_cv.is_finite() {
        gsadf_stat - gsadf_cv
    } else {
        f64::NAN
    };

    let lppls_color = if lppls > 0.7 { app.theme.signal_high } else if lppls > 0.3 { app.theme.signal_mid } else { app.theme.signal_low };
    let gsadf_color = if gsadf_excess.is_finite() && gsadf_excess > 0.0 { app.theme.signal_high } else { app.theme.signal_low };

    let fmt_f = |v: f64| -> String {
        if v.is_nan() || v.is_infinite() { "-".to_string() } else { format!("{:.2}", v) }
    };

    let bubble_text = vec![
        Line::from(vec![
            Span::raw("  LPPLS confidence: "),
            Span::styled(format!("{:.0}%", lppls * 100.0), Style::default().fg(lppls_color).add_modifier(Modifier::BOLD)),
            Span::raw(format!("    tc: {} days    tc std: {}", fmt_f(tc_days), fmt_f(tc_std))),
        ]),
        Line::from(vec![
            Span::raw("  GSADF statistic:  "),
            Span::styled(fmt_f(gsadf_stat), Style::default().fg(gsadf_color).add_modifier(Modifier::BOLD)),
            Span::raw(format!("        95% CV: {}    excess: {}", fmt_f(gsadf_cv), fmt_f(gsadf_excess))),
        ]),
    ];

    let bubble_panel = Paragraph::new(bubble_text)
        .block(
            Block::default()
                .title(" Bubble Detectors ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(app.theme.title)),
        );
    f.render_widget(bubble_panel, chunks[2]);

    // -- Confirmations: two-column layout --
    let rows_data = method_rows(app, scan_idx);

    let details = signals::category_details(&scan.components);
    let n_confirming = details.iter().filter(|(_, v, _)| *v > 0.5).count();
    let confirming_names: Vec<&str> = details
        .iter()
        .filter(|(_, v, _)| *v > 0.5)
        .map(|(name, _, _)| *name)
        .collect();

    let confirm_summary = if n_confirming > 0 {
        format!(
            " Confirmations: {}/5 ({}) ",
            n_confirming,
            confirming_names.join(", ")
        )
    } else {
        " Confirmations: 0/5 ".to_string()
    };

    // Outer block with summary title
    let outer_block = Block::default()
        .title(confirm_summary)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border));
    let inner = outer_block.inner(chunks[3]);
    f.render_widget(outer_block, chunks[3]);

    // Split into left column | 1-char separator | right column
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Length(1),
            Constraint::Percentage(50),
        ])
        .split(inner);

    // Render vertical separator
    let sep_lines: String = (0..cols[1].height)
        .map(|_| "\u{2502}")
        .collect::<Vec<_>>()
        .join("\n");
    let sep = Paragraph::new(sep_lines).style(Style::default().fg(app.theme.border));
    f.render_widget(sep, cols[1]);

    // Assign categories to columns: left = Tail risk + Momentum, right = rest
    let left_cats = ["Tail risk", "Momentum"];

    let mut left_rows: Vec<Row> = Vec::new();
    let mut right_rows: Vec<Row> = Vec::new();
    let mut last_left_cat = "";
    let mut last_right_cat = "";

    for (flat_idx, (cat_name, method_key, signal)) in rows_data.iter().enumerate() {
        let is_left = left_cats.contains(cat_name);
        let (rows, last_cat) = if is_left {
            (&mut left_rows, &mut last_left_cat)
        } else {
            (&mut right_rows, &mut last_right_cat)
        };

        // Category header row when category changes
        if *cat_name != *last_cat {
            *last_cat = cat_name;
            rows.push(
                Row::new(vec![
                    Cell::from(format!(" {}", cat_name)).style(
                        Style::default()
                            .fg(app.theme.title)
                            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                    ),
                    Cell::from(""),
                ]),
            );
        }

        // Method row
        let sig_color = if *signal > 0.7 {
            app.theme.signal_high
        } else if *signal > 0.5 {
            app.theme.signal_mid
        } else if *signal > 0.01 {
            app.theme.text
        } else {
            app.theme.text_dim
        };

        let is_selected = flat_idx == app.method_selected;
        let row_style = if is_selected {
            Style::default()
                .bg(app.theme.selected_bg)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        rows.push(
            Row::new(vec![
                Cell::from(format!("  {}", pretty_name(method_key))),
                Cell::from(format!("{:.2}", signal))
                    .style(Style::default().fg(sig_color)),
            ])
            .style(row_style),
        );
    }

    let col_widths = [Constraint::Fill(1), Constraint::Length(6)];
    let left_table = Table::new(left_rows, col_widths);
    f.render_widget(left_table, cols[0]);
    let right_table = Table::new(right_rows, col_widths);
    f.render_widget(right_table, cols[2]);

    // Footer
    let error_str = match &scan.error {
        Some(e) => format!("  |  Error: {}", e),
        None => String::new(),
    };

    let footer_text = format!(
        " {}{}  |  \u{2190}=back  \u{2192}=inspect method  \u{2191}\u{2193}=select  r=refresh  q=quit",
        scan.timestamp.format("%H:%M:%S UTC"),
        error_str,
    );
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(app.theme.text_dim))
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(app.theme.border)));
    f.render_widget(footer, chunks[4]);
}
