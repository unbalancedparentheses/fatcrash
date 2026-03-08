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

/// Build a signal bar like `████████░░` for a 0.0–1.0 value.
fn signal_bar(value: f64, width: usize) -> String {
    let clamped = value.clamp(0.0, 1.0);
    let filled = (clamped * width as f64).round() as usize;
    let empty = width - filled;
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
}

/// Build the flat list of (category_name, method_key, signal_value) for display.
/// Categories are sorted by their max signal value (hottest first).
/// Methods within each category are sorted by signal value descending.
fn method_rows(app: &App, scan_idx: usize) -> Vec<(&'static str, &'static str, f64)> {
    let scan = match app.scans.get(scan_idx) {
        Some(s) => s,
        None => return vec![],
    };
    let cats = signals::confirmation_categories();

    // Build per-category groups with their max signal for sorting.
    let mut groups: Vec<(f64, Vec<(&'static str, &'static str, f64)>)> = cats
        .iter()
        .map(|(cat_name, keys)| {
            let mut methods: Vec<(&'static str, &'static str, f64)> = keys
                .iter()
                .map(|key| {
                    let val = scan.components.get(*key).copied().unwrap_or(0.0).max(0.0);
                    (*cat_name, *key, val)
                })
                .collect();
            methods.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let max_sig = methods.iter().map(|m| m.2).fold(0.0_f64, f64::max);
            (max_sig, methods)
        })
        .collect();

    groups.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    groups.into_iter().flat_map(|(_, methods)| methods).collect()
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
            Constraint::Length(8),  // sparkline + bubble (side by side)
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

    let gsadf_header = if app.gsadf_computing.contains(&scan.asset) {
        "  |  GSADF: computing..."
    } else if scan.components.contains_key("gsadf_bubble") {
        ""
    } else {
        "  |  GSADF: press g"
    };
    let header_text = format!(
        " {}  |  Status: {}  |  Confirms: {}/5  |  Data points: {}{}",
        scan.asset,
        status,
        scan.signal.n_confirming,
        scan.data_points,
        gsadf_header,
    );
    let header = Paragraph::new(header_text)
        .style(
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(app.theme.border)));
    f.render_widget(header, chunks[0]);

    // Side-by-side: sparkline (left) + bubble panel (right)
    let top_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(55),
            Constraint::Percentage(45),
        ])
        .split(chunks[1]);

    // Sparkline (left)
    let sparkline_area = top_cols[0];
    if !scan.prices.is_empty() {
        let n = scan.prices.len().min(sparkline_area.width.saturating_sub(2) as usize);
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

    // Bubble panel (right)
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

    let dim = Style::default().fg(app.theme.text_dim);
    let is_computing_gsadf = app.gsadf_computing.contains(&scan.asset);
    let gsadf_row = if is_computing_gsadf {
        Row::new(vec![
            Cell::from("GSADF").style(dim),
            Cell::from("computing...").style(Style::default().fg(app.theme.signal_mid).add_modifier(Modifier::BOLD)),
            Cell::from(""),
            Cell::from(""),
        ])
    } else {
        Row::new(vec![
            Cell::from("GSADF").style(dim),
            Cell::from(fmt_f(gsadf_stat)).style(Style::default().fg(gsadf_color).add_modifier(Modifier::BOLD)),
            Cell::from(format!("CV: {}", fmt_f(gsadf_cv))),
            Cell::from(format!("ex: {}", fmt_f(gsadf_excess))).style(dim),
        ])
    };
    let bubble_rows = vec![
        Row::new(vec![
            Cell::from("LPPLS").style(dim),
            Cell::from(format!("{:.0}%", lppls * 100.0)).style(Style::default().fg(lppls_color).add_modifier(Modifier::BOLD)),
            Cell::from(format!("tc: {}", fmt_f(tc_days))),
            Cell::from(format!("\u{00b1}{}", fmt_f(tc_std))).style(dim),
        ]),
        gsadf_row,
    ];

    let bubble_table = Table::new(
        bubble_rows,
        [Constraint::Length(6), Constraint::Length(6), Constraint::Length(10), Constraint::Fill(1)],
    )
    .block(
        Block::default()
            .title(" Bubble Detectors ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(app.theme.title)),
    );
    f.render_widget(bubble_table, top_cols[1]);

    // -- Confirmations: single-column with scrolling --
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

    // Build all display rows, tracking which display-row index corresponds to each method.
    // (display_row_index, is_method, flat_method_index)
    let mut all_rows: Vec<(Row, Option<usize>)> = Vec::new();
    let mut last_cat = "";

    for (flat_idx, (cat_name, method_key, signal)) in rows_data.iter().enumerate() {
        if *cat_name != last_cat {
            last_cat = cat_name;
            all_rows.push((
                Row::new(vec![
                    Cell::from(format!(" {}", cat_name)).style(
                        Style::default()
                            .fg(app.theme.title)
                            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                    ),
                    Cell::from(""),
                    Cell::from(""),
                ]),
                None,
            ));
        }

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
        let is_zero = *signal < 0.01;
        let name_color = if is_zero { app.theme.text_dim } else { app.theme.text };
        let row_style = if is_selected {
            Style::default()
                .bg(app.theme.selected_bg)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        let bar = signal_bar(*signal, 10);

        all_rows.push((
            Row::new(vec![
                Cell::from(format!("  {}", pretty_name(method_key)))
                    .style(Style::default().fg(name_color)),
                Cell::from(bar).style(Style::default().fg(sig_color)),
                Cell::from(if is_zero { "\u{00b7}".to_string() } else { format!("{:.2}", signal) })
                    .style(Style::default().fg(sig_color)),
            ])
            .style(row_style),
            Some(flat_idx),
        ));
    }

    // Find the display-row index of the selected method for scrolling.
    let selected_display_row = all_rows
        .iter()
        .position(|(_, m)| *m == Some(app.method_selected))
        .unwrap_or(0);

    // Visible rows: table area height - borders(2) - no header
    let visible = (chunks[2].height as usize).saturating_sub(2);

    if selected_display_row < app.detail_offset {
        app.detail_offset = selected_display_row;
    } else if visible > 0 && selected_display_row >= app.detail_offset + visible {
        app.detail_offset = selected_display_row - visible + 1;
    }

    let offset = app.detail_offset;
    let end = (offset + visible).min(all_rows.len());
    let visible_rows: Vec<Row> = all_rows
        .into_iter()
        .skip(offset)
        .take(end - offset)
        .map(|(row, _)| row)
        .collect();

    let table = Table::new(
        visible_rows,
        [Constraint::Fill(1), Constraint::Length(10), Constraint::Length(6)],
    )
    .block(
        Block::default()
            .title(confirm_summary)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(app.theme.border)),
    );
    f.render_widget(table, chunks[2]);

    // Footer
    let error_str = match &scan.error {
        Some(e) => format!("  |  Error: {}", e),
        None => String::new(),
    };

    let footer_text = format!(
        " {}{}  |  \u{2190}=back  \u{2192}=inspect method  \u{2191}\u{2193}=select  g=GSADF  r=refresh  q=quit",
        scan.timestamp.format("%H:%M:%S UTC"),
        error_str,
    );
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(app.theme.text_dim))
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(app.theme.border)));
    f.render_widget(footer, chunks[3]);
}
