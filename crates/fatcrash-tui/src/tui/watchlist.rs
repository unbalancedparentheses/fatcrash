use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};

use super::App;

/// Render the main watchlist table view.
pub fn render(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title bar
            Constraint::Min(10),   // main table
            Constraint::Length(3), // status bar
        ])
        .split(f.area());

    // Title bar
    let gsadf_count = app.gsadf_computing.len();
    let scanning_indicator = if app.scanning && app.scan_total > 0 {
        let eta = format_eta(app);
        format!(
            " [scanning {} {}/{}{}]",
            app.scan_current_asset, app.scan_done, app.scan_total, eta
        )
    } else if app.scanning {
        " [scanning...]".to_string()
    } else {
        String::new()
    };
    let title = Paragraph::new(format!(
        " fatcrash-tui  |  window={}  days={}{}",
        app.window, app.days, scanning_indicator
    ))
    .style(Style::default().fg(app.theme.title).add_modifier(Modifier::BOLD))
    .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(app.theme.border)));
    f.render_widget(title, chunks[0]);

    // Main table area
    if app.scans.is_empty() {
        let msg = if app.scanning {
            if app.scan_total > 0 {
                let bar = progress_bar(app.scan_done, app.scan_total, 15);
                let eta = format_eta(app);
                format!(
                    "Scanning {}... ({}/{})  {}{}\n",
                    app.scan_current_asset, app.scan_done, app.scan_total, bar, eta
                )
            } else {
                "Scanning assets... please wait".to_string()
            }
        } else {
            "No data yet. Press r to scan.".to_string()
        };
        let loading = Paragraph::new(msg)
            .alignment(Alignment::Center)
            .style(Style::default().fg(app.theme.text_dim).add_modifier(Modifier::ITALIC))
            .block(
                Block::default()
                    .title(" Watchlist ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(app.theme.border)),
            );
        f.render_widget(loading, chunks[1]);
    } else {
        // Collect sorted indices to avoid holding borrow on app
        let sorted_indices: Vec<usize> = {
            let sorted = app.sorted_scans();
            sorted.iter().map(|(idx, _)| *idx).collect()
        };
        let total_rows = sorted_indices.len();

        // Compute visible rows: table area height - borders(2) - header(1) - header margin(1)
        let visible = (chunks[1].height as usize).saturating_sub(4);

        // Adjust scroll offset to keep selection visible
        if app.selected < app.watchlist_offset {
            app.watchlist_offset = app.selected;
        } else if app.selected >= app.watchlist_offset + visible && visible > 0 {
            app.watchlist_offset = app.selected - visible + 1;
        }

        let offset = app.watchlist_offset;
        let end = (offset + visible).min(total_rows);

        let header_cells = ["Asset", "Sparkline", "LPPLS", "GSADF", "tc", "Status", "Confirms"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(app.theme.header).add_modifier(Modifier::BOLD)));
        let header = Row::new(header_cells).height(1).bottom_margin(1);

        let rows: Vec<Row> = sorted_indices[offset..end]
            .iter()
            .enumerate()
            .map(|(vi, &orig_idx)| {
                let scan = &app.scans[orig_idx];
                let i = offset + vi;

                let lppls = scan.signal.components.get("lppls_confidence").copied().unwrap_or(0.0).max(0.0);
                let gsadf_has_data = scan.components.contains_key("gsadf_bubble");
                // Show raw stat/CV ratio instead of 0-1 signal
                let gsadf_ratio = scan.raw_values.get("gsadf_bubble").map(|rv| {
                    let stat = rv.iter().find(|(k, _)| k == "GSADF statistic").map(|(_, v)| *v).unwrap_or(f64::NAN);
                    let cv = rv.iter().find(|(k, _)| k == "95% critical value").map(|(_, v)| *v).unwrap_or(f64::NAN);
                    if cv.is_finite() && cv != 0.0 { stat / cv } else { 0.0 }
                });
                let tc_days = scan.signal.horizon_days;

                let status = scan.signal.status();
                let status_color = match status {
                    "ALERT" => app.theme.signal_high,
                    "WATCH" => app.theme.signal_mid,
                    _ => app.theme.signal_low,
                };

                let lppls_color = if lppls > 0.7 { app.theme.signal_high } else if lppls > 0.5 { app.theme.signal_mid } else { app.theme.signal_low };
                let ratio_val = gsadf_ratio.unwrap_or(0.0);
                let gsadf_color = if ratio_val >= 1.5 { app.theme.signal_high } else if ratio_val >= 1.0 { app.theme.signal_mid } else { app.theme.signal_low };

                let tc_str = if tc_days.is_finite() && tc_days > 0.0 && lppls >= 0.3 {
                    format!("{:.0}d", tc_days)
                } else {
                    "-".to_string()
                };

                let confirms_str = if status == "QUIET" {
                    String::new()
                } else {
                    format!("{}/5", scan.signal.n_confirming)
                };

                let is_selected = i == app.selected;
                let row_style = if is_selected {
                    Style::default()
                        .bg(app.theme.selected_bg)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                let spark_color = if scan.prices.len() >= 2 {
                    let first = scan.prices[0];
                    let last = scan.prices[scan.prices.len() - 1];
                    if last >= first { app.theme.spark_up } else { app.theme.spark_down }
                } else {
                    app.theme.text_dim
                };

                let is_computing_gsadf = app.gsadf_computing.contains(&scan.asset);
                let error_suffix = if scan.error.is_some() { " !" } else { "" };

                Row::new(vec![
                    Cell::from(format!("{}{}", scan.asset, error_suffix)),
                    Cell::from(sparkline_text(&scan.prices, 20))
                        .style(Style::default().fg(spark_color)),
                    Cell::from(format!("{:.0}%", lppls * 100.0))
                        .style(Style::default().fg(lppls_color)),
                    Cell::from(if is_computing_gsadf {
                        "\u{00b7}\u{00b7}\u{00b7}".to_string()
                    } else if !gsadf_has_data {
                        "-".to_string()
                    } else {
                        format!("{:.2}", ratio_val)
                    })
                        .style(Style::default().fg(if !gsadf_has_data {
                            app.theme.text_dim
                        } else {
                            gsadf_color
                        })),
                    Cell::from(tc_str),
                    Cell::from(if is_computing_gsadf {
                        "GSADF..".to_string()
                    } else {
                        status.to_string()
                    }).style(Style::default().fg(if is_computing_gsadf {
                        app.theme.signal_mid
                    } else {
                        status_color
                    })),
                    Cell::from(confirms_str),
                ])
                .style(row_style)
            })
            .collect();

        let scroll_info = if total_rows > visible {
            format!(" Watchlist ({}/{}) ", app.selected + 1, total_rows)
        } else {
            " Watchlist (ALERT > WATCH > QUIET) ".to_string()
        };

        let table = Table::new(
            rows,
            [
                Constraint::Length(8),  // Asset
                Constraint::Length(22), // Sparkline
                Constraint::Length(7),  // LPPLS
                Constraint::Length(7),  // GSADF
                Constraint::Length(6),  // tc
                Constraint::Length(7),  // Status
                Constraint::Length(9),  // Confirms
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(scroll_info)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(app.theme.border)),
        );
        f.render_widget(table, chunks[1]);
    }

    // Status bar
    let last_scan_str = match app.last_scan {
        Some(ts) => ts.format("%H:%M:%S UTC").to_string(),
        None => "never".to_string(),
    };
    let asset_count = app.scans.len();
    let n_alerts = app.scans.iter().filter(|s| s.signal.status() == "ALERT").count();
    let n_watches = app.scans.iter().filter(|s| s.signal.status() == "WATCH").count();
    let gsadf_status = if gsadf_count > 0 {
        format!("  |  GSADF computing: {}", gsadf_count)
    } else {
        String::new()
    };
    let status = Paragraph::new(format!(
        " Last scan: {}  |  {} assets  |  {} alerts {} watches{}  |  g=GSADF q r w d \u{2190}\u{2192} \u{2191}\u{2193}",
        last_scan_str, asset_count, n_alerts, n_watches, gsadf_status
    ))
    .style(Style::default().fg(app.theme.text_dim))
    .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(app.theme.border)));
    f.render_widget(status, chunks[2]);
}

/// Build a Unicode progress bar like `████████░░░░░░░`.
fn progress_bar(done: usize, total: usize, width: usize) -> String {
    if total == 0 {
        return "\u{2591}".repeat(width);
    }
    let filled = (done * width) / total;
    let empty = width - filled;
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
}

/// Format ETA string like ` ~1m 23s remaining`, or empty if not enough data.
fn format_eta(app: &super::App) -> String {
    if app.scan_done == 0 {
        return String::new();
    }
    let elapsed = match app.scan_started {
        Some(t) => t.elapsed().as_secs_f64(),
        None => return String::new(),
    };
    let remaining_items = app.scan_total.saturating_sub(app.scan_done);
    let eta_secs = (elapsed * remaining_items as f64 / app.scan_done as f64) as u64;
    let mins = eta_secs / 60;
    let secs = eta_secs % 60;
    if mins > 0 {
        format!(" ~{}m {:02}s remaining", mins, secs)
    } else {
        format!(" ~{}s remaining", secs)
    }
}

/// Create a text-based sparkline representation of recent prices.
fn sparkline_text(prices: &[f64], width: usize) -> String {
    if prices.is_empty() {
        return " ".repeat(width);
    }

    let n = prices.len().min(width);
    let tail = &prices[prices.len() - n..];

    let min = tail.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = tail.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range < 1e-10 {
        return "\u{2584}".repeat(n);
    }

    let blocks = [
        '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
        '\u{2588}',
    ];

    let mut result = String::with_capacity(n * 3);
    for &p in tail {
        let normalized = ((p - min) / range * 7.0).round() as usize;
        let idx = normalized.min(7);
        result.push(blocks[idx]);
    }
    result
}
