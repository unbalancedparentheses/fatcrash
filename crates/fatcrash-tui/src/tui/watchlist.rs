use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};

use super::App;

/// Pretty-print a signal key as a human-readable name.
fn pretty_signal_name(key: &str) -> String {
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
    let scanning_indicator = if app.scanning { " [scanning...]" } else { "" };
    let title = Paragraph::new(format!(
        " fatcrash-tui  |  window={}  days={}{}",
        app.window, app.days, scanning_indicator
    ))
    .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
    .block(Block::default().borders(Borders::BOTTOM));
    f.render_widget(title, chunks[0]);

    // Main table area
    if app.scans.is_empty() {
        let msg = if app.scanning {
            "Scanning assets... please wait"
        } else {
            "No data yet. Press r to scan."
        };
        let loading = Paragraph::new(msg)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))
            .block(
                Block::default()
                    .title(" Watchlist ")
                    .borders(Borders::ALL),
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

        let header_cells = ["Asset", "Sparkline", "Prob", "Level", "Top Signal", "Agreeing"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)));
        let header = Row::new(header_cells).height(1).bottom_margin(1);

        let rows: Vec<Row> = sorted_indices[offset..end]
            .iter()
            .enumerate()
            .map(|(vi, &orig_idx)| {
                let scan = &app.scans[orig_idx];
                let i = offset + vi; // actual index in sorted list
                let prob = scan.signal.probability;
                let level = scan.signal.level();

                let level_color = match level {
                    "CRITICAL" | "HIGH" => Color::Red,
                    "ELEVATED" => Color::Yellow,
                    _ => Color::Green,
                };

                let weights = crate::signals::default_weights();
                let top_signal = scan
                    .components
                    .iter()
                    .filter(|(_, v)| v.is_finite() && **v > 0.0)
                    .max_by(|a, b| {
                        let wa = weights.get(a.0.as_str()).copied().unwrap_or(0.0) * a.1;
                        let wb = weights.get(b.0.as_str()).copied().unwrap_or(0.0) * b.1;
                        wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(k, _)| pretty_signal_name(k))
                    .unwrap_or_else(|| "-".to_string());

                let is_selected = i == app.selected;
                let row_style = if is_selected {
                    Style::default()
                        .bg(Color::Rgb(40, 40, 60))
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                let spark_color = if scan.prices.len() >= 2 {
                    let first = scan.prices[0];
                    let last = scan.prices[scan.prices.len() - 1];
                    if last >= first { Color::Green } else { Color::Red }
                } else {
                    Color::DarkGray
                };

                let error_suffix = if scan.error.is_some() { " !" } else { "" };

                Row::new(vec![
                    Cell::from(format!("{}{}", scan.asset, error_suffix)),
                    Cell::from(sparkline_text(&scan.prices, 20))
                        .style(Style::default().fg(spark_color)),
                    Cell::from(format!("{:.1}%", prob * 100.0))
                        .style(Style::default().fg(level_color)),
                    Cell::from(level).style(Style::default().fg(level_color)),
                    Cell::from(top_signal),
                    Cell::from(format!("{}", scan.signal.n_agreeing)),
                ])
                .style(row_style)
            })
            .collect();

        let scroll_info = if total_rows > visible {
            format!(" Watchlist ({}/{}) ", app.selected + 1, total_rows)
        } else {
            " Watchlist (sorted by probability) ".to_string()
        };

        let table = Table::new(
            rows,
            [
                Constraint::Length(8),  // Asset
                Constraint::Length(22), // Sparkline
                Constraint::Length(8),  // Prob
                Constraint::Length(10), // Level
                Constraint::Length(20), // Top Signal
                Constraint::Length(9),  // Agreeing
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(scroll_info)
                .borders(Borders::ALL),
        );
        f.render_widget(table, chunks[1]);
    }

    // Status bar
    let last_scan_str = match app.last_scan {
        Some(ts) => ts.format("%H:%M:%S UTC").to_string(),
        None => "never".to_string(),
    };
    let asset_count = app.scans.len();
    let status = Paragraph::new(format!(
        " Last scan: {}  |  {} assets  |  q=quit  r=refresh  w=window({})  d=days({})  \u{2190}\u{2192}=navigate  \u{2191}\u{2193}=select",
        last_scan_str, asset_count, app.window, app.days
    ))
    .style(Style::default().fg(Color::DarkGray))
    .block(Block::default().borders(Borders::TOP));
    f.render_widget(status, chunks[2]);
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
