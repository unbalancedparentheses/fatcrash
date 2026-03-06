pub mod detail;
pub mod method;
pub mod watchlist;

use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::prelude::*;

use crate::config::Config;
use crate::scanner::{self, AssetScan};

/// Current view in the TUI.
#[derive(Debug, Clone)]
pub enum View {
    Watchlist,
    Detail(usize),
    /// Method drill-down: (scan_index, method_key)
    MethodDetail(usize, String),
}

/// Application state.
pub struct App {
    pub scans: Vec<AssetScan>,
    pub selected: usize,
    /// Selected row index within the detail methods table.
    pub method_selected: usize,
    pub view: View,
    pub scanning: bool,
    pub last_scan: Option<chrono::DateTime<chrono::Utc>>,
    pub config: Config,
    pub window: usize,
    pub days: usize,
    pub use_cache: bool,
    /// Scroll offset for the watchlist table.
    pub watchlist_offset: usize,
    /// Scroll offset for the detail methods table.
    pub detail_offset: usize,
}

impl App {
    pub fn new(config: Config, window: usize, days: usize, use_cache: bool) -> Self {
        Self {
            scans: Vec::new(),
            selected: 0,
            method_selected: 0,
            view: View::Watchlist,
            scanning: false,
            last_scan: None,
            config,
            window,
            days,
            use_cache,
            watchlist_offset: 0,
            detail_offset: 0,
        }
    }

    /// Get scans sorted by probability descending.
    pub fn sorted_scans(&self) -> Vec<(usize, &AssetScan)> {
        let mut indexed: Vec<(usize, &AssetScan)> =
            self.scans.iter().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.signal
                .probability
                .partial_cmp(&a.1.signal.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed
    }
}

/// Run the TUI event loop.
pub fn run(
    config: Config,
    window: usize,
    days: usize,
    use_cache: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(config, window, days, use_cache);

    // Channel for background scan results
    let (tx, rx) = mpsc::channel::<Vec<AssetScan>>();

    // Kick off initial scan
    start_scan(&app, tx.clone());
    app.scanning = true;

    let refresh_interval = Duration::from_secs(app.config.refresh_seconds);
    let mut last_refresh = Instant::now();
    let tick_rate = Duration::from_millis(250);

    loop {
        // Draw
        let view = app.view.clone();
        terminal.draw(|f| {
            match &view {
                View::Watchlist => watchlist::render(f, &mut app),
                View::Detail(idx) => detail::render(f, &mut app, *idx),
                View::MethodDetail(scan_idx, ref key) => method::render(f, &app, *scan_idx, key),
            }
        })?;

        // Check for background scan results (non-blocking)
        if let Ok(scans) = rx.try_recv() {
            app.scans = scans;
            app.scanning = false;
            app.last_scan = Some(chrono::Utc::now());
        }

        // Handle input
        if event::poll(tick_rate)? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => break,
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        if !app.scanning {
                            start_scan(&app, tx.clone());
                            app.scanning = true;
                            last_refresh = Instant::now();
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') => match &app.view {
                        View::Watchlist => {
                            if app.selected > 0 { app.selected -= 1; }
                        }
                        View::Detail(_) => {
                            if app.method_selected > 0 { app.method_selected -= 1; }
                        }
                        _ => {}
                    },
                    KeyCode::Down | KeyCode::Char('j') => match &app.view {
                        View::Watchlist => {
                            let max = app.scans.len().saturating_sub(1);
                            if app.selected < max { app.selected += 1; }
                        }
                        View::Detail(scan_idx) => {
                            let max = detail::method_count(&app, *scan_idx).saturating_sub(1);
                            if app.method_selected < max { app.method_selected += 1; }
                        }
                        _ => {}
                    },
                    KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => match &app.view {
                        View::Watchlist => {
                            if !app.scans.is_empty() {
                                let sorted = app.sorted_scans();
                                if app.selected < sorted.len() {
                                    let original_idx = sorted[app.selected].0;
                                    app.method_selected = 0;
                                    app.view = View::Detail(original_idx);
                                }
                            }
                        }
                        View::Detail(scan_idx) => {
                            if let Some(key) = detail::selected_method_key(&app, *scan_idx) {
                                let idx = *scan_idx;
                                app.view = View::MethodDetail(idx, key);
                            }
                        }
                        _ => {}
                    },
                    KeyCode::Char('w') | KeyCode::Char('W') => {
                        const WINDOWS: &[usize] = &[60, 90, 120, 180, 252];
                        let cur = WINDOWS.iter().position(|&w| w == app.window);
                        app.window = match cur {
                            Some(i) => WINDOWS[(i + 1) % WINDOWS.len()],
                            None => WINDOWS[0],
                        };
                        if !app.scanning {
                            start_scan(&app, tx.clone());
                            app.scanning = true;
                            last_refresh = Instant::now();
                        }
                    }
                    KeyCode::Char('d') | KeyCode::Char('D') => {
                        const DAYS: &[usize] = &[180, 365, 730, 1095];
                        let cur = DAYS.iter().position(|&d| d == app.days);
                        app.days = match cur {
                            Some(i) => DAYS[(i + 1) % DAYS.len()],
                            None => DAYS[0],
                        };
                        if !app.scanning {
                            start_scan(&app, tx.clone());
                            app.scanning = true;
                            last_refresh = Instant::now();
                        }
                    }
                    KeyCode::Esc | KeyCode::Backspace | KeyCode::Left | KeyCode::Char('h') => match &app.view {
                        View::Detail(_) => {
                            app.view = View::Watchlist;
                        }
                        View::MethodDetail(scan_idx, _) => {
                            let idx = *scan_idx;
                            app.view = View::Detail(idx);
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }

        // Auto-refresh
        if !app.scanning && last_refresh.elapsed() >= refresh_interval {
            start_scan(&app, tx.clone());
            app.scanning = true;
            last_refresh = Instant::now();
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

/// Kick off a background scan on a separate thread.
fn start_scan(app: &App, tx: mpsc::Sender<Vec<AssetScan>>) {
    let entries = app.config.watchlist.clone();
    let window = app.window;
    let days = app.days;
    let use_cache = app.use_cache;

    std::thread::spawn(move || {
        let scans = scanner::scan_watchlist(&entries, window, days, use_cache);
        let _ = tx.send(scans);
    });
}
