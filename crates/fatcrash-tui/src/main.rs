use clap::{Parser, Subcommand};

mod cache;
mod config;
mod data;
mod scanner;
#[allow(dead_code)]
mod signals;
mod tui;

#[derive(Parser)]
#[command(name = "fatcrash-tui", about = "Crash probability monitor")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch monitor (TUI or one-shot JSON scan)
    Monitor {
        #[arg(long)]
        config: Option<String>,
        #[arg(long, default_value = "120")]
        window: usize,
        #[arg(long, default_value = "365")]
        days: usize,
        #[arg(long)]
        no_cache: bool,
        #[arg(long)]
        json: bool,
    },
    /// Clear the data cache
    CacheClear,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Monitor {
            config: config_path,
            window,
            days,
            no_cache,
            json,
        } => {
            let cfg = config::load_config(config_path.as_deref());
            let use_cache = !no_cache;
            if json {
                // One-shot scan, print JSON, exit
                let (tx, rx) = std::sync::mpsc::channel();
                scanner::scan_watchlist(&cfg.watchlist, window, days, use_cache, tx);
                let mut scans = Vec::new();
                loop {
                    match rx.recv().unwrap() {
                        scanner::ScanMsg::Progress { done, total, asset } => {
                            eprint!("\rScanning {}... ({}/{})", asset, done, total);
                        }
                        scanner::ScanMsg::PartialResults(s) => {
                            scans = s;
                        }
                        scanner::ScanMsg::GsadfUpdate(updated) => {
                            if let Some(scan) = scans.iter_mut().find(|s| s.asset == updated.asset) {
                                *scan = *updated;
                            }
                        }
                        scanner::ScanMsg::Done => {
                            eprintln!();
                            break;
                        }
                    }
                }
                let output: Vec<serde_json::Value> = scans
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "asset": s.asset,
                            "probability": s.signal.probability,
                            "status": s.signal.status(),
                            "level": s.signal.level(),
                            "n_confirming": s.signal.n_confirming,
                            "confirming_categories": s.signal.confirming_categories,
                            "components": s.signal.components,
                            "data_points": s.data_points,
                            "error": s.error,
                            "timestamp": s.timestamp.to_rfc3339(),
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&output).unwrap());
            } else {
                tui::run(cfg, window, days, use_cache).unwrap();
            }
        }
        Commands::CacheClear => {
            cache::clear_cache();
            println!("Cache cleared.");
        }
    }
}
