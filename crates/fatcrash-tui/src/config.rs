use serde::Deserialize;

use crate::data::DataSource;

#[derive(Debug, Clone)]
pub struct WatchlistEntry {
    pub symbol: String,
    pub source: DataSource,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Config {
    pub watchlist: Vec<WatchlistEntry>,
    pub window: usize,
    pub refresh_seconds: u64,
}

#[derive(Deserialize)]
struct TomlConfig {
    monitor: Option<MonitorConfig>,
}

#[derive(Deserialize)]
struct MonitorConfig {
    window: Option<usize>,
    refresh_seconds: Option<u64>,
    assets: Option<Vec<AssetConfig>>,
}

#[derive(Deserialize)]
struct AssetConfig {
    symbol: String,
    source: String,
    ticker: String,
}

pub const DEFAULT_WATCHLIST: &[(&str, &str, &str)] = &[
    // Crypto
    ("BTC", "yahoo", "BTC-USD"),
    ("ETH", "yahoo", "ETH-USD"),
    ("SOL", "coingecko", "solana"),
    // US indices
    ("SPY", "yahoo", "SPY"),
    ("QQQ", "yahoo", "QQQ"),
    ("IWM", "yahoo", "IWM"),
    ("DIA", "yahoo", "DIA"),
    // International indices
    ("EEM", "yahoo", "EEM"),
    ("FXI", "yahoo", "FXI"),
    ("EWZ", "yahoo", "EWZ"),
    ("EWJ", "yahoo", "EWJ"),
    ("EWG", "yahoo", "EWG"),
    // Mega-cap tech
    ("NVDA", "yahoo", "NVDA"),
    ("AAPL", "yahoo", "AAPL"),
    ("MSFT", "yahoo", "MSFT"),
    ("GOOG", "yahoo", "GOOG"),
    ("AMZN", "yahoo", "AMZN"),
    ("META", "yahoo", "META"),
    ("TSLA", "yahoo", "TSLA"),
    // Commodities
    ("GOLD", "yahoo", "GC=F"),
    ("OIL", "yahoo", "CL=F"),
    ("SILV", "yahoo", "SI=F"),
    ("CPER", "yahoo", "HG=F"),
    ("NGAS", "yahoo", "NG=F"),
    // Bonds / rates
    ("TLT", "yahoo", "TLT"),
    ("HYG", "yahoo", "HYG"),
    ("LQD", "yahoo", "LQD"),
    // Currency
    ("DXY", "yahoo", "DX-Y.NYB"),
    // Sectors
    ("XLF", "yahoo", "XLF"),
    ("XLE", "yahoo", "XLE"),
    ("XLK", "yahoo", "XLK"),
];

pub fn load_config(path: Option<&str>) -> Config {
    if let Some(p) = path {
        if let Ok(content) = std::fs::read_to_string(p) {
            if let Ok(toml_cfg) = toml::from_str::<TomlConfig>(&content) {
                if let Some(mon) = toml_cfg.monitor {
                    let watchlist = mon
                        .assets
                        .unwrap_or_default()
                        .iter()
                        .map(|a| WatchlistEntry {
                            symbol: a.symbol.clone(),
                            source: match a.source.as_str() {
                                "coingecko" => DataSource::CoinGecko {
                                    coin_id: a.ticker.clone(),
                                },
                                _ => DataSource::Yahoo {
                                    ticker: a.ticker.clone(),
                                },
                            },
                        })
                        .collect();
                    return Config {
                        watchlist,
                        window: mon.window.unwrap_or(120),
                        refresh_seconds: mon.refresh_seconds.unwrap_or(300),
                    };
                }
            }
        }
    }
    // Default config
    Config {
        watchlist: DEFAULT_WATCHLIST
            .iter()
            .map(|(sym, src, ticker)| WatchlistEntry {
                symbol: sym.to_string(),
                source: match *src {
                    "coingecko" => DataSource::CoinGecko {
                        coin_id: ticker.to_string(),
                    },
                    _ => DataSource::Yahoo {
                        ticker: ticker.to_string(),
                    },
                },
            })
            .collect(),
        window: 120,
        refresh_seconds: 300,
    }
}
