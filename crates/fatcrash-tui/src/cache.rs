use crate::data::OhlcvBar;
use chrono::NaiveDate;
use std::path::PathBuf;

fn cache_dir() -> PathBuf {
    let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("fatcrash")
}

fn cache_key(source: &str, symbol: &str, scope: &str) -> String {
    use md5::{Digest, Md5};
    let input = format!("{}:{}:{}", source, symbol, scope);
    let hash = Md5::digest(input.as_bytes());
    format!("{:x}", hash)
}

fn cache_path(key: &str) -> PathBuf {
    cache_dir().join(format!("{}.csv", key))
}

pub fn load_cached(source: &str, symbol: &str, scope: &str) -> Option<Vec<OhlcvBar>> {
    let key = cache_key(source, symbol, scope);
    let path = cache_path(&key);
    if !path.exists() {
        return None;
    }
    // Check if < 24h old
    if let Ok(metadata) = std::fs::metadata(&path) {
        if let Ok(modified) = metadata.modified() {
            let age = std::time::SystemTime::now()
                .duration_since(modified)
                .unwrap_or_default();
            if age.as_secs() > 86400 {
                return None; // Stale
            }
        }
    }
    let mut rdr = csv::Reader::from_path(&path).ok()?;
    let mut bars = Vec::new();
    for result in rdr.records() {
        let record = result.ok()?;
        let date = NaiveDate::parse_from_str(record.get(0)?, "%Y-%m-%d").ok()?;
        let open: f64 = record.get(1)?.parse().ok()?;
        let high: f64 = record.get(2)?.parse().ok()?;
        let low: f64 = record.get(3)?.parse().ok()?;
        let close: f64 = record.get(4)?.parse().ok()?;
        let volume: f64 = record.get(5)?.parse().ok()?;
        bars.push(OhlcvBar {
            date,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    if bars.is_empty() {
        None
    } else {
        Some(bars)
    }
}

pub fn save_cache(source: &str, symbol: &str, scope: &str, bars: &[OhlcvBar]) {
    let key = cache_key(source, symbol, scope);
    let path = cache_path(&key);
    let dir = cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    let mut wtr = match csv::Writer::from_path(&path) {
        Ok(w) => w,
        Err(_) => return,
    };
    let _ = wtr.write_record(["date", "open", "high", "low", "close", "volume"]);
    for bar in bars {
        let _ = wtr.write_record([
            bar.date.format("%Y-%m-%d").to_string(),
            bar.open.to_string(),
            bar.high.to_string(),
            bar.low.to_string(),
            bar.close.to_string(),
            bar.volume.to_string(),
        ]);
    }
    let _ = wtr.flush();
}

pub fn clear_cache() {
    let dir = cache_dir();
    if dir.exists() {
        let _ = std::fs::remove_dir_all(&dir);
        println!("Removed {}", dir.display());
    }
}
