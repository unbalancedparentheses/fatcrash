use chrono::NaiveDate;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct OhlcvBar {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone)]
pub enum DataSource {
    Yahoo { ticker: String },
    CoinGecko { coin_id: String },
}

impl std::fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSource::Yahoo { ticker } => write!(f, "yahoo:{}", ticker),
            DataSource::CoinGecko { coin_id } => write!(f, "coingecko:{}", coin_id),
        }
    }
}

pub fn fetch(source: &DataSource, days: usize) -> Result<Vec<OhlcvBar>, String> {
    match source {
        DataSource::Yahoo { ticker } => fetch_yahoo(ticker, days),
        DataSource::CoinGecko { coin_id } => fetch_coingecko(coin_id, days),
    }
}

fn fetch_yahoo(ticker: &str, days: usize) -> Result<Vec<OhlcvBar>, String> {
    let now = chrono::Utc::now().timestamp();
    let period1 = now - (days as i64 + 30) * 86400; // extra buffer
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval=1d",
        ticker, period1, now
    );
    let client = reqwest::blocking::Client::builder()
        .user_agent("fatcrash/0.1")
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;
    let resp = client
        .get(&url)
        .send()
        .map_err(|e| format!("Yahoo fetch error: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Yahoo HTTP {}", resp.status()));
    }
    let body: serde_json::Value =
        resp.json().map_err(|e| format!("Yahoo parse error: {}", e))?;

    let result = &body["chart"]["result"][0];
    let timestamps = result["timestamp"]
        .as_array()
        .ok_or("No timestamps")?;
    let quote = &result["indicators"]["quote"][0];
    let opens = quote["open"].as_array().ok_or("No opens")?;
    let highs = quote["high"].as_array().ok_or("No highs")?;
    let lows = quote["low"].as_array().ok_or("No lows")?;
    let closes = quote["close"].as_array().ok_or("No closes")?;
    let volumes = quote["volume"].as_array().ok_or("No volumes")?;

    let mut bars = Vec::new();
    for i in 0..timestamps.len() {
        let ts = timestamps[i].as_i64().unwrap_or(0);
        let date = chrono::DateTime::from_timestamp(ts, 0)
            .map(|dt| dt.date_naive())
            .unwrap_or_else(|| NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());

        let o = opens.get(i).and_then(|v| v.as_f64());
        let h = highs.get(i).and_then(|v| v.as_f64());
        let l = lows.get(i).and_then(|v| v.as_f64());
        let c = closes.get(i).and_then(|v| v.as_f64());
        let v = volumes
            .get(i)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        if let (Some(o), Some(h), Some(l), Some(c)) = (o, h, l, c) {
            bars.push(OhlcvBar {
                date,
                open: o,
                high: h,
                low: l,
                close: c,
                volume: v,
            });
        }
    }
    Ok(bars)
}

fn fetch_coingecko(coin_id: &str, days: usize) -> Result<Vec<OhlcvBar>, String> {
    let url = format!(
        "https://api.coingecko.com/api/v3/coins/{}/market_chart?vs_currency=usd&days={}&interval=daily",
        coin_id, days
    );
    let client = reqwest::blocking::Client::builder()
        .user_agent("fatcrash/0.1")
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;
    let resp = client
        .get(&url)
        .send()
        .map_err(|e| format!("CoinGecko fetch error: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("CoinGecko HTTP {}", resp.status()));
    }

    #[derive(Deserialize)]
    struct CgResponse {
        prices: Vec<(f64, f64)>,
    }

    let data: CgResponse = resp
        .json()
        .map_err(|e| format!("CoinGecko parse error: {}", e))?;
    let bars: Vec<OhlcvBar> = data
        .prices
        .iter()
        .map(|(ts_ms, price)| {
            let ts = (*ts_ms as i64) / 1000;
            let date = chrono::DateTime::from_timestamp(ts, 0)
                .map(|dt| dt.date_naive())
                .unwrap_or_else(|| NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
            OhlcvBar {
                date,
                open: *price,
                high: *price,
                low: *price,
                close: *price,
                volume: 0.0,
            }
        })
        .collect();
    Ok(bars)
}
