/// Simple OLS slope: beta = cov(x,y) / var(x)
pub fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum();
    let var_x: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();

    if var_x < 1e-15 {
        return f64::NAN;
    }

    cov / var_x
}

/// Compute log returns from a price series.
///
/// Returns a vector of ln(price[i] / price[i-1]) for i in 1..n.
/// Returns an error if any price is non-positive.
pub fn log_returns_slice(prices: &[f64]) -> Result<Vec<f64>, String> {
    let n = prices.len();
    if n < 2 {
        return Ok(vec![]);
    }
    let mut returns = Vec::with_capacity(n - 1);
    for i in 1..n {
        if prices[i - 1] <= 0.0 {
            return Err(format!(
                "Non-positive price at index {}: {}",
                i - 1,
                prices[i - 1]
            ));
        }
        returns.push((prices[i] / prices[i - 1]).ln());
    }
    Ok(returns)
}

/// Compute log prices from a price series.
///
/// Returns a vector of ln(price[i]) for each price.
/// Returns an error if any price is non-positive.
pub fn log_prices_slice(prices: &[f64]) -> Result<Vec<f64>, String> {
    for (i, &p) in prices.iter().enumerate() {
        if p <= 0.0 {
            return Err(format!("Non-positive price at index {}: {}", i, p));
        }
    }
    let log_p: Vec<f64> = prices.iter().map(|p| p.ln()).collect();
    Ok(log_p)
}
