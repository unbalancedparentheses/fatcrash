use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Simple OLS slope: beta = cov(x,y) / var(x)
pub(crate) fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
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

#[pyfunction]
pub fn log_returns<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let n = prices.len();
    if n < 2 {
        return Ok(PyArray1::from_vec(py, vec![]));
    }
    let mut returns = Vec::with_capacity(n - 1);
    for i in 1..n {
        if prices[i - 1] <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Non-positive price at index {}: {}",
                i - 1,
                prices[i - 1]
            )));
        }
        returns.push((prices[i] / prices[i - 1]).ln());
    }
    Ok(PyArray1::from_vec(py, returns))
}

#[pyfunction]
pub fn log_prices<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    for (i, &p) in prices.iter().enumerate() {
        if p <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Non-positive price at index {}: {}",
                i, p
            )));
        }
    }
    let log_p: Vec<f64> = prices.iter().map(|p| p.ln()).collect();
    Ok(PyArray1::from_vec(py, log_p))
}
