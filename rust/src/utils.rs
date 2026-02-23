use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

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
    let log_p: Vec<f64> = prices.iter().map(|p| p.ln()).collect();
    Ok(PyArray1::from_vec(py, log_p))
}
