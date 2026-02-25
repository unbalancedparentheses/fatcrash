use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use super::model::solve_linear;

type LinearSolution = (f64, f64, f64, f64, f64);

/// Expose `solve_linear` to Python so NN methods can compute
/// (A, B, C1, C2, RSS) given network-predicted (tc, m, omega).
#[pyfunction]
#[pyo3(signature = (times, log_prices, tc, m, omega))]
pub fn lppls_solve_linear(
    _py: Python<'_>,
    times: PyReadonlyArray1<'_, f64>,
    log_prices: PyReadonlyArray1<'_, f64>,
    tc: f64,
    m: f64,
    omega: f64,
) -> PyResult<Option<LinearSolution>> {
    let t = times.as_slice()?;
    let lp = log_prices.as_slice()?;
    Ok(solve_linear(t, lp, tc, m, omega))
}
