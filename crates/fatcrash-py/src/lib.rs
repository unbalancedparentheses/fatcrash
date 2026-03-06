use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use fatcrash_core::bubble::gsadf as core_gsadf;
use fatcrash_core::evt::{gev as core_gev, gpd as core_gpd};
use fatcrash_core::lppls::{
    confidence as core_confidence, fitter as core_fitter, model as core_model,
};
use fatcrash_core::regime::{
    csd as core_csd, hamilton as core_hamilton, jump as core_jump,
    realized_var as core_realized_var,
};
use fatcrash_core::tail::{
    absorption as core_absorption, amihud as core_amihud, deh as core_deh, dfa as core_dfa,
    hill as core_hill, hurst as core_hurst, kappa as core_kappa, maxsum as core_maxsum,
    momentum as core_momentum, pickands as core_pickands, qq as core_qq,
    skewness as core_skewness, spectral as core_spectral, velocity as core_velocity,
};

// ── Tail estimators ──────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (data, k=None, use_abs=true))]
fn hill_estimator(data: PyReadonlyArray1<'_, f64>, k: Option<usize>, use_abs: Option<bool>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(core_hill::hill_estimator_slice(data, k, use_abs.unwrap_or(true)))
}

#[pyfunction]
#[pyo3(signature = (data, window, k=None, use_abs=true))]
fn hill_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize, k: Option<usize>, use_abs: Option<bool>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    Ok(PyArray1::from_vec(py, core_hill::hill_rolling_slice(data, window, k, use_abs.unwrap_or(true))))
}

#[pyfunction]
#[pyo3(signature = (data, n_subsamples=10, n_sims=1000))]
fn kappa_metric(data: PyReadonlyArray1<'_, f64>, n_subsamples: Option<usize>, n_sims: Option<usize>) -> PyResult<(f64, f64)> {
    let data = data.as_slice()?;
    Ok(core_kappa::kappa_metric_slice(data, n_subsamples.unwrap_or(10), n_sims.unwrap_or(1000)))
}

#[pyfunction]
#[pyo3(signature = (data, window, n_subsamples=10, n_sims=200))]
fn kappa_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize, n_subsamples: Option<usize>, n_sims: Option<usize>) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let data = data.as_slice()?;
    let n_sub = n_subsamples.unwrap_or(10);
    let ns = n_sims.unwrap_or(200);
    let n = data.len();
    let benchmark = core_kappa::gaussian_kappa_benchmark(window, n_sub, ns);
    let mut result = vec![f64::NAN; n];
    if window > 0 {
        for i in (window - 1)..n {
            let slice = &data[(i + 1 - window)..=i];
            result[i] = core_kappa::compute_kappa(slice, n_sub);
        }
    }
    Ok((PyArray1::from_vec(py, result), benchmark))
}

#[pyfunction]
#[pyo3(signature = (data, n0=30, n1=100, n_sims=500))]
fn taleb_kappa(data: PyReadonlyArray1<'_, f64>, n0: Option<usize>, n1: Option<usize>, n_sims: Option<usize>) -> PyResult<(f64, f64)> {
    let data = data.as_slice()?;
    let n0 = n0.unwrap_or(30);
    let n1_val = n1.unwrap_or(100).min(data.len() / 2);
    let ns = n_sims.unwrap_or(500);
    Ok(core_kappa::taleb_kappa_slice(data, n0, n1_val, ns))
}

#[pyfunction]
#[pyo3(signature = (data, window, n0=30, n1=100, n_sims=100))]
fn taleb_kappa_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize, n0: Option<usize>, n1: Option<usize>, n_sims: Option<usize>) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let data = data.as_slice()?;
    let n0 = n0.unwrap_or(30);
    let n1_val = n1.unwrap_or(100).min(window / 2);
    let ns = n_sims.unwrap_or(100);
    let len = data.len();
    let benchmark = core_kappa::gaussian_taleb_kappa_benchmark(window, n0, n1_val, ns);
    let mut result = vec![f64::NAN; len];
    if window > 0 && n1_val > n0 {
        for i in (window - 1)..len {
            let slice = &data[(i + 1 - window)..=i];
            result[i] = core_kappa::compute_taleb_kappa(slice, n0, n1_val);
        }
    }
    Ok((PyArray1::from_vec(py, result), benchmark))
}

#[pyfunction]
#[pyo3(signature = (data, k=None, use_abs=true))]
fn pickands_estimator(data: PyReadonlyArray1<'_, f64>, k: Option<usize>, use_abs: Option<bool>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(core_pickands::pickands_estimator_slice(data, k, use_abs.unwrap_or(true)))
}

#[pyfunction]
#[pyo3(signature = (data, window, k=None, use_abs=true))]
fn pickands_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize, k: Option<usize>, use_abs: Option<bool>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    let use_abs = use_abs.unwrap_or(true);
    if window > n || window < 4 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        let slice = &data[(i + 1 - window)..=i];
        result[i] = core_pickands::pickands_estimator_slice(slice, k, use_abs);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (data,))]
fn hurst_exponent(data: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(core_hurst::compute_hurst(data))
}

#[pyfunction]
#[pyo3(signature = (data, window))]
fn hurst_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    Ok(PyArray1::from_vec(py, core_hurst::hurst_rolling_slice(data, window)))
}

#[pyfunction]
#[pyo3(signature = (data,))]
fn dfa_exponent(data: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(core_dfa::compute_dfa(data))
}

#[pyfunction]
#[pyo3(signature = (data, window))]
fn dfa_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    if window > n || window < 32 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        result[i] = core_dfa::compute_dfa(&data[(i + 1 - window)..=i]);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (data, k=None, use_abs=true))]
fn deh_estimator(data: PyReadonlyArray1<'_, f64>, k: Option<usize>, use_abs: Option<bool>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(core_deh::deh_estimator_slice(data, k, use_abs.unwrap_or(true)))
}

#[pyfunction]
#[pyo3(signature = (data, window, k=None, use_abs=true))]
fn deh_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize, k: Option<usize>, use_abs: Option<bool>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    let use_abs = use_abs.unwrap_or(true);
    if window > n || window < 4 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        result[i] = core_deh::deh_estimator_slice(&data[(i + 1 - window)..=i], k, use_abs);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (data, k=None, use_abs=true))]
fn qq_estimator(data: PyReadonlyArray1<'_, f64>, k: Option<usize>, use_abs: Option<bool>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(core_qq::qq_estimator_slice(data, k, use_abs.unwrap_or(true)))
}

#[pyfunction]
#[pyo3(signature = (data, window, k=None, use_abs=true))]
fn qq_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize, k: Option<usize>, use_abs: Option<bool>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    let use_abs = use_abs.unwrap_or(true);
    if window > n || window < 4 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        result[i] = core_qq::qq_estimator_slice(&data[(i + 1 - window)..=i], k, use_abs);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (data,))]
fn maxsum_ratio(data: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(core_maxsum::compute_maxsum(data))
}

#[pyfunction]
#[pyo3(signature = (data, window))]
fn maxsum_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    if window > n || window < 2 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        result[i] = core_maxsum::compute_maxsum(&data[(i + 1 - window)..=i]);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (data, bandwidth_exp=0.65))]
fn spectral_exponent(data: PyReadonlyArray1<'_, f64>, bandwidth_exp: Option<f64>) -> PyResult<f64> {
    let data = data.as_slice()?;
    Ok(core_spectral::compute_spectral(data, bandwidth_exp.unwrap_or(0.65)))
}

#[pyfunction]
#[pyo3(signature = (data, window, bandwidth_exp=0.65))]
fn spectral_rolling<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize, bandwidth_exp: Option<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let n = data.len();
    let bw = bandwidth_exp.unwrap_or(0.65);
    if window > n || window < 32 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        result[i] = core_spectral::compute_spectral(&data[(i + 1 - window)..=i], bw);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (prices, lookback=252))]
fn momentum_score(prices: PyReadonlyArray1<'_, f64>, lookback: usize) -> PyResult<f64> {
    let prices = prices.as_slice()?;
    Ok(core_momentum::compute_momentum(prices, lookback))
}

#[pyfunction]
#[pyo3(signature = (prices, lookback=252, window=504))]
fn momentum_rolling<'py>(py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, lookback: usize, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let n = prices.len();
    if window > n || window <= lookback || lookback == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        result[i] = core_momentum::compute_momentum(&prices[(i + 1 - window)..=i], lookback);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (prices, short_lookback=21, long_lookback=252))]
fn momentum_reversal(prices: PyReadonlyArray1<'_, f64>, short_lookback: usize, long_lookback: usize) -> PyResult<f64> {
    let prices = prices.as_slice()?;
    Ok(core_momentum::compute_reversal(prices, short_lookback, long_lookback))
}

#[pyfunction]
#[pyo3(signature = (prices, short_lookback=21, long_lookback=252, window=504))]
fn momentum_reversal_rolling<'py>(py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, short_lookback: usize, long_lookback: usize, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let n = prices.len();
    if window > n || window <= long_lookback || long_lookback <= short_lookback {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        result[i] = core_momentum::compute_reversal(&prices[(i + 1 - window)..=i], short_lookback, long_lookback);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, vol_window=21, lag=5))]
fn price_velocity(returns: PyReadonlyArray1<'_, f64>, vol_window: usize, lag: usize) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    Ok(core_velocity::compute_velocity(returns, vol_window, lag))
}

#[pyfunction]
#[pyo3(signature = (returns, vol_window=21, lag=5, window=252))]
fn price_velocity_rolling<'py>(py: Python<'py>, returns: PyReadonlyArray1<'py, f64>, vol_window: usize, lag: usize, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    let min_required = vol_window + lag;
    if window > n || window < min_required || vol_window < 2 || lag == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        result[i] = core_velocity::compute_velocity(&returns[(i + 1 - window)..=i], vol_window, lag);
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window=63))]
fn realized_skewness(returns: PyReadonlyArray1<'_, f64>, window: usize) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    Ok(core_skewness::compute_realized_skewness(returns, window))
}

#[pyfunction]
#[pyo3(signature = (returns, window=63, step=1))]
fn realized_skewness_rolling<'py>(py: Python<'py>, returns: PyReadonlyArray1<'py, f64>, window: usize, step: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    if window < 3 || window > n || step == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n {
        result[i] = core_skewness::compute_realized_skewness(&returns[(i + 1 - window)..=i], window);
        i += step;
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, volume, window=21))]
fn amihud_illiquidity(returns: PyReadonlyArray1<'_, f64>, volume: PyReadonlyArray1<'_, f64>, window: usize) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    let volume = volume.as_slice()?;
    if returns.len() != volume.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("returns and volume must have the same length"));
    }
    Ok(core_amihud::compute_amihud(returns, volume, window))
}

#[pyfunction]
#[pyo3(signature = (returns, volume, window=21, step=1))]
fn amihud_illiquidity_rolling<'py>(py: Python<'py>, returns: PyReadonlyArray1<'py, f64>, volume: PyReadonlyArray1<'py, f64>, window: usize, step: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let volume = volume.as_slice()?;
    let n = returns.len();
    if returns.len() != volume.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("returns and volume must have the same length"));
    }
    if window == 0 || window > n || step == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n {
        result[i] = core_amihud::compute_amihud(&returns[(i + 1 - window)..=i], &volume[(i + 1 - window)..=i], window);
        i += step;
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns_list, window=63, n_components=1))]
fn absorption_ratio(returns_list: Vec<PyReadonlyArray1<'_, f64>>, window: usize, n_components: usize) -> PyResult<f64> {
    let slices: Vec<Vec<f64>> = returns_list.iter().map(|r| r.as_slice().map(|s| s.to_vec())).collect::<Result<Vec<_>, _>>()?;
    if slices.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Need at least 2 assets for absorption ratio"));
    }
    let n = slices[0].len();
    for s in &slices {
        if s.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err("All return arrays must have the same length"));
        }
    }
    let refs: Vec<&[f64]> = slices.iter().map(|s| s.as_slice()).collect();
    Ok(core_absorption::compute_absorption_ratio(&refs, window, n_components))
}

#[pyfunction]
#[pyo3(signature = (returns_list, window=63, n_components=1, step=1))]
fn absorption_ratio_rolling<'py>(py: Python<'py>, returns_list: Vec<PyReadonlyArray1<'py, f64>>, window: usize, n_components: usize, step: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slices: Vec<Vec<f64>> = returns_list.iter().map(|r| r.as_slice().map(|s| s.to_vec())).collect::<Result<Vec<_>, _>>()?;
    if slices.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Need at least 2 assets for absorption ratio"));
    }
    let n = slices[0].len();
    for s in &slices {
        if s.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err("All return arrays must have the same length"));
        }
    }
    if window == 0 || window > n || step == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n {
        let sub_slices: Vec<&[f64]> = slices.iter().map(|s| &s[(i + 1 - window)..=i]).collect();
        result[i] = core_absorption::compute_absorption_ratio(&sub_slices, window, n_components);
        i += step;
    }
    Ok(PyArray1::from_vec(py, result))
}

// ── EVT ──────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (data, quantile=0.95))]
fn gpd_fit(data: PyReadonlyArray1<'_, f64>, quantile: Option<f64>) -> PyResult<(f64, f64, f64, usize)> {
    let data = data.as_slice()?;
    core_gpd::gpd_fit_slice(data, quantile.unwrap_or(0.95))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(signature = (data, p=0.99, quantile=0.95))]
fn gpd_var_es(data: PyReadonlyArray1<'_, f64>, p: Option<f64>, quantile: Option<f64>) -> PyResult<(f64, f64)> {
    let data = data.as_slice()?;
    core_gpd::gpd_var_es_slice(data, p.unwrap_or(0.99), quantile.unwrap_or(0.95))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

#[pyfunction]
fn gev_fit(data: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64, f64)> {
    let data = data.as_slice()?;
    core_gev::gev_fit_slice(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

// ── LPPLS ────────────────────────────────────────────────

#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (times, log_prices, tc_range=None, pop_size=50, n_generations=40, seed=42))]
fn lppls_fit(
    _py: Python<'_>,
    times: PyReadonlyArray1<'_, f64>,
    log_prices: PyReadonlyArray1<'_, f64>,
    tc_range: Option<(f64, f64)>,
    pop_size: Option<usize>,
    n_generations: Option<usize>,
    seed: Option<u64>,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
    let t = times.as_slice()?;
    let lp = log_prices.as_slice()?;
    core_fitter::lppls_fit_slice(t, lp, tc_range, pop_size, n_generations, seed)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (times, log_prices, min_window=60, max_window=750, n_windows=50, n_candidates=30))]
fn lppls_confidence<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    log_prices: PyReadonlyArray1<'py, f64>,
    min_window: Option<usize>,
    max_window: Option<usize>,
    n_windows: Option<usize>,
    n_candidates: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let t = times.as_slice()?;
    let lp = log_prices.as_slice()?;
    let n = t.len();
    let max_win_clamped = max_window.map(|w| w.min(n));
    let (conf, mean, std) = py.allow_threads(|| {
        core_confidence::lppls_confidence_slice(t, lp, min_window, max_win_clamped, n_windows, n_candidates)
    });
    Ok((PyArray1::from_vec(py, conf), PyArray1::from_vec(py, mean), PyArray1::from_vec(py, std)))
}

#[pyfunction]
#[pyo3(signature = (times, log_prices, tc, m, omega))]
fn lppls_solve_linear(
    _py: Python<'_>,
    times: PyReadonlyArray1<'_, f64>,
    log_prices: PyReadonlyArray1<'_, f64>,
    tc: f64,
    m: f64,
    omega: f64,
) -> PyResult<Option<(f64, f64, f64, f64, f64)>> {
    let t = times.as_slice()?;
    let lp = log_prices.as_slice()?;
    Ok(core_model::solve_linear(t, lp, tc, m, omega))
}

// ── Multiscale ───────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (signals_1d, signals_3d, signals_7d))]
fn multiscale_signals<'py>(
    py: Python<'py>,
    signals_1d: PyReadonlyArray1<'py, f64>,
    signals_3d: PyReadonlyArray1<'py, f64>,
    signals_7d: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s1 = signals_1d.as_slice()?;
    let s3 = signals_3d.as_slice()?;
    let s7 = signals_7d.as_slice()?;
    Ok(PyArray1::from_vec(py, fatcrash_core::multiscale::multiscale_signals_slice(s1, s3, s7)))
}

// ── Bubble detection ─────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (data, min_window=None, n_sims=200, seed=42))]
fn gsadf_test<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    min_window: Option<usize>,
    n_sims: Option<usize>,
    seed: Option<u64>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, (f64, f64, f64))> {
    let data_slice = data.as_slice()?;
    let (gsadf, bsadf, cvs) = py.allow_threads(|| {
        core_gsadf::gsadf_test_slice(data_slice, min_window, n_sims, seed)
    });
    Ok((gsadf, PyArray1::from_vec(py, bsadf), cvs))
}

#[pyfunction]
#[pyo3(signature = (data, window, min_window=None))]
fn gsadf_rolling<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    min_window: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let n = data_slice.len();
    let min_win = min_window
        .unwrap_or_else(|| (0.01 * window as f64 + 1.8 * (window as f64).sqrt()).floor() as usize)
        .max(4);
    if window > n || window < min_win + 1 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]));
    }
    let result: Vec<f64> = py.allow_threads(|| {
        use rayon::prelude::*;
        let data_vec = data_slice.to_vec();
        (0..n).into_par_iter().map(|i| {
            if i < window - 1 { return f64::NAN; }
            let slice = &data_vec[(i + 1 - window)..=i];
            let bsadf = core_gsadf::bsadf_sequence(slice, min_win);
            let gsadf = bsadf.iter().filter(|x| x.is_finite()).copied().fold(f64::NEG_INFINITY, f64::max);
            if gsadf.is_finite() { gsadf } else { f64::NAN }
        }).collect()
    });
    Ok(PyArray1::from_vec(py, result))
}

// ── Regime detection — realized variance ─────────────────

#[pyfunction]
#[pyo3(signature = (returns, window=21))]
fn realized_variance(returns: PyReadonlyArray1<'_, f64>, window: usize) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    if window == 0 || window > n { return Ok(f64::NAN); }
    Ok(core_realized_var::compute_realized_variance(&returns[(n - window)..]))
}

#[pyfunction]
#[pyo3(signature = (high, low, window=21))]
fn realized_variance_parkinson(high: PyReadonlyArray1<'_, f64>, low: PyReadonlyArray1<'_, f64>, window: usize) -> PyResult<f64> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let n = high.len();
    if window == 0 || window > n || low.len() != n { return Ok(f64::NAN); }
    Ok(core_realized_var::compute_realized_variance_parkinson(&high[(n - window)..], &low[(n - window)..]))
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, window=21))]
fn realized_variance_gk(open: PyReadonlyArray1<'_, f64>, high: PyReadonlyArray1<'_, f64>, low: PyReadonlyArray1<'_, f64>, close: PyReadonlyArray1<'_, f64>, window: usize) -> PyResult<f64> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let n = open.len();
    if window == 0 || window > n || high.len() != n || low.len() != n || close.len() != n { return Ok(f64::NAN); }
    Ok(core_realized_var::compute_realized_variance_gk(&open[(n - window)..], &high[(n - window)..], &low[(n - window)..], &close[(n - window)..]))
}

#[pyfunction]
#[pyo3(signature = (returns, window=21, step=1))]
fn realized_variance_rolling<'py>(py: Python<'py>, returns: PyReadonlyArray1<'py, f64>, window: usize, step: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    let step = step.max(1);
    if window == 0 || window > n { return Ok(PyArray1::from_vec(py, vec![f64::NAN; n])); }
    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n {
        result[i] = core_realized_var::compute_realized_variance(&returns[(i + 1 - window)..=i]);
        i += step;
    }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, window=21, step=1))]
fn realized_variance_parkinson_rolling<'py>(py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, window: usize, step: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let n = high.len();
    let step = step.max(1);
    if window == 0 || window > n || low.len() != n { return Ok(PyArray1::from_vec(py, vec![f64::NAN; n])); }
    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n {
        result[i] = core_realized_var::compute_realized_variance_parkinson(&high[(i + 1 - window)..=i], &low[(i + 1 - window)..=i]);
        i += step;
    }
    Ok(PyArray1::from_vec(py, result))
}

// ── Regime detection — jump risk ─────────────────────────

#[pyfunction]
#[pyo3(signature = (returns, window=21))]
fn bipower_variation(returns: PyReadonlyArray1<'_, f64>, window: usize) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    if window < 2 || window > n { return Ok(f64::NAN); }
    Ok(core_jump::compute_bipower_variation(&returns[(n - window)..]))
}

#[pyfunction]
#[pyo3(signature = (returns, window=21))]
fn jump_variance(returns: PyReadonlyArray1<'_, f64>, window: usize) -> PyResult<f64> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    if window < 2 || window > n { return Ok(f64::NAN); }
    Ok(core_jump::compute_jump_variance(&returns[(n - window)..]))
}

#[pyfunction]
#[pyo3(signature = (returns, window=63))]
fn jump_test(returns: PyReadonlyArray1<'_, f64>, window: usize) -> PyResult<(f64, f64)> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    if window < 3 || window > n { return Ok((f64::NAN, f64::NAN)); }
    Ok(core_jump::compute_jump_test(&returns[(n - window)..]))
}

#[pyfunction]
#[pyo3(signature = (returns, window=21, step=1))]
fn bipower_variation_rolling<'py>(py: Python<'py>, returns: PyReadonlyArray1<'py, f64>, window: usize, step: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    let step = step.max(1);
    if window < 2 || window > n { return Ok(PyArray1::from_vec(py, vec![f64::NAN; n])); }
    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n { result[i] = core_jump::compute_bipower_variation(&returns[(i + 1 - window)..=i]); i += step; }
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window=21, step=1))]
fn jump_variance_rolling<'py>(py: Python<'py>, returns: PyReadonlyArray1<'py, f64>, window: usize, step: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let n = returns.len();
    let step = step.max(1);
    if window < 2 || window > n { return Ok(PyArray1::from_vec(py, vec![f64::NAN; n])); }
    let mut result = vec![f64::NAN; n];
    let mut i = window - 1;
    while i < n { result[i] = core_jump::compute_jump_variance(&returns[(i + 1 - window)..=i]); i += step; }
    Ok(PyArray1::from_vec(py, result))
}

// ── Regime detection — critical slowing down ─────────────

#[pyfunction]
#[pyo3(signature = (data, window=252))]
fn rolling_ar1<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    Ok(PyArray1::from_vec(py, core_csd::compute_rolling_ar1(data, window)))
}

#[pyfunction]
#[pyo3(signature = (data, window=252))]
fn rolling_variance<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    Ok(PyArray1::from_vec(py, core_csd::compute_rolling_variance(data, window)))
}

#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (data, window=252, roc_window=63))]
fn csd_indicator<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, window: usize, roc_window: usize) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let n = data.len();
    let ar1 = core_csd::compute_rolling_ar1(data, window);
    let var = core_csd::compute_rolling_variance(data, window);
    let ar1_roc = core_csd::rate_of_change(&ar1, roc_window);
    let var_roc = core_csd::rate_of_change(&var, roc_window);
    let mut csd_signal = vec![0.0_f64; n];
    for i in 0..n {
        if !ar1_roc[i].is_nan() && !var_roc[i].is_nan() && ar1_roc[i] > 0.0 && var_roc[i] > 0.0 {
            csd_signal[i] = 1.0;
        }
    }
    Ok((PyArray1::from_vec(py, ar1_roc), PyArray1::from_vec(py, var_roc), PyArray1::from_vec(py, csd_signal)))
}

// ── Regime detection — Hamilton filter ───────────────────

#[pyfunction]
#[pyo3(signature = (data, mu, sigma, p00, p11))]
fn hamilton_filter<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, mu: [f64; 2], sigma: [f64; 2], p00: f64, p11: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    Ok(PyArray1::from_vec(py, core_hamilton::forward_filter(data, mu, sigma, p00, p11)))
}

#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (data, n_restarts=10))]
fn hamilton_fit<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, n_restarts: usize) -> PyResult<(f64, f64, f64, f64, f64, f64, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let (mu0, sigma0, mu1, sigma1, p00, p11, filtered) = core_hamilton::hamilton_fit_slice(data, Some(n_restarts));
    Ok((mu0, sigma0, mu1, sigma1, p00, p11, PyArray1::from_vec(py, filtered)))
}

#[pyfunction]
#[pyo3(signature = (data, mu, sigma, p00, p11))]
fn hamilton_smooth<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, mu: [f64; 2], sigma: [f64; 2], p00: f64, p11: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    Ok(PyArray1::from_vec(py, core_hamilton::smooth(data, mu, sigma, p00, p11)))
}

// ── Utils ────────────────────────────────────────────────

#[pyfunction]
fn log_returns<'py>(py: Python<'py>, prices: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    match fatcrash_core::utils::log_returns_slice(prices) {
        Ok(v) => Ok(PyArray1::from_vec(py, v)),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

#[pyfunction]
fn log_prices<'py>(py: Python<'py>, prices: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    match fatcrash_core::utils::log_prices_slice(prices) {
        Ok(v) => Ok(PyArray1::from_vec(py, v)),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

// ── Module registration ──────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tail estimators
    m.add_function(wrap_pyfunction!(hill_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(hill_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(kappa_metric, m)?)?;
    m.add_function(wrap_pyfunction!(kappa_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(taleb_kappa, m)?)?;
    m.add_function(wrap_pyfunction!(taleb_kappa_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(pickands_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(pickands_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(hurst_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(hurst_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(dfa_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(dfa_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(deh_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(deh_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(qq_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(qq_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(maxsum_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(maxsum_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(momentum_score, m)?)?;
    m.add_function(wrap_pyfunction!(momentum_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(momentum_reversal, m)?)?;
    m.add_function(wrap_pyfunction!(momentum_reversal_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(price_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(price_velocity_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(realized_skewness, m)?)?;
    m.add_function(wrap_pyfunction!(realized_skewness_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(amihud_illiquidity, m)?)?;
    m.add_function(wrap_pyfunction!(amihud_illiquidity_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(absorption_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(absorption_ratio_rolling, m)?)?;

    // EVT
    m.add_function(wrap_pyfunction!(gpd_fit, m)?)?;
    m.add_function(wrap_pyfunction!(gpd_var_es, m)?)?;
    m.add_function(wrap_pyfunction!(gev_fit, m)?)?;

    // LPPLS
    m.add_function(wrap_pyfunction!(lppls_fit, m)?)?;
    m.add_function(wrap_pyfunction!(lppls_confidence, m)?)?;
    m.add_function(wrap_pyfunction!(lppls_solve_linear, m)?)?;

    // Multiscale
    m.add_function(wrap_pyfunction!(multiscale_signals, m)?)?;

    // Bubble detection
    m.add_function(wrap_pyfunction!(gsadf_test, m)?)?;
    m.add_function(wrap_pyfunction!(gsadf_rolling, m)?)?;

    // Regime detection — realized variance
    m.add_function(wrap_pyfunction!(realized_variance, m)?)?;
    m.add_function(wrap_pyfunction!(realized_variance_parkinson, m)?)?;
    m.add_function(wrap_pyfunction!(realized_variance_gk, m)?)?;
    m.add_function(wrap_pyfunction!(realized_variance_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(realized_variance_parkinson_rolling, m)?)?;

    // Regime detection — jump risk
    m.add_function(wrap_pyfunction!(bipower_variation, m)?)?;
    m.add_function(wrap_pyfunction!(jump_variance, m)?)?;
    m.add_function(wrap_pyfunction!(jump_test, m)?)?;
    m.add_function(wrap_pyfunction!(bipower_variation_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(jump_variance_rolling, m)?)?;

    // Regime detection — critical slowing down
    m.add_function(wrap_pyfunction!(rolling_ar1, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_variance, m)?)?;
    m.add_function(wrap_pyfunction!(csd_indicator, m)?)?;

    // Regime detection — Hamilton filter
    m.add_function(wrap_pyfunction!(hamilton_filter, m)?)?;
    m.add_function(wrap_pyfunction!(hamilton_fit, m)?)?;
    m.add_function(wrap_pyfunction!(hamilton_smooth, m)?)?;

    // Utils
    m.add_function(wrap_pyfunction!(log_returns, m)?)?;
    m.add_function(wrap_pyfunction!(log_prices, m)?)?;

    Ok(())
}
