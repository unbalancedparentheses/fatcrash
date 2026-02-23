use pyo3::prelude::*;

mod bubble;
mod evt;
mod lppls;
mod multiscale;
mod tail;
mod utils;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tail estimators
    m.add_function(wrap_pyfunction!(tail::hill::hill_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(tail::hill::hill_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::kappa::kappa_metric, m)?)?;
    m.add_function(wrap_pyfunction!(tail::kappa::kappa_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::kappa::taleb_kappa, m)?)?;
    m.add_function(wrap_pyfunction!(tail::kappa::taleb_kappa_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::pickands::pickands_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(tail::pickands::pickands_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::hurst::hurst_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(tail::hurst::hurst_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::dfa::dfa_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(tail::dfa::dfa_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::deh::deh_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(tail::deh::deh_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::qq::qq_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(tail::qq::qq_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::maxsum::maxsum_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(tail::maxsum::maxsum_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::spectral::spectral_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(tail::spectral::spectral_rolling, m)?)?;

    // EVT
    m.add_function(wrap_pyfunction!(evt::gpd::gpd_fit, m)?)?;
    m.add_function(wrap_pyfunction!(evt::gpd::gpd_var_es, m)?)?;
    m.add_function(wrap_pyfunction!(evt::gev::gev_fit, m)?)?;

    // LPPLS
    m.add_function(wrap_pyfunction!(lppls::fitter::lppls_fit, m)?)?;
    m.add_function(wrap_pyfunction!(lppls::confidence::lppls_confidence, m)?)?;

    // Multiscale
    m.add_function(wrap_pyfunction!(multiscale::multiscale_signals, m)?)?;

    // Bubble detection
    m.add_function(wrap_pyfunction!(bubble::gsadf::gsadf_test, m)?)?;
    m.add_function(wrap_pyfunction!(bubble::gsadf::gsadf_rolling, m)?)?;

    // Utils
    m.add_function(wrap_pyfunction!(utils::log_returns, m)?)?;
    m.add_function(wrap_pyfunction!(utils::log_prices, m)?)?;

    Ok(())
}
