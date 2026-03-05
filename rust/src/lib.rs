use pyo3::prelude::*;

mod bubble;
mod evt;
mod lppls;
mod multiscale;
mod regime;
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
    m.add_function(wrap_pyfunction!(tail::momentum::momentum_score, m)?)?;
    m.add_function(wrap_pyfunction!(tail::momentum::momentum_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(tail::momentum::momentum_reversal, m)?)?;
    m.add_function(wrap_pyfunction!(
        tail::momentum::momentum_reversal_rolling,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tail::velocity::price_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(
        tail::velocity::price_velocity_rolling,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tail::skewness::realized_skewness, m)?)?;
    m.add_function(wrap_pyfunction!(
        tail::skewness::realized_skewness_rolling,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tail::amihud::amihud_illiquidity, m)?)?;
    m.add_function(wrap_pyfunction!(
        tail::amihud::amihud_illiquidity_rolling,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(tail::absorption::absorption_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(
        tail::absorption::absorption_ratio_rolling,
        m
    )?)?;

    // EVT
    m.add_function(wrap_pyfunction!(evt::gpd::gpd_fit, m)?)?;
    m.add_function(wrap_pyfunction!(evt::gpd::gpd_var_es, m)?)?;
    m.add_function(wrap_pyfunction!(evt::gev::gev_fit, m)?)?;

    // LPPLS
    m.add_function(wrap_pyfunction!(lppls::fitter::lppls_fit, m)?)?;
    m.add_function(wrap_pyfunction!(lppls::confidence::lppls_confidence, m)?)?;
    m.add_function(wrap_pyfunction!(
        lppls::solve_linear_py::lppls_solve_linear,
        m
    )?)?;

    // Multiscale
    m.add_function(wrap_pyfunction!(multiscale::multiscale_signals, m)?)?;

    // Bubble detection
    m.add_function(wrap_pyfunction!(bubble::gsadf::gsadf_test, m)?)?;
    m.add_function(wrap_pyfunction!(bubble::gsadf::gsadf_rolling, m)?)?;

    // Regime detection — realized variance
    m.add_function(wrap_pyfunction!(regime::realized_var::realized_variance, m)?)?;
    m.add_function(wrap_pyfunction!(
        regime::realized_var::realized_variance_parkinson,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        regime::realized_var::realized_variance_gk,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        regime::realized_var::realized_variance_rolling,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        regime::realized_var::realized_variance_parkinson_rolling,
        m
    )?)?;

    // Regime detection — jump risk
    m.add_function(wrap_pyfunction!(regime::jump::bipower_variation, m)?)?;
    m.add_function(wrap_pyfunction!(regime::jump::jump_variance, m)?)?;
    m.add_function(wrap_pyfunction!(regime::jump::jump_test, m)?)?;
    m.add_function(wrap_pyfunction!(
        regime::jump::bipower_variation_rolling,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        regime::jump::jump_variance_rolling,
        m
    )?)?;

    // Regime detection — critical slowing down
    m.add_function(wrap_pyfunction!(regime::csd::rolling_ar1, m)?)?;
    m.add_function(wrap_pyfunction!(regime::csd::rolling_variance, m)?)?;
    m.add_function(wrap_pyfunction!(regime::csd::csd_indicator, m)?)?;

    // Regime detection — Hamilton filter
    m.add_function(wrap_pyfunction!(regime::hamilton::hamilton_filter, m)?)?;
    m.add_function(wrap_pyfunction!(regime::hamilton::hamilton_fit, m)?)?;
    m.add_function(wrap_pyfunction!(regime::hamilton::hamilton_smooth, m)?)?;

    // Utils
    m.add_function(wrap_pyfunction!(utils::log_returns, m)?)?;
    m.add_function(wrap_pyfunction!(utils::log_prices, m)?)?;

    Ok(())
}
