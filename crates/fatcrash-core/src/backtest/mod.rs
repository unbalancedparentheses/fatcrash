//! Forecast evaluation with honest uncertainty when tail events are rare.
//!
//! - [`e_backtest`] — Anytime-valid ES backtest that needs no VaR forecast
//!   (Wang & Yu, 2026). Monitor daily, stop whenever.
//! - [`forecast_cs`] — Anytime-valid confidence sequence comparing two tail
//!   quantile forecasts (Coelho, 2026). Its effective sample size is the number
//!   of exceedances, not the number of days.

pub mod e_backtest;
pub mod forecast_cs;
