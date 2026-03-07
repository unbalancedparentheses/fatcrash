//! Market regime detection and variance decomposition.
//!
//! - [`hamilton`] — Hamilton (1989) 2-state Gaussian HMM with EM estimation,
//!   log-space forward filter, and Kim (1994) smoother. Multiple random
//!   restarts parallelized with [`rayon`].
//! - [`csd`] — Critical Slowing Down: rolling AR(1) and variance as
//!   early-warning signals for regime transitions (Scheffer et al., 2009).
//! - [`realized_var`] — Realized variance estimators: simple (sum of squared
//!   returns), Parkinson (high-low range), and Garman-Klass (OHLC).
//!   All annualized with a 252-day convention.
//! - [`jump`] — BNS jump decomposition: bipower variation, tripower
//!   quarticity, and the Barndorff-Nielsen & Shephard z-test for jumps.

pub mod csd;
pub mod hamilton;
pub mod jump;
pub mod realized_var;
