//! Tail estimation, persistence analysis, and microstructure signals.
//!
//! 14 submodules covering three aspects of tail behavior:
//!
//! **Tail index estimation** — How heavy are the tails?
//! - [`hill`] — Hill estimator (Fréchet domain only, most efficient).
//! - [`pickands`] — Pickands estimator (all three EVT domains).
//! - [`deh`] — Dekkers-Einmahl-de Haan moment estimator (all domains).
//! - [`qq`] — QQ-plot regression estimator.
//! - [`kappa`] — Taleb's kappa (CLT convergence rate) and max-stability kappa.
//! - [`maxsum`] — Maximum-to-sum ratio (infinite variance diagnostic).
//!
//! **Persistence & long memory** — Are shocks transient or persistent?
//! - [`hurst`] — Hurst exponent via R/S analysis.
//! - [`dfa`] — Detrended Fluctuation Analysis.
//! - [`spectral`] — GPH spectral exponent (frequency domain long memory).
//!
//! **Microstructure & flow signals**
//! - [`momentum`] — Trailing momentum and reversal detection.
//! - [`velocity`] — Rate of change of realized volatility.
//! - [`skewness`] — Realized skewness.
//! - [`amihud`] — Amihud illiquidity ratio.
//! - [`absorption`] — Absorption ratio (systemic risk from PCA).

pub mod absorption;
pub mod amihud;
pub mod deh;
pub mod dfa;
pub mod hill;
pub mod hurst;
pub mod kappa;
pub mod maxsum;
pub mod momentum;
pub mod pickands;
pub mod qq;
pub mod skewness;
pub mod spectral;
pub mod velocity;
