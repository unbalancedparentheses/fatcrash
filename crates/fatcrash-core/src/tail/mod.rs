//! Tail estimation, persistence analysis, and microstructure signals.
//!
//! 18 submodules covering three aspects of tail behavior:
//!
//! **Tail index estimation** — How heavy are the tails?
//! - [`hill`] — Hill estimator (Fréchet domain only, most efficient).
//! - [`pickands`] — Pickands estimator (all three EVT domains).
//! - [`deh`] — Dekkers-Einmahl-de Haan moment estimator (all domains).
//! - [`qq`] — QQ-plot regression estimator.
//! - [`kappa`] — Taleb's kappa (CLT convergence rate) and max-stability kappa.
//! - [`maxsum`] — Maximum-to-sum ratio (infinite variance diagnostic).
//! - [`conservative`] — Rescaled Hill that rarely overestimates alpha (Gösgens et al. 2026).
//! - [`reduced_bias`] — Power-mean, Lehmer-mean and reduced-bias EVI estimators.
//! - [`persistence`] — Tail persistence E[X | X > Q(u)] / Q(u) and its limit 1/(1 - xi).
//! - [`homogeneity`] — Permutation test of a common tail index across groups (Yang 2026).
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
pub mod conservative;
pub mod deh;
pub mod dfa;
pub mod hill;
pub mod homogeneity;
pub mod hurst;
pub mod kappa;
pub mod maxsum;
pub mod momentum;
pub mod persistence;
pub mod pickands;
pub mod qq;
pub mod reduced_bias;
pub mod skewness;
pub mod spectral;
pub mod velocity;
