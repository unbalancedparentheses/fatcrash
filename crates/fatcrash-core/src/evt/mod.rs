//! Extreme Value Theory: GPD (Peaks-over-Threshold) and GEV (block maxima).
//!
//! - [`gpd`] — Generalized Pareto Distribution fitted via MLE to exceedances
//!   over a threshold. Produces Value-at-Risk and Expected Shortfall estimates.
//! - [`gev`] — Generalized Extreme Value distribution fitted to block maxima
//!   via adaptive coordinate descent. Classifies the tail as Fréchet (heavy),
//!   Gumbel (exponential), or Weibull (bounded).
//!
//! # References
//!
//! - Balkema & de Haan (1974), Pickands (1975) — GPD limit theorem.
//! - McNeil, Frey & Embrechts (2005). *Quantitative Risk Management*.

pub mod gev;
pub mod gpd;
