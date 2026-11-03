//! Extreme Value Theory: GPD (Peaks-over-Threshold) and GEV (block maxima).
//!
//! - [`gpd`] — Generalized Pareto Distribution fitted via MLE to exceedances
//!   over a threshold. Produces Value-at-Risk and Expected Shortfall estimates.
//! - [`gev`] — Generalized Extreme Value distribution fitted to block maxima
//!   via adaptive coordinate descent. Classifies the tail as Fréchet (heavy),
//!   Gumbel (exponential), or Weibull (bounded).
//! - [`expectile`] — Extreme expectiles and expectile-based ES (coherent and
//!   elicitable), extrapolated with Hill and Weissman.
//! - [`mixture`] — Threshold-free lognormal–GPD mixture fitted by EM
//!   (Bee & Santi 2026), with VaR/ES from the fitted mixture.
//!
//! # References
//!
//! - Balkema & de Haan (1974), Pickands (1975) — GPD limit theorem.
//! - McNeil, Frey & Embrechts (2005). *Quantitative Risk Management*.

pub mod expectile;
pub mod gev;
pub mod gpd;
pub mod mixture;
