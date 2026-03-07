//! Explosive unit root testing for bubble detection.
//!
//! - [`gsadf`] — Generalized Sup ADF test (Phillips, Shi & Yu, 2015).
//!   Computes backward SADF sequences and the GSADF statistic, compared
//!   against Monte Carlo critical values under the null of a driftless
//!   random walk. Parallelized with [`rayon`].

pub mod gsadf;
