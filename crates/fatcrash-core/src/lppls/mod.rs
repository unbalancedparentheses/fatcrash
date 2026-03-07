//! Log-Periodic Power Law Singularity (LPPLS) bubble detection.
//!
//! The LPPLS equation models super-exponential price growth with log-periodic
//! oscillations — the fingerprint of a speculative bubble approaching a
//! critical point `tc`:
//!
//! `ln p(t) = A + B|tc - t|^m + C|tc - t|^m cos(ω ln|tc - t| + φ)`
//!
//! - [`model`] — LPPLS parameter struct and OLS solver for the linear
//!   parameters (A, B, C1, C2) given nonlinear parameters (tc, m, ω).
//! - [`fitter`] — Full CMA-ES (Hansen 2001) optimizer for the 3 nonlinear
//!   parameters with rank-μ covariance adaptation.
//! - [`filter`] — Sornette filter: m ∈ \[0.1, 0.9\], ω ∈ \[6, 13\], B < 0.
//! - [`confidence`] — DS LPPLS confidence indicator: fraction of sub-windows
//!   producing a qualifying fit. Parallelized with [`rayon`].
//!
//! # Reference
//!
//! Sornette, D. (2003). *Why Stock Markets Crash*. Princeton University Press.

pub mod confidence;
pub mod filter;
pub mod fitter;
pub mod model;
