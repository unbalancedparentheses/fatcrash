//! Fat-tail crash detection library.
//!
//! `fatcrash-core` provides 18+ crash detection methods across five families,
//! each measuring a different property of the distribution or dynamics:
//!
//! - [`tail`] — Tail index estimation (Hill, Pickands, DEH, QQ), convergence
//!   diagnostics (Taleb kappa, max-to-sum), persistence (Hurst, DFA, spectral),
//!   and microstructure signals (momentum, velocity, Amihud, absorption ratio).
//! - [`evt`] — Extreme Value Theory: GPD (Peaks-over-Threshold) for VaR/ES
//!   and GEV (block maxima) for tail classification.
//! - [`lppls`] — Log-Periodic Power Law Singularity bubble detection with
//!   CMA-ES fitting, Sornette filter, and multi-window confidence indicator.
//! - [`bubble`] — GSADF explosive unit root test (Phillips-Shi-Yu 2015).
//! - [`regime`] — Hamilton 2-state HMM, Critical Slowing Down, realized
//!   variance (simple, Parkinson, Garman-Klass), and BNS jump decomposition.
//!
//! Heavy computations (GSADF Monte Carlo, Hamilton EM restarts, LPPLS
//! confidence) are parallelized with [`rayon`].

pub mod bubble;
pub mod evt;
pub mod lppls;
pub mod multiscale;
pub mod regime;
pub mod tail;
pub mod utils;
