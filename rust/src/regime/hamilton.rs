//! Hamilton Filter — 2-state Hidden Markov Model for regime detection.
//!
//! TODO: Implement forward filter (Hamilton filter):
//!   - States: s_t ∈ {0=normal, 1=stressed}
//!   - Observation model: y_t | s_t=j ~ N(μ_j, σ_j²)
//!   - Transition matrix P with p00 and p11
//!   - Forward pass: predict → density → Bayesian update
//!   - Output: P(s_t = stressed | data_1..t)
//!
//! TODO: Parameter estimation via EM (Baum-Welch):
//!   - E-step: run forward-backward, compute smoothed probabilities
//!   - M-step: update μ_j, σ_j, P[i,j] from weighted sufficient statistics
//!   - Multiple random restarts to avoid local optima
//!   - Work in log-space for numerical stability
//!   - Clip filtered probabilities to [1e-10, 1 - 1e-10]
//!
//! Reference: Hamilton (1989), "A New Approach to the Economic Analysis
//! of Nonstationary Time Series and the Business Cycle." Econometrica.
