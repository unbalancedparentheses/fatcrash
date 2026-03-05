//! Critical Slowing Down — early warning for regime transitions.
//!
//! TODO: Implement rolling AR(1) coefficient estimation:
//!   - Fit x_t = α + β*x_{t-1} + ε via OLS over rolling window W=252
//!   - β trending toward 1 = system losing resilience
//!
//! TODO: Implement rolling variance tracking:
//!   - Use Welford's online algorithm for numerically stable computation
//!   - Track 63-day rate of change of variance
//!
//! TODO: Combined CSD indicator:
//!   - Dual increase (rising β AND rising variance) = elevated risk
//!   - Apply to credit spreads, VIX, and cross-asset correlation eigenvalue
//!
//! Reference: Scheffer et al. (2009), "Early-Warning Signals for Critical
//! Transitions." Nature, 461, 53-59.
