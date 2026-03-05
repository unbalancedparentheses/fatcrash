//! BNS bipower variation and jump risk decomposition.
//!
//! TODO: Bipower variation (robust to jumps):
//!   BV_t = (π/2) * (1/(W-1)) * Σ |r_i| * |r_{i-1}|
//!
//! TODO: Jump variance decomposition:
//!   CV_t = BV_t                      (continuous component)
//!   JV_t = max(RV_t - BV_t, 0)      (jump component)
//!
//! TODO: BNS jump test statistic:
//!   z_jump = (RV_t - BV_t) / sqrt(variance of RV - BV)
//!   Use realized tri-power quarticity for denominator variance
//!
//! TODO: Jump risk premium proxy:
//!   JRP_t = JV_t (when full option chain is unavailable)
//!   Spikes around market dislocations (Flash Crash, Lehman, COVID)
//!   Rises faster than continuous vol during sudden regime transitions
//!
//! Reference: Barndorff-Nielsen & Shephard (2004), "Power and Bipower
//! Variation with Stochastic Volatility and Jumps."
//! Bollerslev & Todorov (2011), "Tails, Fears, and Risk Premia."
