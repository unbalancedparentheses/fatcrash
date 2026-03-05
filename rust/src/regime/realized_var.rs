//! Realized variance estimators.
//!
//! TODO: Simple realized variance from daily returns:
//!   RV_t = (252/W) * Σ r_i²
//!   Default W = 21 (monthly), 63 (quarterly)
//!
//! TODO: Parkinson estimator (from OHLC, no intraday needed):
//!   RV_Park = (252/(4*ln2*W)) * Σ (ln(H_i/L_i))²
//!   More efficient than simple RV (~5x lower variance)
//!
//! TODO: Garman-Klass estimator (most efficient from OHLC):
//!   GK_i = 0.5*(ln(H/L))² - (2*ln2-1)*(ln(C/O))²
//!   RV_GK = 252 * (1/W) * Σ GK_i
//!
//! TODO: Rolling computation with configurable window
//!
//! These are needed for VRP computation (implied variance - realized variance)
//! and for the jump risk decomposition (BNS bipower variation).
