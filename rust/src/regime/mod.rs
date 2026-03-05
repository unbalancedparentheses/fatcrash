//! Market regime detection algorithms.
//!
//! Realized variance estimators, jump risk decomposition,
//! critical slowing down indicators, and Hamilton filter HMM.

pub mod csd;
pub mod hamilton;
pub mod jump;
pub mod realized_var;
