//! Market regime detection algorithms.
//!
//! Stubs for computationally intensive regime signal algorithms.
//! These will be exposed to Python via PyO3 once implemented.

pub mod csd;
pub mod hamilton;
pub mod hawkes;
pub mod jump;
pub mod realized_var;
