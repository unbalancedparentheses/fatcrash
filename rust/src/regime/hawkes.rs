//! Hawkes process branching ratio estimation.
//!
//! TODO: Implement self-exciting point process model:
//!   - Intensity: λ(t) = μ + α * Σ_{t_i < t} exp(-β*(t - t_i))
//!   - Branching ratio: n = α/β (n < 1 = stable; n → 1 = critical)
//!
//! TODO: MLE via L-BFGS-B:
//!   - Log-likelihood: Σ_i ln λ(t_i) - μ*T - (α/β)*Σ_i(1 - exp(-β*(T-t_i)))
//!   - Analytic gradient available
//!   - Bounds: μ > 0, α ≥ 0, β > 0
//!   - Use Kahan compensated summation for numerical stability on long series
//!
//! TODO: Rolling estimation on 252-day windows
//!   - Event definition: VIX threshold crossings or large order flow imbalances
//!   - Rising branching ratio approaching 1 = warning signal
//!
//! Reference: Hawkes (1971); Bacry, Mastromatteo, Muzy (2015),
//! "Hawkes Processes in Finance."
