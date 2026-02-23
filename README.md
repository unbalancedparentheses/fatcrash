# fatcrash

Crash detection via fat-tail statistics. Works with any asset: Bitcoin, gold, S&P 500, etc.

Python + Rust (PyO3) — performance-critical numerical work in Rust, everything else in Python.

## Architecture

Full pipeline: LPPLS bubble detection, Extreme Value Theory, Hill estimator, Kappa metric, multiscale analysis, with optional Deep LPPLS.

### What goes in Rust (via PyO3)

| Component | Why Rust |
|-----------|----------|
| **LPPLS fitter** | O(1000) nonlinear fits per anchor date, CMA-ES + LM. Most expensive computation. |
| **LPPLS confidence** | Nested window fits parallelized with `rayon` |
| **GEV/GPD fitting** | Rolling EVT (re-fit at each timestep) needs speed |
| **Hill estimator** | Simple math but called at every rolling step |
| **Kappa metric** | Gaussian benchmarks via simulation |

Key Rust crates: `pyo3`, `numpy` (PyO3), `nalgebra`, `ndarray`, `rayon`.

### What stays in Python

- Data ingestion (CCXT, HTTP, CSV/Parquet I/O)
- Visualization (plotly, matplotlib)
- CLI (typer) and service (FastAPI)
- Deep LPPLS (PyTorch)
- Signal aggregation logic
- Notebooks

## Data Flow

```
Exchange/CSV → ingest.py → transforms.py → [log_prices, log_returns]
                                                    │
                    ┌───────────┬───────────┬───────┴───────┐
                    ▼           ▼           ▼               ▼
               LPPLS(Rust)  EVT(Rust)  Tail(Rust)    DeepLPPLS(Torch)
               tc, confidence VaR, ES   Hill α, κ     tc prediction
                    │           │           │               │
                    └───────────┴───────────┴───────┬───────┘
                                                    ▼
                                         signals.py (aggregate)
                                         → CrashSignal(prob, horizon)
                                                    │
                                    ┌───────────┬───┴───────┐
                                    ▼           ▼           ▼
                                  CLI       FastAPI     Notebook
```

## Signal Aggregation Weights

| Indicator | Weight | Logic |
|-----------|--------|-------|
| LPPLS confidence (DS LPPLS) | 0.30 | Fraction of qualifying fits |
| LPPLS tc proximity | 0.15 | How close predicted crash to now |
| GPD VaR exceedance | 0.15 | Current returns vs tail risk |
| Kappa regime change | 0.10 | Rolling kappa shifts |
| Hill tail thinning | 0.10 | Declining alpha = thicker tails |
| Deep LPPLS | 0.10 | Neural tc prediction |
| Multiscale agreement | 0.10 | Cross-timeframe confirmation |

## LPPLS Design

The hardest component. Key decisions:
1. **Reduce 7→3 nonlinear params**: Solve (A, B, C) analytically via OLS for each candidate (tc, m, ω)
2. **CMA-ES global search** → **LM local refinement** (standard in literature)
3. **Sornette filter**: m ∈ [0.1, 0.9], ω ∈ [2, 25], damping ratio check
4. **Nested windows with rayon**: ~1000 (t1, t2) pairs per anchor date, embarrassingly parallel

## Implementation Phases

### Phase 1: Skeleton + Data Layer
- Repo structure, maturin build, CI
- `data/ingest.py`, `transforms.py`, `cache.py`
- Minimal Rust `lib.rs` verifying PyO3 bridge
- Notebook 01 (data exploration)

### Phase 2: Tail Estimators (Rust)
- `hill.rs` with Taleb bias corrections
- `kappa.rs` with Gaussian benchmarks
- PyO3 bindings, Python wrapper, viz
- Notebook 04

### Phase 3: EVT (Rust)
- `gev.rs` (MLE via L-BFGS)
- `gpd.rs` (Grimshaw method, threshold selection)
- Rolling VaR/ES, Python wrapper, viz
- Notebook 03

### Phase 4: LPPLS Core (Rust) — hardest phase
- `model.rs`, `filter.rs`, `fitter.rs`, `confidence.rs`
- CMA-ES + LM hybrid fitting
- Rayon-parallelized nested windows
- Validate against 2017/2021 BTC bubbles
- Notebook 02

### Phase 5: Aggregation + CLI
- `multiscale.rs`, `signals.py`, `calibration.py`
- CLI commands: detect, backtest, plot
- Historical backtest notebook

### Phase 6: Deep LPPLS + Service (optional)
- P-LNN / M-LNN PyTorch implementation
- FastAPI service with webhooks
- Notebook 07

## Verification

1. **Build**: `maturin develop` compiles Rust + installs Python package
2. **Unit tests**: `cargo test` (Rust) + `pytest` (Python) — test against known distributions
3. **Backtest**: Run against historical crashes — should show elevated signals in preceding weeks
4. **CLI smoke test**: `fatcrash detect --asset BTC` fetches live data and produces output
5. **False positive check**: Quiet periods shouldn't show persistent high crash probability

## Development

```bash
# Install
maturin develop --release

# Test
cargo test
pytest

# CLI
fatcrash detect --asset BTC
fatcrash backtest --asset BTC --start 2017-01-01 --end 2018-06-01
fatcrash plot --asset GOLD --indicator lppls
fatcrash serve --port 8000
```
