# Architecture & Contributing

## Cargo Workspace Structure

fatcrash is a Cargo workspace with 3 crates:

```
fatcrash/
├── crates/
│   ├── fatcrash-core/        # Pure Rust computation library (no PyO3)
│   │   └── src/
│   │       ├── tail/         # Hill, Pickands, DEH, QQ, Kappa, MaxSum, Hurst,
│   │       │                 # DFA, Spectral, Momentum, Velocity, Amihud
│   │       ├── evt/          # GPD, GEV
│   │       ├── lppls/        # LPPLS fit (CMA-ES), confidence, Sornette filter
│   │       ├── bubble/       # GSADF
│   │       ├── regime/       # Hamilton HMM, CSD, RealizedVar, Jump
│   │       └── multiscale.rs
│   ├── fatcrash-py/          # Thin PyO3 wrappers (cdylib)
│   │   └── src/lib.rs        # #[pymodule] → fatcrash._core
│   └── fatcrash-tui/         # Native Rust TUI monitor (ratatui)
│       └── src/
│           ├── main.rs       # CLI (clap)
│           ├── scanner.rs    # Runs all 18 methods via fatcrash-core
│           ├── signals.rs    # Signal aggregation + LPPLS-first architecture
│           ├── data.rs       # Yahoo Finance + CoinGecko HTTP fetching
│           ├── cache.rs      # CSV cache (~/.cache/fatcrash/)
│           ├── config.rs     # Watchlist (31 assets)
│           └── tui/          # Watchlist → Detail → Method drill-down views
└── python/                   # Python package
    └── fatcrash/
        ├── __init__.py       # Top-level convenience imports
        ├── _core.so          # Compiled Rust extension (built by maturin)
        ├── _core.pyi         # Type stubs for the Rust extension
        ├── indicators/       # Python wrappers returning dataclasses
        │   ├── tail_indicator.py
        │   ├── evt_indicator.py
        │   ├── bubble_indicator.py
        │   ├── lppls_indicator.py
        │   ├── regime_indicator.py
        │   └── vol_indicator.py
        ├── aggregator/       # Signal combination and calibration
        │   ├── signals.py
        │   └── calibration.py
        ├── data/             # Data ingestion and transforms
        │   ├── ingest.py
        │   ├── transforms.py
        │   └── cache.py
        ├── nn/               # Neural network methods (PyTorch)
        └── cli/              # CLI commands
```

## How the Rust → Python Bridge Works

```
fatcrash-core (pure Rust)
    ↓
fatcrash-py (PyO3 #[pyfunction] wrappers)
    ↓  built by maturin → fatcrash/_core.so
Python: fatcrash._core.hill_estimator(returns, k=None)
    ↓  wrapped by
Python: fatcrash.indicators.tail_indicator.estimate_tail_index(returns, k=None) → TailEstimate
    ↓  re-exported by
Python: fatcrash.estimate_tail_index(returns, k=None) → TailEstimate
```

**Layer 1: `fatcrash-core`** — Pure Rust. No PyO3 dependency. All algorithms implemented here. Can be used as a standalone Rust library. Uses rayon for parallelism (CMA-ES, GSADF Monte Carlo, Hamilton EM restarts).

**Layer 2: `fatcrash-py`** — Thin PyO3 wrappers. Each function in `src/lib.rs` takes numpy arrays (via `PyReadonlyArrayDyn`) and returns Python-compatible types (floats, tuples, lists). Registered as a `#[pymodule]` named `_core`.

**Layer 3: Python wrappers** — Each `indicators/*.py` file imports from `fatcrash._core` and wraps results in dataclasses with named fields, interpretation logic (e.g., `is_fat_tail`, `regime` labels), and docstrings. These provide the user-facing API.

**Layer 4: Re-exports** — `fatcrash/__init__.py` re-exports everything so users can do `from fatcrash import estimate_tail_index`.

**Type stubs:** `_core.pyi` provides type hints for the Rust extension, enabling IDE autocomplete and type checking.

## How to Add a New Method

### Step 1: Rust implementation in `fatcrash-core`

Create the estimator in the appropriate module (e.g., `crates/fatcrash-core/src/tail/my_method.rs`):

```rust
pub fn my_estimator(data: &[f64], param: usize) -> f64 {
    // implementation
}

pub fn my_estimator_rolling(data: &[f64], window: usize, param: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    for i in window..n {
        result[i] = my_estimator(&data[i - window..i], param);
    }
    result
}
```

Register in the module's `mod.rs` and re-export from `lib.rs`.

### Step 2: PyO3 binding in `fatcrash-py`

Add a `#[pyfunction]` in `crates/fatcrash-py/src/lib.rs`:

```rust
#[pyfunction]
#[pyo3(signature = (data, *, param = 100))]
fn my_estimator(data: PyReadonlyArrayDyn<f64>, param: usize) -> f64 {
    let slice = data.as_slice().expect("contiguous array required");
    fatcrash_core::tail::my_method::my_estimator(slice, param)
}
```

Register it in the `#[pymodule]` function:

```rust
m.add_function(wrap_pyfunction!(my_estimator, m)?)?;
```

### Step 3: Python wrapper

Create or extend `python/fatcrash/indicators/tail_indicator.py`:

```python
from fatcrash._core import my_estimator as _my_estimator

@dataclass
class MyEstimate:
    value: float
    is_significant: bool

def estimate_my_method(data: npt.NDArray[np.float64], param: int = 100) -> MyEstimate:
    """Estimate my method."""
    value = _my_estimator(data, param=param)
    return MyEstimate(value=value, is_significant=value > some_threshold)
```

### Step 4: Add to exports

Add to `python/fatcrash/indicators/__init__.py` and `python/fatcrash/__init__.py`.

### Step 5: Signal converter

Add a signal converter in `python/fatcrash/aggregator/signals.py`:

```python
def my_method_signal(value: float) -> float:
    """Convert my method output to [0, 1] signal."""
    if np.isnan(value):
        return 0.0
    return np.clip(value / some_max, 0.0, 1.0)
```

### Step 6: Weight

Add to `DEFAULT_WEIGHTS` in `signals.py`:

```python
DEFAULT_WEIGHTS = {
    ...
    "my_method_signal": 0.05,
}
```

### Step 7: Type stubs

Add the function signature to `python/fatcrash/_core.pyi`.

### Step 8: Tests

- Rust: `cargo test` in `crates/fatcrash-core/`
- Python bridge: add test in `tests/python/test_rust_bridge.py`
- Integration: add to `tests/python/test_methods_comparison.py`

## Testing

```bash
# Rust tests
cargo test

# Build Python extension
maturin develop --release

# Python tests
pytest tests/python/

# Full pipeline
cargo test && maturin develop --release && pytest
```

## TUI Architecture

The TUI (`fatcrash-tui`) is a standalone Rust binary that uses `fatcrash-core` directly (no Python).

**Data flow:**
1. `data.rs` fetches OHLCV from Yahoo Finance / CoinGecko
2. `cache.rs` manages CSV cache in `~/.cache/fatcrash/` (24h expiry)
3. `scanner.rs` runs all 18 methods on each asset via `fatcrash-core`
4. `signals.rs` aggregates results using LPPLS-first architecture
5. `tui/` renders three views using ratatui:
   - **Watchlist**: All 31 assets sorted by crash probability
   - **Detail**: Single asset with all 18 methods, signal values, weights
   - **Method drill-down**: Raw intermediate values for a specific method

**LPPLS-first architecture:** If neither LPPLS confidence nor GSADF detects a bubble (both < 0.1), crash probability is set to zero. Other methods (tail, regime, liquidity) only contribute when a bubble signal fires — they act as confirmation/denial multipliers.

**Background scanning:** Data fetching and computation happen on background threads (tokio). The TUI never blocks on computation. Auto-refresh every 5 minutes.
