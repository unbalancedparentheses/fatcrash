# fatcrash

Crash detection via fat-tail statistics. Works with any asset: Bitcoin, gold, S&P 500, forex, etc.

Python + Rust (PyO3) — performance-critical numerical work in Rust, everything else in Python.

## The Methods

### Hill Estimator — "How fat are the tails?"

The Hill estimator measures the **tail index** (alpha) of a distribution. Financial returns don't follow a normal distribution — they have "fat tails", meaning extreme events happen far more often than a bell curve predicts. The 2008 crash, COVID crash, Bitcoin's -83% drops — these aren't 6-sigma events under normality, they're regular features of fat-tailed distributions.

The Hill estimator fits a power law to the tail: P(X > x) ~ x^(-alpha). Lower alpha = fatter tails = more extreme events.

- **alpha < 2**: Infinite variance (very dangerous — Cauchy-like)
- **alpha 2-4**: Fat tails, finite variance but infinite kurtosis
- **alpha > 4**: Relatively thin tails

Most financial assets have alpha between 2 and 4. When alpha drops over time, it means the tail is getting fatter — a warning sign.

### Kappa Metric — "How far from Gaussian?"

Taleb's kappa metric directly measures how much a distribution deviates from Gaussian (normal). It works by splitting the data into subsamples, computing the max of each, and comparing the mean of those maxima to the overall max.

For a Gaussian distribution, kappa equals a known benchmark. For fat-tailed distributions, kappa falls below the benchmark. The ratio kappa/benchmark tells you:

- **Ratio ~ 1.0**: Behaves like Gaussian (safe, predictable)
- **Ratio < 0.8**: Significantly fatter tails than Gaussian
- **Ratio < 0.5**: Extremely fat tails (crisis regime)

### EVT (Extreme Value Theory) — "What's the worst that can happen?"

EVT is the mathematically rigorous framework for modeling extreme events. Instead of fitting a distribution to all the data (where the bulk drowns out the tails), EVT fits only to the extremes.

**GPD (Generalized Pareto Distribution)**: Fit to losses that exceed a threshold. Gives you:
- **VaR (Value at Risk)**: "There's a 1% chance of losing more than X"
- **ES (Expected Shortfall)**: "If we do lose more than VaR, the average loss is Y"

**GEV (Generalized Extreme Value)**: Fit to block maxima (e.g., worst loss per month). The shape parameter xi tells you the tail type:
- **xi > 0**: Frechet — fat tail, power-law decay (typical for financial data)
- **xi ~ 0**: Gumbel — exponential tail decay
- **xi < 0**: Weibull — bounded tail

### LPPLS — "Is this a bubble?"

The Log-Periodic Power Law Singularity model detects **bubbles before they burst**. The theory (Didier Sornette, ETH Zurich): during a bubble, prices follow a specific pattern — super-exponential growth with accelerating oscillations that converge toward a critical time tc (the most likely crash date).

The LPPLS equation: `ln(p(t)) = A + B(tc-t)^m + C(tc-t)^m cos(w*ln(tc-t) + phi)`

Key parameters:
- **tc**: Predicted critical time (crash date)
- **m**: Power law exponent (must be 0.1-0.9 for a valid bubble)
- **omega**: Log-periodic frequency (must be 2-25)
- **B < 0**: Required — indicates super-exponential growth

The **DS LPPLS confidence indicator** fits this model across many different time windows. If a high fraction of windows produce valid bubble fits, confidence is high.

### Deep LPPLS — "Neural network bubble detection"

A physics-informed neural network (P-LNN) that predicts LPPLS parameters directly from a price window. The loss function combines reconstruction error with LPPLS physics constraints (Sornette filter violations are penalized). Optional, requires PyTorch.

## Accuracy Results

Tested on 35 historical drawdowns across BTC (15% threshold), SPY (8%), and Gold (8%):

| Method | Small (<15%) | Medium (15-30%) | Major (>30%) | Overall |
|--------|:---:|:---:|:---:|:---:|
| **LPPLS** | **100%** | **100%** | **88%** | **97%** |
| **Kappa** | 57% | 47% | 38% | 49% |
| **GPD VaR** | 40% | 55% | 0% | 42% |
| **Hill** | 29% | 29% | 25% | 28% |

Key findings:
- **LPPLS is the best single method** (97%) — it detects the bubble regime itself, not just tail statistics
- **Kappa is the best tail-based method** — more robust than Hill alone
- **GPD VaR** works well for medium corrections but struggles with major crashes (the pre-crash period is itself volatile)
- **Hill alone is unreliable** (28%) — too noisy as a standalone signal, but useful in the aggregate
- **Combining methods improves reliability** — the aggregator produces more stable signals than any single method

## Data Flow

```
Exchange/CSV/FRED → ingest.py → transforms.py → [log_prices, log_returns]
                                                        |
                    +----------+----------+-------------+
                    v          v          v              v
               LPPLS(Rust) EVT(Rust) Tail(Rust)   DeepLPPLS(Torch)
               tc, conf    VaR, ES   Hill a, k     tc prediction
                    |          |          |              |
                    +----------+----------+------+------+
                                                 v
                                      signals.py (aggregate)
                                      -> CrashSignal(prob, horizon)
                                                 |
                                    +----------+-+----------+
                                    v          v            v
                                  CLI       FastAPI      Notebook
```

## Sample Data

Bundled offline data (no internet needed):

| Asset | Period | Days | Source |
|-------|--------|------|--------|
| BTC | 2014-2025 | 4,124 | Yahoo Finance |
| SPY | 1999-2025 | 6,570 | options_backtester |
| Gold | 2000-2025 | 6,441 | Yahoo Finance |
| GBP/USD | 1971-2025 | 13,791 | forex-centuries (FRED) |
| Macro signals | 2007-2025 | 5,020 | FRED (VIX, yield curve, HY spread, GDP) |

```python
from fatcrash.data.ingest import from_sample, from_fred

btc = from_sample("btc")           # Offline
vix = from_fred("VIXCLS")          # Live from FRED
spy = from_sample("spy")           # Offline
```

## Development

```bash
# Setup
make setup

# Build (compiles Rust, installs Python package)
make build

# Run all tests (7 Rust + 55 Python)
make test

# Lint
make lint

# Quick detect
make detect

# CLI
fatcrash detect --asset BTC --source sample
fatcrash backtest --asset BTC --start 2017-01-01 --end 2018-06-01
fatcrash plot --asset GOLD --indicator hill
fatcrash serve --port 8000
```

## Architecture

### What goes in Rust (via PyO3)

| Component | Why Rust |
|-----------|----------|
| **LPPLS fitter** | O(1000) nonlinear fits per anchor date. Most expensive computation. |
| **LPPLS confidence** | Nested window fits parallelized with `rayon` |
| **GEV/GPD fitting** | Rolling EVT (re-fit at each timestep) needs speed |
| **Hill estimator** | Called at every rolling step |
| **Kappa metric** | Gaussian benchmarks via Monte Carlo simulation |

### What stays in Python

- Data ingestion (Yahoo, CoinGecko, CCXT, FRED, CSV/Parquet)
- Visualization (plotly, matplotlib)
- CLI (typer) and service (FastAPI)
- Deep LPPLS (PyTorch)
- Signal aggregation and calibration
- 9 Jupyter notebooks

## Signal Aggregation

| Indicator | Weight | Logic |
|-----------|--------|-------|
| LPPLS confidence | 0.30 | Fraction of qualifying fits |
| LPPLS tc proximity | 0.15 | How close predicted crash to now |
| GPD VaR exceedance | 0.15 | Current returns vs tail risk |
| Kappa regime change | 0.10 | Rolling kappa deviation from benchmark |
| Hill tail thinning | 0.10 | Declining alpha = thickening tails |
| Deep LPPLS | 0.10 | Neural tc prediction |
| Multiscale agreement | 0.10 | Cross-timeframe confirmation |
