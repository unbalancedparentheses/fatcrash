# fatcrash

Crash detection via fat-tail statistics. Works with any asset: Bitcoin, gold, S&P 500, forex, etc.

Python + Rust (PyO3) — performance-critical numerical work in Rust, everything else in Python. 7 methods, 97 tests, tested on 500 years of data across 138 countries.

## The Methods

### Hill Estimator — "How fat are the tails?"

The Hill estimator measures the **tail index** (alpha) of a distribution. Financial returns don't follow a normal distribution — they have "fat tails", meaning extreme events happen far more often than a bell curve predicts. The 2008 crash, COVID crash, Bitcoin's -83% drops — these aren't 6-sigma events under normality, they're regular features of fat-tailed distributions.

The Hill estimator fits a power law to the tail: P(X > x) ~ x^(-alpha). Lower alpha = fatter tails = more extreme events.

- **alpha < 2**: Infinite variance (very dangerous — Cauchy-like)
- **alpha 2-4**: Fat tails, finite variance but infinite kurtosis
- **alpha > 4**: Relatively thin tails

### Pickands Estimator — "More robust tail measurement"

Like Hill but works for all tail types (heavy, light, bounded), not just heavy. Uses three order statistics instead of Hill's sum, making it less sensitive to the choice of k. Positive gamma = heavy tail, zero = exponential, negative = bounded.

### Kappa Metric — "How far from Gaussian?"

Taleb's kappa metric directly measures how much a distribution deviates from Gaussian. It splits data into subsamples, computes the max of each, and compares the mean of those maxima to the overall max. The ratio kappa/benchmark tells you:

- **Ratio ~ 1.0**: Behaves like Gaussian (safe, predictable)
- **Ratio < 0.8**: Significantly fatter tails than Gaussian
- **Ratio < 0.5**: Extremely fat tails (crisis regime)

### Hurst Exponent — "Is the market trending?"

Measures long-range dependence via R/S analysis. H > 0.5 means trending (persistent), H = 0.5 means random walk, H < 0.5 means mean-reverting. A trending market combined with a bubble signal is a strong crash warning.

### EVT (Extreme Value Theory) — "What's the worst that can happen?"

The mathematically rigorous framework for modeling extreme events. Fits only the extremes, not the bulk.

- **GPD**: Fit to losses exceeding a threshold → VaR and Expected Shortfall
- **GEV**: Fit to block maxima → tail shape classification (Frechet/Gumbel/Weibull)

### LPPLS — "Is this a bubble?"

The Log-Periodic Power Law Singularity model detects **bubbles before they burst** (Didier Sornette, ETH Zurich). During a bubble, prices follow super-exponential growth with accelerating oscillations converging toward a critical time tc. The DS LPPLS confidence indicator fits across many time windows — high fraction of valid fits = high confidence.

### GSADF — "Is there explosive growth?"

The Generalized Sup ADF test (Phillips-Shi-Yu, 2015) — the standard econometric bubble test. Tests for explosive unit root behavior. Complements LPPLS: LPPLS detects bubble *shape*, GSADF detects *explosive growth*. Includes Monte Carlo critical values (parallelized with rayon).

## Accuracy Results

### Individual methods on 39 drawdowns (BTC, SPY, Gold)

| Method | Small (<15%) | Medium (15-30%) | Major (>30%) | Overall |
|--------|:---:|:---:|:---:|:---:|
| **LPPLS** | **100%** | **100%** | **100%** | **100%** |
| Hurst | 57% | 65% | 50% | 59% |
| Kappa | 57% | 47% | 38% | 49% |
| Pickands | 43% | 53% | 50% | 49% |
| GPD VaR | 40% | 55% | — | 42% |
| GSADF | 14% | 59% | 38% | 38% |
| Hill | 29% | 29% | 25% | 28% |

### Combined detector (all 7 methods with agreement bonus)

| Method | Small | Medium | Major | Overall |
|--------|:---:|:---:|:---:|:---:|
| **COMBINED** | **64%** | **94%** | **75%** | **79%** |

When the combined detector says HIGH or CRITICAL, it's almost always right. Agreement across independent method categories (bubble, tail, regime) boosts the signal.

### 6/6 known GBP/USD crises detected

1976 IMF Crisis, 1985 Plaza Accord, 1992 Black Wednesday, 2008 Financial Crisis, 2016 Brexit, 2022 Truss Mini-Budget.

## 500 Years of Forex Data

### FRED Daily (12 currency pairs, 1971-2025)

| Pair | Hill alpha | Pickands | Hurst H | GSADF bubble? |
|------|:---------:|:-------:|:------:|:---:|
| AUD/USD | 2.58 | 1.02 | 0.56 | YES |
| GBP/USD | 4.13 | 0.06 | 0.58 | YES |
| JPY/USD | 3.94 | -0.23 | 0.58 | YES |
| CAD/USD | 3.84 | 0.30 | 0.57 | YES |
| CNY/USD | 2.79 | 0.55 | 0.59 | YES |
| MXN/USD | 2.04 | 0.60 | 0.56 | YES |
| BRL/USD | 2.80 | 0.44 | 0.56 | YES |
| KRW/USD | 1.90 | 1.01 | 0.67 | YES |
| INR/USD | 2.62 | 0.45 | 0.57 | YES |
| EUR/USD | 4.88 | 0.06 | 0.56 | no |
| CHF/USD | 3.81 | -0.32 | 0.57 | no |
| NZD/USD | 2.89 | 0.68 | 0.57 | no |

**Every single pair has Hurst H > 0.55** — forex markets are universally persistent. 9/12 show explosive bubble episodes. 10/12 have heavy tails.

### Clio Infra Yearly (30 countries, 1500-2013)

| Country | Years | Hill alpha | Hurst H | Verdict |
|---------|:-----:|:---------:|:------:|---------|
| Germany | 153 | 0.52 | 0.56 | EXTREME, persistent |
| Austria | 104 | 0.63 | 0.61 | EXTREME, persistent |
| Belgium | 114 | 0.89 | 0.64 | EXTREME, persistent |
| Finland | 100 | 0.94 | 0.58 | EXTREME, persistent |
| Argentina | 102 | 1.28 | 0.71 | EXTREME, persistent |
| Mexico | 113 | 1.06 | 0.70 | EXTREME, persistent |
| Italy | 95 | 0.77 | 0.80 | EXTREME, persistent |
| Portugal | 88 | 0.98 | 0.85 | EXTREME, persistent |
| Greece | 87 | 0.77 | 0.76 | EXTREME, persistent |
| UK | 223 | 2.42 | 0.47 | fat-tail |
| Canada | 100 | 3.70 | 0.50 | fat-tail |

**19/30 countries have alpha < 2** (infinite variance). **25/30 have Hurst > 0.5** (persistent). Only 1/30 has alpha > 4.

Italy (H=0.80) and Portugal (H=0.85) show the strongest persistence — their exchange rates trend for years at a time over century-scale data.

### The headline number

**71% of countries have exchange rate distributions with infinite variance.** The median alpha across 138 countries is 1.57. Standard risk models (VaR under normality, Modern Portfolio Theory, CAPM) assume finite variance. For 71% of the world's currencies, this assumption is empirically false.

## Data Flow

```
Exchange/CSV/FRED → ingest.py → transforms.py → [log_prices, log_returns]
                                                        |
                    +----------+----------+-------------+
                    v          v          v              v
               LPPLS(Rust) EVT(Rust) Tail(Rust)   DeepLPPLS(Torch)
               GSADF(Rust)           Pickands      Hurst
               tc, conf    VaR, ES   Hill,Kappa    H exponent
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

## Signal Aggregation

Groups methods into 4 independent categories. When 3+ categories agree, probability gets a +15% bonus.

| Category | Methods | What it catches |
|----------|---------|-----------------|
| **Bubble** | LPPLS, GSADF, Deep LPPLS | Super-exponential growth, explosive roots |
| **Tail** | Kappa, Hill, Pickands, GPD VaR | Tail thickening, regime shifts |
| **Regime** | Hurst | Trending vs mean-reverting |
| **Structure** | Multiscale, LPPLS tc proximity | Cross-timeframe, timing |

## Development

```bash
# Setup
make setup

# Build (compiles Rust, installs Python package)
make build

# Run all tests (18 Rust + 79 Python = 97 total)
make test

# Lint (zero clippy warnings)
make lint

# CLI
fatcrash detect --asset BTC --source sample
fatcrash backtest --asset BTC --start 2017-01-01 --end 2018-06-01
fatcrash plot --asset GOLD --indicator hill
fatcrash serve --port 8000

# Full analysis
python analysis/accuracy_report.py
```

## Architecture

### Rust (via PyO3) — performance-critical

| Component | Why Rust |
|-----------|----------|
| LPPLS fitter (CMA-ES) | O(1000) nonlinear fits per anchor date |
| LPPLS confidence | Nested windows parallelized with rayon |
| GSADF test | O(n^2) BSADF + Monte Carlo, parallelized with rayon |
| GEV/GPD fitting | Rolling EVT needs speed |
| Hill, Pickands, Kappa, Hurst | Called at every rolling step |

### Python — everything else

- Data ingestion (Yahoo, CoinGecko, CCXT, FRED, CSV/Parquet)
- Visualization (plotly, matplotlib)
- CLI (typer) and service (FastAPI)
- Deep LPPLS (PyTorch, with save/load for pretrained weights)
- Signal aggregation with agreement bonus
- 9 Jupyter notebooks
