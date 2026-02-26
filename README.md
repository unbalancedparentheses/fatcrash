# fatcrash

**71% of countries have exchange rate distributions with infinite variance.**

The median tail index across 138 countries is alpha = 1.57. Standard risk models assume finite variance (alpha > 2) and often finite kurtosis (alpha > 4). For the majority of the world's currencies, these assumptions are empirically false. fatcrash detects crashes by measuring what actually matters: the tail.

Python + Rust (PyO3). 17 methods. 246 tests. 500 years of data.

```python
from fatcrash.data.ingest import from_sample
from fatcrash.data.transforms import log_returns
from fatcrash._core import (
    hill_estimator, taleb_kappa, dfa_exponent,
    deh_estimator, maxsum_ratio, spectral_exponent,
)

btc = from_sample("btc")
ret = log_returns(btc)

hill_estimator(ret)      # 2.87 — infinite kurtosis
taleb_kappa(ret)         # (0.34, 0.09) — CLT barely operates
dfa_exponent(ret)        # 0.56 — persistent dynamics
deh_estimator(ret)       # 0.31 — heavy-tailed (gamma > 0)
maxsum_ratio(ret)        # 0.003 — single obs doesn't dominate (alpha > 2)
spectral_exponent(ret)   # 0.04 — weak long memory
```

```bash
fatcrash detect --asset BTC --source sample
fatcrash backtest --asset BTC --start 2017-01-01 --end 2018-06-01
```

### Data Sources and Cache

By default, network sources are cached to `~/.cache/fatcrash` to avoid rate limits and speed up repeated runs.

```bash
fatcrash detect --asset BTC --source yahoo --days 365
fatcrash detect --asset BTC --source yahoo --start 2020-01-01 --end 2021-01-01
fatcrash detect --asset BTC --source coingecko --no-use-cache
fatcrash cache-clear
```

> **DISCLAIMER:** This software is for academic research and educational purposes only. It does not constitute financial advice. No warranty is provided regarding the accuracy of predictions. Do not use for investment decisions.

## Why This Exists

### The ergodicity problem

Classical finance evaluates gambles by their expected value — the ensemble average over all possible outcomes at a single point in time. Peters (2019) showed this is the wrong quantity for a single agent who must live through outcomes sequentially. In a multiplicative process like investing, the time-average growth rate and the ensemble-average growth rate diverge. The process is non-ergodic.

A gamble that pays +50% or -40% with equal probability has positive expected value (+5% per round) but negative time-average growth: log(1.5 * 0.6) / 2 = -5.3% per round. Over enough rounds, a single participant goes bankrupt with certainty despite the "positive EV."

Fat tails amplify this divergence. When alpha < 2 (infinite variance), the ensemble average is dominated by rare outcomes that no individual trajectory will realize. The sample mean does not converge at the rate prescribed by the CLT. Taleb (2020) quantifies this: for alpha near 1, the CLT does not operate at any practical sample size.

### What breaks

Standard risk metrics presuppose:

1. **Finite variance** — requires alpha > 2. Violated for 71% of countries.
2. **Rapid CLT convergence** — requires alpha > 4. Violated for 97% of countries.
3. **Ergodicity** — ensemble average = time average. Violated for all multiplicative processes with fat tails.

VaR under normality, Sharpe ratios, CAPM betas, mean-variance optimization — all of these produce nonsense when the underlying distribution has infinite variance. fatcrash measures the tail directly so you know which regime you're in.

## Results

### Crash detection: precision, recall, and F1 (39 drawdowns across BTC, SPY, Gold)

All accuracy numbers are in-sample on historical data. Methods are tested on both crash windows (true positives) and non-crash windows (false positives) sampled at least 180 days from any crash.

| Method | Precision | Recall | F1 | Notes |
|--------|:---------:|:------:|:--:|-------|
| **LPPLS** | **37%** | **74%** | **50%** | Tightened: Nielsen omega [6,13] + tc constraint |
| **LPPLS confidence** | **29%** | **90%** | **43%** | Multi-window aggregation (rayon-parallelized) |
| GSADF | 38% | 38% | 38% | Best for medium/major crashes |
| **DFA** | **22%** | **82%** | **34%** | Best non-bubble method |
| Hurst | 19% | 59% | 28% | DFA handles non-stationarity better |
| Pickands | 19% | 49% | 27% | |
| Kappa | 19% | 49% | 27% | |
| DEH | 18% | 46% | 26% | Most useful on major crashes (62%) |
| Spectral | 22% | 28% | 25% | |
| Taleb Kappa | 20% | 33% | 25% | 50% recall on major crashes |
| QQ | 16% | 38% | 23% | |
| GPD VaR | 12% | 42% | 19% | |
| Max-to-Sum | 12% | 31% | 18% | |
| Hill | 12% | 28% | 16% | Unreliable alone, contributes to ensemble |

**Why precision is low for tail/regime methods:** These methods detect distributional regime shifts (tail thickening, persistent dynamics), not crash-specific patterns. They fire in many non-crash periods because fat tails and persistence are pervasive in financial data. This is by design — they measure the *distributional regime*, not a specific crash. LPPLS and GSADF have higher precision because they detect bubble-specific structure.

**The Sornette–Bouchaud debate on precision vs recall:**

Sornette (the LPPLS inventor) argues that LPPLS is deliberately tuned for **high recall at the cost of precision** because the cost function is asymmetric — missing a crash is far more expensive than a false alarm. He calls false positives "failed predictions" and argues they are inevitable: bubbles can end in slow deflation rather than sharp crashes. His 2024 paper with Nielsen ([arXiv:2405.12803](https://arxiv.org/abs/2405.12803)) introduced the tightened omega [6,13] range specifically to improve precision without sacrificing recall.

Bouchaud takes a more skeptical view. In his work at CFM and in papers with Potters, he emphasizes that fat-tail estimators (Hill, etc.) measure **unconditional** properties of returns and are poor at **conditional** crash prediction. His point is exactly what the data shows: tail estimators have decent recall but terrible precision because fat tails are always present, not just before crashes. He favors portfolio-level risk measures (drawdown control, volatility targeting) over point-in-time crash prediction.

Both perspectives are reflected in fatcrash: LPPLS targets the mechanism (Sornette's approach), tail estimators measure the regime (which Bouchaud correctly notes is always fat-tailed), and the aggregator combines both — using Sornette-style bubble detection as the primary signal and Bouchaud-style regime measurement as confirmation.

### Recall by crash size

| Method | Small (<15%) | Medium (15-30%) | Major (>30%) |
|--------|:---:|:---:|:---:|
| LPPLS confidence | 93% | 94% | 75% |
| LPPLS | 86% | 71% | 62% |
| DFA | 86% | 88% | 62% |
| Hurst | 57% | 65% | 50% |
| GSADF | 14% | 59% | 38% |
| DEH | 43% | 41% | 62% |
| Taleb Kappa | 21% | 35% | 50% |

### Combined detector

| | Small | Medium | Major | Overall |
|--------|:---:|:---:|:---:|:---:|
| **All methods + agreement bonus** | **64%** | **94%** | **75%** | **79%** |

When 3+ independent method categories (bubble, tail, regime, structure) agree, the probability gets a +15% bonus. No single method is reliable alone; the ensemble is.

### 6/6 known GBP/USD crises detected

1976 IMF Crisis, 1985 Plaza Accord, 1992 Black Wednesday, 2008 Financial Crisis, 2016 Brexit, 2022 Truss Mini-Budget.

## 500 Years of Forex Data

### FRED Daily (12 currency pairs, 1971-2025)

| Pair | Hill alpha | QQ alpha | DEH gamma | Hurst H | DFA alpha | GSADF bubble? |
|------|:---------:|:-------:|:--------:|:------:|:--------:|:---:|
| AUD/USD | 2.58 | 2.30 | 0.44 | 0.56 | 0.56 | YES |
| GBP/USD | 4.13 | 4.11 | 0.19 | 0.58 | 0.55 | YES |
| JPY/USD | 3.94 | 4.02 | 0.18 | 0.58 | 0.58 | YES |
| CAD/USD | 3.84 | 3.58 | 0.27 | 0.57 | 0.53 | YES |
| CNY/USD | 2.79 | 1.70 | 0.69 | 0.59 | 0.71 | YES |
| MXN/USD | 2.04 | 1.98 | 0.44 | 0.56 | 0.57 | YES |
| BRL/USD | 2.80 | 3.12 | 0.15 | 0.56 | 0.58 | YES |
| KRW/USD | 1.90 | 1.93 | 0.44 | 0.67 | 0.60 | YES |
| INR/USD | 2.62 | 2.56 | 0.34 | 0.57 | 0.58 | YES |
| EUR/USD | 4.88 | 4.90 | 0.12 | 0.56 | 0.54 | no |
| CHF/USD | 3.81 | 3.59 | 0.28 | 0.57 | 0.54 | no |
| NZD/USD | 2.89 | 2.46 | 0.44 | 0.57 | 0.56 | no |

Universals across all 12 pairs:
- **Fat tails**: DEH gamma > 0 for 12/12. Mean Hill alpha = 3.19, mean QQ alpha = 3.02 — two independent estimators converge.
- **Persistence**: Hurst H > 0.55 for 12/12, DFA alpha > 0.5 for 12/12.
- **Bubbles**: 9/12 show explosive episodes via GSADF.

KRW/USD has the fattest tails (Hill alpha = 1.90, approaching infinite variance). CNY/USD shows the strongest persistence (DFA = 0.71) — consistent with managed float dynamics.

### Clio Infra Yearly (30 countries, 1500-2013)

| Country | Years | Hill alpha | Hurst H | Taleb kappa | Verdict |
|---------|:-----:|:---------:|:------:|:----------:|---------|
| Germany | 153 | 0.52 | 0.56 | 1.00 | EXTREME, persistent |
| Austria | 104 | 0.63 | 0.61 | 1.00 | EXTREME, persistent |
| Belgium | 114 | 0.89 | 0.64 | 0.86 | EXTREME, persistent |
| Finland | 100 | 0.94 | 0.58 | 0.43 | EXTREME, persistent |
| Argentina | 102 | 1.28 | 0.71 | 1.00 | EXTREME, persistent |
| Mexico | 113 | 1.06 | 0.70 | 0.92 | EXTREME, persistent |
| Italy | 95 | 0.77 | 0.80 | 0.95 | EXTREME, persistent |
| Portugal | 88 | 0.98 | 0.85 | 1.00 | EXTREME, persistent |
| Greece | 87 | 0.77 | 0.76 | 0.81 | EXTREME, persistent |
| UK | 223 | 2.42 | 0.47 | 0.04 | fat-tail |
| Canada | 100 | 3.70 | 0.50 | 0.00 | fat-tail |

Summary across 30 countries:
- 19/30 have alpha < 2 (infinite variance)
- 28/30 have DFA > 0.5, 25/30 have Hurst > 0.5 (persistent)
- 20/30 have DEH gamma > 0, 28/30 have QQ alpha < 4 (heavy tails confirmed by all estimators)

Germany, Austria, Argentina, and Portugal saturate at Taleb kappa = 1.0 — Cauchy-like behavior where the CLT does not operate at any practical sample size. Italy (H = 0.80, DFA = 1.44) and Portugal (H = 0.85) show the strongest persistence over century-scale data.

## The 17 Methods

### Overview

| # | Method | Category | What it measures | Key output |
|---|--------|----------|-----------------|------------|
| 1 | Hill | Tail | Tail index from order statistics | alpha (< 2 = infinite variance) |
| 2 | Pickands | Tail | Extreme value index, all domains | gamma (> 0 = heavy tail) |
| 3 | DEH | Tail | Tail index via moment estimator | gamma (> 0 = heavy tail) |
| 4 | QQ | Tail | Tail index from QQ-plot slope | alpha (< 4 = fat tail) |
| 5 | Taleb kappa | Tail | CLT convergence rate | kappa (0 = Gaussian, 1 = Cauchy) |
| 6 | Max-stability kappa | Tail | Block-max concentration | kappa vs benchmark |
| 7 | Max-to-Sum | Tail | Infinite variance diagnostic | ratio (> 0 if alpha < 2) |
| 8 | Hurst | Regime | Persistence via R/S analysis | H (> 0.5 = trending) |
| 9 | DFA | Regime | Persistence, non-stationary-robust | alpha (> 0.5 = trending) |
| 10 | Spectral | Regime | Long memory from frequency domain | d (> 0 = long memory) |
| 11 | GPD | EVT | Tail risk (VaR, ES) | VaR, Expected Shortfall |
| 12 | GEV | EVT | Block maxima classification | Frechet / Gumbel / Weibull |
| 13 | LPPLS + GSADF | Bubble | Bubble shape + explosive unit roots | critical time, confidence |
| 14 | M-LNN | Bubble (NN) | Per-series LPPLS via neural network | tc, m, omega, confidence |
| 15 | P-LNN | Bubble (NN) | Pre-trained LPPLS (~700x faster) | tc, m, omega, confidence |
| 16 | HLPPL | Bubble (NN) | Dual-stream transformer + sentiment | bubble score [0, 1] |
| 17 | DTCAI | Bubble (NN) | LPPLS reliability classifier + DTC | DTCAI score [0, 1] |

### Tail estimation

**Hill estimator** (Hill, 1975). Estimates alpha from the k largest order statistics: alpha = [1/k * sum log(X_(i) / X_(k+1))]^(-1). The tail index governs tail decay: P(X > x) ~ x^(-alpha). Alpha < 2 means infinite variance; alpha < 4 means infinite kurtosis. Includes Huisman et al. (2001) small-sample bias correction.

**Pickands estimator** (Pickands, 1975). Estimates gamma = 1/alpha using three order statistics: gamma = log((X_(k) - X_(2k)) / (X_(2k) - X_(4k))) / log(2). Valid for all three domains of attraction (Frechet, Gumbel, Weibull), unlike Hill which assumes heavy tails.

**DEH moment estimator** (Dekkers, Einmahl & de Haan, 1989). Uses first and second moments of log-spacings: gamma = M1 + 1 - (1/2)(1 - M1^2/M2)^(-1). Valid for all domains of attraction. Complements Hill (heavy-tail only) and Pickands (higher variance).

**QQ estimator.** Regresses log(X_(i)) vs -log(i/(k+1)) for the k largest observations. Slope = 1/alpha. Simple, visual, good for regime change detection in rolling windows.

**Taleb's kappa** (Taleb, 2019). Measures how fast the sample mean converges: kappa = 2 - log(n/n0) / log(M(n)/M(n0)), where M(n) = E[|S_n - E[S_n]|]. Under the CLT, M(n) ~ sqrt(n), giving kappa = 0. For Cauchy, M(n) ~ n, giving kappa = 1. Answers what asymptotic theory cannot: *how many observations do you actually need?*

**Max-stability kappa.** Partitions data into blocks, computes mean-of-block-maxima / global-maximum. For Gaussian data this ratio is near a Monte Carlo benchmark; for fat-tailed data, a single extreme dominates and the ratio drops.

**Maximum-to-Sum ratio.** R_n = max(|X_i|) / sum(|X_i|). Converges to zero for thin tails (alpha > 2); stays positive when alpha < 2. The simplest diagnostic for whether variance exists.

### Long-range dependence

**Hurst exponent** (Hurst, 1951). Persistence via rescaled range (R/S) analysis. H = 0.5 is a random walk; H > 0.5 means trends persist; H < 0.5 means mean-reversion.

**DFA** (Peng et al., 1994). Detrended fluctuation analysis: divides into windows, removes linear trend per window, regresses log(RMS of residuals) vs log(window size). Handles non-stationarity better than R/S — best non-bubble crash detector (82% recall, 34% F1).

**Spectral exponent** (Geweke & Porter-Hudak, 1983). Estimates long-memory parameter d from the periodogram near frequency zero: f(lambda) ~ |lambda|^(1-2d). Relation to Hurst: d = H - 0.5. Confirms persistence from the frequency domain.

### Extreme value theory

**GPD** (Balkema & de Haan, 1974). Fits exceedances over a threshold to the Generalized Pareto Distribution. Yields VaR and Expected Shortfall at arbitrary confidence levels.

**GEV** (Fisher & Tippett, 1928). Fits block maxima to the Generalized Extreme Value distribution. Classifies into Frechet (xi > 0, heavy tail), Gumbel (xi = 0, exponential), or Weibull (xi < 0, bounded).

### Bubble detection

**LPPLS** (Sornette, 2003). Models bubble dynamics as a power law with log-periodic oscillations: log(p(t)) = A + B|tc-t|^m + C|tc-t|^m * cos(omega*log|tc-t| + phi). The critical time tc is the predicted crash date. Confidence measured by fitting across many windows. Nonlinear optimization via CMA-ES in Rust.

**GSADF** (Phillips, Shi & Yu, 2015). Detects explosive unit root behavior — the econometric signature of bubbles. The supremum of recursive ADF statistics over all feasible subsamples, with Monte Carlo critical values. Complements LPPLS: LPPLS detects bubble *shape*, GSADF detects *explosive growth*.

### Neural network methods (requires `pip install fatcrash[deep]`)

Three neural network approaches to LPPLS fitting from recent 2024-2025 papers. All share a common pattern: the network predicts nonlinear LPPLS parameters (tc, m, omega), then linear parameters (A, B, C1, C2) are solved analytically via OLS. This physics-informed decomposition makes training stable and predictions interpretable.

**M-LNN** — Mono-LPPLS Neural Network (Nielsen, Sornette & Raissi, 2024). One small network (2 hidden layers, 64 units each) trained per time series. Minimizes reconstruction MSE between the LPPLS fit and observed log-prices. Works on variable-length input. Slower than P-LNN but more flexible — no pre-training required.

```python
from fatcrash.nn.mlnn import fit_mlnn
result = fit_mlnn(times, log_prices, epochs=200, lr=1e-2)
# result.tc, result.m, result.omega, result.is_bubble, result.confidence
```

**P-LNN** — Poly-LPPLS Neural Network (Nielsen, Sornette & Raissi, 2024). Pre-trained on 100K synthetic LPPLS series, ~700x faster than CMA-ES at inference. A deeper network (4 hidden layers: 512-256-128-64) maps a 252-observation min-max normalized price window to (tc, m, omega) in a single forward pass. Three variants trained with white noise, AR(1) noise, or both.

```python
from fatcrash.nn.plnn import train_plnn, predict_plnn
model = train_plnn(variant="P-LNN-100K", n_samples=100_000)
result = predict_plnn(model, times, log_prices)
```

**HLPPL** — Hyped LPPL (Cao, Shao, Yan & Geman, 2025). Dual-stream transformer that fuses price dynamics with market sentiment. Stream 1 (TemporalEncoder) processes price and LPPLS features through a 2-layer, 4-head TransformerEncoder. Stream 2 (SentimentEncoder) processes volume-derived hype features through a 1-layer, 2-head TransformerEncoder. The fused representation produces a crash probability in [0, 1]. Works with OHLCV data — no external NLP sentiment feed required (uses volume-based proxies: volume z-score, momentum, absolute return z-score, combined hype index).

```python
from fatcrash.nn.hlppl import train_hlppl, predict_hlppl
model = train_hlppl(train_dfs, crash_labels, window=60, epochs=50)
result = predict_hlppl(model, ohlcv_df)  # result.bubble_score
```

**DTCAI** — Distance-to-Crash AI (Lee, Jeong, Park & Ahn, 2025). Trains a classifier (ANN, Random Forest, or Logistic Regression) on 7 LPPLS parameters to assess the reliability of each LPPLS fit. The DTCAI score combines the Distance-to-Crash ratio (how far into the bubble the current window extends) with the AI reliability probability: DTCAI = DTC * P. Addresses a key LPPLS weakness: it always produces a tc estimate even when no bubble exists. Uses the Bree & Joseph (2013) crash criterion for labeling.

```python
from fatcrash.nn.dtcai import train_dtcai, predict_dtcai
from fatcrash.nn.dtcai_data import generate_dtcai_dataset
dataset = generate_dtcai_dataset(prices, window_size=504, n_fits_per_window=10)
model = train_dtcai(dataset, model_type="ANN", epochs=50)
result = predict_dtcai(model, times, log_prices)  # result.dtcai
```

## Signal Aggregation

Methods grouped into 4 independent categories. When 3+ categories agree, probability gets a +15% bonus.

| Category | Methods | What it detects |
|----------|---------|-----------------|
| **Bubble** | LPPLS, GSADF, M-LNN, P-LNN, HLPPL, DTCAI | Super-exponential growth, explosive unit roots |
| **Tail** | Hill, Pickands, DEH, QQ, Taleb Kappa, Max-Stability Kappa, Max-to-Sum, GPD | Tail thickening, distributional regime shifts |
| **Regime** | Hurst, DFA, Spectral | Transition from mean-reverting to persistent dynamics |
| **Structure** | Multiscale, LPPLS tc proximity | Cross-timeframe agreement, timing |

Indicators are also computed at daily, 3-day, and weekly frequencies. A signal at one scale may be noise; a signal across all three is structural.

## Beyond Valuations: Revenue & Profit

These methods were built for market prices, but most transfer to fundamental data like revenue or profit growth. The key distinction: **market prices** reflect collective speculative behavior (reflexivity, herding, positive feedback loops), while **revenue/profit** reflects real economic activity (customer demand, operational execution, competitive dynamics).

### What transfers

| Method | Works on revenue/profit? | Why |
|--------|:-:|-----|
| Hill, DEH, QQ, Pickands | Yes | Tail thickness is a property of any distribution. Revenue growth rates have fat tails — Gabaix (2011) showed that idiosyncratic firm-level shocks drive aggregate fluctuations precisely because firm-size distributions are fat-tailed. |
| Kappa, Taleb kappa | Yes | Measures departure from Gaussian max-stability. Works on any data where you suspect non-Gaussian extremes. |
| Max-to-Sum ratio | Yes | A single quarter where revenue drops 50% dominating the total sum = same math as a single market crash day. |
| GPD / GEV | Yes | EVT is distribution-agnostic. Fit GPD to the worst quarterly revenue declines for valid tail risk estimates. |
| Hurst, DFA, Spectral | Yes | Revenue series often show strong persistence (H > 0.5) due to contracts and customer stickiness. A shift from persistent to anti-persistent could signal fundamental deterioration. |
| GSADF | Partially | Detects unsustainable exponential growth. Could flag "revenue bubbles" — growth rates that imply a company would need to capture 100% of its addressable market. |
| LPPLS, LPPLS confidence | No | Models speculative bubble dynamics (herding, log-periodic oscillations). Revenue doesn't exhibit these patterns — it's driven by real economic activity, not reflexive speculation. |

### Practical application

For a company's quarterly revenue time series:

```python
import numpy as np
from fatcrash._core import hill_estimator, dfa_exponent, hurst_exponent, gsadf_test

# Quarterly revenue growth rates
growth = np.diff(np.log(quarterly_revenue))

hill_estimator(growth)    # Tail index — are revenue shocks fat-tailed?
dfa_exponent(growth)      # Persistence — is growth momentum persistent or mean-reverting?
hurst_exponent(growth)    # Same question, different method
gsadf_test(quarterly_revenue)  # Is revenue growth explosive/unsustainable?
```

The challenge: quarterly data gives ~80 observations over 20 years (vs ~5,000 daily prices). Tail estimators need at least ~100 data points to be reliable. Use monthly revenue or longer history when possible.

## Architecture

```
Rust (PyO3, _core.so)                Python
┌────────────────────────────┐       ┌──────────────────────────────────┐
│ Tail: Hill, Pickands, DEH, │       │ indicators/                      │
│       QQ, Kappa, Taleb,    │──────▶│   tail_indicator.py              │
│       MaxSum, Hurst, DFA,  │       │   lppls_indicator.py             │
│       Spectral             │       │   bubble_indicator.py            │
│                            │       │   evt_indicator.py               │
│ EVT:  GPD, GEV             │       │                                  │
│                            │       │ nn/                              │
│ LPPLS: fit, confidence,    │──────▶│   mlnn.py      (M-LNN)          │
│        solve_linear        │       │   plnn.py      (P-LNN)          │
│                            │       │   hlppl.py     (HLPPL)          │
│ Bubble: GSADF              │       │   dtcai.py     (DTCAI)          │
│                            │       │   crash_labels.py (crash detect) │
│ Multiscale                 │       │   dtcai_data.py (dataset gen)   │
│                            │       │   lppls_torch.py (shared)       │
│                            │       │   synthetic.py  (data gen)      │
│                            │       │   sentiment.py  (volume proxy)  │
│                            │       │                                  │
│ rayon: parallel CMA-ES,    │       │ aggregator/signals.py            │
│        GSADF, confidence   │       │ cli/ viz/ service/ data/         │
└────────────────────────────┘       └──────────────────────────────────┘
```

All 13 classical estimators are implemented in Rust and exposed to Python via PyO3. The computationally intensive methods (LPPLS CMA-ES, GSADF, confidence) use rayon for parallelization. The 4 neural network methods are in Python (PyTorch/sklearn) and call `lppls_solve_linear` from Rust for the analytical linear parameter solve.

| Component | Language | Why |
|-----------|----------|-----|
| LPPLS fitter (CMA-ES) | Rust | O(1000) nonlinear fits per anchor date |
| LPPLS confidence | Rust | Nested windows parallelized with rayon |
| GSADF test | Rust | O(n^2) BSADF + Monte Carlo, parallelized |
| GEV/GPD fitting | Rust | Rolling EVT needs speed |
| All tail & regime estimators | Rust | Called at every rolling window step |
| M-LNN, P-LNN, HLPPL, DTCAI | Python (PyTorch/sklearn) | GPU support, autograd for training |
| Data ingestion, viz, CLI | Python | Ecosystem (pandas, plotly, typer, FastAPI) |

## Sample Data

Bundled offline (no internet required):

| Asset | Period | Days | Source |
|-------|--------|------|--------|
| BTC | 2014-2025 | 4,124 | Yahoo Finance |
| SPY | 1999-2025 | 6,570 | options_backtester |
| Gold | 2000-2025 | 6,441 | Yahoo Finance |
| GBP/USD | 1971-2025 | 13,791 | forex-centuries (FRED) |
| Macro signals | 2007-2025 | 5,020 | FRED (VIX, yield curve, HY spread, GDP) |

## Development

Requires [Nix](https://nixos.org/) with flakes enabled.

```bash
nix develop                  # Enter dev shell (Rust, Python 3.13, maturin, uv)
make setup                   # Install Python deps + build Rust extension
make build                   # Recompile Rust, install into venv
make test                    # 35 Rust + 211 Python = 246 tests
make lint                    # cargo clippy + cargo fmt --check
python analysis/accuracy_report.py   # Full analysis across all methods and timescales
```

With [direnv](https://direnv.net/): `direnv allow` and the shell activates on `cd`.

For neural network methods:

```bash
pip install fatcrash[deep]   # Adds PyTorch dependency
```

## References

### Ergodicity

- Peters, O. (2019). "The Ergodicity Problem in Economics." *Nature Physics*, 15, 1216-1221. [DOI:10.1038/s41567-019-0732-0](https://doi.org/10.1038/s41567-019-0732-0)
- Peters, O. & Gell-Mann, M. (2016). "Evaluating Gambles Using Dynamics." *Chaos*, 26(2), 023103. [arXiv:1405.0585](https://arxiv.org/abs/1405.0585)
- Peters, O. (2011). "The Time Resolution of the St Petersburg Paradox." *Phil. Trans. R. Soc. A*, 369(1956), 4913-4931. [arXiv:1011.4404](https://arxiv.org/abs/1011.4404)

### Tail Estimation

- Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of a Distribution." *Ann. Statist.*, 3(5), 1163-1174.
- Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics." *Ann. Statist.*, 3(1), 119-131.
- Dekkers, A.L.M., Einmahl, J.H.J. & de Haan, L. (1989). "A Moment Estimator for the Index of an Extreme-Value Distribution." *Ann. Statist.*, 17(4), 1833-1855.
- Taleb, N.N. (2019). "How Much Data Do You Need? An Operational, Pre-Asymptotic Metric for Fat-tailedness." *Int. J. Forecasting*, 35(2), 677-686. [arXiv:1802.05495](https://arxiv.org/abs/1802.05495)
- Taleb, N.N. (2020). *Statistical Consequences of Fat Tails.* STEM Academic Press. [arXiv:2001.10488](https://arxiv.org/abs/2001.10488)

### Extreme Value Theory

- Embrechts, P., Kluppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance.* Springer.
- Balkema, A.A. & de Haan, L. (1974). "Residual Life Time at Great Age." *Ann. Probab.*, 2(5), 792-804.
- Fisher, R.A. & Tippett, L.H.C. (1928). "Limiting Forms of the Frequency Distribution of the Largest or Smallest Member of a Sample." *Proc. Cambridge Phil. Soc.*, 24(2), 180-190.

### Long-Range Dependence

- Hurst, H.E. (1951). "Long-Term Storage Capacity of Reservoirs." *Trans. ASCE*, 116, 770-799.
- Peng, C.-K. et al. (1994). "Mosaic Organization of DNA Nucleotides." *Physical Review E*, 49(2), 1685-1689.
- Geweke, J. & Porter-Hudak, S. (1983). "The Estimation and Application of Long Memory Time Series Models." *J. Time Series Analysis*, 4(4), 221-238.
- Lo, A.W. (1991). "Long-Term Memory in Stock Market Prices." *Econometrica*, 59(5), 1279-1313.

### Bubble Detection

- Sornette, D. (2003). *Why Stock Markets Crash.* Princeton University Press.
- Sornette, D. et al. (2015). "Real-Time Prediction and Post-Mortem Analysis of the Shanghai 2015 Stock Market Bubble and Crash." *J. Investment Strategies*, 4(4), 77-95.
- Phillips, P.C.B., Shi, S. & Yu, J. (2015). "Testing for Multiple Bubbles." *Int. Econ. Rev.*, 56(4), 1043-1078.
- Phillips, P.C.B., Wu, Y. & Yu, J. (2011). "Explosive Behavior in the 1990s NASDAQ." *Int. Econ. Rev.*, 52(1), 201-226.
- Hansen, N. (2006). "The CMA Evolution Strategy: A Comparing Review." In *Towards a New Evolutionary Computation*, Springer, 75-102.

### Neural Network Methods

- Nielsen, M., Sornette, D. & Raissi, M. (2024). "Deep Learning for LPPLS: M-LNN and P-LNN." [arXiv:2405.12803](https://arxiv.org/abs/2405.12803) — **Implemented** (M-LNN, P-LNN)
- Cao, G., Shao, L., Yan, H. & Geman, H. (2025). "HLPPL: Hyped LPPL with Dual-Stream Transformer." [arXiv:2510.10878](https://arxiv.org/abs/2510.10878) — **Implemented**
- Ma, J. & Li, C. (2024). "Detecting Market Bubbles: A Generalized LPPLS Neural Network Model." *Economics Letters*, 244, 112003. [DOI:10.1016/j.econlet.2024.112003](https://doi.org/10.1016/j.econlet.2024.112003) — Future work (extends P-LNN, paywalled)
- Lee, G., Jeong, M., Park, T. & Ahn, K. (2025). "More Than Ex-Post Fitting: LPPL and Its AI-Based Classification." *Humanities and Social Sciences Communications*, 12, 236. [DOI:10.1038/s41599-025-05920-7](https://doi.org/10.1038/s41599-025-05920-7) — **Implemented** (DTCAI)
- Sakurai, Y. & Chen, Z. (2024). "Forecasting Tail Risk via Neural Networks with Asymptotic Expansions." *IMF Working Paper* WP/24/99. [IMF](https://www.imf.org/en/Publications/WP/Issues/2024/05/10/Forecasting-Tail-Risk-via-Neural-Networks-with-Asymptotic-Expansions-548841) — Future work (CoFiE-NN, VaR-focused)

### Fat Tails in Finance

- Mandelbrot, B.B. (1963). "The Variation of Certain Speculative Prices." *J. Business*, 36(4), 394-419.
- Gabaix, X. (2009). "Power Laws in Economics and Finance." *Ann. Rev. Econ.*, 1, 255-294.
- Mandelbrot, B.B. & Taleb, N.N. (2010). "Random Jump, Not Random Walk." In *The Known, the Unknown, and the Unknowable in Financial Risk Management.* Princeton University Press.
