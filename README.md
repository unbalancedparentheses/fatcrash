# fatcrash

**71% of countries have exchange rate distributions with infinite variance.**

The median tail index across 138 countries is alpha = 1.57. Standard risk models assume finite variance (alpha > 2) and often finite kurtosis (alpha > 4). For the majority of the world's currencies, these assumptions are empirically false. fatcrash detects crashes by measuring what actually matters: the tail.

Python + Rust (PyO3). 17 crash detection methods + 5 regime detection algorithms + 7 market regime signal families. 500 years of data.

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

For FRED forex analysis, the `load_fred_forex()` helper loads all 23 currency pairs from the [forex-centuries](https://github.com/unbalancedparentheses/forex-centuries) dataset:

```python
from fatcrash.data.ingest import load_fred_forex

pairs = load_fred_forex()           # dict of 23 DataFrames
aud = load_fred_forex("AUD_USD")    # single pair
```

Requires `git clone https://github.com/unbalancedparentheses/forex-centuries ~/projects/forex-centuries` or set `FOREX_CENTURIES_DIR`.

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

### Crash detection: precision, recall, and F1

#### Core dataset (39 drawdowns across BTC, SPY, Gold)

All accuracy numbers are in-sample on historical data. Methods are tested on both crash windows (120 days before a drawdown peak) and non-crash windows (random 120-day stretches at least 180 days from any crash). 39 crash windows, 150 non-crash windows.

| Method | TP | FP | FN | TN | Precision | Recall | F1 |
|--------|:--:|:--:|:--:|:--:|:---------:|:------:|:--:|
| **LPPLS** | **35** | **40** | **4** | **110** | **47%** | **90%** | **61%** |
| **LPPLS confidence** | **17** | **16** | **22** | **134** | **52%** | **44%** | **47%** |
| M-LNN | 22 | 47 | 17 | 103 | 32% | 56% | 41% |
| GSADF | 15 | 24 | 24 | 126 | 38% | 38% | 38% |
| **DFA** | **32** | **115** | **7** | **35** | **22%** | **82%** | **34%** |
| CSD (on vol) | 20 | 59 | 19 | 91 | 25% | 51% | 34% |
| RV spike | 21 | 71 | 18 | 79 | 23% | 54% | 32% |
| Hamilton | 16 | 57 | 23 | 93 | 22% | 41% | 29% |
| Hurst | 23 | 101 | 16 | 49 | 19% | 59% | 28% |
| Pickands | 19 | 82 | 20 | 68 | 19% | 49% | 27% |
| Kappa | 19 | 83 | 20 | 67 | 19% | 49% | 27% |
| DEH | 18 | 81 | 21 | 69 | 18% | 46% | 26% |
| Spectral | 11 | 38 | 28 | 112 | 22% | 28% | 25% |
| Taleb Kappa | 13 | 53 | 26 | 97 | 20% | 33% | 25% |
| P-LNN | 9 | 30 | 30 | 120 | 23% | 23% | 23% |
| QQ | 15 | 79 | 24 | 71 | 16% | 38% | 23% |
| GPD VaR | 10 | 73 | 14 | 66 | 12% | 42% | 19% |
| Max-to-Sum | 12 | 86 | 27 | 64 | 12% | 31% | 18% |
| Hill | 11 | 84 | 28 | 66 | 12% | 28% | 16% |
**Why precision is low for tail/regime methods:** These methods detect distributional regime shifts (tail thickening, persistent dynamics), not crash-specific patterns. They fire in many non-crash periods because fat tails and persistence are pervasive in financial data. This is by design — they measure the *distributional regime*, not a specific crash. LPPLS and GSADF have higher precision because they detect bubble-specific structure.

**The Sornette–Bouchaud debate on precision vs recall:**

Sornette (the LPPLS inventor) argues that LPPLS is deliberately tuned for **high recall at the cost of precision** because the cost function is asymmetric — missing a crash is far more expensive than a false alarm. He calls false positives "failed predictions" and argues they are inevitable: bubbles can end in slow deflation rather than sharp crashes. His 2024 paper with Nielsen ([arXiv:2405.12803](https://arxiv.org/abs/2405.12803)) introduced the tightened omega [6,13] range specifically to improve precision without sacrificing recall.

Bouchaud takes a more skeptical view. In his work at CFM and in papers with Potters, he emphasizes that fat-tail estimators (Hill, etc.) measure **unconditional** properties of returns and are poor at **conditional** crash prediction. His point is exactly what the data shows: tail estimators have decent recall but terrible precision because fat tails are always present, not just before crashes. He favors portfolio-level risk measures (drawdown control, volatility targeting) over point-in-time crash prediction.

Both perspectives are reflected in fatcrash: LPPLS targets the mechanism (Sornette's approach), tail estimators measure the regime (which Bouchaud correctly notes is always fat-tailed), and the aggregator combines both — using Sornette-style bubble detection as the primary signal and Bouchaud-style regime measurement as confirmation.

**NN method notes:** M-LNN (per-series fitting) is the strongest NN approach at F1=41%, comparable to GSADF. P-LNN (pre-trained, ~700x faster) underperforms at F1=23%.

#### Extended dataset (FRED forex + options backtester)

To test generalization beyond the three core assets, we evaluate on 23 FRED forex pairs (1971–2025) and 6 options backtester series. 57 crash windows, 481 non-crash windows.

| Method | TP | FP | FN | TN | Precision | Recall | F1 |
|--------|:--:|:--:|:--:|:--:|:---------:|:------:|:--:|
| LPPLS confidence | 31 | 45 | 26 | 436 | 41% | 54% | 47% |
| LPPLS | 51 | 116 | 6 | 365 | 31% | 89% | 46% |
| GSADF | 40 | 91 | 17 | 390 | 31% | 70% | 43% |
| M-LNN | 35 | 136 | 22 | 345 | 20% | 61% | 31% |
| QQ | 37 | 215 | 20 | 251 | 15% | 65% | 24% |
| RV spike | 37 | 237 | 20 | 244 | 14% | 65% | 22% |
| Max-to-Sum | 33 | 210 | 24 | 261 | 14% | 58% | 22% |
| Kappa | 32 | 220 | 25 | 261 | 13% | 56% | 21% |
| Hill | 28 | 202 | 29 | 279 | 12% | 49% | 20% |
| Hurst | 40 | 314 | 17 | 160 | 11% | 70% | 19% |
| Spectral | 17 | 118 | 40 | 356 | 13% | 30% | 18% |
| DFA | 41 | 374 | 16 | 100 | 10% | 72% | 17% |
| Pickands | 26 | 232 | 31 | 230 | 10% | 46% | 17% |
| GPD VaR | 17 | 169 | 10 | 196 | 9% | 63% | 16% |
| CSD (on vol) | 19 | 158 | 38 | 323 | 11% | 33% | 16% |
| DEH | 25 | 233 | 32 | 233 | 10% | 44% | 16% |
| Taleb Kappa | 21 | 187 | 36 | 280 | 10% | 37% | 16% |
| Hamilton | 22 | 212 | 35 | 269 | 9% | 39% | 15% |
| P-LNN | 5 | 52 | 52 | 429 | 9% | 9% | 9% |

LPPLS maintains 89% recall on the extended dataset. GSADF jumps from F1=38% to F1=43% — forex pairs provide more explosive episodes for GSADF to detect. DFA's precision drops (10% vs 22%) because forex data shows persistent dynamics even in non-crash periods.

#### Combined dataset (96 crash windows, 631 non-crash windows)

Merging core (BTC/SPY/Gold) and extended (FRED/OptsBT) datasets:

| Method | Precision | Recall | F1 |
|--------|:---------:|:------:|:--:|
| **LPPLS** | **36%** | **90%** | **51%** |
| LPPLS confidence | 44% | 50% | 47% |
| GSADF | 32% | 57% | 41% |
| M-LNN | 24% | 59% | 34% |
| RV spike | 16% | 60% | 25% |
| QQ | 15% | 54% | 24% |
| Kappa | 14% | 53% | 23% |
| DFA | 13% | 76% | 22% |
| CSD (on vol) | 15% | 41% | 22% |
| Hurst | 13% | 66% | 22% |
| Max-to-Sum | 13% | 47% | 21% |
| Spectral | 15% | 29% | 20% |
| Pickands | 13% | 47% | 20% |
| DEH | 12% | 45% | 19% |
| Hamilton | 12% | 40% | 19% |
| Hill | 12% | 41% | 19% |
| Taleb Kappa | 12% | 35% | 18% |
| GPD VaR | 10% | 53% | 17% |
| P-LNN | 15% | 15% | 15% |

LPPLS recall holds at 90% across 96 crash windows spanning crypto, equities, commodities, and 23 forex pairs. This is the strongest evidence that LPPLS detects a universal bubble-to-crash mechanism, not an asset-specific pattern.

### Recall by crash size (BTC, SPY, Gold)

| Method | Small (<15%) | Medium (15-30%) | Major (>30%) |
|--------|:---:|:---:|:---:|
| LPPLS | 86% | 100% | 75% |
| DFA | 86% | 88% | 62% |
| Hurst | 57% | 65% | 50% |
| M-LNN | 57% | 65% | 38% |
| RV spike | 50% | 53% | 62% |
| CSD (on vol) | 50% | 47% | 62% |
| LPPLS confidence | 43% | 59% | 12% |
| Hamilton | 29% | 29% | **88%** |
| DEH | 43% | 41% | 62% |
| Taleb Kappa | 21% | 35% | 50% |
| GSADF | 14% | 59% | 38% |

LPPLS catches 100% of medium crashes (15-30% drawdowns). LPPLS confidence drops to 12% on major crashes — the multi-window aggregation is too conservative for the fastest, most violent drawdowns. **Hamilton has 88% recall on major crashes** — it's the best single method for detecting large (>30%) drawdowns, though it rarely fires on smaller crashes. DEH and Taleb Kappa improve on major crashes (62% and 50%), detecting the tail-thickening that precedes large moves.

### Weighted ensemble aggregator

Combines all methods via weighted average + category agreement bonus. When 3+ independent categories (bubble, tail, regime, structure) agree, probability gets a +15% bonus.

**Core dataset (39 crash, 150 non-crash):**

| Threshold | Level | Precision | Recall | F1 |
|:---------:|-------|:---------:|:------:|:--:|
| 0.3 | ELEVATED+ | 23% | 82% | 36% |
| 0.4 | >40% | 30% | 69% | 42% |
| **0.5** | **HIGH+** | **54%** | **56%** | **55%** |
| 0.7 | CRITICAL+ | 50% | 3% | 5% |

**Extended dataset (57 crash, 481 non-crash):**

| Threshold | Level | Precision | Recall | F1 |
|:---------:|-------|:---------:|:------:|:--:|
| 0.3 | ELEVATED+ | 14% | 95% | 25% |
| 0.4 | >40% | 20% | 81% | 32% |
| **0.5** | **HIGH+** | **36%** | **58%** | **44%** |
| 0.7 | CRITICAL+ | 17% | 2% | 3% |

At threshold 0.5, the aggregator achieves P=54%, R=56%, F1=55% on the core dataset. On the extended dataset it maintains F1=44% with P=36%, R=58%. The best individual method (LPPLS, F1=61%) still outperforms the ensemble on F1; the ensemble's advantage is balanced precision/recall rather than LPPLS's high-recall-low-precision profile.

### 6/6 known GBP/USD crises detected

1976 IMF Crisis, 1985 Plaza Accord, 1992 Black Wednesday, 2008 Financial Crisis, 2016 Brexit, 2022 Truss Mini-Budget.

| Crisis | Hill alpha | K/bench | Status |
|--------|:---------:|:-------:|:------:|
| 1976 IMF Crisis | 2.51 | 0.71 | DETECTED |
| 1985 Plaza Accord | 2.93 | 0.80 | DETECTED |
| 1992 Black Wednesday | 4.08 | 0.88 | DETECTED |
| 2008 Financial Crisis | 2.78 | 0.56 | DETECTED |
| 2016 Brexit Vote | 1.92 | 0.48 | DETECTED |
| 2022 Truss Mini-Budget | 2.81 | 0.92 | DETECTED |

### GBP/USD by decade (1971-2025)

| Decade | N days | Hill alpha | Kappa | K/bench | Taleb K | VaR 95% | Worst day |
|--------|:------:|:---------:|:-----:|:-------:|:-------:|:-------:|:---------:|
| 1970s | 2,247 | 2.92 | 0.648 | 0.78 | 0.238 | 6.2% | -3.8% |
| 1980s | 2,508 | 4.36 | 0.567 | 0.68 | 0.336 | 3.8% | -3.8% |
| 1990s | 2,515 | 4.51 | 0.686 | 0.83 | 0.116 | 3.4% | -3.3% |
| 2000s | 2,516 | 2.90 | 0.474 | 0.57 | 0.324 | 5.5% | -5.0% |
| 2010s | 2,501 | 3.86 | 0.293 | 0.35 | 0.504 | 2.0% | -8.2% |
| 2020s | 1,498 | 3.39 | 0.574 | 0.71 | 0.236 | 3.1% | -3.1% |

The 2010s are notable: lowest kappa/benchmark ratio (0.35) yet the worst single-day loss (-8.2%, Brexit). This is the signature of a regime where extreme events dominate — low tail index and high concentration of risk in a single observation.

## 500 Years of Forex Data

### FRED Daily (23 currency pairs, 1971-2025)

| Pair | Hill alpha | QQ alpha | DEH gamma | Hurst H | DFA alpha | GSADF bubble? |
|------|:---------:|:-------:|:--------:|:------:|:--------:|:---:|
| VEF/USD | 1.20 | 0.82 | 1.06 | 0.53 | 0.82 | YES |
| HKD/USD | 1.73 | 2.12 | 0.24 | 0.54 | 0.62 | YES |
| KRW/USD | 1.90 | 1.93 | 0.44 | 0.67 | 0.60 | YES |
| MXN/USD | 2.04 | 1.98 | 0.44 | 0.56 | 0.57 | YES |
| LKR/USD | 2.14 | 1.97 | 0.51 | 0.58 | 0.66 | YES |
| TWD/USD | 2.31 | 2.62 | 0.21 | 0.60 | 0.63 | YES |
| THB/USD | 2.38 | 2.43 | 0.33 | 0.58 | 0.59 | YES |
| MYR/USD | 2.42 | 2.46 | 0.33 | 0.58 | 0.60 | YES |
| AUD/USD | 2.58 | 2.30 | 0.44 | 0.56 | 0.56 | YES |
| INR/USD | 2.62 | 2.56 | 0.34 | 0.57 | 0.58 | YES |
| CNY/USD | 2.79 | 1.70 | 0.69 | 0.59 | 0.71 | YES |
| BRL/USD | 2.80 | 3.12 | 0.15 | 0.56 | 0.58 | YES |
| NZD/USD | 2.89 | 2.46 | 0.44 | 0.57 | 0.56 | no |
| ZAR/USD | 3.19 | 3.43 | 0.15 | 0.58 | 0.54 | YES |
| NOK/USD | 3.39 | 3.44 | 0.22 | 0.57 | 0.53 | YES |
| SEK/USD | 3.50 | 2.88 | 0.41 | 0.58 | 0.55 | no |
| SGD/USD | 3.59 | 3.66 | 0.18 | 0.56 | 0.53 | YES |
| CHF/USD | 3.81 | 3.59 | 0.28 | 0.57 | 0.54 | no |
| CAD/USD | 3.84 | 3.58 | 0.27 | 0.57 | 0.53 | YES |
| DKK/USD | 3.84 | 3.23 | 0.37 | 0.58 | 0.55 | no |
| JPY/USD | 3.94 | 4.02 | 0.18 | 0.58 | 0.58 | YES |
| GBP/USD | 4.13 | 4.11 | 0.19 | 0.58 | 0.55 | YES |
| EUR/USD | 4.88 | 4.90 | 0.12 | 0.56 | 0.54 | no |

Universals across all 23 pairs:
- **Fat tails**: DEH gamma > 0 for 23/23. Mean Hill alpha = 2.95, mean QQ alpha = 2.84 — two independent estimators converge.
- **Persistence**: Hurst H > 0.5 for 23/23, DFA alpha > 0.5 for 23/23.
- **Bubbles**: 18/23 show explosive episodes via GSADF.

VEF/USD (Venezuela) is the extreme case: Hill alpha = 1.20, QQ alpha = 0.82, DEH gamma = 1.06 — every estimator confirms infinite variance. KRW/USD has the fattest tails among liquid pairs (Hill alpha = 1.90). CNY/USD shows the strongest persistence (DFA = 0.71) — consistent with managed float dynamics.

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

### Cross-method summary

**FRED Daily Forex (23 pairs):**

| Metric | Value |
|--------|:-----:|
| DEH gamma > 0 (heavy tails) | 23/23 |
| Hurst H > 0.5 (persistent) | 23/23 |
| DFA alpha > 0.5 (persistent) | 23/23 |
| Hill alpha < 4 (fat tails) | 21/23 |
| QQ alpha < 4 (fat tails) | 20/23 |
| Spectral d > 0 (long memory) | 19/23 |
| Taleb kappa > 0.1 (fat) | 19/23 |
| Pickands > 0 (heavy tails) | 18/23 |
| GSADF bubble detected | 18/23 |
| Mean Hill alpha | 2.95 |
| Mean QQ alpha | 2.84 |
| Mean Hurst H | 0.575 |
| Mean DFA alpha | 0.588 |
| Mean DEH gamma | 0.346 |
| Mean Taleb kappa | 0.330 |

**Clio Infra Yearly (top 30 countries):**

| Metric | Value |
|--------|:-----:|
| Hill alpha < 4 (fat tails) | 29/30 |
| DFA > 0.5 (persistent) | 28/30 |
| QQ alpha < 4 (fat tails) | 28/30 |
| Hurst > 0.5 (persistent) | 25/30 |
| Taleb kappa > 0.1 (fat) | 20/29 |
| DEH > 0 (heavy tails) | 20/30 |
| Hill alpha < 2 (infinite var) | 19/30 |
| Pickands > 0 (heavy tails) | 19/30 |
| Spectral d > 0 (long memory) | 18/30 |
| Mean Hill alpha | 1.56 |
| Mean QQ alpha | 1.57 |
| Mean Hurst H | 0.615 |
| Mean DFA alpha | 0.883 |
| Mean Taleb kappa | 0.560 |

Method agreement: Pickands xi > 0, DEH gamma > 0, Hill alpha < 4, and QQ alpha < 4 all detect heavy tails from independent angles — order statistics, moments, QQ-slope, and three-quantile methods. Hurst H > 0.5, DFA alpha > 0.5, and Spectral d > 0 all confirm persistence from R/S analysis, detrended fluctuation, and the frequency domain respectively. When multiple independent methods converge on the same conclusion, the evidence is robust.

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
| 11 | Momentum | Regime | 3-12 month trailing return + reversal | momentum, reversal signal |
| 12 | GPD | EVT | Tail risk (VaR, ES) | VaR, Expected Shortfall |
| 13 | GEV | EVT | Block maxima classification | Frechet / Gumbel / Weibull |
| 14 | LPPLS + GSADF | Bubble | Bubble shape + explosive unit roots | critical time, confidence |
| 15 | M-LNN | Bubble (NN) | Per-series LPPLS via neural network | tc, m, omega, confidence |
| 16 | P-LNN | Bubble (NN) | Pre-trained LPPLS (~700x faster) | tc, m, omega, confidence |
| 17 | Price velocity | Structure | Volatility acceleration (cascade detection) | velocity signal |

**Regime detection algorithms** (implemented in Rust, F1 on core dataset):

| # | Algorithm | What it computes | F1 | Notes |
|---|-----------|-----------------|:--:|-------|
| R-A1 | RV Spike | Short-term RV vs long-term baseline (21d / 126d) | 32% | Detects volatility regime change |
| R-A2 | CSD on Volatility | AR(1) + variance of rolling RV series | 34% | Tipping point detection (Scheffer 2009) |
| R-A3 | Hamilton Filter | 2-state HMM, EM estimation, Kim smoother | 29% | 88% recall on major (>30%) crashes |
| R-A4 | Realized Variance | Simple, Parkinson, Garman-Klass estimators | — | Foundation for RV spike and CSD |
| R-A5 | Jump Risk (BNS) | Bipower variation, jump variance, z-test | — | Building block (poor at daily frequency alone) |

**Market regime signal families** (macro/microstructure — see [Market Regime Signals](#market-regime-signals)):

| # | Family | Signals | What it detects |
|---|--------|---------|-----------------|
| R1 | Risk Premium | VRP, RV spike | Variance and tail-risk compensation |
| R2 | Liquidity | SOFR-OIS, TED, Amihud, xccy basis, Treasury/equity vol ratio | Funding stress, market illiquidity |
| R3 | Volatility Regime | VIX slope, SKEW, MOVE, VVIX | Vol term structure inversion, vol-of-vol |
| R4 | Credit & Macro | OFR FSI, STLFSI, credit spreads, EBP, yield curve, term premium | Credit stress, macro deterioration |
| R5 | Structure & Flows | Cross-asset eigenvalue, COT, ETF flows | Diversification breakdown, positioning extremes |
| R6 | Contagion | CoVaR, MES, SRISK | Systemic risk, institution-level stress |
| R7 | Sentiment | FOMC tone, news uncertainty | Policy uncertainty, narrative shifts |

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

**Momentum** (Jegadeesh & Titman, 1993). Trailing log return over 3, 6, and 12-month windows. The strongest documented anomaly in finance: 3-12 month winners continue to outperform, generating ~9.5% cumulative excess returns. Momentum *reversal* — when long-term momentum is positive but short-term turns negative — is a crash precursor. The divergence between 12-month and 1-month momentum captures the transition from bubble buildup to unwind. Scowcroft & Sefton (2005) showed momentum is driven by industry effects in large-cap and stock-specific effects in small-cap.

**Price velocity** (cascade detector). Rate of change of realized volatility: velocity = (vol[t] - vol[t-lag]) / vol[t-lag]. Detects forced-liquidation cascades where volatility itself accelerates — the signature of events like Volmageddon (Feb 5, 2018: XIV lost 97%, VIX spiked 116% in hours due to $5B in forced inverse-vol product rebalancing) or the Sep 2019 repo market blowup (collateral cleared at 600bps over, largest 1-day move on record). High positive velocity = cascade forming.

### Constant volatility strategy

Position sizing via inverse volatility targeting: weight = target_vol / realized_vol. Dalgaard (2016, CBS thesis) tested tail-hedging strategies on S&P 500 data and found that simple put monetization strategies do NOT reduce drawdowns and have LOWER returns/Sharpe than an unhedged index. The constant volatility strategy, however, DOES reduce drawdowns while earning higher returns — mechanically cutting exposure when vol spikes (exactly when cascades hit) and increasing exposure during calm periods.

**Rebalance risk signal** (Rattray, Harvey & Van Hemert, 2018). Mechanical rebalancing is negative convexity — it buys into drawdowns that continue. When DFA detects trending behavior (alpha > 0.5) and momentum is negative (active drawdown), rebalancing into the position is dangerous. A 10% trend-following allocation reduces drawdowns by ~5 percentage points.

### Extreme value theory

**GPD** (Balkema & de Haan, 1974). Fits exceedances over a threshold to the Generalized Pareto Distribution. Yields VaR and Expected Shortfall at arbitrary confidence levels.

**GEV** (Fisher & Tippett, 1928). Fits block maxima to the Generalized Extreme Value distribution. Classifies into Frechet (xi > 0, heavy tail), Gumbel (xi = 0, exponential), or Weibull (xi < 0, bounded).

### Bubble detection

**LPPLS** (Sornette, 2003). Models bubble dynamics as a power law with log-periodic oscillations: log(p(t)) = A + B|tc-t|^m + C|tc-t|^m * cos(omega*log|tc-t| + phi). The critical time tc is the predicted crash date. Confidence measured by fitting across many windows. Nonlinear optimization via CMA-ES in Rust.

**GSADF** (Phillips, Shi & Yu, 2015). Detects explosive unit root behavior — the econometric signature of bubbles. The supremum of recursive ADF statistics over all feasible subsamples, with Monte Carlo critical values. Complements LPPLS: LPPLS detects bubble *shape*, GSADF detects *explosive growth*.

### Neural network methods (requires `pip install fatcrash[deep]`)

Two neural network approaches to LPPLS fitting from Nielsen, Sornette & Raissi (2024). Both share a common pattern: the network predicts nonlinear LPPLS parameters (tc, m, omega), then linear parameters (A, B, C1, C2) are solved analytically via OLS. This physics-informed decomposition makes training stable and predictions interpretable.

**M-LNN** — Mono-LPPLS Neural Network (Nielsen, Sornette & Raissi, 2024). One small network (2 hidden layers, 64 units each) trained per time series. Minimizes reconstruction MSE between the LPPLS fit and observed log-prices. Works on variable-length input. Slower than P-LNN but more flexible — no pre-training required. **F1=41%, Recall=56%** — the strongest NN method.

```python
from fatcrash.nn.mlnn import fit_mlnn
result = fit_mlnn(times, log_prices, epochs=200, lr=1e-2)
# result.tc, result.m, result.omega, result.is_bubble, result.confidence
```

**P-LNN** — Poly-LPPLS Neural Network (Nielsen, Sornette & Raissi, 2024). Pre-trained on 100K synthetic LPPLS series, ~700x faster than CMA-ES at inference. A deeper network (4 hidden layers: 512-256-128-64) maps a 252-observation min-max normalized price window to (tc, m, omega) in a single forward pass. Three variants trained with white noise, AR(1) noise, or both. **F1=23%, Recall=23%** — fast but lower accuracy.

```python
from fatcrash.nn.plnn import train_plnn, predict_plnn
model = train_plnn(variant="P-LNN-100K", n_samples=100_000)
result = predict_plnn(model, times, log_prices)
```

## Signal Aggregation

Methods grouped into 4 independent categories. When 3+ categories agree, probability gets a +15% bonus.

| Category | Methods | What it detects |
|----------|---------|-----------------|
| **Bubble** | LPPLS, GSADF, M-LNN, P-LNN | Super-exponential growth, explosive unit roots |
| **Tail** | Hill, Pickands, DEH, QQ, Taleb Kappa, Max-Stability Kappa, Max-to-Sum, GPD | Tail thickening, distributional regime shifts |
| **Regime** | Hurst, DFA, Spectral, Momentum reversal, CSD (on vol), Hamilton | Transition from mean-reverting to persistent dynamics, trend breaks, tipping points |
| **Structure** | Multiscale, LPPLS tc proximity, Price velocity | Cross-timeframe agreement, timing, cascade detection |

Indicators are also computed at daily, 3-day, and weekly frequencies. A signal at one scale may be noise; a signal across all three is structural.

## Market Regime Signals

Complementing the 17 crash detection methods above, fatcrash includes a regime signal framework that combines macro, funding, credit, and microstructure indicators to classify the current market state as risk-on, neutral, or risk-off. These signals answer a different question than tail estimators: not "are tails thickening?" but "is the macro-financial environment shifting toward stress?"

The goal is transparent, inspectable signals grounded in the academic literature — not black-box scores. Each signal is normalized to a common scale (z-score or percentile rank), aggregated into thematic buckets, and combined into a single regime score.

### Signal Families

**Risk Premium** — Variance risk premium (VRP) and jump/tail-risk premium. VRP = implied variance minus realized variance. When VRP is high, investors are paying more for variance protection — a sign of fear. When VRP goes negative (realized vol exceeds implied), the market is already in acute stress. Jump risk premium isolates the compensation for discontinuous moves using BNS bipower variation (Barndorff-Nielsen & Shephard, 2004). References: Bollerslev, Tauchen & Zhou (2009); Bekaert & Hoerova (2014); Bollerslev & Todorov (2011).

**Liquidity & Funding Stress** — SOFR-OIS spread (pure funding stress), TED spread (discontinued April 2022, substitute with SOFR vs 3M T-bill), Amihud illiquidity (|return|/volume), cross-currency basis (deviation from covered interest parity — negative USD basis = global dollar shortage), Treasury/equity realized vol ratio (when Treasury vol exceeds equity vol, the safe haven is broken). References: Brunnermeier & Pedersen (2009); Pastor & Stambaugh (2003); Flood, Liechty & Piontek (2015).

**Volatility Regime** — VIX term structure slope (backwardation = near-term fear), SKEW (tail risk pricing), MOVE (bond vol), VVIX (vol-of-vol). Negative VIX slope is the single fastest stress signal — it inverts within hours of a shock. VVIX captures uncertainty about uncertainty itself.

**Credit & Macro** — OFR Financial Stress Index, STLFSI, credit spreads (HY OAS, BBB OAS, Baa-Aaa), excess bond premium (EBP — the component of credit spreads driven by risk appetite, not default risk; Gilchrist & Zakrajsek, 2012), yield curve (10Y-2Y, 10Y-3M), ACM term premium. EBP is the most predictive component for recessions. The 10Y-3M spread has inverted before every US recession since 1960.

**Structure & Flows** — Cross-asset correlation eigenvalue (largest eigenvalue of rolling correlation matrix — when it approaches 1.0, all assets move together and diversification breaks down), CFTC COT net speculative positioning, ETF flows. Rising eigenvalue + rising variance is the critical slowing down signature.

**Contagion & Systemic Risk** — CoVaR (system VaR conditional on institution distress; Adrian & Brunnermeier, 2016), MES (marginal expected shortfall), SRISK (expected capital shortfall in crisis; Brownlees & Engle, 2017). Pre-computed data available from NYU Stern V-Lab.

**Sentiment** — FOMC communication tone (using Loughran-McDonald financial sentiment dictionary), Baker-Bloom-Davis Economic Policy Uncertainty index (FRED: `USEPUINDXD`).

### Key Algorithms

**Realized Variance.** From daily returns: `RV_t = (252/W) * sum(r_i^2)`. Parkinson estimator (from OHLC): `RV_Park = (252/(4*ln2*W)) * sum(ln(H/L)^2)`. Garman-Klass (most efficient from OHLC): `GK_i = 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2`.

**Variance Risk Premium.** `VRP_t = (VIX/100)^2/12 - RV_monthly`. VRP > 0 is normal (paying for protection). VRP < 0 means realized vol exceeds implied (acute stress).

**RV Spike Signal.** Compares short-term realized variance (21-day) to a long-term baseline (126-day): `ratio = RV_short / RV_long`. When ratio > 2x, volatility is spiking — the signature of regime change. More robust than BNS JV/RV fraction at daily frequency (BNS bipower variation was designed for 5-minute intraday data). F1=32%.

**Jump Risk Decomposition.** BNS bipower variation: `BV_t = (pi/2) * (1/(W-1)) * sum(|r_i| * |r_{i-1}|)`. Jump variance: `JV_t = max(RV_t - BV_t, 0)`. Building block for the RV spike signal. At daily frequency, the JV/RV fraction alone doesn't discriminate (Bollerslev & Todorov, 2011 showed it dominates return predictability at *intraday* frequency).

**Hamilton Filter (2-state HMM).** States: normal (s=0) and stressed (s=1). Each state has its own mean and variance. Forward filter: predict → compute densities → update via Bayes' rule. Parameter estimation via EM (Baum-Welch). Use log-space for numerical stability. Multiple random restarts (parallelized via rayon) to avoid local optima. Overall F1=29%, but **88% recall on major (>30%) crashes** — the best single method for detecting large drawdowns. Backward-looking: it confirms you're in a crisis, it doesn't predict one.

**Critical Slowing Down.** Near a tipping point, systems recover more slowly from perturbations (Scheffer et al., 2009). Observable: rising AR(1) coefficient AND rising variance simultaneously. **Important:** CSD must be applied to the *volatility* series (rolling RV), not raw returns — raw returns have near-zero autocorrelation, so AR(1) on returns is meaningless. Volatility has strong clustering (autocorrelation), so CSD on the vol series detects genuine regime transitions. Uses windows of 63/21 (vs the original 252/63 which was too large). F1=34%, 51% recall.

**Hawkes Process Branching Ratio** (future work). Intensity: `lambda(t) = mu + alpha * sum(exp(-beta*(t-t_i)))`. Branching ratio `n = alpha/beta`. Not yet implemented — the self-excitation signal is largely captured by CSD + velocity. Would require an L-BFGS-B optimizer and event-timestamp input rather than price series.

### Regime Scoring

Signals are normalized (rolling z-score or percentile rank), aggregated into 7 buckets (mean of member signals), then combined:

```
regime_score = -0.30 * risk_premium
             - 0.25 * liquidity
             - 0.20 * volatility
             - 0.15 * credit_macro
             - 0.10 * structure_flows
             - w_contagion * contagion
             - w_sentiment * sentiment
```

Negative weights: higher stress → more risk-off. Smooth with EMA (alpha=0.2) to prevent noisy label switching.

**Labels:** risk_on (score >= +0.50), neutral (-0.50 < score < +0.50), risk_off (score <= -0.50).

**Confidence:** `min(1, |score| / 2)`. Top-3 buckets by absolute impact reported as contributors.

### Historical Calibration

| Signal | Sept 2008 (Lehman) | Mar 2020 (COVID) | Feb 2018 (Volmageddon) | Sept 2019 (Repo) | 2022 (Rate Shock) | 2013 (Taper Tantrum) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| VIX | ~80 | ~85 | 14→37 | ~18 | ~35 | ~21 |
| HY OAS | >1000 bps | ~1000 bps | ~330 bps | ~400 bps | ~600 bps | ~500 bps |
| SOFR/TED | ~450 bps | ~50 bps | unchanged | repo 600 bps | ~20 bps | ~15 bps |
| VRP | negative | negative | brief inversion | normal | positive | positive |
| Cross-asset λ | ~1.0 | ~1.0 | brief spike | low | moderate | low |
| **Expected** | **risk-off** | **risk-off** | **brief risk-off (vol only)** | **brief risk-off (liquidity only)** | **neutral→risk-off** | **neutral** |

2008 and 2020 are unambiguous (all buckets fire). Feb 2018 (Volmageddon) is the purest vol event — only the volatility bucket fires. Sept 2019 repo crisis is the mirror: only liquidity fires. These single-bucket events test that no one bucket dominates the regime call.

### Regime Data Sources

| Source | Signals | Frequency | Access |
|--------|---------|-----------|--------|
| FRED API | VIX, spreads, yield curve, STLFSI, EPU | Daily/Weekly | Free |
| OFR | Financial Stress Index | Daily | Free |
| Cboe | VIX futures, SKEW, VVIX | Daily | Free (research) |
| NY Fed | ACM term premium | Monthly | Free |
| CFTC | COT net speculative positions | Weekly (3-day lag) | Free |
| NYU V-Lab | CoVaR, MES, SRISK | Daily | Free |

**Frequency alignment:** maintain all signals at daily resolution. Forward-fill weekly/monthly series. Mark signals stale after 5 business days.

**Warmup periods:** Rolling z-score requires 63 days minimum. Rolling percentile requires 126 days. Hamilton filter requires 252 days. Full signal availability for FRED-based signals starts ~2005.

### Theoretical Frameworks

**Hamilton's regime-switching** (Hamilton, 1989) treats regimes as latent discrete states inferred from data via a hidden Markov model. Statistically rigorous but backward-looking — it tells you which regime you're probably in, not when the next transition will occur.

**Sornette's LPPL framework** treats regime endings as deterministic critical points driven by endogenous positive feedback. Forward-looking but sensitive to fitting assumptions. Already implemented in fatcrash's bubble detection methods.

**Critical slowing down** (Scheffer et al., 2009): near tipping points, systems recover more slowly. Observable signatures: rising variance + rising autocorrelation. The empirical complement to Sornette's theoretical prediction.

**Hawkes processes** (Bacry, Muzy): extreme events cluster via self-excitation. Branching ratio approaching 1 signals criticality. Middle ground between LPPL (mechanistic) and Hamilton (statistical). Not yet implemented — signal largely captured by CSD + velocity.

**Rough volatility** (Gatheral & Rosenbaum, 2018): realized vol has Hurst exponent H ≈ 0.1 — rougher than a random walk. Standard GARCH underestimates short-term vol clustering.

**Intermediary asset pricing** (Adrian, Shin; He, Krishnamurthy): intermediary balance-sheet constraints drive risk premia and liquidity simultaneously. Explains why credit spreads, VRP, and funding stress co-move at regime transitions.

**Minsky's Financial Instability Hypothesis:** stability is destabilizing — calm encourages leverage migration from hedge to speculative to Ponzi finance. Low VRP + compressed credit spreads + rising leverage = Minsky warning even when no individual signal crosses a threshold.

**Financial network contagion** (Acemoglu, Ozdaglar; Elliott, Golub, Jackson): dense interconnection is "robust yet fragile." DebtRank (Battiston et al., 2012) quantifies systemic importance through the balance-sheet network.

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

**What to watch for:**
- **Hill alpha dropping below 3**: Revenue shocks are getting more extreme. The distribution is shifting toward heavier tails.
- **DFA shifting from > 0.5 to < 0.5**: Growth momentum is breaking down. Revenue used to be self-reinforcing; now it's mean-reverting.
- **Max-to-Sum ratio rising**: A single quarter is starting to dominate the entire history — either a massive win or a massive loss.
- **GPD VaR spiking**: The worst-case quarterly decline is getting worse, even accounting for the fat tails.

These methods won't tell you *why* revenue is deteriorating — you still need business context for that. But they can tell you *that* something structural has changed in the data before it becomes obvious in the headline numbers.

## Architecture

```
Rust (PyO3, _core.so)                Python
┌────────────────────────────┐       ┌──────────────────────────────────┐
│ Tail: Hill, Pickands, DEH, │       │ indicators/                      │
│       QQ, Kappa, Taleb,    │──────▶│   tail_indicator.py              │
│       MaxSum, Hurst, DFA,  │       │   vol_indicator.py               │
│       Spectral, Momentum,  │       │   lppls_indicator.py             │
│       Velocity             │       │   bubble_indicator.py            │
│                            │       │   evt_indicator.py               │
│ EVT:  GPD, GEV             │       │   regime_indicator.py (NEW)      │
│                            │       │                                  │
│ LPPLS: fit, confidence,    │──────▶│ nn/                              │
│        solve_linear        │       │   mlnn.py      (M-LNN)          │
│                            │       │   plnn.py      (P-LNN)          │
│ Bubble: GSADF              │       │   lppls_torch.py (shared)       │
│                            │       │   synthetic.py  (data gen)      │
│ Multiscale                 │       │                                  │
│                            │       │ aggregator/signals.py            │
│ Regime:                    │       │   + regime signal categories     │
│   RealizedVar, Jump, CSD,  │       │                                  │
│   Hamilton                 │       │ data/ingest.py                   │
│                            │       │   + from_fred_macro()            │
│ rayon: parallel CMA-ES,    │       │   + from_ofr_fsi()              │
│        GSADF, confidence   │       │ cli/ viz/ service/               │
└────────────────────────────┘       └──────────────────────────────────┘
```

All 15 classical estimators and 4 regime detection algorithms are implemented in Rust and exposed to Python via PyO3. The computationally intensive methods (LPPLS CMA-ES, GSADF, confidence, Hamilton EM) use rayon for parallelization. The 2 neural network methods are in Python (PyTorch) and call `lppls_solve_linear` from Rust for the analytical linear parameter solve.

| Component | Language | Why |
|-----------|----------|-----|
| LPPLS fitter (CMA-ES) | Rust | O(1000) nonlinear fits per anchor date |
| LPPLS confidence | Rust | Nested windows parallelized with rayon |
| GSADF test | Rust | O(n^2) BSADF + Monte Carlo, parallelized |
| GEV/GPD fitting | Rust | Rolling EVT needs speed |
| All tail & regime estimators | Rust | Called at every rolling window step |
| M-LNN, P-LNN | Python (PyTorch) | GPU support, autograd for training |
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

For extended analysis, `load_fred_forex()` loads all 23 FRED daily currency pairs from the [forex-centuries](https://github.com/unbalancedparentheses/forex-centuries) repository.

## Development

Requires [Nix](https://nixos.org/) with flakes enabled.

```bash
nix develop                  # Enter dev shell (Rust, Python 3.13, maturin, uv)
make setup                   # Install Python deps + build Rust extension
make build                   # Recompile Rust, install into venv
make test                    # Run all Rust + Python tests
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
- Ma, J. & Li, C. (2024). "Detecting Market Bubbles: A Generalized LPPLS Neural Network Model." *Economics Letters*, 244, 112003. [DOI:10.1016/j.econlet.2024.112003](https://doi.org/10.1016/j.econlet.2024.112003) — Future work (extends P-LNN, paywalled)
- Sakurai, Y. & Chen, Z. (2024). "Forecasting Tail Risk via Neural Networks with Asymptotic Expansions." *IMF Working Paper* WP/24/99. [IMF](https://www.imf.org/en/Publications/WP/Issues/2024/05/10/Forecasting-Tail-Risk-via-Neural-Networks-with-Asymptotic-Expansions-548841) — Future work (CoFiE-NN, VaR-focused)

### Momentum & Trend-Following

- Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65-91.
- Scowcroft, A. & Sefton, J. (2005). "Understanding Momentum." *Financial Analysts Journal*, 61(2), 64-82.
- Rattray, S., Harvey, C.R. & Van Hemert, O. (2018). "Strategic Rebalancing." *Journal of Portfolio Management*, 44(4), 18-31. [SSRN:2993107](https://ssrn.com/abstract=2993107)

### Tail Hedging & Volatility Targeting

- Dalgaard, K.T. (2016). "Tail-Risk Hedging: An Empirical Study." Copenhagen Business School (MSc thesis). — Constant volatility strategy beats put monetization.
- Bhansali, V. (2014). "Monetization of Tail Risk." *Financial Analysts Journal*, 70(1), 65-80. [DOI:10.2469/faj.v70.n1.5](https://doi.org/10.2469/faj.v70.n1.5)
- Cole, C.S. (2020). "Volatility and the Allegory of the Prisoner's Dilemma." Artemis Capital Management. — Long volatility as portfolio "rebounder"; stock-bond negative correlation is historically rare.

### Asset Class Returns

- Jordà, Ò., Knoll, K., Kuvshinov, D., Schularick, M. & Taylor, A.M. (2019). "The Rate of Return on Everything, 1870-2015." *Quarterly Journal of Economics*, 134(3), 1225-1298. [DOI:10.1093/qje/qjz012](https://doi.org/10.1093/qje/qjz012) — Housing and equities both ~7% real returns; housing at half the volatility.

### Fat Tails in Finance

- Mandelbrot, B.B. (1963). "The Variation of Certain Speculative Prices." *J. Business*, 36(4), 394-419.
- Gabaix, X. (2009). "Power Laws in Economics and Finance." *Ann. Rev. Econ.*, 1, 255-294.
- Mandelbrot, B.B. & Taleb, N.N. (2010). "Random Jump, Not Random Walk." In *The Known, the Unknown, and the Unknowable in Financial Risk Management.* Princeton University Press.

### Variance Risk Premium & Tail-Risk Premia

- Bollerslev, T., Tauchen, G. & Zhou, H. (2009). "Expected Stock Returns and Variance Risk Premia." *Review of Financial Studies*, 22(11), 4463-4492.
- Bollerslev, T., Marrone, J., Xu, L. & Zhou, H. (2014). "Stock Return Predictability and Variance Risk Premia: International Evidence." *J. Financial and Quantitative Analysis*, 49(3), 511-540.
- Bekaert, G. & Hoerova, M. (2014). "The VIX, the Variance Premium and Stock Market Volatility." *J. Econometrics*, 183(2), 181-190. NBER WP 18995.
- Bollerslev, T. & Todorov, V. (2011). "Tails, Fears, and Risk Premia." *J. Finance*, 66(6), 2165-2211.
- Barndorff-Nielsen, O.E. & Shephard, N. (2004). "Power and Bipower Variation with Stochastic Volatility and Jumps." *J. Financial Econometrics*, 2(1), 1-37.

### Liquidity & Funding Stress

- Pastor, L. & Stambaugh, R. (2003). "Liquidity Risk and Expected Stock Returns." *J. Political Economy*, 111(3), 642-685. NBER WP 8462.
- Flood, M., Liechty, J. & Piontek, K. (2015). "Systemwide Commonalities in Market Liquidity." OFR Working Paper.
- Brunnermeier, M. & Pedersen, L.H. (2009). "Market Liquidity and Funding Liquidity." *Review of Financial Studies*, 22(6), 2201-2238.

### Credit Spreads & Macro

- Gilchrist, S. & Zakrajsek, E. (2012). "Credit Spreads and Business Cycle Fluctuations." *American Economic Review*, 102(4), 1692-1720. NBER WP 17021.
- Cochrane, J. & Piazzesi, M. (2005). "Bond Risk Premia." NBER WP 9178.
- Adrian, T., Crump, R. & Moench, E. (2013). "Pricing the Term Structure with Linear Regressions." NY Fed Staff Report 340.

### Regime-Switching & Structural Breaks

- Hamilton, J. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357-384.
- Ang, A. & Bekaert, G. (2002). "International Asset Allocation with Regime Shifts." *Review of Financial Studies*, 15(4), 1137-1187.
- Ang, A. & Timmermann, A. (2012). "Regime Changes and Financial Markets." *Annual Review of Financial Economics*, 4, 313-337.

### Critical Transitions & Tipping Points

- Scheffer, M. et al. (2009). "Early-Warning Signals for Critical Transitions." *Nature*, 461, 53-59.
- Scheffer, M. et al. (2001). "Catastrophic Shifts in Ecosystems." *Nature*, 413, 591-596.

### Hawkes Processes

- Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015). "Hawkes Processes in Finance." *Market Microstructure and Liquidity*, 1(1).
- Filimonov, V. & Sornette, D. (2012). "Quantifying Reflexivity in Financial Markets." *Physical Review E*, 85, 056108.

### Rough Volatility

- Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). "Volatility is Rough." *Quantitative Finance*, 18(6), 933-949.

### Intermediary Asset Pricing

- He, Z. & Krishnamurthy, A. (2013). "Intermediary Asset Pricing." *American Economic Review*, 103(2), 732-770.
- Adrian, T. & Shin, H.S. (2014). "Procyclical Leverage and Endogenous Risk." *J. Political Economy*, 122(1).

### Systemic Risk

- Adrian, T. & Brunnermeier, M. (2016). "CoVaR." *American Economic Review*, 106(7), 1705-1741.
- Brownlees, C. & Engle, R. (2017). "SRISK: A Conditional Capital Shortfall Measure of Systemic Risk." *Review of Financial Studies*, 30(1), 48-79.
- Acemoglu, D., Ozdaglar, A. & Tahbaz-Salehi, A. (2015). "Systemic Risk and Stability in Financial Networks." *American Economic Review*, 105(2), 564-608.
- Battiston, S. et al. (2012). "DebtRank: Too Central to Fail?" *Scientific Reports*, 2, 541.

### Financial Networks & Contagion

- Elliott, M., Golub, B. & Jackson, M.O. (2014). "Financial Networks and Contagion." *American Economic Review*, 104(10), 3115-3153.
- Forbes, K. & Rigobon, R. (2002). "No Contagion, Only Interdependence." *J. Finance*, 57(5), 2223-2261.
