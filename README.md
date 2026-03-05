# fatcrash

**71% of countries have exchange rate distributions with infinite variance.**

The median tail index across 138 countries is alpha = 1.57. Standard risk models assume finite variance (alpha > 2) and often finite kurtosis (alpha > 4). For the majority of the world's currencies, these assumptions are empirically false. fatcrash detects crashes by measuring what actually matters: the tail.

Python + Rust (PyO3). 18 crash detection methods across 5 families + 7 market regime signal families. 500 years of data.

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

## Methods

18 methods organized into 5 families. Each family detects a different type of signal — the aggregator combines them and rewards cross-family agreement.

### Family 1: Bubble Detection

Detects super-exponential growth and explosive dynamics. The strongest individual predictors.

| Method | What it measures | Key output | F1 |
|--------|-----------------|------------|:--:|
| LPPLS | Log-periodic power law bubble shape | Critical time tc, confidence | 61% |
| LPPLS confidence | Multi-window LPPLS robustness | Confidence [0,1] | 47% |
| GSADF | Explosive unit root (Phillips-Shi-Yu) | Test statistic vs CV | 38% |
| M-LNN | Neural network LPPLS fitting (per-series) | tc, m, omega, confidence | 41% |
| P-LNN | Pre-trained neural network (~700x faster) | tc, m, omega, confidence | 23% |

**LPPLS** (Sornette, 2003). Models bubble dynamics as a power law with log-periodic oscillations: log(p(t)) = A + B|tc-t|^m + C|tc-t|^m * cos(omega*log|tc-t| + phi). The critical time tc is the predicted crash date. Confidence measured by fitting across many windows. Nonlinear optimization via CMA-ES in Rust.

**GSADF** (Phillips, Shi & Yu, 2015). Detects explosive unit root behavior — the econometric signature of bubbles. The supremum of recursive ADF statistics over all feasible subsamples, with Monte Carlo critical values. Complements LPPLS: LPPLS detects bubble *shape*, GSADF detects *explosive growth*.

**M-LNN / P-LNN** (Nielsen, Sornette & Raissi, 2024). Neural network approaches to LPPLS fitting. Both predict nonlinear parameters (tc, m, omega), then solve linear parameters via OLS. M-LNN trains one small network per series (F1=41%). P-LNN is pre-trained on 100K synthetic series, ~700x faster (F1=23%). Requires `pip install fatcrash[deep]`.

### Family 2: Tail Estimation

Measures how fat the tails are — the shape of the distribution, not specific crash timing. Fat tails mean extreme events are far more likely than Gaussian models predict.

| Method | What it measures | Key output | F1 |
|--------|-----------------|------------|:--:|
| Hill | Tail index from order statistics | alpha (< 2 = infinite variance) | 16% |
| Pickands | Extreme value index, all domains | gamma (> 0 = heavy tail) | 27% |
| DEH | Tail index via moment estimator | gamma (> 0 = heavy tail) | 26% |
| QQ | Tail index from QQ-plot slope | alpha (< 4 = fat tail) | 23% |
| Taleb kappa | CLT convergence rate | kappa (0 = Gaussian, 1 = Cauchy) | 25% |
| Max-stability kappa | Block-max concentration | kappa vs benchmark | 27% |
| Max-to-Sum | Infinite variance diagnostic | ratio (> 0 if alpha < 2) | 18% |

**Hill estimator** (Hill, 1975). alpha = [1/k * sum log(X_(i) / X_(k+1))]^(-1). The tail index governs tail decay: P(X > x) ~ x^(-alpha). Alpha < 2 means infinite variance; alpha < 4 means infinite kurtosis. Includes Huisman et al. (2001) small-sample bias correction.

**Pickands estimator** (Pickands, 1975). gamma = log((X_(k) - X_(2k)) / (X_(2k) - X_(4k))) / log(2). Valid for all three domains of attraction (Frechet, Gumbel, Weibull), unlike Hill which assumes heavy tails.

**DEH moment estimator** (Dekkers, Einmahl & de Haan, 1989). gamma = M1 + 1 - (1/2)(1 - M1^2/M2)^(-1). Valid for all domains of attraction. Complements Hill (heavy-tail only) and Pickands (higher variance).

**QQ estimator.** Regresses log(X_(i)) vs -log(i/(k+1)) for the k largest observations. Slope = 1/alpha. Good for regime change detection in rolling windows.

**Taleb's kappa** (Taleb, 2019). kappa = 2 - log(n/n0) / log(M(n)/M(n0)). Under the CLT, M(n) ~ sqrt(n), giving kappa = 0. For Cauchy, M(n) ~ n, giving kappa = 1. Answers: *how many observations do you actually need for the sample mean to converge?*

**Max-stability kappa.** Partitions data into blocks, computes mean-of-block-maxima / global-maximum. For Gaussian data this ratio is near a Monte Carlo benchmark; for fat-tailed data, a single extreme dominates and the ratio drops.

**Maximum-to-Sum ratio.** R_n = max(|X_i|) / sum(|X_i|). Converges to zero for thin tails (alpha > 2); stays positive when alpha < 2. The simplest diagnostic for whether variance exists.

### Family 3: Extreme Value Theory

Directly models the distribution of extreme losses. Produces actionable risk numbers (VaR, ES).

| Method | What it measures | Key output | F1 |
|--------|-----------------|------------|:--:|
| GPD | Tail risk from exceedances | VaR, Expected Shortfall | 19% |
| GEV | Block maxima classification | Frechet / Gumbel / Weibull | — |

**GPD** (Balkema & de Haan, 1974). Fits exceedances over a threshold to the Generalized Pareto Distribution. Yields VaR and Expected Shortfall at arbitrary confidence levels.

**GEV** (Fisher & Tippett, 1928). Fits block maxima to the Generalized Extreme Value distribution. Classifies into Frechet (xi > 0, heavy tail), Gumbel (xi = 0, exponential), or Weibull (xi < 0, bounded).

### Family 4: Regime & Persistence

Detects transitions in market dynamics — from calm to turbulent, mean-reverting to trending, stable to critical. These methods answer "is the regime shifting?" rather than "how fat are the tails?"

| Method | What it measures | Key output | F1 |
|--------|-----------------|------------|:--:|
| DFA | Persistence, non-stationary-robust | alpha (> 0.5 = trending) | 34% |
| CSD on volatility | Tipping point detection | AR(1) + variance rising | 34% |
| RV spike | Short-term vol vs baseline | ratio (> 2x = regime change) | 32% |
| Hamilton filter | 2-state HMM regime classification | P(stressed) | 29% |
| Hurst | Persistence via R/S analysis | H (> 0.5 = trending) | 28% |
| Spectral | Long memory from frequency domain | d (> 0 = long memory) | 25% |
| Momentum reversal | Trend breakdown detection | reversal signal | — |
| Price velocity | Volatility acceleration | velocity (cascade detection) | — |

**DFA** (Peng et al., 1994). Detrended fluctuation analysis: divides into windows, removes linear trend, regresses log(RMS of residuals) vs log(window size). Handles non-stationarity better than R/S — best non-bubble crash detector (82% recall, 34% F1).

**Critical Slowing Down** (Scheffer et al., 2009). Near a tipping point, systems recover more slowly. Observable: rising AR(1) coefficient AND rising variance simultaneously. Applied to the *volatility* series (rolling RV), not raw returns — raw returns have near-zero autocorrelation. Volatility has strong clustering, so CSD on the vol series detects genuine regime transitions. F1=34%, 51% recall.

**RV Spike.** Compares short-term realized variance (21-day) to a long-term baseline (126-day): `ratio = RV_short / RV_long`. When ratio > 2x, volatility is spiking. More robust than BNS JV/RV fraction at daily frequency. F1=32%.

**Hamilton Filter** (Hamilton, 1989). 2-state HMM with EM estimation and Kim smoother. Overall F1=29%, but **88% recall on major (>30%) crashes**. Backward-looking: it confirms you're in a crisis, it doesn't predict one.

**Hurst exponent** (Hurst, 1951). Persistence via rescaled range (R/S) analysis. H = 0.5 is a random walk; H > 0.5 means trends persist; H < 0.5 means mean-reversion.

**Spectral exponent** (Geweke & Porter-Hudak, 1983). Estimates long-memory parameter d from the periodogram near frequency zero. Relation to Hurst: d = H - 0.5. Confirms persistence from the frequency domain.

**Momentum** (Jegadeesh & Titman, 1993). Trailing log return over 3, 6, and 12-month windows. Momentum *reversal* — when long-term momentum is positive but short-term turns negative — is a crash precursor.

**Price velocity** (cascade detector). Rate of change of realized volatility: velocity = (vol[t] - vol[t-lag]) / vol[t-lag]. Detects forced-liquidation cascades where volatility itself accelerates — the signature of Volmageddon (Feb 5, 2018: XIV lost 97%, VIX spiked 116%) or the Sep 2019 repo blowup.

**Building blocks** (not standalone signals): Realized variance (simple, Parkinson, Garman-Klass estimators), bipower variation and jump variance decomposition (BNS). Foundation for RV spike and CSD.

### Family 5: Liquidity

Volume-based signals that detect liquidity dry-ups. Brings a dimension no price-only method can capture.

| Method | What it measures | Key output | F1 |
|--------|-----------------|------------|:--:|
| Amihud illiquidity | Price impact per unit volume | |return|/volume ratio | 32% |

**Amihud illiquidity** (Amihud, 2002). mean(|return| / volume) over a rolling window. Higher values = less liquid = harder to trade without moving price. Illiquidity spikes precede and accompany crashes because market makers widen spreads and pull quotes. The only method in fatcrash that uses volume data. F1=32%, with 71% recall on small (<15%) crashes — catches liquidity dry-ups that price-only methods miss.

### Additional Rust primitives

The Rust layer also exposes **realized skewness** (rolling third moment / std^3, captures distributional asymmetry) and **absorption ratio** (fraction of variance in top eigenvector of cross-asset correlation matrix, measures systemic coupling). Both are available as `_core` functions but not included in the aggregator — realized skewness overlaps with tail estimators (F1=30% core, drops to 19% extended), and absorption ratio requires multi-asset alignment at test time (F1=29%).

### Constant volatility strategy

Position sizing via inverse volatility targeting: weight = target_vol / realized_vol. Dalgaard (2016, CBS thesis) tested tail-hedging strategies on S&P 500 data and found that simple put monetization strategies do NOT reduce drawdowns and have LOWER returns/Sharpe than an unhedged index. The constant volatility strategy, however, DOES reduce drawdowns while earning higher returns — mechanically cutting exposure when vol spikes and increasing exposure during calm periods.

**Rebalance risk signal** (Rattray, Harvey & Van Hemert, 2018). Mechanical rebalancing is negative convexity — it buys into drawdowns that continue. When DFA detects trending behavior (alpha > 0.5) and momentum is negative, rebalancing into the position is dangerous. A 10% trend-following allocation reduces drawdowns by ~5 percentage points.

## Signal Aggregation

Methods grouped into 5 independent families (matching the method categories above). When 3+ families have elevated signals, probability gets a +15% bonus.

| Family | Methods | What it detects |
|--------|---------|-----------------|
| **Bubble** | LPPLS, GSADF, M-LNN, P-LNN | Super-exponential growth, explosive unit roots |
| **Tail** | Hill, Pickands, DEH, QQ, Taleb Kappa, Max-Stability Kappa, Max-to-Sum, GPD | Tail thickening, distributional regime shifts |
| **Regime** | Hurst, DFA, Spectral, Momentum reversal, CSD (on vol), Hamilton | Persistence shifts, tipping points, regime transitions |
| **Structure** | Multiscale, LPPLS tc proximity, Price velocity, RV spike | Cross-timeframe agreement, timing, cascade detection |
| **Liquidity** | Amihud illiquidity | Volume-based liquidity dry-ups |

Indicators are also computed at daily, 3-day, and weekly frequencies. A signal at one scale may be noise; a signal across all three is structural.

## Results

### Crash detection: precision, recall, and F1

All accuracy numbers are in-sample on historical data. Methods are tested on both crash windows (120 days before a drawdown peak) and non-crash windows (random 120-day stretches at least 180 days from any crash).

#### Core dataset (39 drawdowns across BTC, SPY, Gold)

39 crash windows, 150 non-crash windows.

| Method | TP | FP | FN | TN | Precision | Recall | F1 |
|--------|:--:|:--:|:--:|:--:|:---------:|:------:|:--:|
| **LPPLS** | **35** | **40** | **4** | **110** | **47%** | **90%** | **61%** |
| **LPPLS confidence** | **17** | **16** | **22** | **134** | **52%** | **44%** | **47%** |
| M-LNN | 22 | 47 | 17 | 103 | 32% | 56% | 41% |
| GSADF | 15 | 24 | 24 | 126 | 38% | 38% | 38% |
| **DFA** | **32** | **115** | **7** | **35** | **22%** | **82%** | **34%** |
| CSD (on vol) | 20 | 59 | 19 | 91 | 25% | 51% | 34% |
| Amihud | 19 | 60 | 20 | 90 | 24% | 49% | 32% |
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

#### Extended dataset (FRED forex + options backtester)

23 FRED forex pairs (1971-2025) and 6 options backtester series. 57 crash windows, 481 non-crash windows.

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

| Method | Precision | Recall | F1 |
|--------|:---------:|:------:|:--:|
| **LPPLS** | **36%** | **90%** | **51%** |
| LPPLS confidence | 44% | 50% | 47% |
| GSADF | 32% | 57% | 41% |
| M-LNN | 24% | 59% | 34% |
| Amihud | 21% | 49% | 30% |
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

LPPLS recall holds at 90% across 96 crash windows spanning crypto, equities, commodities, and 23 forex pairs.

### Recall by crash size

| Method | Small (<15%) | Medium (15-30%) | Major (>30%) |
|--------|:---:|:---:|:---:|
| LPPLS | 86% | 100% | 75% |
| DFA | 86% | 88% | 62% |
| Hurst | 57% | 65% | 50% |
| M-LNN | 57% | 65% | 38% |
| Amihud | **71%** | 47% | 12% |
| RV spike | 50% | 53% | 62% |
| CSD (on vol) | 50% | 47% | 62% |
| LPPLS confidence | 43% | 59% | 12% |
| Hamilton | 29% | 29% | **88%** |
| DEH | 43% | 41% | 62% |
| Taleb Kappa | 21% | 35% | 50% |
| GSADF | 14% | 59% | 38% |

LPPLS catches 100% of medium crashes. **Hamilton has 88% recall on major crashes** — the best single method for detecting large (>30%) drawdowns, though it rarely fires on smaller crashes.

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

### Why precision is low for tail/regime methods

Tail and regime methods detect distributional shifts (tail thickening, persistent dynamics), not crash-specific patterns. They fire in many non-crash periods because fat tails and persistence are pervasive in financial data. This is by design — they measure the *distributional regime*, not a specific crash. LPPLS and GSADF have higher precision because they detect bubble-specific structure.

**The Sornette-Bouchaud debate:** Sornette (the LPPLS inventor) argues for high recall at the cost of precision because missing a crash is far more expensive than a false alarm. Bouchaud emphasizes that fat-tail estimators measure unconditional properties and are poor at conditional crash prediction. Both perspectives are reflected in fatcrash: LPPLS targets the mechanism (Sornette), tail estimators measure the regime (Bouchaud), and the aggregator combines both.

## 500 Years of Forex Data

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

19/30 have alpha < 2 (infinite variance). Germany, Austria, Argentina, and Portugal saturate at Taleb kappa = 1.0 — Cauchy-like behavior where the CLT does not operate at any practical sample size.

### Cross-method agreement

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

Multiple independent methods — order statistics, moments, QQ-slope, R/S analysis, detrended fluctuation, frequency domain — all converge on the same conclusions: fat tails and persistence are universal across timescales.

## Market Regime Signals

Beyond crash detection from price data, fatcrash includes a framework for macro/microstructure regime classification. These signals answer a different question: not "are tails thickening?" but "is the macro-financial environment shifting toward stress?"

Each signal is normalized (z-score or percentile rank), aggregated into thematic buckets, and combined into a single regime score. The goal is transparent, inspectable signals grounded in academic literature.

### Signal families

| # | Family | Signals | What it detects |
|---|--------|---------|-----------------|
| R1 | Risk Premium | VRP, RV spike | Variance and tail-risk compensation |
| R2 | Liquidity | SOFR-OIS, TED, **Amihud** (implemented), xccy basis | Funding stress, market illiquidity |
| R3 | Volatility Regime | VIX slope, SKEW, MOVE, VVIX | Vol term structure inversion, vol-of-vol |
| R4 | Credit & Macro | OFR FSI, credit spreads, EBP, yield curve | Credit stress, macro deterioration |
| R5 | Structure & Flows | Cross-asset eigenvalue, COT, ETF flows | Diversification breakdown, positioning extremes |
| R6 | Contagion | CoVaR, MES, SRISK | Systemic risk, institution-level stress |
| R7 | Sentiment | FOMC tone, news uncertainty | Policy uncertainty, narrative shifts |

**Risk Premium** — VRP = implied variance minus realized variance. VRP > 0 is normal (paying for protection). VRP < 0 means realized vol exceeds implied — acute stress. References: Bollerslev, Tauchen & Zhou (2009); Bekaert & Hoerova (2014).

**Liquidity & Funding Stress** — SOFR-OIS spread (pure funding stress), Amihud illiquidity (|return|/volume), cross-currency basis (negative USD basis = global dollar shortage). References: Brunnermeier & Pedersen (2009); Pastor & Stambaugh (2003).

**Volatility Regime** — VIX term structure slope (backwardation = near-term fear), SKEW (tail risk pricing), MOVE (bond vol), VVIX (vol-of-vol). Negative VIX slope is the single fastest stress signal.

**Credit & Macro** — OFR Financial Stress Index, credit spreads (HY OAS, Baa-Aaa), excess bond premium (EBP — the component driven by risk appetite, not default risk; Gilchrist & Zakrajsek, 2012), yield curve (10Y-2Y, 10Y-3M). The 10Y-3M spread has inverted before every US recession since 1960.

**Structure & Flows** — Cross-asset correlation eigenvalue (when it approaches 1.0, all assets move together and diversification breaks down), CFTC COT positioning, ETF flows.

**Contagion** — CoVaR (Adrian & Brunnermeier, 2016), MES, SRISK (Brownlees & Engle, 2017). Pre-computed data from NYU Stern V-Lab.

**Sentiment** — FOMC communication tone (Loughran-McDonald dictionary), Baker-Bloom-Davis Economic Policy Uncertainty index.

### Regime scoring

```
regime_score = -0.30 * risk_premium
             - 0.25 * liquidity
             - 0.20 * volatility
             - 0.15 * credit_macro
             - 0.10 * structure_flows
```

Negative weights: higher stress → more risk-off. Smoothed with EMA (alpha=0.2).

**Labels:** risk_on (score >= +0.50), neutral, risk_off (score <= -0.50).

### Historical calibration

| Signal | Sept 2008 (Lehman) | Mar 2020 (COVID) | Feb 2018 (Volmageddon) | Sept 2019 (Repo) |
|--------|:---:|:---:|:---:|:---:|
| VIX | ~80 | ~85 | 14→37 | ~18 |
| HY OAS | >1000 bps | ~1000 bps | ~330 bps | ~400 bps |
| SOFR/TED | ~450 bps | ~50 bps | unchanged | repo 600 bps |
| VRP | negative | negative | brief inversion | normal |
| Cross-asset λ | ~1.0 | ~1.0 | brief spike | low |
| **Expected** | **risk-off** | **risk-off** | **brief risk-off** | **brief risk-off** |

2008 and 2020 are unambiguous (all buckets fire). Volmageddon is the purest vol event — only the volatility bucket fires. The repo crisis is the mirror: only liquidity fires.

### Data sources

| Source | Signals | Access |
|--------|---------|--------|
| FRED API | VIX, spreads, yield curve, STLFSI, EPU | Free |
| OFR | Financial Stress Index | Free |
| Cboe | VIX futures, SKEW, VVIX | Free (research) |
| CFTC | COT positioning | Free |
| NYU V-Lab | CoVaR, MES, SRISK | Free |

### Theoretical background

**Hamilton's regime-switching** (1989): regimes as latent discrete states via HMM. Rigorous but backward-looking.

**Sornette's LPPL framework**: regime endings as deterministic critical points driven by endogenous positive feedback. Forward-looking but sensitive to fitting.

**Critical slowing down** (Scheffer et al., 2009): near tipping points, rising variance + rising autocorrelation. The empirical complement to Sornette's theoretical prediction.

**Hawkes processes** (Bacry, Muzy): extreme events cluster via self-excitation. Branching ratio approaching 1 signals criticality. Not yet implemented — signal largely captured by CSD + velocity.

**Rough volatility** (Gatheral & Rosenbaum, 2018): realized vol has Hurst exponent H ≈ 0.1, rougher than a random walk.

**Minsky's Financial Instability Hypothesis**: stability is destabilizing — calm encourages leverage migration from hedge to speculative to Ponzi finance.

## Beyond Market Prices

These methods were built for prices, but most transfer to fundamental data like revenue or profit growth.

| Method | Works on revenue? | Why |
|--------|:-:|-----|
| Hill, DEH, QQ, Pickands | Yes | Tail thickness is a property of any distribution |
| Kappa, Taleb kappa | Yes | Measures departure from Gaussian max-stability |
| Max-to-Sum ratio | Yes | A single quarter where revenue drops 50% = same math |
| GPD / GEV | Yes | EVT is distribution-agnostic |
| Hurst, DFA, Spectral | Yes | Revenue persistence (H > 0.5) from contracts/stickiness |
| GSADF | Partially | Could flag unsustainable exponential growth |
| LPPLS | No | Models speculative dynamics, not real economic activity |

```python
import numpy as np
from fatcrash._core import hill_estimator, dfa_exponent, gsadf_test

growth = np.diff(np.log(quarterly_revenue))
hill_estimator(growth)            # Are revenue shocks fat-tailed?
dfa_exponent(growth)              # Is growth persistent or mean-reverting?
gsadf_test(quarterly_revenue)     # Is revenue growth explosive?
```

## Architecture

```
Rust (PyO3, _core.so)                Python
┌────────────────────────────┐       ┌──────────────────────────────────┐
│ Tail: Hill, Pickands, DEH, │       │ indicators/                      │
│       QQ, Kappa, Taleb,    │──────▶│   tail_indicator.py              │
│       MaxSum, Hurst, DFA,  │       │   vol_indicator.py               │
│       Spectral, Momentum,  │       │   lppls_indicator.py             │
│       Velocity, Skewness,  │       │   bubble_indicator.py            │
│       Amihud, Absorption   │       │                                  │
│                            │       │   evt_indicator.py               │
│ EVT:  GPD, GEV             │       │   regime_indicator.py            │
│                            │       │                                  │
│ LPPLS: fit, confidence,    │──────▶│ nn/                              │
│        solve_linear        │       │   mlnn.py      (M-LNN)          │
│                            │       │   plnn.py      (P-LNN)          │
│ Bubble: GSADF              │       │   lppls_torch.py (shared)       │
│                            │       │   synthetic.py  (data gen)      │
│ Multiscale                 │       │                                  │
│                            │       │ aggregator/signals.py            │
│ Regime:                    │       │                                  │
│   RealizedVar, Jump, CSD,  │       │ data/ingest.py                   │
│   Hamilton                 │       │ cli/ viz/ service/               │
│                            │       │                                  │
│ rayon: parallel CMA-ES,    │       │                                  │
│        GSADF, confidence   │       │                                  │
└────────────────────────────┘       └──────────────────────────────────┘
```

All estimators and regime algorithms are in Rust, exposed via PyO3. Computationally intensive methods (LPPLS CMA-ES, GSADF, confidence, Hamilton EM) use rayon for parallelization. NN methods are in Python (PyTorch).

| Component | Language | Why |
|-----------|----------|-----|
| LPPLS fitter (CMA-ES) | Rust | O(1000) nonlinear fits per anchor date |
| LPPLS confidence | Rust | Nested windows parallelized with rayon |
| GSADF test | Rust | O(n^2) BSADF + Monte Carlo, parallelized |
| GEV/GPD fitting | Rust | Rolling EVT needs speed |
| All tail & regime estimators | Rust | Called at every rolling window step |
| M-LNN, P-LNN | Python (PyTorch) | GPU support, autograd for training |
| Data ingestion, viz, CLI | Python | Ecosystem (pandas, plotly, typer, FastAPI) |

## Data Sources

### Bundled sample data (no internet required)

| Asset | Period | Days | Source |
|-------|--------|------|--------|
| BTC | 2014-2025 | 4,124 | Yahoo Finance |
| SPY | 1999-2025 | 6,570 | options_backtester |
| Gold | 2000-2025 | 6,441 | Yahoo Finance |

### Network sources

```bash
fatcrash detect --asset BTC --source yahoo --days 365
fatcrash detect --asset BTC --source yahoo --start 2020-01-01 --end 2021-01-01
fatcrash detect --asset BTC --source coingecko --no-use-cache
fatcrash cache-clear
```

Network sources are cached to `~/.cache/fatcrash` by default.

### FRED forex (23 pairs, 1971-2025)

```python
from fatcrash.data.ingest import load_fred_forex

pairs = load_fred_forex()           # dict of 23 DataFrames
aud = load_fred_forex("AUD_USD")    # single pair
```

Requires `git clone https://github.com/unbalancedparentheses/forex-centuries ~/projects/forex-centuries` or set `FOREX_CENTURIES_DIR`.
