# fatcrash

**71% of countries have exchange rate distributions with infinite variance.**

The median tail index across 138 countries is alpha = 1.57. Standard risk models — VaR under normality, Sharpe ratios, CAPM — assume finite variance (alpha > 2) and often finite kurtosis (alpha > 4). For the majority of the world's currencies, these assumptions are empirically false.

fatcrash detects crashes via fat-tail statistics. Python + Rust (PyO3). 13 methods, 180 tests, 500 years of data.

```python
from fatcrash.data.ingest import from_sample
from fatcrash.data.transforms import log_returns
from fatcrash._core import hill_estimator, taleb_kappa, lppls_fit

btc = from_sample("btc")
returns = log_returns(btc)

hill_estimator(returns)                        # 2.87 — infinite kurtosis
taleb_kappa(returns)                           # (0.34, 0.09) — CLT barely operates
```

```bash
fatcrash detect --asset BTC --source sample
fatcrash backtest --asset BTC --start 2017-01-01 --end 2018-06-01
```

## Results

### Crash detection accuracy (39 drawdowns across BTC, SPY, Gold)

| Method | Small (<15%) | Medium (15-30%) | Major (>30%) | Overall |
|--------|:---:|:---:|:---:|:---:|
| **LPPLS** | **100%** | **100%** | **100%** | **100%** |
| **DFA** | **86%** | **88%** | **62%** | **82%** |
| Hurst | 57% | 65% | 50% | 59% |
| Max-Stability Kappa | 57% | 47% | 38% | 49% |
| Pickands | 43% | 53% | 50% | 49% |
| DEH | 43% | 41% | 62% | 46% |
| GPD VaR | 40% | 55% | 0% | 42% |
| GSADF | 14% | 59% | 38% | 38% |
| QQ | 36% | 35% | 50% | 38% |
| Taleb Kappa | 21% | 35% | 50% | 33% |
| Max-to-Sum | 36% | 29% | 25% | 31% |
| Spectral | 21% | 29% | 38% | 28% |
| Hill | 29% | 29% | 25% | 28% |

LPPLS detects all 39 drawdowns. DFA is the best non-bubble method at 82% — detrended fluctuation analysis handles non-stationarity better than R/S Hurst. DEH is most useful on major crashes (62%). Hill alpha alone is unreliable (28%) but contributes to the aggregate.

### Combined detector

| | Small | Medium | Major | Overall |
|--------|:---:|:---:|:---:|:---:|
| **All methods + agreement bonus** | **64%** | **94%** | **75%** | **79%** |

When 3+ independent method categories (bubble, tail, regime) agree, the probability gets a +15% bonus. With 13 methods across 4 categories, the agreement signal is more robust.

### 6/6 known GBP/USD crises detected

1976 IMF Crisis, 1985 Plaza Accord, 1992 Black Wednesday, 2008 Financial Crisis, 2016 Brexit, 2022 Truss Mini-Budget.

## 500 Years of Forex Data

### FRED Daily (12 currency pairs, 1971-2025)

| Pair | Hill | Pickands | Hurst | DFA | DEH | QQ | MaxSum | Spectral | GSADF? |
|------|:----:|:-------:|:-----:|:---:|:---:|:--:|:------:|:--------:|:------:|
| AUD/USD | 2.58 | 1.02 | 0.56 | 0.56 | 0.44 | 2.30 | 0.003 | 0.001 | YES |
| GBP/USD | 4.13 | 0.06 | 0.58 | 0.55 | 0.19 | 4.11 | 0.001 | 0.009 | YES |
| JPY/USD | 3.94 | -0.23 | 0.58 | 0.58 | 0.18 | 4.02 | 0.002 | 0.028 | YES |
| CAD/USD | 3.84 | 0.30 | 0.57 | 0.53 | 0.27 | 3.58 | 0.001 | -0.022 | YES |
| CNY/USD | 2.79 | 0.55 | 0.59 | 0.71 | 0.69 | 1.70 | 0.038 | 0.027 | YES |
| MXN/USD | 2.04 | 0.60 | 0.56 | 0.57 | 0.44 | 1.98 | 0.005 | 0.052 | YES |
| BRL/USD | 2.80 | 0.44 | 0.56 | 0.58 | 0.15 | 3.12 | 0.002 | 0.081 | YES |
| KRW/USD | 1.90 | 1.01 | 0.67 | 0.60 | 0.44 | 1.93 | 0.006 | 0.094 | YES |
| INR/USD | 2.62 | 0.45 | 0.57 | 0.58 | 0.34 | 2.56 | 0.004 | 0.026 | YES |
| EUR/USD | 4.88 | 0.06 | 0.56 | 0.54 | 0.12 | 4.90 | 0.002 | 0.008 | no |
| CHF/USD | 3.81 | -0.32 | 0.57 | 0.54 | 0.28 | 3.59 | 0.002 | 0.023 | no |
| NZD/USD | 2.89 | 0.68 | 0.57 | 0.56 | 0.44 | 2.46 | 0.003 | 0.007 | no |

All 12 pairs: Hurst H > 0.55 and DFA > 0.5 (universal persistence). DEH > 0 for all 12 (heavy tails confirmed). 9/12 show GSADF bubbles. Mean QQ alpha = 3.02 — consistent with Hill mean of 3.19. Mean Spectral d = 0.03 confirms weak long memory from the frequency domain.

### Clio Infra Yearly (30 countries, 1500-2013)

| Country | Years | Hill alpha | Hurst H | Taleb kappa | Verdict |
|---------|:-----:|:---------:|:------:|:------:|---------|
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

19/30 countries have alpha < 2 (infinite variance). 25/30 have Hurst > 0.5, 28/30 have DFA > 0.5 (persistent). DEH > 0 for 20/30, QQ alpha < 4 for 28/30. Germany, Austria, Argentina, and Portugal saturate at Taleb kappa = 1.0 — Cauchy-like behavior where the CLT does not operate at any practical sample size.

Italy (H=0.80, DFA=1.44) and Portugal (H=0.85) show the strongest persistence over century-scale data.

## Why This Exists

### The ergodicity problem

Classical finance evaluates gambles by their expected value — the ensemble average over all possible outcomes at a single point in time. Peters (2019) showed that this is the wrong quantity for a single agent who must live through outcomes sequentially. In a multiplicative process like investing, the time-average growth rate and the ensemble-average growth rate are not equal. The process is non-ergodic.

A gamble that pays +50% or -40% with equal probability has a positive expected value (+5% per round) but a negative time-average growth rate: log(1.5 * 0.6) / 2 = -5.3% per round. Over enough rounds, a single participant goes bankrupt with certainty despite the "positive EV."

This divergence is amplified by fat tails. When alpha < 2 (infinite variance), the ensemble average is dominated by rare outcomes that no individual trajectory will realize. The sample mean does not converge at the rate prescribed by the CLT. Taleb (2020) quantifies this: for alpha near 1, the CLT does not operate at any practical sample size.

### Implications

Standard risk metrics presuppose:

1. Finite variance (requires alpha > 2)
2. Rapid convergence of the sample mean (requires alpha > 4)
3. Ergodicity (ensemble average = time average)

Our data shows 71% of countries have alpha < 2, and 97% have alpha < 4. All three presuppositions are violated for most financial instruments.

## The Methods

### Tail estimation

**Hill estimator** (Hill, 1975). Estimates the tail index alpha from the k largest order statistics: alpha_hat = [1/k * sum log(X_(i) / X_(k+1))]^(-1). The tail index governs tail decay: P(X > x) ~ x^(-alpha). Alpha < 2 means infinite variance; alpha < 4 means infinite kurtosis.

**Pickands estimator** (Pickands, 1975). Estimates the extreme value index gamma = 1/alpha using three order statistics: gamma_hat = log((X_(k) - X_(2k)) / (X_(2k) - X_(4k))) / log(2). Valid for all three domains of attraction (Frechet, Gumbel, Weibull), unlike Hill which assumes heavy tails. Less efficient but more robust.

**Taleb's kappa** (Taleb, 2019). Measures the rate of convergence of the Mean Absolute Deviation of partial sums: kappa = 2 - log(n/n0) / log(M(n)/M(n0)), where M(n) = E[|S_n - E[S_n]|]. Under the CLT, M(n) ~ sqrt(n), giving kappa = 0. For Cauchy, M(n) ~ n, giving kappa = 1. This answers a question asymptotic theory cannot: how many observations do you actually need? (BTC: 0.34, SPY: 0.03, Gold: 0.00.)

**Max-stability kappa.** Partitions data into k blocks, computes the mean of block maxima divided by the global maximum. For Gaussian data this ratio is near the Monte Carlo benchmark; for fat-tailed data, a single extreme observation dominates and the ratio drops.

**DEH moment estimator** (Dekkers, Einmahl & de Haan, 1989). Tail index estimator valid for all domains of attraction, not just heavy tails. Uses first and second moments of log-spacings: gamma = M1 + 1 - (1/2)(1 - M1^2/M2)^(-1). Complements Hill (heavy-tail only) and Pickands (less efficient).

**QQ estimator.** Tail index from the slope of a log-log QQ plot against exponential quantiles. For the k largest observations, regress log(X_(i)) vs -log(i/(k+1)). Slope = 1/alpha. Simple, visual, good for tracking tail index regime changes over time.

**Maximum-to-Sum ratio.** R_n = max(|X_i|) / sum(|X_i|) — a direct diagnostic for infinite variance. R_n -> 0 for thin tails (alpha > 2); R_n stays positive when alpha < 2. The simplest test of whether variance exists.

### Long-range dependence

**Hurst exponent** (Hurst, 1951). Quantifies persistence via rescaled range analysis. H = 0.5 is a random walk; H > 0.5 means trends persist; H < 0.5 means mean-reversion. All 12 FRED daily forex pairs show H > 0.55.

**DFA** (Detrended Fluctuation Analysis; Peng et al., 1994). Alternative to Hurst R/S that handles non-stationarity better. Divides into windows, fits linear trend per window, computes RMS of residuals, regresses log(RMS) vs log(window_size). Returns alpha exponent: 0.5 = white noise, >0.5 persistent. Best non-bubble crash detector at 82% accuracy.

**Spectral exponent** (Geweke & Porter-Hudak, 1983). Estimates long-memory parameter d from the periodogram near frequency zero: f(lambda) ~ |lambda|^(1-2d). Relationship to Hurst: d = H - 0.5. Complements Hurst (time domain) and DFA (detrended) from the frequency domain.

### Extreme value theory

**GPD** (Balkema & de Haan, 1974). Fits exceedances over a threshold to the Generalized Pareto Distribution: F_u(x) = 1 - (1 + xi*x/sigma)^(-1/xi). Yields VaR and Expected Shortfall at arbitrary confidence levels.

**GEV** (Fisher & Tippett, 1928). Fits block maxima to the Generalized Extreme Value distribution. Classifies into Frechet (xi > 0, heavy tail), Gumbel (xi = 0, exponential), or Weibull (xi < 0, bounded).

### Bubble detection

**LPPLS** (Sornette, 2003). Models bubble dynamics as a power law with log-periodic oscillations: log(p(t)) = A + B|tc-t|^m + C|tc-t|^m * cos(omega*log|tc-t| + phi). The critical time tc is the predicted crash date. The DS confidence indicator fits across many windows; the fraction of valid fits is the confidence measure. Nonlinear optimization via CMA-ES (Hansen, 2006) in Rust — each anchor date needs O(1000) evaluations.

**Deep LPPLS.** A PyTorch neural network that recognizes LPPLS-like patterns without explicit parametric fitting. Faster and more robust to deviations from the strict LPPLS form, at the cost of interpretability.

**GSADF** (Phillips, Shi & Yu, 2015). Detects explosive unit root behavior — the econometric signature of bubbles. The test statistic is the supremum of recursive ADF statistics over all feasible subsamples. Monte Carlo critical values under the null of a random walk. O(n^2) computation parallelized with rayon. Complements LPPLS: LPPLS detects bubble *shape*, GSADF detects *explosive growth*.

### Multiscale

Indicators computed at daily, 3-day, and weekly frequencies. A signal at one scale may be noise; a signal across all three is structural.

## Signal Aggregation

Methods grouped into 4 independent categories. When 3+ categories agree, probability gets a +15% bonus.

| Category | Methods | Signal |
|----------|---------|--------|
| **Bubble** | LPPLS, GSADF, Deep LPPLS | Super-exponential growth, explosive unit roots |
| **Tail** | Taleb Kappa, Max-Stability Kappa, Hill, Pickands, DEH, QQ, Max-to-Sum, GPD VaR | Tail thickening, distributional regime shifts |
| **Regime** | Hurst, DFA, Spectral | Transition from mean-reverting to persistent dynamics |
| **Structure** | Multiscale, LPPLS tc proximity | Cross-timeframe agreement, timing |

## Architecture

### Rust (via PyO3)

| Component | Why Rust |
|-----------|----------|
| LPPLS fitter (CMA-ES) | O(1000) nonlinear fits per anchor date |
| LPPLS confidence | Nested windows parallelized with rayon |
| GSADF test | O(n^2) BSADF + Monte Carlo, parallelized with rayon |
| GEV/GPD fitting | Rolling EVT needs speed |
| Hill, Pickands, DEH, QQ, Max-to-Sum, Taleb Kappa, Max-Stability Kappa, Hurst, DFA, Spectral | Called at every rolling step |

### Python

Data ingestion (Yahoo, CoinGecko, CCXT, FRED, CSV/Parquet), visualization (plotly, matplotlib), CLI (typer), service (FastAPI), Deep LPPLS (PyTorch), signal aggregation, 9 Jupyter notebooks.

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
make test                    # 35 Rust + 145 Python = 180 tests
make lint                    # cargo clippy + cargo fmt --check
python analysis/accuracy_report.py   # Full analysis
```

With [direnv](https://direnv.net/): `direnv allow` and the shell activates on `cd`.

## References

### Ergodicity

- Peters, O. (2019). "The Ergodicity Problem in Economics." *Nature Physics*, 15, 1216-1221. [DOI:10.1038/s41567-019-0732-0](https://doi.org/10.1038/s41567-019-0732-0)
- Peters, O. & Gell-Mann, M. (2016). "Evaluating Gambles Using Dynamics." *Chaos*, 26(2), 023103. [arXiv:1405.0585](https://arxiv.org/abs/1405.0585)
- Peters, O. (2011). "The Time Resolution of the St Petersburg Paradox." *Phil. Trans. R. Soc. A*, 369(1956), 4913-4931. [arXiv:1011.4404](https://arxiv.org/abs/1011.4404)

### Tail Estimation

- Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of a Distribution." *Ann. Statist.*, 3(5), 1163-1174.
- Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics." *Ann. Statist.*, 3(1), 119-131.
- Taleb, N.N. (2019). "How Much Data Do You Need? An Operational, Pre-Asymptotic Metric for Fat-tailedness." *Int. J. Forecasting*, 35(2), 677-686. [arXiv:1802.05495](https://arxiv.org/abs/1802.05495)
- Taleb, N.N. (2020). *Statistical Consequences of Fat Tails.* STEM Academic Press. [arXiv:2001.10488](https://arxiv.org/abs/2001.10488)

### Extreme Value Theory

- Embrechts, P., Kluppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance.* Springer.
- Balkema, A.A. & de Haan, L. (1974). "Residual Life Time at Great Age." *Ann. Probab.*, 2(5), 792-804.
- Fisher, R.A. & Tippett, L.H.C. (1928). "Limiting Forms of the Frequency Distribution of the Largest or Smallest Member of a Sample." *Proc. Cambridge Phil. Soc.*, 24(2), 180-190.

### Bubble Detection

- Sornette, D. (2003). *Why Stock Markets Crash.* Princeton University Press.
- Sornette, D. et al. (2015). "Real-Time Prediction and Post-Mortem Analysis of the Shanghai 2015 Stock Market Bubble and Crash." *J. Investment Strategies*, 4(4), 77-95.
- Phillips, P.C.B., Shi, S. & Yu, J. (2015). "Testing for Multiple Bubbles." *Int. Econ. Rev.*, 56(4), 1043-1078.
- Phillips, P.C.B., Wu, Y. & Yu, J. (2011). "Explosive Behavior in the 1990s NASDAQ." *Int. Econ. Rev.*, 52(1), 201-226.
- Hansen, N. (2006). "The CMA Evolution Strategy: A Comparing Review." In *Towards a New Evolutionary Computation*, Springer, 75-102.

### Tail Estimation (additional)

- Dekkers, A.L.M., Einmahl, J.H.J. & de Haan, L. (1989). "A Moment Estimator for the Index of an Extreme-Value Distribution." *Ann. Statist.*, 17(4), 1833-1855.

### Long-Range Dependence

- Hurst, H.E. (1951). "Long-Term Storage Capacity of Reservoirs." *Trans. ASCE*, 116, 770-799.
- Lo, A.W. (1991). "Long-Term Memory in Stock Market Prices." *Econometrica*, 59(5), 1279-1313.
- Peng, C.-K. et al. (1994). "Mosaic Organization of DNA Nucleotides." *Physical Review E*, 49(2), 1685-1689.
- Geweke, J. & Porter-Hudak, S. (1983). "The Estimation and Application of Long Memory Time Series Models." *J. Time Series Analysis*, 4(4), 221-238.

### Fat Tails in Finance

- Mandelbrot, B.B. (1963). "The Variation of Certain Speculative Prices." *J. Business*, 36(4), 394-419.
- Gabaix, X. (2009). "Power Laws in Economics and Finance." *Ann. Rev. Econ.*, 1, 255-294.
- Mandelbrot, B.B. & Taleb, N.N. (2010). "Random Jump, Not Random Walk." In *The Known, the Unknown, and the Unknowable in Financial Risk Management.* Princeton University Press.
