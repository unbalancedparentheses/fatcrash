# fatcrash

Crash detection via fat-tail statistics. Works with any asset: Bitcoin, gold, S&P 500, forex, etc.

Python + Rust (PyO3) — performance-critical numerical work in Rust, everything else in Python. 8 methods, 99 tests, tested on 500 years of data across 138 countries.

## Motivation

### The ergodicity problem

Classical financial theory evaluates gambles by their expected value — the ensemble average over all possible outcomes at a single point in time. Peters (2019) demonstrated that this is the wrong quantity for a single agent who must live through outcomes sequentially. In a multiplicative process such as investing, the time-average growth rate and the ensemble-average growth rate are not equal. The process is non-ergodic.

Consider a gamble that pays +50% or -40% with equal probability. The ensemble average yields +5% per round. The time-average growth rate is log(1.5 × 0.6)/2 = -5.3% per round. Over a sufficient number of rounds, a single participant goes bankrupt with probability 1, despite the positive expected value. This is not a paradox — it is a consequence of the multiplicative dynamics that govern wealth.

The divergence between ensemble and time averages is amplified by fat tails. When the tail index alpha < 2 (infinite variance), the ensemble average is dominated by rare, extreme outcomes that no individual trajectory is likely to realize. Under such distributions, the sample mean does not converge to the population mean at the rate prescribed by the Central Limit Theorem. Taleb (2020) quantifies this through the kappa metric: for alpha near 1 (Cauchy-like), the CLT effectively does not operate at any practical sample size.

### Implications for risk measurement

Standard risk metrics — Value at Risk under normality, Sharpe ratios, CAPM betas — are computed from ensemble averages of thin-tailed models. They presuppose that:

1. Variance is finite (requires alpha > 2)
2. The sample mean converges rapidly (requires alpha > 4 for finite kurtosis)
3. Returns are ergodic (the ensemble average equals the time average)

Our empirical results show that 71% of countries have exchange rate distributions with alpha < 2 (infinite variance), and 97% have alpha < 4 (infinite kurtosis). For the majority of financial instruments, all three presuppositions are violated.

### What fatcrash measures

The methods implemented here target quantities relevant to the time-average investor:

1. **Tail index estimation** (Hill, Pickands, Taleb kappa, max-stability kappa) — characterizes the severity of the ensemble/time-average divergence
2. **Bubble detection** (LPPLS, GSADF, Deep LPPLS) — identifies super-exponential growth regimes that precede multiplicative ruin events
3. **Long-range dependence** (Hurst exponent) — detects persistent dynamics that amplify tail risk
4. **Extreme value quantification** (GPD, GEV) — estimates tail risk directly from the empirical distribution of extremes
5. **Multiscale analysis** — filters false positives by requiring signal agreement across multiple observation frequencies

## The Methods

### Hill Estimator

The Hill estimator (Hill, 1975) provides a consistent estimate of the tail index alpha for distributions with regularly varying tails. Given order statistics X_(1) >= X_(2) >= ... >= X_(n), the estimator is:

```
alpha_hat = [1/k * sum_{i=1}^{k} log(X_(i) / X_(k+1))]^(-1)
```

where k is the number of upper order statistics used. The tail index governs the rate of tail decay: P(X > x) ~ x^(-alpha) as x -> infinity.

- **alpha < 2**: Infinite variance. The sample variance does not converge. Ensemble-based risk metrics are undefined.
- **alpha in [2, 4)**: Finite variance, infinite kurtosis. The CLT operates but convergence is slow.
- **alpha >= 4**: Finite fourth moment. Standard statistical methods are approximately valid.

### Pickands Estimator

The Pickands estimator (Pickands, 1975) estimates the extreme value index gamma = 1/alpha using three order statistics:

```
gamma_hat = log((X_(k) - X_(2k)) / (X_(2k) - X_(4k))) / log(2)
```

Unlike the Hill estimator, which assumes heavy tails (gamma > 0), the Pickands estimator is valid for all three domains of attraction: Frechet (gamma > 0, heavy tails), Gumbel (gamma = 0, exponential tails), and Weibull (gamma < 0, bounded tails). It is less efficient than Hill when the heavy-tail assumption holds, but more robust when it does not.

### Taleb's Kappa

Taleb (2019) introduced an operational metric for fat-tailedness based on the rate of convergence of the Mean Absolute Deviation of partial sums. For i.i.d. random variables X_i with finite mean:

```
M(n) = E[|S_n - E[S_n]|],  where S_n = X_1 + ... + X_n

kappa(n_0, n) = 2 - log(n / n_0) / log(M(n) / M(n_0))
```

Under the CLT, M(n) ~ sqrt(n), giving kappa = 0. For a Cauchy distribution, M(n) ~ n, giving kappa = 1. The kappa metric provides a pre-asymptotic, operational measure of how many observations are required for statistical stability — a question that classical asymptotic theory cannot answer for fat-tailed distributions.

Empirical values: BTC = 0.34, SPY = 0.03, Gold = 0.00. Bitcoin requires substantially more data for reliable statistical inference than equities or commodities.

### Max-Stability Kappa

A non-parametric tail detector based on the ratio of subsample maxima to the global maximum. The data is partitioned into k non-overlapping blocks; the max-stability kappa is:

```
kappa_ms = (1/k * sum_{i=1}^{k} max(|X_j| : j in block_i)) / max(|X_j| : j = 1..n)
```

For Gaussian data, subsample maxima are close to the global maximum (kappa_ms near the Monte Carlo benchmark). For fat-tailed data, a single extreme observation dominates, pulling kappa_ms well below benchmark. The Gaussian benchmark is computed via Monte Carlo simulation with matched sample size and block count.

### Hurst Exponent

The Hurst exponent (Hurst, 1951) quantifies long-range dependence via rescaled range (R/S) analysis. For a time series of length n, the R/S statistic scales as n^H:

- **H = 0.5**: No long-range dependence (random walk, independent increments)
- **H > 0.5**: Persistent (positive autocorrelation at long lags; trends continue)
- **H < 0.5**: Anti-persistent (mean-reverting)

Lo (1991) applied R/S analysis to stock returns and found evidence of long memory. In our forex data, all 12 FRED daily pairs exhibit H > 0.55, consistent with momentum and carry trade dynamics at the daily frequency.

### Extreme Value Theory (EVT)

EVT provides a mathematically rigorous framework for modeling the distribution of extremes, independent of assumptions about the bulk distribution.

**Generalized Pareto Distribution (GPD):** For exceedances over a high threshold u, the Balkema-de Haan-Pickands theorem (Balkema & de Haan, 1974) establishes that the excess distribution converges to a GPD:

```
F_u(x) = 1 - (1 + xi * x / sigma)^(-1/xi)
```

where xi is the shape parameter and sigma the scale. This yields direct estimates of Value at Risk and Expected Shortfall at arbitrary confidence levels.

**Generalized Extreme Value (GEV):** The Fisher-Tippett-Gnedenko theorem (Fisher & Tippett, 1928) establishes that the distribution of block maxima converges to a GEV distribution, classified into three types: Frechet (xi > 0, heavy tail), Gumbel (xi = 0, exponential tail), and Weibull (xi < 0, bounded tail).

### LPPLS (Log-Periodic Power Law Singularity)

The LPPLS model (Sornette, 2003) describes the price dynamics during a bubble as a power law decorated with log-periodic oscillations:

```
log(p(t)) = A + B|t_c - t|^m + C|t_c - t|^m * cos(omega * log|t_c - t| + phi)
```

where t_c is the critical time (predicted crash date), m is the power law exponent, and omega is the log-periodic angular frequency. Valid fits satisfy 0.1 <= m <= 0.9 and 2 <= omega <= 25 (Sornette et al., 2015).

The DS LPPLS confidence indicator fits the model across many overlapping time windows with varying start dates and lengths. The fraction of windows producing valid fits provides a probability-like confidence measure.

The nonlinear optimization of (t_c, m, omega) is performed via **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy, Hansen 2006), a derivative-free evolutionary strategy suited to the multimodal, noisy LPPLS objective landscape. Each anchor date requires O(1000) candidate evaluations — this computational load motivates the Rust implementation.

### Deep LPPLS

A PyTorch neural network trained to recognize LPPLS-like patterns in price windows without explicit parametric fitting. The classical LPPLS approach is interpretable (it returns t_c, m, omega), but sensitive to initialization and the strict functional form. Deep LPPLS trades interpretability for robustness: it detects bubble-like dynamics even when the price trajectory deviates from the parametric model. Supports save/load for pretrained weights.

### GSADF (Generalized Sup ADF)

The Generalized Sup Augmented Dickey-Fuller test (Phillips, Shi & Yu, 2015) detects explosive behavior in time series — a hallmark of asset price bubbles. It extends the SADF test (Phillips, Wu & Yu, 2011) by allowing multiple bubble episodes and date-stamping their start and end.

The test statistic is the supremum of recursively computed ADF statistics over all feasible subsamples. Critical values are obtained via Monte Carlo simulation under the null of a random walk with drift. The O(n^2) computation of the backward sup ADF (BSADF) sequence and the Monte Carlo simulations are parallelized with rayon.

GSADF complements LPPLS: LPPLS detects the characteristic *shape* of a bubble (super-exponential growth with log-periodic oscillations), while GSADF detects *explosive unit root behavior* without imposing a parametric form.

### Multiscale Signals

Tail and bubble indicators are computed at daily, 3-day, and weekly observation frequencies. The multiscale aggregator requires signal agreement across at least two scales before elevating the crash probability. A signal that appears at a single frequency may reflect microstructure noise; a signal that persists across scales reflects structural dynamics in the underlying process.

## Accuracy Results

### Individual methods on 39 drawdowns (BTC, SPY, Gold)

| Method | Small (<15%) | Medium (15-30%) | Major (>30%) | Overall |
|--------|:---:|:---:|:---:|:---:|
| **LPPLS** | **100%** | **100%** | **100%** | **100%** |
| Hurst | 57% | 65% | 50% | 59% |
| Max-Stability Kappa | 57% | 47% | 38% | 49% |
| Pickands | 43% | 53% | 50% | 49% |
| GPD VaR | 40% | 55% | 0% | 42% |
| GSADF | 14% | 59% | 38% | 38% |
| Taleb Kappa | 21% | 35% | 50% | 33% |
| Hill | 29% | 29% | 25% | 28% |

### Combined detector (all methods with agreement bonus)

| Method | Small | Medium | Major | Overall |
|--------|:---:|:---:|:---:|:---:|
| **COMBINED** | **64%** | **94%** | **75%** | **79%** |

When the combined detector signals HIGH or CRITICAL, agreement across independent method categories (bubble, tail, regime) provides strong confirmation.

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

All 12 pairs exhibit Hurst H > 0.55, indicating universal persistence in forex markets. 9/12 show explosive bubble episodes (GSADF). 10/12 have heavy tails (Pickands xi > 0 or Hill alpha < 4).

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

19/30 countries have alpha < 2 (infinite variance). 25/30 have Hurst > 0.5 (persistent). Only 1/30 has alpha > 4.

Italy (H=0.80) and Portugal (H=0.85) exhibit the strongest persistence — their exchange rates trend for years at a time over century-scale data.

### The headline number

**71% of countries have exchange rate distributions with infinite variance.** The median tail index across 138 countries is alpha = 1.57. Standard risk models assume finite variance (alpha > 2) and often finite kurtosis (alpha > 4). For the majority of the world's currencies, these assumptions are empirically false.

## Data Flow

```
Exchange/CSV/FRED -> ingest.py -> transforms.py -> [log_prices, log_returns]
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

Bundled offline data (no internet required):

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

Methods are grouped into 4 independent categories. When 3+ categories agree, the crash probability receives a +15% bonus.

| Category | Methods | What it detects |
|----------|---------|-----------------|
| **Bubble** | LPPLS, GSADF, Deep LPPLS | Super-exponential growth, explosive unit roots |
| **Tail** | Taleb Kappa, Max-Stability Kappa, Hill, Pickands, GPD VaR | Tail thickening, distributional regime shifts |
| **Regime** | Hurst | Transition from mean-reverting to persistent dynamics |
| **Structure** | Multiscale, LPPLS t_c proximity | Cross-timeframe agreement, timing |

## Development

Requires [Nix](https://nixos.org/) with flakes enabled.

```bash
# Enter dev shell (Rust, Python 3.13, maturin, uv)
nix develop

# Install Python deps and build Rust extension
make setup

# Build (compiles Rust, installs Python package)
make build

# Run all tests (20 Rust + 79 Python = 99 total)
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

If you use [direnv](https://direnv.net/), `cd` into the project and run `direnv allow` — the shell activates automatically.

## Architecture

### Rust (via PyO3)

| Component | Why Rust |
|-----------|----------|
| LPPLS fitter (CMA-ES) | O(1000) nonlinear fits per anchor date |
| LPPLS confidence | Nested windows parallelized with rayon |
| GSADF test | O(n^2) BSADF + Monte Carlo, parallelized with rayon |
| GEV/GPD fitting | Rolling EVT needs speed |
| Hill, Pickands, Taleb Kappa, Max-Stability Kappa, Hurst | Called at every rolling step |

### Python

- Data ingestion (Yahoo, CoinGecko, CCXT, FRED, CSV/Parquet)
- Visualization (plotly, matplotlib)
- CLI (typer) and service (FastAPI)
- Deep LPPLS (PyTorch, with save/load for pretrained weights)
- Signal aggregation with agreement bonus
- 9 Jupyter notebooks

## References

### Ergodicity and Non-Ergodic Economics

- Peters, O. (2019). "The Ergodicity Problem in Economics." *Nature Physics*, 15, 1216-1221. [DOI:10.1038/s41567-019-0732-0](https://doi.org/10.1038/s41567-019-0732-0)
- Peters, O. & Gell-Mann, M. (2016). "Evaluating Gambles Using Dynamics." *Chaos*, 26(2), 023103. [arXiv:1405.0585](https://arxiv.org/abs/1405.0585)
- Peters, O. (2011). "The Time Resolution of the St Petersburg Paradox." *Philosophical Transactions of the Royal Society A*, 369(1956), 4913-4931. [arXiv:1011.4404](https://arxiv.org/abs/1011.4404)
- Peters, O. & Adamou, A. (2021). "The Ergodicity Problem in Economics." [Ergodicity Economics lecture notes](https://ergodicityeconomics.com/lecture-notes/)

### Tail Estimation

- Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of a Distribution." *Annals of Statistics*, 3(5), 1163-1174.
- Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics." *Annals of Statistics*, 3(1), 119-131.
- Taleb, N.N. (2019). "How Much Data Do You Need? An Operational, Pre-Asymptotic Metric for Fat-tailedness." *International Journal of Forecasting*, 35(2), 677-686. [arXiv:1802.05495](https://arxiv.org/abs/1802.05495)
- Taleb, N.N. (2020). *Statistical Consequences of Fat Tails: Real World Preasymptotics, Epistemology, and Applications.* STEM Academic Press. [arXiv:2001.10488](https://arxiv.org/abs/2001.10488)

### Extreme Value Theory

- Embrechts, P., Kluppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance.* Springer.
- Balkema, A.A. & de Haan, L. (1974). "Residual Life Time at Great Age." *Annals of Probability*, 2(5), 792-804.
- Fisher, R.A. & Tippett, L.H.C. (1928). "Limiting Forms of the Frequency Distribution of the Largest or Smallest Member of a Sample." *Proceedings of the Cambridge Philosophical Society*, 24(2), 180-190.

### Bubble Detection

- Sornette, D. (2003). *Why Stock Markets Crash: Critical Events in Complex Financial Systems.* Princeton University Press.
- Sornette, D., Demos, G., Zhang, Q., Cauwels, P., Filimonov, V. & Zhang, Q. (2015). "Real-Time Prediction and Post-Mortem Analysis of the Shanghai 2015 Stock Market Bubble and Crash." *Journal of Investment Strategies*, 4(4), 77-95.
- Phillips, P.C.B., Shi, S. & Yu, J. (2015). "Testing for Multiple Bubbles: Historical Episodes of Exuberance and Collapse in the S&P 500." *International Economic Review*, 56(4), 1043-1078.
- Phillips, P.C.B., Wu, Y. & Yu, J. (2011). "Explosive Behavior in the 1990s NASDAQ: When Did Exuberance Escalate Asset Values?" *International Economic Review*, 52(1), 201-226.
- Hansen, N. (2006). "The CMA Evolution Strategy: A Comparing Review." In *Towards a New Evolutionary Computation*, Springer, 75-102.

### Long-Range Dependence

- Hurst, H.E. (1951). "Long-Term Storage Capacity of Reservoirs." *Transactions of the American Society of Civil Engineers*, 116, 770-799.
- Lo, A.W. (1991). "Long-Term Memory in Stock Market Prices." *Econometrica*, 59(5), 1279-1313.

### Fat Tails in Finance

- Mandelbrot, B.B. (1963). "The Variation of Certain Speculative Prices." *Journal of Business*, 36(4), 394-419.
- Gabaix, X. (2009). "Power Laws in Economics and Finance." *Annual Review of Economics*, 1, 255-294.
- Mandelbrot, B.B. & Taleb, N.N. (2010). "Random Jump, Not Random Walk." In *The Known, the Unknown, and the Unknowable in Financial Risk Management.* Princeton University Press.
