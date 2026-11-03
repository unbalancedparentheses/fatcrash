# Method Deep-Dive

18 crash detection methods across 5 families, plus market regime signals. Each method measures a different property of the distribution or dynamics — the ensemble captures signals that no single method can.

---

## Family 1: Bubble Detection

### LPPLS (Log-Periodic Power Law Singularity)

**What it measures:** Super-exponential growth with log-periodic oscillations — the fingerprint of a speculative bubble.

**Formula:**

$$\ln p(t) = A + B|t_c - t|^m + C|t_c - t|^m \cos(\omega \ln|t_c - t| + \phi)$$

Where $t_c$ is the critical time (predicted crash date), $m$ is the power-law exponent, $\omega$ is the log-periodic frequency.

**Parameters:**
- `tc_range`: Bounds for critical time search (default: auto)
- `pop_size`: CMA-ES population (default: 50)
- `n_generations`: Optimization generations (default: 40)

**Interpretation:**
- $t_c > 0$: Crash predicted in $t_c$ days
- $m \in [0.1, 0.9]$: Valid power-law exponent
- $\omega \in [6, 13]$: Valid log-periodic frequency
- $B < 0$: Super-exponential growth (required for bubble)
- All three conditions = Sornette filter passes = `is_bubble = True`

**Reference:** Sornette, D. (2003). *Why Stock Markets Crash*. Princeton University Press.

**Limitations:** Sensitive to fitting window. Contract rolls in futures corrupt bubble shapes. Not applicable to fundamental data.

### GSADF (Generalized Sup ADF)

**What it measures:** Explosive unit roots — price dynamics that are too fast to be explained by a random walk.

**Formula:**

Augmented Dickey-Fuller regression: $\Delta y_t = \delta y_{t-1} + \sum \varphi_j \Delta y_{t-j} + \varepsilon_t$

GSADF = sup over all start/end subsamples of the ADF t-statistic. Compared to Monte Carlo critical values under the null of a driftless random walk.

**Parameters:**
- `min_window`: Minimum regression window (default: PSY rule of thumb)
- `n_sims`: Monte Carlo simulations (default: 200)

**Interpretation:**
- `gsadf_stat > cv_95`: Explosive behavior detected at 95% confidence
- `bsadf_sequence`: Time series of backward SADF — shows *when* explosive episodes start/end

**Reference:** Phillips, P.C.B., Shi, S., & Yu, J. (2015). "Testing for Multiple Bubbles." *International Economic Review*, 56(4).

**Limitations:** $O(n^2)$ computation. Constant F1=23% across all window sizes. LPPLS confidence subsumes it.

---

## Family 2: Tail Estimation

### Hill Estimator

**What it measures:** Tail index $\alpha$ — the rate at which the tail probability decays.

**Formula:**

$$\alpha = \left[\frac{1}{k} \sum_{i=1}^{k} \ln\frac{X_{(i)}}{X_{(k+1)}}\right]^{-1}$$

Where $X_{(1)} \geq X_{(2)} \geq \ldots$ are order statistics and $k = \lfloor\sqrt{n}\rfloor$ by default.

**Parameters:**
- `k`: Number of order statistics (default: $\sqrt{n}$)

**Interpretation:**
- $\alpha < 2$: Infinite variance — VaR, Sharpe ratios, correlations are meaningless
- $\alpha < 4$: Infinite kurtosis — CLT convergence extremely slow
- $\alpha > 4$: Finite kurtosis

**Reference:** Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of a Distribution." *Annals of Statistics*, 3(5).

**Limitations:** Assumes heavy tails (Fréchet domain). Biased for small $k$. For all-domain estimation, use Pickands or DEH.

### Pickands Estimator

**What it measures:** Extreme value index $\gamma$, valid for all three domains of attraction.

**Formula:**

$$\gamma = \frac{\ln\left(\frac{X_{(k)} - X_{(2k)}}{X_{(2k)} - X_{(4k)}}\right)}{\ln 2}$$

**Interpretation:**
- $\gamma > 0$: Heavy tails (Fréchet) — power-law decay
- $\gamma \approx 0$: Exponential tails (Gumbel)
- $\gamma < 0$: Bounded support (Weibull)

**Reference:** Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics." *Annals of Statistics*, 3(1).

**Limitations:** Higher variance than Hill for heavy tails. Requires $4k$ observations minimum.

### DEH Moment Estimator

**What it measures:** Extreme value index via moments. Valid for all domains of attraction.

**Formula:**

$$M_j = \frac{1}{k} \sum_{i=1}^{k} \left(\ln X_{(i)} - \ln X_{(k+1)}\right)^j, \quad j = 1, 2$$

$$\gamma = M_1 + 1 - \frac{1}{2}\left(1 - \frac{M_1^2}{M_2}\right)^{-1}$$

**Reference:** Dekkers, A.L.M., Einmahl, J.H.J., & de Haan, L. (1989). "A Moment Estimator for the Index of an Extreme-Value Distribution." *Annals of Statistics*, 17(4).

**Limitations:** Sensitive to the choice of $k$.

### QQ Estimator

**What it measures:** Tail index from QQ-plot slope.

**Formula:**

Regress $\ln X_{(i)}$ vs $-\ln(i/(k+1))$ for the $k$ largest observations. Slope $= 1/\alpha$.

**Interpretation:** Same as Hill ($\alpha < 2$ = infinite variance, $\alpha < 4$ = infinite kurtosis).

**Limitations:** Sensitive to outliers in the largest observation.

### Taleb's Kappa

**What it measures:** How fast the sample mean converges. Quantifies the practical failure of the CLT.

**Formula:**

$$\kappa = 2 - \frac{\ln(n/n_0)}{\ln(M(n)/M(n_0))}$$

Where $M(n)$ = mean absolute deviation computed on $n$ observations.

Under CLT: $M(n) \sim \sqrt{n}$ → $\kappa = 0$.
For Cauchy: $M(n) \sim n$ → $\kappa = 1$.

**Parameters:**
- `n0`: Small subsample size (default: 30)
- `n1`: Large subsample size (default: 100)

**Interpretation:**
- $\kappa \approx 0$: Gaussian — CLT works normally
- $\kappa \approx 0.5$: Very fat — need thousands of observations for convergence
- $\kappa \approx 1.0$: Cauchy-like — sample mean never converges

**Reference:** Taleb, N.N. (2019). "How Much Data Do You Need? An Operational, Pre-Asymptotic Metric for Fat-Tailedness." *International Journal of Forecasting*.

### Max-Stability Kappa

**What it measures:** Block-max concentration. Whether a single extreme dominates the sample.

**Formula:**

Partition data into blocks, compute $\text{mean(block\_maxima)} / \text{global\_maximum}$. Compare to Monte Carlo Gaussian benchmark.

**Interpretation:** $\kappa < \text{benchmark}$ → fat tails (single extreme dominates more than expected under normality).

### Maximum-to-Sum Ratio

**What it measures:** Whether variance is infinite.

**Formula:**

$$R_n = \frac{\max_i |X_i|}{\sum_i |X_i|}$$

**Interpretation:**
- $R_n \to 0$ as $n \to \infty$ for $\alpha > 2$ (finite variance)
- $R_n$ stays positive for $\alpha < 2$ (infinite variance)
- Threshold: $R_n > 0.05$ suggests infinite variance

**Limitations:** Simplest diagnostic — use alongside Hill/Pickands for confirmation.

---

### Conservative Hill (rescaled)

**What it measures:** A lower-confidence tail index: an $\alpha$ that rarely overestimates the true value. Overestimating $\alpha$ underestimates crash probabilities.

**Formula:** With $H_k$ the Hill log-spacing mean,

$$\hat\alpha^{(\theta)} = \frac{1-\theta}{H_k}, \qquad -\log P(\hat\alpha^{(\theta)} > \alpha) \sim k f(\theta), \quad f(\theta) = -\theta - \log(1-\theta)$$

Given a target overestimation probability $\delta$: $\lambda = \log(1/\delta)/k$ and $\theta = 1 + W_0(-e^{-(1+\lambda)})$.

**Parameters:** `k` (default $\sqrt n$), `delta` (default 0.05). Python: `conservative_hill(data, k, delta)` → `(alpha, theta)`.

**Reference:** Gösgens, M., van Parys, B. & Zwart, B. (2026). "Large and Moderate Deviations for Conservative Tail-Index Estimation." arXiv:2609.31127.

**Limitations:** Large-deviations rate, not an exact finite-sample bound. Assumes iid data; volatility clustering lowers the effective $k$. Among scale-invariant estimators using the same $k$ order statistics, none is less biased at the same decay rate.

### Power-mean, Lehmer-mean, and reduced-bias EVI estimators

**What it measures:** $\xi = 1/\alpha$ with the leading Hill bias removed, so the estimate is much flatter in $k$.

**Formula:** With log-excesses $V_i = \ln X_{(i)} - \ln X_{(k+1)}$ and $M^{(p)} = \frac1k\sum V_i^p$:

$$\hat\xi_{PM}(p) = \left(\frac{M^{(p)}}{\Gamma(p+1)}\right)^{1/p}, \qquad \hat\xi_{L}(p) = \frac{M^{(p)}}{p\,M^{(p-1)}}$$

Both equal Hill at $p=1$. Reduced-bias versions multiply by $1 - c(p,\hat\rho)\,\hat\beta (n/k)^{\hat\rho}$; the nonlinear version uses $2 - \exp\{c\,\hat\beta (n/k)^{\hat\rho}\}$. The second-order parameters $(\hat\rho,\hat\beta)$ are estimated at $k_1 = n^{0.999}$ (Fraga Alves et al. 2003; Gomes & Martins 2002).

**Parameters:** `kind` ∈ {`hill`, `pm`, `nrb_pm`, `lehmer`}, `p`, `tau` (0 for $-1\le\rho<0$). Python: `reduced_bias_evi`, `power_mean_evi`, `lehmer_evi`, `second_order_params`.

**Reference:** Henriques-Rodrigues, L., Caeiro, F. & Gomes, M.I. (2026). "Nonlinear Reduced-Bias Extreme Value Index Estimators: A Comparative Study." arXiv:2610.34308.

**Limitations:** Needs ≥ 20 positive values; the second-order estimates are noisy on short rolling windows. Best on full histories or long windows, where a larger $k$ is usable.

### Tail persistence

**What it measures:** How far observations overshoot an extreme threshold, in units of that threshold. This is the ES/VaR ratio.

**Formula:**

$$P(u) = \frac{E[X \mid X > Q(u)]}{Q(u)}, \qquad \lim_{u\to1} P(u) = \frac{1}{1-\xi} \quad (0<\xi<1)$$

**Interpretation:** $P \approx 1$ for light tails; $P = 1.5$ for $\alpha = 3$; $P \to \infty$ as $\alpha \to 1$.

**Reference:** Rawat, N., Yadav, S.K., Sharma, D. & Yadav, R. (2026). "Tail persistence of extremes at finite and asymptotic levels with Bayesian hierarchical inference." arXiv:2611.03288.

**Limitations:** The asymptotic value $1/(1-\xi)$ is a fixed transform of $\xi$, so it adds no information beyond Hill/DEH and is not in the aggregator. Only the empirical finite-level $P(u)$ (`tail_persistence`, `tail_persistence_rolling`) carries anything new. The paper is written for environmental data; the Bayesian hierarchical model is not implemented.

### Tail-index homogeneity test

**What it measures:** Whether the tail index really differs across groups (countries, regimes, before/after) or along a covariate, without estimating any conditional tail index.

**Method:** Under a common $\alpha$, log exceedances $\ln(Y/w)$ are asymptotically Exp($\alpha$) regardless of the covariate. The test measures dependence between log exceedances and the covariate with squared distance covariance, calibrated by permutation (plus-one p-value). The group version keeps the top `k_per_group` of *each* group with its own threshold. Without this, a single global threshold would select mostly from the fattest-tailed group and the test would lose power.

**Python:** `tail_homogeneity_test(values, groups, k_per_group)` and `tail_homogeneity_covariate_test(values, covariate, k)` → `(statistic, p_value, n_tail)`.

**Reference:** Yang, T.T. (2026). "Testing Tail-Index Homogeneity via Exceedance-Covariate Independence." arXiv:2610.21019.

**Limitations:** The permutation null assumes independent observations. Decluster or subsample daily returns first.

## Family 3: Extreme Value Theory

### GPD (Generalized Pareto Distribution)

**What it measures:** The distribution of losses exceeding a threshold. Produces VaR and Expected Shortfall.

**Formula:**

For exceedances $y = x - u$ over threshold $u$:

$$G(y) = \begin{cases} 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi} & \text{if } \xi \neq 0 \\ 1 - \exp(-y/\sigma) & \text{if } \xi = 0 \end{cases}$$

VaR at level $p$:

$$\text{VaR}_p = u + \frac{\sigma}{\xi}\left[\left(\frac{n}{N_u(1-p)}\right)^{\xi} - 1\right]$$

Expected Shortfall:

$$\text{ES}_p = \frac{\text{VaR}_p}{1 - \xi} + \frac{\sigma - \xi u}{1 - \xi}$$

**Parameters:**
- `quantile`: Threshold quantile (default: 0.95 = top 5% of losses)
- `p`: VaR confidence level (default: 0.99)

**Reference:** Balkema, A.A. & de Haan, L. (1974); McNeil, A.J., Frey, R. & Embrechts, P. (2005). *Quantitative Risk Management*.

### GEV (Generalized Extreme Value)

**What it measures:** Distribution of block maxima. Classifies tail type.

**Formula:**

$$H(x) = \begin{cases} \exp\left[-\left(1 + \xi\frac{x-\mu}{\sigma}\right)^{-1/\xi}\right] & \text{if } \xi \neq 0 \\ \exp\left[-\exp\left(-\frac{x-\mu}{\sigma}\right)\right] & \text{if } \xi = 0 \end{cases}$$

- $\xi > 0$: Fréchet (heavy tails)
- $\xi = 0$: Gumbel (exponential tails)
- $\xi < 0$: Weibull (bounded support)

**Reference:** Fisher, R.A. & Tippett, L.H.C. (1928).

### Lognormal–GPD mixture (threshold-free)

**What it measures:** The whole loss distribution as a lognormal body plus a zero-location GPD tail. No threshold to tune.

**Formula:**

$$f(x) = p\,\mathrm{LN}(x;\mu,s) + (1-p)\,\mathrm{GPD}(x;\beta,\xi), \qquad \xi > 0$$

The mixture's tail is the GPD's, so $\xi$ is the tail index of the losses. It is fitted by EM: weighted lognormal MLE in closed form plus a weighted GPD MLE. VaR comes from inverting the mixture CDF; ES uses closed-form stop-loss transforms of both components.

**Python:** `lognormal_gpd_mixture(data)` → `(p_body, mu, s, xi, beta, loglik, converged)`; `lognormal_gpd_mixture_var_es(data, p)`.

**Reference:** Bee, M. & Santi, F. (2026). "A multivariate unsupervised lognormal-GPD mixture for dependent losses." arXiv:2611.02148 (univariate model implemented; the paper's multivariate Gumbel-copula extension is not).

**Limitations:** EM finds a local maximum. It is slower than POT GPD, so it suits full-history fits better than every rolling step.

### Extreme expectiles

**What it measures:** The $\tau$-expectile of losses and the expected loss beyond it. Expectiles are the only risk measure that is both coherent and elicitable, and unlike VaR they respond to the size of tail losses.

**Formula:** For $0<\gamma<1$, $e_\tau / q_\tau \to (1/\gamma - 1)^{-\gamma}$, giving

$$\hat e_\tau = (1/\hat\gamma - 1)^{-\hat\gamma}\,\hat q_\tau, \qquad \hat q_\tau = X_{(k+1)}\left(\frac{k}{n(1-\tau)}\right)^{\hat\gamma}, \qquad \mathrm{XES}_\tau \approx \frac{\hat e_\tau}{1-\hat\gamma}$$

with Hill $\hat\gamma$ and the Weissman quantile.

**Python:** `expectile_var_es(data, tau=0.99)` → `(expectile, xes, gamma)`.

**Reference:** Daouia, A., Girard, S. & Stupfler, G. (2018). "Estimation of tail risk based on extreme expectiles." *JRSS B*, 80(2).

---

## Family 4: Regime & Persistence

### Critical Slowing Down (CSD)

**What it measures:** Proximity to a tipping point. Near critical transitions, systems recover more slowly — observable as rising AR(1) and rising variance simultaneously.

**Method:** Compute rolling AR(1) coefficient and rolling variance of the *volatility* series (not raw returns). When both are increasing, the system is approaching a tipping point.

**Parameters:**
- `rv_window`: Window for realized variance (default: 21)
- `csd_window`: Window for AR(1) and variance (default: 63)
- `roc_window`: Rate-of-change window (default: 21)

**Interpretation:**
- Both AR(1) and variance rising → `warning = True` → approaching tipping point
- Only one rising → partial signal
- Neither rising → normal conditions

**Reference:** Scheffer, M. et al. (2009). "Early-warning signals for critical transitions." *Nature*, 461.

**Important:** Apply to volatility series, not raw returns. Raw returns have near-zero autocorrelation.

### RV Spike

**What it measures:** Short-term volatility regime change vs long-term baseline.

**Formula:**

$$\text{ratio} = \frac{RV_{\text{short}}}{RV_{\text{long}}}$$

**Interpretation:**
- ratio $\approx 1$: Normal volatility
- ratio $> 2$: Volatility spiking — regime change onset
- ratio $> 3$: Severe spike

### Hamilton Filter

**What it measures:** 2-state Gaussian HMM. Classifies the current observation into normal or stressed regime.

**Method:** Baum-Welch EM with Kim (1994) smoother. 10 random restarts to avoid local optima. State 1 is always the higher-volatility (stressed) state.

**Interpretation:**
- $P(\text{stressed}) > 0.5$: Currently in stress regime
- **88% recall on major (>30%) crashes** — the best single method for large drawdowns
- Backward-looking: confirms crisis, doesn't predict it

**Reference:** Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*, 57(2).

### Momentum & Reversal

**What it measures:** Trailing momentum and momentum breakdown.

Momentum = trailing log return over lookback period.
Reversal = long-term momentum minus short-term momentum. When long-term is positive but short-term turns negative, the trend is breaking.

**Reference:** Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*, 48(1).

### Price Velocity

**What it measures:** Rate of change of realized volatility. Detects forced-liquidation cascades.

**Formula:**

$$\text{velocity} = \frac{\sigma_t - \sigma_{t-\text{lag}}}{\sigma_{t-\text{lag}}}$$

velocity $> 1.0$ means vol has doubled — the signature of Volmageddon (Feb 5, 2018) or the Sep 2019 repo blowup.

---

## Family 5: Persistence

### Hurst Exponent

**What it measures:** Persistence via rescaled range (R/S) analysis.

**Formula:**

For a range of block sizes $n$:

$$\mathbb{E}[R(n)/S(n)] \sim n^H$$

Estimate $H$ from log-log regression of $R/S$ vs $n$.

**Reference:** Hurst, H.E. (1951). "Long-term Storage Capacity of Reservoirs." *Transactions of ASCE*, 116.

**Limitations:** Financial data is almost always persistent ($H > 0.5$), so this rarely distinguishes crash from non-crash.

### DFA (Detrended Fluctuation Analysis)

**What it measures:** Long-range correlations, handling non-stationarity.

**Method:** Compute cumulative sum, divide into boxes of size $n$, detrend each box, compute RMS fluctuation $F(n)$. DFA exponent $\alpha$ from log-log regression of $F(n)$ vs $n$.

**Reference:** Peng, C.K. et al. (1994). "Mosaic Organization of DNA Nucleotides." *Physical Review E*, 49(2).

**Limitations:** Overfires on forex/futures — persistence is the norm, not crash-specific.

### Spectral Exponent

**What it measures:** Long-memory parameter $d$ from the periodogram (frequency domain).

**Formula:**

$$\ln I(\omega_j) = c - 2d \ln \omega_j + \varepsilon_j$$

Relation to Hurst: $d = H - 0.5$.

**Reference:** Geweke, J. & Porter-Hudak, S. (1983).

---

## Evaluation & Backtesting

### Effective sample size is the number of tail events

When comparing tail forecasts, the score difference is known in advance on every day without an exceedance. The information is in the exceedances, so they are the effective sample size, not the number of days (Coelho 2026). The same applies to crash detection: recall is estimated from the crash windows. `analysis/accuracy_report.py` now prints a Wilson 95% interval for recall. For example, 90% recall over 96 crashes is 82–94%, and 82% over 39 crash windows is 67–91%. Crash windows that share a market-wide event are not independent, so even these intervals are optimistic.

### Anytime-valid comparison of quantile forecasts

`compare_quantile_forecasts(y, q_a, q_b, tau)` gives an empirical-Bernstein confidence sequence $(L_t, U_t)$ for the running mean quantile-score difference. It is valid simultaneously over all $t$ with no distributional assumption. $L_t>0$ means forecast B is better; $U_t<0$ means A is. Unlike a Diebold–Mariano test, it does not systematically favour the less conservative forecaster when exceedances are rare.

**Reference:** Coelho, R. (2026). "Comparing tail forecasts: the number of exceedances is the sample size." arXiv:2610.40366.

### ES e-backtest without VaR forecasts

`es_e_backtest(losses, es_forecast, n_cal)` monitors $H_0: \mathrm{ES}_p \le r$ with an e-process: reject the first time it exceeds $1/\alpha$, checked as often as you like. A binomial lower confidence bound for VaR from the calibration window replaces the missing VaR forecast, and the e-process takes the infimum of the ES betting product over all VaR values in $[z_\delta, r]$.

**Reference:** Wang, R. & Yu, J. (2026). "E-Backtesting Expected Shortfall without Value-at-Risk Forecasts under Time-series Models." arXiv:2610.17681 (iid version implemented; the AR-GARCH time-series version is not).

**Limitations:** Assumes iid losses: use standardized residuals, not raw returns.

### Crash-window backtest of the 2026 methods

See the README section "Crash-window results" and `analysis/tail_methods_2026_backtest.py`. In short, the tail-shape estimators (conservative Hill, reduced-bias EVI, tail persistence, mixture ξ, homogeneity gating) have zero or negative lift before crashes. Expectile ES has positive lift (+17% at 120d) and complements GPD VaR.

## Signal Aggregation

Methods are grouped into independent categories. When 3+ categories have elevated signals (> 0.5), the aggregator applies a +15% bonus to crash probability.

**Categories:** bubble, tail, regime, structure, risk_premium, liquidity, vol_regime, credit_macro, structure_flows, contagion, sentiment.

**Weights** are informed by L1-regularized logistic regression on 406 crash windows. Methods with negative learned weights (Hill, max-to-sum, GPD) are zeroed out — they hurt the ensemble.

The strongest individual signals:
1. LPPLS confidence (weight: 0.22)
2. GSADF bubble (weight: 0.16)
3. Kappa regime (weight: 0.12)
