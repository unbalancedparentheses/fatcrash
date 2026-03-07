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

## Signal Aggregation

Methods are grouped into independent categories. When 3+ categories have elevated signals (> 0.5), the aggregator applies a +15% bonus to crash probability.

**Categories:** bubble, tail, regime, structure, risk_premium, liquidity, vol_regime, credit_macro, structure_flows, contagion, sentiment.

**Weights** are informed by L1-regularized logistic regression on 406 crash windows. Methods with negative learned weights (Hill, max-to-sum, GPD) are zeroed out — they hurt the ensemble.

The strongest individual signals:
1. LPPLS confidence (weight: 0.22)
2. GSADF bubble (weight: 0.16)
3. Kappa regime (weight: 0.12)
