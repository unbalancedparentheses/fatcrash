# Python API Reference

All functions are importable from the top-level `fatcrash` package:

```python
from fatcrash import estimate_tail_index, detect_bubble, aggregate_signals
```

Or from subpackages:

```python
from fatcrash.indicators import estimate_tail_index
from fatcrash.data import from_sample, log_returns
from fatcrash.aggregator import aggregate_signals, CrashSignal
```

---

## Data Ingestion (`fatcrash.data.ingest`)

### `from_sample(asset: str = "btc") -> pd.DataFrame`

Load bundled sample data (no internet required).

**Available assets:** `"btc"` (2014-2025), `"spy"` (1999-2025), `"gold"` (2000-2025), `"signals"`.

Returns DataFrame with DatetimeIndex and at least a `close` column.

### `from_yahoo(ticker, start="2015-01-01", end=None, use_cache=True) -> pd.DataFrame`

Fetch OHLCV data from Yahoo Finance. Cached to `~/.cache/fatcrash/`.

### `from_coingecko(coin_id="bitcoin", vs_currency="usd", days=365, use_cache=True) -> pd.DataFrame`

Fetch daily prices from CoinGecko (free, no API key).

### `from_csv(path, date_col="date", price_col="close") -> pd.DataFrame`

Load OHLCV data from a CSV file.

### `from_fred(series, start="2007-01-01", end=None, use_cache=True) -> pd.DataFrame`

Fetch macro data from FRED. `series` can be a single ID or list of IDs.

Common series: `VIXCLS` (VIX), `T10Y2Y` (yield curve), `BAMLH0A0HYM2` (HY spread), `DGS10` (10Y Treasury).

### `from_fred_macro(start="2005-01-01", end=None, use_cache=True) -> pd.DataFrame`

Fetch key macro series (VIX, HY OAS, BBB OAS, yield curves, TED, BAA10Y, STLFSI) in one call.

---

## Data Transforms (`fatcrash.data.transforms`)

### `log_returns(df, col="close") -> np.ndarray`

Compute log returns: $\ln(p_t / p_{t-1})$. Returns array of length `n-1`.

### `log_prices(df, col="close") -> np.ndarray`

Compute $\ln(\text{price})$ series. Same length as input.

### `time_index(df) -> np.ndarray`

Convert DatetimeIndex to float64 array (days from start). Used as input for LPPLS fitting.

### `negative_returns(returns) -> np.ndarray`

Extract negative returns as positive values (losses) for tail analysis. Equivalent to `-returns[returns < 0]`.

### `block_maxima(returns, block_size=21) -> np.ndarray`

Compute block maxima of absolute returns for GEV fitting. Default `block_size=21` = monthly blocks for daily data.

---

## Tail Estimators (`fatcrash.indicators.tail_indicator`)

### `estimate_tail_index(returns, k=None) -> TailEstimate`

Hill estimator for tail index $\alpha$.

**Returns:** `TailEstimate(alpha: float, is_fat_tail: bool)`

| $\alpha$ | Interpretation |
|:-----:|----------------|
| $< 2$ | Infinite variance — VaR, Sharpe ratios are meaningless |
| $< 4$ | Infinite kurtosis — CLT convergence is extremely slow |
| $> 4$ | Finite kurtosis — standard risk metrics apply (but may still be fat-tailed) |

### `rolling_tail_index(returns, window=252, k=None) -> np.ndarray`

Rolling Hill $\alpha$. Returns array of same length, NaN where insufficient data.

### `estimate_kappa(returns, n_subsamples=10) -> KappaEstimate`

Max-stability kappa: subsample-max ratio vs Gaussian benchmark.

**Returns:** `KappaEstimate(kappa: float, gaussian_benchmark: float, is_fat_tail: bool)`

`is_fat_tail` is True when `kappa < benchmark` — a single extreme dominates the sample more than it would for Gaussian data.

### `rolling_kappa(returns, window=252, n_subsamples=10) -> tuple[np.ndarray, float]`

Returns `(kappa_series, gaussian_benchmark)`.

### `estimate_taleb_kappa(returns, n0=30, n1=100) -> TalebKappaEstimate`

MAD convergence rate. Measures how fast the sample mean converges.

**Returns:** `TalebKappaEstimate(kappa: float, gaussian_benchmark: float, is_fat_tail: bool)`

| $\kappa$ | Interpretation |
|:-----:|----------------|
| $\approx 0$ | Gaussian — CLT operates normally |
| $\approx 0.5$ | Very fat — CLT convergence extremely slow |
| $\approx 1.0$ | Cauchy-like — sample mean does not converge |

### `rolling_taleb_kappa(returns, window=252, n0=30, n1=100) -> tuple[np.ndarray, float]`

Returns `(kappa_series, gaussian_benchmark)`.

### `estimate_pickands(returns, k=None) -> PickandsEstimate`

Pickands extreme value index. Valid for all three domains of attraction.

**Returns:** `PickandsEstimate(gamma: float, tail_type: str)`

| $\gamma$ | tail_type | Interpretation |
|:-----:|-----------|----------------|
| $> 0$ | `"heavy"` | Fréchet domain — power-law tails |
| $\approx 0$ | `"light"` | Gumbel domain — exponential tails |
| $< 0$ | `"bounded"` | Weibull domain — finite support |

### `rolling_pickands(returns, window=252, k=None) -> np.ndarray`

### `estimate_hurst(data) -> HurstEstimate`

Hurst exponent via R/S analysis.

**Returns:** `HurstEstimate(h: float, regime: str)`

| $H$ | regime | Interpretation |
|:---:|--------|----------------|
| $> 0.5$ | `"trending"` | Persistent dynamics — trends continue |
| $\approx 0.5$ | `"random_walk"` | No memory |
| $< 0.5$ | `"mean_reverting"` | Anti-persistent — mean reversion |

### `rolling_hurst(data, window=252) -> np.ndarray`

### `estimate_dfa(data) -> DFAEstimate`

Detrended Fluctuation Analysis exponent. Handles non-stationarity better than R/S Hurst.

**Returns:** `DFAEstimate(alpha: float, regime: str)`

| $\alpha$ | regime | Interpretation |
|:-----:|--------|----------------|
| $> 0.5$ | `"persistent"` | Long-range correlations |
| $\approx 0.5$ | `"white_noise"` | Uncorrelated |
| $< 0.5$ | `"anti_persistent"` | Anti-correlated |

### `rolling_dfa(data, window=252) -> np.ndarray`

### `estimate_deh(returns, k=None) -> DEHEstimate`

Dekkers-Einmahl-de Haan moment estimator. Valid for all domains of attraction.

**Returns:** `DEHEstimate(gamma: float, tail_type: str)` — same interpretation as Pickands.

### `rolling_deh(returns, window=252, k=None) -> np.ndarray`

### `estimate_qq(returns, k=None) -> QQEstimate`

Tail index from QQ-plot slope regression.

**Returns:** `QQEstimate(alpha: float, is_fat_tail: bool)` — same thresholds as Hill.

### `rolling_qq(returns, window=252, k=None) -> np.ndarray`

### `estimate_maxsum(data) -> MaxSumEstimate`

Maximum-to-Sum ratio diagnostic.

**Returns:** `MaxSumEstimate(ratio: float, is_infinite_variance: bool)`

High ratio (> 0.05) suggests $\alpha < 2$. The simplest diagnostic for whether variance exists.

### `rolling_maxsum(data, window=252) -> np.ndarray`

### `estimate_spectral(data, bandwidth_exp=0.65) -> SpectralEstimate`

GPH spectral exponent for long-memory detection.

**Returns:** `SpectralEstimate(d: float, regime: str)`

| $d$ | regime | Interpretation |
|:---:|--------|----------------|
| $> 0$ | `"long_memory"` | Shocks persist |
| $\approx 0$ | `"short_memory"` | No long-range dependence |
| $< 0$ | `"anti_persistent"` | Shocks dissipate |

### `estimate_momentum(prices, lookback=252) -> MomentumEstimate`

Trailing log return over lookback period.

**Returns:** `MomentumEstimate(momentum: float, lookback: int, direction: str)`

### `rolling_momentum(prices, lookback=252, window=504) -> np.ndarray`

### `estimate_reversal(prices, short_lookback=21, long_lookback=252) -> ReversalEstimate`

Momentum reversal: divergence between short and long-term momentum.

**Returns:** `ReversalEstimate(reversal: float, is_reversing: bool)`

When `is_reversing` is True (reversal > 0.1), long-term momentum is positive but short-term has turned negative — a crash precursor.

### `rolling_reversal(prices, short_lookback=21, long_lookback=252, window=504) -> np.ndarray`

### `estimate_velocity(returns, vol_window=21, lag=5) -> VelocityEstimate`

Rate of change of realized volatility. Detects cascade dynamics.

**Returns:** `VelocityEstimate(velocity: float, is_accelerating: bool)`

`is_accelerating` is True when velocity $> 1.0$ (vol doubled).

### `rolling_velocity(returns, vol_window=21, lag=5, window=252) -> np.ndarray`

---

## EVT (`fatcrash.indicators.evt_indicator`)

### `fit_gpd(returns, quantile=0.95) -> GPDResult`

Fit Generalized Pareto Distribution to the tail of the return distribution.

**Returns:** `GPDResult(sigma: float, xi: float, threshold: float, n_exceedances: int)`

$\xi > 0$ means heavy tails (Fréchet domain). $\xi = 0$ means exponential tails. $\xi < 0$ means bounded support.

### `fit_gev(block_maxima) -> GEVResult`

Fit Generalized Extreme Value distribution to block maxima.

**Returns:** `GEVResult(mu: float, sigma: float, xi: float)`

### `compute_var_es(returns, p=0.99, quantile=0.95) -> RiskMetrics`

Compute Value-at-Risk and Expected Shortfall from GPD tail fit.

**Returns:** `RiskMetrics(var: float, es: float)`

- `var`: The loss threshold at confidence level `p` (e.g., 99% VaR)
- `es`: Expected loss given that loss exceeds VaR (always >= VaR)

### `rolling_var_es(returns, window=252, p=0.99, quantile=0.95) -> tuple[np.ndarray, np.ndarray]`

Returns `(var_series, es_series)`.

---

## Bubble Detection (`fatcrash.indicators.bubble_indicator`)

### `detect_bubble(prices, min_window=None, n_sims=200, seed=42) -> GSADFResult`

GSADF test for explosive bubble detection (Phillips-Shi-Yu 2015).

**Parameters:**
- `prices`: Price levels (not returns)
- `min_window`: Minimum regression window (default: PSY rule of thumb)
- `n_sims`: Monte Carlo simulations for critical values

**Returns:** `GSADFResult(gsadf_stat, bsadf_sequence, critical_values, is_bubble_90, is_bubble_95, is_bubble_99)`

| Condition | Interpretation |
|-----------|----------------|
| `stat > cv_95` | Bubble detected at 95% confidence |
| `stat > cv_99` | Bubble detected at 99% confidence |

### `rolling_bubble_detection(prices, window=252, min_window=None) -> np.ndarray`

Rolling GSADF statistic for continuous monitoring. Returns array of GSADF stats.

---

## LPPLS (`fatcrash.indicators.lppls_indicator`)

### `fit_lppls(times, log_prices, tc_range=None, pop_size=50, n_generations=40) -> LPPLSResult`

Fit Log-Periodic Power Law Singularity model.

**Parameters:**
- `times`: Float64 array from `time_index()`
- `log_prices`: Float64 array from `log_prices()`
- `tc_range`: Optional `(min, max)` bounds for critical time

**Returns:** `LPPLSResult(tc, m, omega, a, b, c1, c2, rss, r2, is_bubble)`

Sornette filter (`is_bubble`): $m \in [0.1, 0.9]$, $\omega \in [6.0, 13.0]$, $B < 0$.

| Field | Interpretation |
|-------|----------------|
| `tc` | Predicted crash date (days from end of fitting window) |
| `m` | Power-law exponent (0.1-0.9 for valid bubble) |
| `omega` | Log-periodic frequency (6-13 for valid bubble) |
| `is_bubble` | Passes Sornette filter — all conditions met |

### `compute_confidence(times, log_prices, min_window=60, max_window=750, n_windows=50, n_candidates=30) -> tuple[np.ndarray, np.ndarray, np.ndarray]`

DS LPPLS confidence indicator. Fits LPPLS across many sub-windows.

**Returns:** `(confidence, tc_mean, tc_std)` arrays.

---

## Regime Detection (`fatcrash.indicators.regime_indicator`)

### `estimate_realized_variance(returns, window=21) -> RealizedVarianceEstimate`

**Returns:** `RealizedVarianceEstimate(rv: float, rv_annualized_vol: float)`

### `estimate_rv_spike(returns, short_window=21, long_window=126) -> RVSpikeEstimate`

Compare short-term RV to long-term baseline.

**Returns:** `RVSpikeEstimate(rv_short, rv_long, ratio)`

| ratio | Interpretation |
|:-----:|----------------|
| ~1.0 | Normal volatility |
| > 2.0 | Vol spike — regime change onset |
| > 3.0 | Severe vol spike |

### `estimate_jump_risk(returns, window=21) -> JumpRiskEstimate`

BNS jump variance decomposition.

**Returns:** `JumpRiskEstimate(rv, bv, jv, jump_fraction)`

### `estimate_jump_test(returns, window=63) -> JumpTestResult`

BNS z-test for jumps.

**Returns:** `JumpTestResult(z_stat, jv, significant)` — `significant` if z > 1.96.

### `estimate_csd(data, window=252, roc_window=63) -> CSDEstimate`

Critical Slowing Down detection. Pass a *volatility* series, not raw returns.

**Returns:** `CSDEstimate(ar1_rising: bool, var_rising: bool, warning: bool)`

`warning` is True when both AR(1) and variance are rising — approaching a tipping point.

### `estimate_csd_on_vol(returns, rv_window=21, csd_window=63, roc_window=21) -> CSDEstimate | None`

CSD on the rolling realized variance series (recommended approach). Returns None if insufficient data.

### `estimate_hamilton(data, n_restarts=10) -> HamiltonEstimate`

2-state HMM regime classification via EM.

**Returns:** `HamiltonEstimate(mu_normal, sigma_normal, mu_stressed, sigma_stressed, p00, p11, prob_stressed, filtered_probs)`

| P(stressed) | Interpretation |
|:-----------:|----------------|
| > 0.5 | Currently in stress regime |
| > 0.8 | High-confidence stress regime |
| < 0.2 | Normal regime |

### `rolling_rv(returns, window=21, step=1) -> np.ndarray`

### `rolling_csd(data, window=252, roc_window=63) -> tuple[np.ndarray, np.ndarray, np.ndarray]`

Returns `(ar1_roc, var_roc, csd_signal)`.

### `rolling_ar1_coeff(data, window=252) -> np.ndarray`

---

## Vol Strategy (`fatcrash.indicators.vol_indicator`)

### `constant_vol_weight(returns, target_vol=0.15, window=63, max_leverage=2.0) -> ConstantVolResult`

Position sizing for constant volatility targeting: `weight = target_vol / realized_vol`.

**Returns:** `ConstantVolResult(weight: float, current_vol: float, target_vol: float)`

### `rebalance_risk(dfa_alpha, momentum) -> RebalanceRiskResult`

Assess risk of mechanical rebalancing given current regime.

**Returns:** `RebalanceRiskResult(risk: float, dfa_component: float, momentum_component: float)`

`risk` is $[0, 1]$. High risk when DFA detects trending ($\alpha > 0.5$) AND momentum is negative — rebalancing buys into a continuing drawdown.

---

## Signal Aggregation (`fatcrash.aggregator`)

### `aggregate_signals(components, weights=None) -> CrashSignal`

Combine indicator signals into a crash probability.

**Parameters:**
- `components`: Dict mapping signal name to value in [0, 1]
- `weights`: Optional custom weights (default: `DEFAULT_WEIGHTS`)

**Returns:** `CrashSignal(probability, horizon_days, components, n_agreeing)`

| probability | level | Interpretation |
|:-----------:|-------|----------------|
| > 0.7 | `CRITICAL` | Multiple families agree — high crash risk |
| > 0.5 | `HIGH` | Significant risk — consider reducing exposure |
| > 0.3 | `ELEVATED` | Some warning signs — monitor closely |
| <= 0.3 | `LOW` | Normal conditions |

### `CrashSignal`

Dataclass with fields: `probability`, `horizon_days`, `components`, `n_agreeing`, and property `level`.

### `RegimeSignal`

Dataclass with fields: `label` (`"risk_on"` / `"neutral"` / `"risk_off"`), `score`, `confidence`, `buckets`, `components`.

### `calibrate_weights(signals_history, crash_labels, method="logistic") -> dict[str, float]`

Calibrate aggregation weights from historical data. Methods: `"logistic"` (L1-regularized logistic regression) or `"equal"`.

### `DEFAULT_WEIGHTS`

Dict mapping signal names to their default weights, informed by L1-regularized logistic regression.

---

## Threshold Interpretation Summary

| Indicator | Threshold | Meaning |
|-----------|-----------|---------|
| Hill $\alpha$ | $< 2$ | Infinite variance |
| Hill $\alpha$ | $< 4$ | Infinite kurtosis |
| Kappa | $< \text{benchmark}$ | Fat tails (single extreme dominates) |
| Taleb $\kappa$ | $\approx 0$ | Gaussian |
| Taleb $\kappa$ | $\approx 1$ | Cauchy (CLT fails) |
| Pickands $\gamma$ | $> 0$ | Heavy tails (Fréchet) |
| Pickands $\gamma$ | $= 0$ | Exponential tails (Gumbel) |
| Pickands $\gamma$ | $< 0$ | Bounded support (Weibull) |
| Hurst $H$ | $> 0.5$ | Trending / persistent |
| Hurst $H$ | $= 0.5$ | Random walk |
| Hurst $H$ | $< 0.5$ | Mean-reverting |
| DFA $\alpha$ | $> 0.5$ | Persistent / long-range correlations |
| DFA $\alpha$ | $= 0.5$ | White noise |
| LPPLS | $m \in [0.1, 0.9]$, $\omega \in [6, 13]$, $B < 0$ | Valid bubble |
| GSADF | stat $> cv_{95}$ | Explosive bubble (95% confidence) |
| Hamilton | $P(\text{stressed}) > 0.5$ | Stress regime |
| CSD | AR(1) + variance both rising | Approaching tipping point |
| Crash signal | $> 0.7$ | CRITICAL |
| Crash signal | $> 0.5$ | HIGH |
| Crash signal | $> 0.3$ | ELEVATED |
