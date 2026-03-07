# Quickstart

## Installation

```bash
# From PyPI (when published)
pip install fatcrash

# Development install (builds Rust extensions)
cd fatcrash
maturin develop --release
pip install -e "python/[dev]"
```

## 5-Minute Tutorial

```python
from fatcrash import from_sample, log_returns, estimate_tail_index, detect_bubble, aggregate_signals

# 1. Load data
btc = from_sample("btc")
returns = log_returns(btc)

# 2. Compute indicators
tail = estimate_tail_index(returns)
print(f"Hill alpha: {tail.alpha:.2f}, fat tails: {tail.is_fat_tail}")

bubble = detect_bubble(btc["close"].values)
print(f"GSADF: {bubble.gsadf_stat:.2f}, bubble at 95%: {bubble.is_bubble_95}")

# 3. Aggregate into crash probability
signal = aggregate_signals({"hill_thinning": 0.5, "gsadf_bubble": 0.8})
print(f"Crash probability: {signal.probability:.0%}, level: {signal.level}")
```

## Three Workflows

### 1. Quick Check: Are Tails Fat?

Load sample data, run tail estimators, interpret.

```python
from fatcrash import (
    from_sample, log_returns,
    estimate_tail_index, estimate_kappa, estimate_taleb_kappa,
    estimate_pickands, estimate_deh, estimate_hurst,
)

btc = from_sample("btc")
ret = log_returns(btc)

tail = estimate_tail_index(ret)
print(f"Hill alpha: {tail.alpha:.2f}")
# alpha < 2 → infinite variance
# alpha < 4 → infinite kurtosis

kappa = estimate_kappa(ret)
print(f"Kappa: {kappa.kappa:.3f} vs benchmark {kappa.gaussian_benchmark:.3f}")
# kappa < benchmark → fat tails

tk = estimate_taleb_kappa(ret)
print(f"Taleb kappa: {tk.kappa:.3f}")
# 0 = Gaussian, 1 = Cauchy

pickands = estimate_pickands(ret)
print(f"Pickands gamma: {pickands.gamma:.3f} ({pickands.tail_type})")

deh = estimate_deh(ret)
print(f"DEH gamma: {deh.gamma:.3f} ({deh.tail_type})")

hurst = estimate_hurst(ret)
print(f"Hurst H: {hurst.h:.3f} ({hurst.regime})")
```

### 2. Crash Detection: Full Pipeline

Load data, run all methods, aggregate into crash probability.

```python
import numpy as np
from fatcrash import (
    from_sample, log_returns, log_prices, time_index, negative_returns,
    estimate_tail_index, estimate_kappa, estimate_taleb_kappa,
    estimate_pickands, estimate_deh, estimate_qq, estimate_maxsum,
    estimate_hurst, estimate_dfa, estimate_spectral,
    estimate_momentum, estimate_reversal, estimate_velocity,
    fit_gpd, compute_var_es, detect_bubble,
    fit_lppls, compute_confidence,
    estimate_rv_spike, estimate_csd_on_vol, estimate_hamilton,
    aggregate_signals,
)
from fatcrash.aggregator.signals import (
    kappa_regime_signal, taleb_kappa_signal, gsadf_signal,
    dfa_signal, momentum_reversal_signal, velocity_signal,
    rv_spike_signal, hamilton_stress_signal,
)

btc = from_sample("btc")
ret = log_returns(btc)
prices = btc["close"].values
lp = log_prices(btc)
ti = time_index(btc)

# Tail estimators
kappa = estimate_kappa(ret)
tk = estimate_taleb_kappa(ret)
dfa = estimate_dfa(ret)
reversal = estimate_reversal(prices)
vel = estimate_velocity(ret)

# Bubble detection
bubble = detect_bubble(prices)

# LPPLS (needs log prices and time index, skip last point for alignment)
lppls = fit_lppls(ti[:-1], lp[:-1])

# Regime detection
rv = estimate_rv_spike(ret)
csd = estimate_csd_on_vol(ret)
hmm = estimate_hamilton(ret)

# Convert to signals [0, 1]
components = {
    "kappa_regime": kappa_regime_signal(kappa.kappa, kappa.gaussian_benchmark),
    "taleb_kappa": taleb_kappa_signal(tk.kappa, tk.gaussian_benchmark),
    "gsadf_bubble": gsadf_signal(bubble.gsadf_stat, bubble.critical_values.cv_95),
    "dfa_trending": dfa_signal(dfa.alpha),
    "momentum_reversal": momentum_reversal_signal(reversal.reversal),
    "velocity_spike": velocity_signal(vel.velocity),
    "rv_spike": rv_spike_signal(rv.rv_short, rv.rv_long),
    "hamilton_stress": hamilton_stress_signal(hmm.prob_stressed),
    "lppls_confidence": lppls.r2 if lppls.is_bubble else 0.0,
}

crash = aggregate_signals(components)
print(f"Crash probability: {crash.probability:.0%}")
print(f"Level: {crash.level}")
print(f"Categories agreeing: {crash.n_agreeing}")
```

### 3. Rolling Monitoring

Compute rolling indicators for time-series visualization.

```python
import numpy as np
from fatcrash import (
    from_sample, log_returns,
    rolling_tail_index, rolling_kappa, rolling_hurst,
    rolling_dfa, rolling_pickands, rolling_var_es,
)

btc = from_sample("btc")
ret = log_returns(btc)

# Rolling tail indicators (252-day window = 1 year)
alpha_series = rolling_tail_index(ret, window=252)
kappa_series, kappa_bench = rolling_kappa(ret, window=252)
hurst_series = rolling_hurst(ret, window=252)
dfa_series = rolling_dfa(ret, window=252)
pickands_series = rolling_pickands(ret, window=252)

# Rolling risk metrics
var_series, es_series = rolling_var_es(ret, window=252)

# Plot (requires matplotlib)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(alpha_series, label="Hill alpha")
axes[0].axhline(2, color="r", ls="--", label="Infinite variance")
axes[0].axhline(4, color="orange", ls="--", label="Infinite kurtosis")
axes[0].legend()
axes[0].set_title("Rolling Hill Alpha (252d)")

axes[1].plot(hurst_series, label="Hurst H")
axes[1].axhline(0.5, color="gray", ls="--", label="Random walk")
axes[1].legend()
axes[1].set_title("Rolling Hurst Exponent (252d)")

axes[2].plot(var_series, label="VaR 99%")
axes[2].plot(es_series, label="ES 99%")
axes[2].legend()
axes[2].set_title("Rolling GPD VaR & ES (252d)")

plt.tight_layout()
plt.show()
```

## Available Sample Data

```python
from fatcrash import from_sample

btc  = from_sample("btc")      # BTC 2014-2025 (4,124 days)
spy  = from_sample("spy")      # SPY 1999-2025 (6,570 days)
gold = from_sample("gold")     # Gold 2000-2025 (6,441 days)
```

## Network Data Sources

```python
from fatcrash import from_yahoo, from_coingecko, from_fred, from_fred_macro

# Yahoo Finance
aapl = from_yahoo("AAPL", start="2020-01-01")

# CoinGecko (free, no API key)
eth = from_coingecko("ethereum", days=365)

# FRED macro data
vix = from_fred("VIXCLS")
macro = from_fred_macro()  # VIX, HY spread, yield curve, etc.
```
