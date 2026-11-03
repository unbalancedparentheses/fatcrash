"""Is expectile ES a crash signal in its own right, or just volatility?

Uses the same crash and non-crash windows as `tail_methods_2026_backtest.py`.
For each window, computes log ratios (pre-crash window / baseline window) of:
volatility, downside semivolatility, the report's RV-spike ratio, GPD VaR, and
expectile ES. Then reports:

1. AUC of each ratio for separating crash from non-crash windows.
2. Lift of each binary "ratio > 0" rule.
3. Correlation of the expectile ES ratio with the volatility ratios.
4. Logistic-regression likelihood-ratio tests: does expectile ES add
   information once volatility (or downside volatility) is known?

Run:
    python analysis/expectile_es_check.py
"""

import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from accuracy_report import find_drawdowns, sample_non_crash_windows  # noqa: E402
from tail_methods_2026_backtest import iter_datasets  # noqa: E402
from fatcrash._core import expectile_var_es, gpd_var_es  # noqa: E402
from fatcrash.data.transforms import log_returns  # noqa: E402
from fatcrash.indicators.regime_indicator import estimate_rv_spike  # noqa: E402

FEATURES = ["vol", "downside_vol", "rv_spike", "gpd_var", "expectile_es"]


def _safe(f):
    try:
        v = float(f())
    except (ValueError, OverflowError, ZeroDivisionError):
        return float("nan")
    return v if np.isfinite(v) and v > 0 else float("nan")


def window_stats(ret):
    downside = np.minimum(ret, 0.0)
    rv = estimate_rv_spike(ret, short_window=21, long_window=min(len(ret), 63)) if len(ret) >= 63 else None
    return {
        "vol": _safe(lambda: np.std(ret)),
        "downside_vol": _safe(lambda: np.sqrt(np.mean(downside**2))),
        "rv_spike": _safe(lambda: rv.ratio) if rv else float("nan"),
        "gpd_var": _safe(lambda: gpd_var_es(ret, p=0.95, quantile=0.80)[0]),
        "expectile_es": _safe(lambda: expectile_var_es(ret, tau=0.95)[1]),
    }


def features(df, end_idx, window):
    pre_start = max(0, end_idx - window)
    base_end = max(0, pre_start - 30)
    base_start = max(0, base_end - window)
    min_len = max(20, window // 2)
    if base_end - base_start < min_len or end_idx - pre_start < min_len:
        return None
    pre = log_returns(df.iloc[pre_start:end_idx])
    base = log_returns(df.iloc[base_start:base_end])
    if len(pre) < max(15, window // 2) or len(base) < max(15, window // 2):
        return None
    sp, sb = window_stats(pre), window_stats(base)
    return {k: np.log(sp[k] / sb[k]) for k in FEATURES}


def collect(window, datasets):
    rows, labels = [], []
    for _, _, df, min_dd, min_apart, n_nc in datasets:
        events = find_drawdowns(df, min_dd=min_dd, min_apart=min_apart)
        for ev in events:
            f = features(df, ev["peak_idx"], window)
            if f:
                rows.append(f)
                labels.append(1)
        for nc in sample_non_crash_windows(df, events, n_samples=n_nc, window=window, seed=42):
            f = features(df, nc["center_idx"], window)
            if f:
                rows.append(f)
                labels.append(0)
    X = {k: np.array([r[k] for r in rows]) for k in FEATURES}
    return X, np.array(labels)


def auc(x, y):
    """Mann–Whitney AUC with a normal-approximation 95% CI (Hanley & McNeil)."""
    m = np.isfinite(x)
    x, y = x[m], y[m]
    pos, neg = x[y == 1], x[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan"), 0
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    a = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    return a, se, n1


def lift(x, y):
    m = np.isfinite(x)
    x, y = x[m], y[m]
    fire = x > 0
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    rec, fpr = fire[y == 1].mean(), fire[y == 0].mean()
    pooled = fire.mean()
    se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    return rec, fpr, rec - fpr, erfc(abs(rec - fpr) / se / sqrt(2)) if se > 0 else float("nan")


def logit_loglik(X, y, iters=50):
    """Logistic regression by Newton's method; returns the maximized log-likelihood."""
    Xd = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xd @ b))
        g = Xd.T @ (y - p)
        H = (Xd * (p * (1 - p))[:, None]).T @ Xd
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-Xd @ b)), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), b


def lr_test(X, y, base, extra):
    cols = base + [extra]
    m = np.all(np.isfinite(np.column_stack([X[c] for c in cols])), axis=1)
    yb = y[m]
    ll0, _ = logit_loglik(np.column_stack([X[c][m] for c in base]), yb)
    ll1, b = logit_loglik(np.column_stack([X[c][m] for c in cols]), yb)
    stat = 2 * (ll1 - ll0)
    # chi-square with 1 df
    p = erfc(sqrt(max(stat, 0) / 2))
    return stat, p, b[-1], int(m.sum())


def main():
    datasets = list(iter_datasets())
    for window in [30, 60, 120]:
        X, y = collect(window, datasets)
        print(f"\n=== {window}d: {int(y.sum())} crash / {int((1 - y).sum())} non-crash windows ===")
        print(f"  {'Feature':<14} {'AUC':>5} {'95% CI':>13} {'n_crash':>7}   {'Recall':>6} {'FPR':>5} {'Lift':>5} {'p':>6}")
        for k in FEATURES:
            a, se, n1 = auc(X[k], y)
            if n1 == 0:
                print(f"  {k:<14} (not enough data)")
                continue
            rec, fpr, lft, p = lift(X[k], y)
            print(f"  {k:<14} {a:>5.2f} {a - 1.96 * se:>6.2f}–{a + 1.96 * se:<5.2f} {n1:>7}   "
                  f"{rec:>6.0%} {fpr:>5.0%} {lft:>+5.0%} {p:>6.3f}")
        m = np.isfinite(X["expectile_es"]) & np.isfinite(X["vol"]) & np.isfinite(X["downside_vol"])
        print(f"  corr(expectile_es, vol) = {np.corrcoef(X['expectile_es'][m], X['vol'][m])[0, 1]:.2f}, "
              f"corr(expectile_es, downside_vol) = {np.corrcoef(X['expectile_es'][m], X['downside_vol'][m])[0, 1]:.2f}")
        for base in (["vol"], ["downside_vol"], ["vol", "downside_vol"]):
            stat, p, coef, n = lr_test(X, y, base, "expectile_es")
            print(f"  ES added to {' + '.join(base):<20}: LR = {stat:5.2f}, p = {p:.3f}, coef = {coef:+.2f} (n={n})")
        stat, p, coef, n = lr_test(X, y, ["expectile_es"], "downside_vol")
        print(f"  downside_vol added to ES          : LR = {stat:5.2f}, p = {p:.3f}, coef = {coef:+.2f} (n={n})")


if __name__ == "__main__":
    main()
