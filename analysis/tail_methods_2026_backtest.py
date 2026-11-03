"""Crash-window backtest of the 2026 tail methods against the classic tail estimators.

Uses the crash and non-crash windows of `accuracy_report.py` (same drawdown
finder, thresholds and seeds). Each method is scored by comparing the
pre-crash window to an earlier baseline of the same length, as in the main
report. Hill, DEH, QQ, Pickands and GPD VaR are re-scored on the same windows so the
comparison is like for like.

Run:
    python analysis/tail_methods_2026_backtest.py

Datasets are those available locally: bundled BTC/SPY/Gold, plus FRED forex
and LBMA from ~/projects/forex-centuries if present. Databento futures and
options-backtester series are used only if their directories exist.

All numbers are in-sample on historical data. This is not financial advice.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from accuracy_report import (  # noqa: E402
    _DATABENTO_DIR,
    _DATABENTO_FUTURES,
    _LBMA_DIR,
    _LBMA_FILES,
    compute_metrics,
    find_drawdowns,
    fred_min_dd,
    load_databento_continuous,
    load_fred_forex,
    load_lbma,
    sample_non_crash_windows,
)
from fatcrash._core import (  # noqa: E402
    conservative_hill,
    deh_estimator,
    expectile_var_es,
    gpd_var_es,
    hill_estimator,
    lognormal_gpd_mixture,
    pickands_estimator,
    qq_estimator,
    reduced_bias_evi,
    tail_homogeneity_test,
    tail_persistence,
)
from fatcrash.data.ingest import from_csv, from_sample  # noqa: E402
from fatcrash.data.transforms import log_returns  # noqa: E402

BASELINE = ["hill", "deh", "qq", "pickands", "gpd_var"]
NEW = [
    "cons_hill", "rb_hill", "nrb_pm", "rb_lehmer",
    "tail_persist", "mixture_xi", "expectile_xes", "tail_shift",
]
METHODS = BASELINE + NEW


def _nan_safe(f):
    try:
        v = f()
    except (ValueError, OverflowError):
        return float("nan")
    return float(v)


def _compare(pre, base, higher_is_signal=True):
    if np.isnan(pre) or np.isnan(base):
        return None
    return pre > base if higher_is_signal else pre < base


def score_window(df, end_idx, window, diag):
    """Score all methods on [end-window, end) against the window 30 days earlier."""
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

    r = {}
    # Baselines: same rules as accuracy_report.py.
    r["hill"] = _compare(hill_estimator(pre), hill_estimator(base), higher_is_signal=False)
    r["deh"] = _compare(deh_estimator(pre), deh_estimator(base))
    r["qq"] = _compare(qq_estimator(pre), qq_estimator(base), higher_is_signal=False)
    r["pickands"] = _compare(pickands_estimator(pre), pickands_estimator(base))
    r["gpd_var"] = _compare(
        _nan_safe(lambda: gpd_var_es(pre, p=0.95, quantile=0.80)[0]),
        _nan_safe(lambda: gpd_var_es(base, p=0.95, quantile=0.80)[0]),
    )

    # Tail index: lower alpha / higher xi in the pre-crash window = thickening.
    r["cons_hill"] = _compare(
        conservative_hill(pre)[0], conservative_hill(base)[0], higher_is_signal=False
    )
    for name, kind, p in [("rb_hill", "hill", 1.0), ("nrb_pm", "nrb_pm", 0.5), ("rb_lehmer", "lehmer", 2.0)]:
        r[name] = _compare(
            _nan_safe(lambda d=pre: reduced_bias_evi(d, kind=kind, p=p)),
            _nan_safe(lambda d=base: reduced_bias_evi(d, kind=kind, p=p)),
        )

    # Overshoot beyond the 90% quantile of |returns|.
    r["tail_persist"] = _compare(tail_persistence(pre, u=0.9), tail_persistence(base, u=0.9))

    # Threshold-free mixture tail index (needs >= 30 losses per window).
    def mix_xi(d):
        try:
            return lognormal_gpd_mixture(d, max_iter=200, tol=1e-6)[3]
        except ValueError:
            return float("nan")

    xi_pre, xi_base = mix_xi(pre), mix_xi(base)
    r["mixture_xi"] = _compare(xi_pre, xi_base)
    if not np.isnan(xi_pre):
        diag["mixture_fits"] += 1
        diag["mixture_at_bound"] += xi_pre <= 1.1e-4

    # Expectile-based expected shortfall at 95% (higher = more tail risk).
    r["expectile_xes"] = _compare(
        _nan_safe(lambda: expectile_var_es(pre, tau=0.95)[1]),
        _nan_safe(lambda: expectile_var_es(base, tau=0.95)[1]),
    )

    # Significance-gated thickening: Yang test says the tail index changed
    # (p < 0.05) and Hill says it got heavier.
    vals = np.concatenate([pre, base])
    groups = np.repeat([0, 1], [len(pre), len(base)]).astype(np.int64)
    k = max(5, int(np.sqrt(min(len(pre), len(base)))))
    try:
        _, pval, _ = tail_homogeneity_test(vals, groups, k, n_perm=199, seed=0)
        thicker = hill_estimator(pre) < hill_estimator(base)
        r["tail_shift"] = bool(pval < 0.05 and thicker)
    except ValueError:
        r["tail_shift"] = None
    return r


def iter_datasets():
    """Yield (source, name, df, min_dd, min_apart, n_non_crash) like accuracy_report."""
    for name, thr in [("btc", 0.15), ("spy", 0.08), ("gold", 0.08)]:
        yield "core", name.upper(), from_sample(name), thr, 90, 50
    fred_dir = Path("/Users/unbalancedparen/projects/forex-centuries/data/sources/fred/daily")
    skip = {"fred_usd_broad_index.csv", "fred_usd_major_index.csv"}
    if fred_dir.is_dir():
        for path in sorted(fred_dir.glob("fred_*.csv")):
            if path.name in skip:
                continue
            df = load_fred_forex(path)
            if df is not None:
                yield "FRED", path.stem, df, fred_min_dd(path.name), 90, 20
    opts_dir = Path("/Users/unbalancedparen/projects/options_backtester/tests/data")
    if opts_dir.is_dir():
        for fname in ["spy_crisis_stocks.csv", "spy_covid_stocks.csv", "spy_bear_stocks.csv",
                      "spy_lowvol_stocks.csv", "qqq_2020_stocks.csv", "iwm_2020_stocks.csv"]:
            path = opts_dir / fname
            if path.exists():
                df = from_csv(str(path), date_col="date", price_col="close")
                if len(df) >= 60:
                    yield "OptsBT", fname, df, 0.08, 30, 10
    if _DATABENTO_DIR.is_dir():
        for fname, root, display, min_dd in _DATABENTO_FUTURES:
            path = _DATABENTO_DIR / fname
            if path.exists():
                df = load_databento_continuous(path, root)
                if df is not None:
                    yield "Databento", display, df, min_dd, 90, 20
    if _LBMA_DIR.is_dir():
        for fname, col, display, min_dd in _LBMA_FILES:
            path = _LBMA_DIR / fname
            if path.exists():
                df = load_lbma(path, col)
                if df is not None:
                    yield "LBMA", display, df, min_dd, 90, 20


def lift(x):
    """Recall minus false-positive rate, with a two-proportion z-test p-value.

    A signal that fires as often outside crashes as before them has zero lift,
    whatever its recall. Windows overlap across assets (shared crises), so the
    p-value is optimistic.
    """
    from math import erfc, sqrt

    n1, n2 = x["tp"] + x["fn"], x["fp"] + x["tn"]
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    p1, p2 = x["tp"] / n1, x["fp"] / n2
    pooled = (x["tp"] + x["fp"]) / (n1 + n2)
    se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    pval = erfc(abs(p1 - p2) / se / sqrt(2)) if se > 0 else float("nan")
    return p2, p1 - p2, pval


def run(window, datasets):
    tp, fp = [], []
    diag = {"mixture_fits": 0, "mixture_at_bound": 0, "n_crash": 0, "n_non_crash": 0}
    for source, name, df, min_dd, min_apart, n_nc in datasets:
        events = find_drawdowns(df, min_dd=min_dd, min_apart=min_apart)
        for ev in events:
            res = score_window(df, ev["peak_idx"], window, diag)
            if res:
                diag["n_crash"] += 1
                tp += [(m, v) for m, v in res.items() if v is not None]
        for nc in sample_non_crash_windows(df, events, n_samples=n_nc, window=window, seed=42):
            res = score_window(df, nc["center_idx"], window, diag)
            if res:
                diag["n_non_crash"] += 1
                fp += [(m, v) for m, v in res.items() if v is not None]
    return compute_metrics(tp, fp, METHODS), diag


def main():
    datasets = list(iter_datasets())
    sources = {}
    for src, *_ in datasets:
        sources[src] = sources.get(src, 0) + 1
    print("Datasets:", ", ".join(f"{k}={v}" for k, v in sources.items()))

    summary = {}
    for window in [30, 60, 120]:
        metrics, diag = run(window, datasets)
        summary[window] = metrics
        base_rate = diag["n_crash"] / max(1, diag["n_crash"] + diag["n_non_crash"])
        print(f"\n=== {window}d window: {diag['n_crash']} crash / {diag['n_non_crash']} non-crash windows "
              f"(precision of a coin flip = {base_rate:.0%}) ===")
        print(f"  {'Method':<14} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  {'Prec':>5} {'Recall':>6} {'F1':>5}  "
              f"{'Recall 95% CI':>13}  {'FPR':>4} {'Lift':>5} {'p':>6}")
        for m in sorted(METHODS, key=lambda m: -metrics[m]["f1"]):
            x = metrics[m]
            if x["tp"] + x["fn"] == 0:
                print(f"  {m:<14} (not enough data at this window)")
                continue
            lo, hi = x["recall_ci"]
            fpr, lft, pval = lift(x)
            tag = "" if m in BASELINE else " *"
            print(f"  {m + tag:<14} {x['tp']:>4} {x['fp']:>4} {x['fn']:>4} {x['tn']:>4}  "
                  f"{x['precision']:>5.0%} {x['recall']:>6.0%} {x['f1']:>5.0%}  {lo:>5.0%}–{hi:<5.0%}  "
                  f"{fpr:>4.0%} {lft:>+5.0%} {pval:>6.3f}")
        if diag["mixture_fits"]:
            print(f"  mixture xi at its lower bound (body/tail not separable) in "
                  f"{diag['mixture_at_bound']}/{diag['mixture_fits']} pre-crash/non-crash fits")
        print("  * = new method. Lift = recall - FPR; p = two-proportion z-test (optimistic: windows overlap).")

    print("\n=== F1 by window ===")
    print(f"  {'Method':<14} {'30d':>5} {'60d':>5} {'120d':>5}")
    for m in METHODS:
        row = [summary[w][m] for w in (30, 60, 120)]
        cells = [f"{r['f1']:>5.0%}" if r["tp"] + r["fn"] else "   --" for r in row]
        print(f"  {m:<14} " + " ".join(cells))


if __name__ == "__main__":
    main()
