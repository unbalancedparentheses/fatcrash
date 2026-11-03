"""Which candidate predictors improve the crash ensemble?

Runs the accuracy report's full method suite (`test_method_on_drawdown`) on the
crash and non-crash windows of `tail_methods_2026_backtest.py`, caches the
aggregator components plus three candidate predictors, then scores each
candidate alone (AUC) and inside the ensemble.

Candidates:
- vol_ratio: pre-window vs baseline-window volatility,
  clip(std_pre / std_base - 1, 0, 1).
- vn_gsadf: GSADF on the volatility-normalized log price sum(r_t / sigma_t),
  sigma_t = std of the previous 20 returns (Boswijk, Yu & Zu, arXiv:2405.02087).
  Removes the volatility bursts that make plain GSADF fire.
- cusum_vol: Page CUSUM of standardized log squared returns in the pre window
  against the baseline window (sequential volatility-shift monitor, in the
  spirit of Horváth, Trapani & Wang, arXiv:2404.17885).

Ensemble variants swap a candidate into the slot of the signal it would replace
(rv_spike or gsadf_bubble), at that slot's weight and at a higher weight, and
optionally drop pickands/deh/qq_thinning (zero lift in the tail backtest).

Uncertainty: overlapping windows that share a crisis are not independent
(Shi, Zhang & Shi, arXiv:2609.30721), so the bootstrap resamples calendar
quarters (all windows ending in the same quarter move together), and the
AUC difference versus `current` is paired.

Run:
    python analysis/ensemble_variants_check.py [--cache PATH] [--workers N]
"""

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from accuracy_report import find_drawdowns, sample_non_crash_windows, test_method_on_drawdown  # noqa: E402
from fatcrash._core import gsadf_test  # noqa: E402
from fatcrash.aggregator.signals import DEFAULT_WEIGHTS, aggregate_signals, gsadf_signal  # noqa: E402
from fatcrash.data.transforms import log_returns  # noqa: E402
from tail_methods_2026_backtest import iter_datasets  # noqa: E402

WINDOWS = [30, 60, 120]
TAIL_SHAPE = ["pickands_thinning", "deh_thinning", "qq_thinning"]
THRESHOLDS = [0.3, 0.4, 0.5]

_DATASETS = None


def _datasets():
    global _DATASETS
    if _DATASETS is None:
        _DATASETS = list(iter_datasets())
    return _DATASETS


def _windows(df, end_idx, window):
    pre_start = max(0, end_idx - window)
    base_end = max(0, pre_start - 30)
    base_start = max(0, base_end - window)
    return pre_start, base_start, base_end


def candidates(df, end_idx, window):
    """The three candidate signals, each in [0, 1] (NaN when not computable)."""
    pre_start, base_start, base_end = _windows(df, end_idx, window)
    close = df["close"].values
    r = np.diff(np.log(close))  # r[i] is the return into day i + 1
    pre = r[max(0, pre_start - 1):end_idx - 1]
    base = r[max(0, base_start - 1):base_end - 1]
    out = {"vol_ratio": float("nan"), "vn_gsadf": float("nan"), "cusum_vol": float("nan")}
    if len(pre) < 10 or len(base) < 10 or np.std(base) == 0:
        return out
    out["vol_ratio"] = float(np.clip(np.std(pre) / np.std(base) - 1.0, 0.0, 1.0))

    # Page CUSUM on standardized log squared returns, baseline-calibrated.
    c = 1e-12
    lb = np.log(base**2 + c)
    mu, sd = lb.mean(), lb.std()
    if sd > 0:
        u = (np.log(pre**2 + c) - mu) / sd
        s_t, s_max = 0.0, 0.0
        for x in u:
            s_t = max(0.0, s_t + x - 0.5)
            s_max = max(s_max, s_t)
        # Threshold grows with window length so long windows do not saturate.
        out["cusum_vol"] = float(np.clip(s_max / (2.0 + 0.2 * len(pre)), 0.0, 1.0))

    # Volatility-normalized GSADF (needs 20 returns of history before each step).
    if end_idx - pre_start >= 50 and pre_start >= 21:
        seg = r[pre_start - 21:end_idx - 1]
        sig = np.array([seg[i - 20:i].std() for i in range(20, len(seg))])
        z = seg[20:] / np.where(sig > 0, sig, np.nan)
        if np.all(np.isfinite(z)):
            y = np.concatenate([[0.0], np.cumsum(z)])
            stat, _, (_, cv95, _) = gsadf_test(y, min_window=None, n_sims=30, seed=42)
            out["vn_gsadf"] = float(gsadf_signal(stat, cv95))
    return out


def _job(args):
    ds_idx, end_idx, window, label = args
    df = _datasets()[ds_idx][2]
    _, comps = test_method_on_drawdown(df, end_idx, window=window)
    if not comps:
        return None
    return {
        "label": label,
        "date": df.index[min(end_idx, len(df) - 1)],
        "comps": comps,
        "cand": candidates(df, end_idx, window),
    }


def collect(window, workers):
    jobs = []
    for i, (_, _, df, min_dd, min_apart, n_nc) in enumerate(_datasets()):
        events = find_drawdowns(df, min_dd=min_dd, min_apart=min_apart)
        jobs += [(i, ev["peak_idx"], window, 1) for ev in events]
        jobs += [(i, nc["center_idx"], window, 0)
                 for nc in sample_non_crash_windows(df, events, n_samples=n_nc, window=window, seed=42)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return [r for r in ex.map(_job, jobs, chunksize=4) if r is not None]


def variant(row, drop_tail=False, slot=None, weight=None):
    """Ensemble probability with an optional candidate swapped into a slot."""
    c = dict(row["comps"])
    w = dict(DEFAULT_WEIGHTS)
    if drop_tail:
        for k in TAIL_SHAPE:
            c.pop(k, None)
    if slot:
        cand, target = slot
        c.pop(target, None)
        v = row["cand"][cand]
        if not np.isnan(v):
            c[target] = v
        if weight is not None:
            w[target] = weight
    return aggregate_signals(c, weights=w).probability


def auc(score, y):
    m = np.isfinite(score)
    score, y = score[m], y[m]
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Probability a crash window outscores a non-crash one (ties count half).
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def cluster_boot(clusters, n_boot, rng):
    """Index arrays resampling whole clusters with replacement."""
    ids = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in ids}
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        yield np.concatenate([members[c] for c in pick])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path(__file__).parent / ".ensemble_components.pkl"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    cache = Path(args.cache)
    if cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        data = {}
        for w in WINDOWS:
            print(f"collecting {w}d components ...", flush=True)
            data[w] = collect(w, args.workers)
        cache.write_bytes(pickle.dumps(data))

    rng = np.random.default_rng(0)
    for w in WINDOWS:
        rows = data[w]
        y = np.array([r["label"] for r in rows])
        quarters = np.array([f"{r['date'].year}Q{(r['date'].month - 1) // 3 + 1}" for r in rows])
        boots = list(cluster_boot(quarters, 400, rng))
        print(f"\n=== {w}d: {int(y.sum())} crash / {int((1 - y).sum())} non-crash windows, "
              f"{len(np.unique(quarters))} quarter clusters ===")

        def ci(score, ref=None):
            a = auc(score, y)
            if ref is None:
                d = [auc(score[b], y[b]) for b in boots]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                return f"{a:.3f} ({lo:.3f}–{hi:.3f})"
            d = [auc(score[b], y[b]) - auc(ref[b], y[b]) for b in boots]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            return f"{a:.3f}  Δ {a - auc(ref, y):+.3f} ({lo:+.3f} to {hi:+.3f})"

        print("  Single signals (AUC, quarter-clustered 95% CI):")
        singles = {
            "rv_spike (current)": [r["comps"].get("rv_spike", np.nan) for r in rows],
            "gsadf (current)": [r["comps"].get("gsadf_bubble", np.nan) for r in rows],
            "vol_ratio": [r["cand"]["vol_ratio"] for r in rows],
            "cusum_vol": [r["cand"]["cusum_vol"] for r in rows],
            "vn_gsadf": [r["cand"]["vn_gsadf"] for r in rows],
        }
        for name, vals in singles.items():
            v = np.array(vals, dtype=float)
            print(f"    {name:<20} {ci(v)}   n={int(np.isfinite(v).sum())}")

        variants = {
            "current": {},
            "drop tail-shape": {"drop_tail": True},
            "vol_ratio→rv_spike": {"slot": ("vol_ratio", "rv_spike")},
            "vol_ratio→rv_spike w=.12": {"slot": ("vol_ratio", "rv_spike"), "weight": 0.12},
            "cusum→rv_spike w=.12": {"slot": ("cusum_vol", "rv_spike"), "weight": 0.12},
            "vn_gsadf→gsadf": {"slot": ("vn_gsadf", "gsadf_bubble")},
        }
        scores = {k: np.array([variant(r, **kw) for r in rows]) for k, kw in variants.items()}
        print("  Ensemble variants (AUC, Δ vs current with paired quarter-clustered 95% CI):")
        for k, sc in scores.items():
            print(f"    {k:<26} {ci(sc, None if k == 'current' else scores['current'])}")


if __name__ == "__main__":
    main()
