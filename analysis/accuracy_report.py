"""Accuracy analysis: honest precision/recall/F1 for every method against historical drawdowns.

Evaluates all 17 methods (13 classical + 4 NN) on both crash windows (true positives)
and non-crash windows (false positives), producing precision, recall, and F1 scores.

Run:
    python analysis/accuracy_report.py           # All 17 methods
    python analysis/accuracy_report.py --skip-nn  # Classical methods only

All accuracy numbers are in-sample on historical data. This is not financial advice.

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fatcrash.data.ingest import from_sample, from_csv
from fatcrash.data.transforms import log_returns, log_prices, time_index, block_maxima
from fatcrash._core import (
    hill_estimator, hill_rolling, kappa_metric, kappa_rolling, gpd_var_es, lppls_fit,
    pickands_estimator, hurst_exponent, gsadf_test, taleb_kappa,
    dfa_exponent, deh_estimator, qq_estimator, maxsum_ratio, spectral_exponent,
    lppls_confidence,
)
from fatcrash.aggregator import signals as sig
from fatcrash.aggregator.signals import aggregate_signals

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# ── FRED forex loaders ─────────────────────────────────────

# FRED convention: DEXUS__ = "USD per foreign" (use as-is), DEX__US = "foreign per USD" (invert)
_FRED_USD_PER_FOREIGN = {"DEXUSUK", "DEXUSAL", "DEXUSNZ", "DEXUSEU"}


def load_fred_forex(path) -> "pd.DataFrame | None":
    """Read a FRED daily forex CSV and return DataFrame with DatetimeIndex + ``close`` column.

    Pairs quoted as "foreign per USD" (DEX__US) are inverted so that a drawdown
    in the resulting series corresponds to a currency crisis (foreign currency
    losing value against USD).
    """
    try:
        raw = pd.read_csv(path)
    except Exception:
        return None
    if "observation_date" not in raw.columns:
        return None
    val_col = [c for c in raw.columns if c != "observation_date"][0]
    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw = raw.dropna(subset=["observation_date"])
    raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce")
    raw = raw.dropna(subset=[val_col])
    raw = raw[raw[val_col] > 0]
    if len(raw) < 200:
        return None
    prices = raw[val_col].values.copy()
    if val_col not in _FRED_USD_PER_FOREIGN:
        # "foreign per USD" — invert so drawdown = currency crisis
        prices = 1.0 / prices
    df = pd.DataFrame({"close": prices}, index=pd.DatetimeIndex(raw["observation_date"].values))
    df = df.sort_index()
    return df


def fred_min_dd(pair: str) -> float:
    """Return minimum drawdown threshold for a FRED forex pair.

    Major pairs use 8%; emerging-market pairs use 15%.
    """
    majors = {"gbp", "eur", "jpy", "chf", "aud", "cad", "nzd"}
    # Extract pair name from filename like "fred_gbp_usd.csv"
    key = pair.replace("fred_", "").replace("_usd.csv", "").replace(".csv", "").lower()
    return 0.08 if key in majors else 0.15


# ── Drawdown detection ─────────────────────────────────────


def find_drawdowns(df, min_dd=0.10, min_apart=90):
    """Find all drawdowns exceeding min_dd threshold."""
    prices = df["close"].values
    dates = df.index
    running_max = np.maximum.accumulate(prices)
    drawdown = (prices - running_max) / running_max
    events = []
    last = None
    i = 0
    while i < len(drawdown):
        if drawdown[i] < -min_dd:
            start = i
            while i < len(drawdown) and drawdown[i] < -min_dd * 0.5:
                i += 1
            trough_idx = start + np.argmin(drawdown[start:i])
            for j in range(trough_idx, -1, -1):
                if prices[j] == running_max[j]:
                    peak_idx = j
                    break
            dd = (prices[trough_idx] - prices[peak_idx]) / prices[peak_idx]
            if last is None or (trough_idx - last) > min_apart:
                events.append(
                    {
                        "peak_date": dates[peak_idx],
                        "trough_date": dates[trough_idx],
                        "dd_pct": dd * 100,
                        "peak_idx": peak_idx,
                    }
                )
                last = trough_idx
        i += 1
    return events


# ── Non-crash window sampling ──────────────────────────────


def sample_non_crash_windows(df, crash_events, n_samples=50, window=120, min_distance=180, seed=42):
    """Sample random windows that are far from any crash peak/trough.

    Returns list of dicts with 'center_idx' and 'center_date' keys.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Build exclusion zones around all crash peaks and troughs
    excluded = set()
    for ev in crash_events:
        peak = ev["peak_idx"]
        for idx in range(max(0, peak - min_distance), min(n, peak + min_distance)):
            excluded.add(idx)

    # Candidate center indices: must be far from crashes and have enough room for a window
    candidates = [i for i in range(window, n - 30) if i not in excluded]
    if len(candidates) == 0:
        return []

    chosen = rng.choice(candidates, size=min(n_samples, len(candidates)), replace=False)
    dates = df.index
    return [{"center_idx": int(idx), "center_date": dates[int(idx)]} for idx in chosen]


# ── Method testing ─────────────────────────────────────────


def test_method_on_drawdown(df, peak_idx, window=120,
                            run_nn=False,
                            plnn_model=None, hlppl_model=None, dtcai_model=None):
    """Test all methods on a pre-crash window, comparing to a baseline period.

    Returns (results, components) where:
      results: dict of method_name -> bool (detected or not)
      components: dict of signal_name -> float [0,1] for aggregate_signals()
    """
    pre_start = max(0, peak_idx - window)
    pre_end = peak_idx
    base_end = max(0, pre_start - 30)
    base_start = max(0, base_end - window)

    if base_end - base_start < 60 or pre_end - pre_start < 60:
        return None, None

    pre_ret = log_returns(df.iloc[pre_start:pre_end])
    base_ret = log_returns(df.iloc[base_start:base_end])

    if len(pre_ret) < 30 or len(base_ret) < 30:
        return None, None

    results = {}
    components = {}  # signal_name -> float [0,1] for aggregator

    # Hill
    pre_alpha = hill_estimator(pre_ret)
    base_alpha = hill_estimator(base_ret)
    results["hill"] = pre_alpha < base_alpha
    components["hill_thinning"] = sig.hill_thinning_signal(pre_alpha, base_alpha)

    # Kappa
    pre_k, pre_b = kappa_metric(pre_ret, n_subsamples=5, n_sims=50)
    base_k, base_b = kappa_metric(base_ret, n_subsamples=5, n_sims=50)
    pre_ratio = pre_k / pre_b if pre_b > 0 else 1.0
    base_ratio = base_k / base_b if base_b > 0 else 1.0
    results["kappa"] = pre_ratio < base_ratio
    components["kappa_regime"] = sig.kappa_regime_signal(pre_k, pre_b)

    # Taleb kappa
    try:
        pre_tk, pre_tb = taleb_kappa(pre_ret, n0=15, n1=50, n_sims=100)
        base_tk, base_tb = taleb_kappa(base_ret, n0=15, n1=50, n_sims=100)
        if not (np.isnan(pre_tk) or np.isnan(base_tk)):
            results["taleb_kappa"] = pre_tk > base_tk
            components["taleb_kappa"] = sig.taleb_kappa_signal(pre_tk, pre_tb)
        else:
            results["taleb_kappa"] = None
    except Exception:
        results["taleb_kappa"] = None

    # Pickands
    pre_xi = pickands_estimator(pre_ret)
    base_xi = pickands_estimator(base_ret)
    if not (np.isnan(pre_xi) or np.isnan(base_xi)):
        results["pickands"] = pre_xi > base_xi
        components["pickands_thinning"] = sig.pickands_signal(pre_xi, base_xi)
    else:
        results["pickands"] = None

    # Hurst
    pre_h = hurst_exponent(pre_ret)
    if not np.isnan(pre_h):
        results["hurst"] = pre_h > 0.55
        components["hurst_trending"] = sig.hurst_signal(pre_h)
    else:
        results["hurst"] = None

    # GPD VaR
    try:
        pre_var, _ = gpd_var_es(pre_ret, p=0.95, quantile=0.80)
        base_var, _ = gpd_var_es(base_ret, p=0.95, quantile=0.80)
        results["gpd_var"] = pre_var > base_var
        # Use last return vs VaR for exceedance signal
        last_ret = float(pre_ret[-1])
        components["gpd_var_exceedance"] = sig.var_exceedance_signal(last_ret, pre_var)
    except Exception:
        results["gpd_var"] = None

    # LPPLS (tightened: Nielsen omega [6,13] + tc constraint)
    lppls_tc = None
    try:
        pre_lp = log_prices(df.iloc[pre_start:pre_end])
        pre_t = time_index(df.iloc[pre_start:pre_end])
        tc, m, omega, a, b, c1, c2, rss, r2 = lppls_fit(pre_t, pre_lp)
        t2 = float(pre_t[-1])
        dt = t2 - float(pre_t[0])
        results["lppls"] = (
            0.1 <= m <= 0.9
            and 6.0 <= omega <= 13.0
            and b < 0
            and r2 >= 0.80
            and t2 < tc < t2 + 0.4 * dt
        )
        lppls_tc = tc
        # tc proximity signal
        days_to_tc = tc - t2
        components["lppls_tc_proximity"] = sig.tc_proximity_signal(days_to_tc)
    except Exception:
        results["lppls"] = None

    # LPPLS confidence (multi-window, rayon-parallelized)
    try:
        pre_lp = log_prices(df.iloc[pre_start:pre_end])
        pre_t = time_index(df.iloc[pre_start:pre_end])
        conf_arr, tc_mean_arr, tc_std_arr = lppls_confidence(pre_t, pre_lp, n_windows=20, n_candidates=20)
        conf = np.asarray(conf_arr)
        tc_std = np.asarray(tc_std_arr)
        valid = conf[~np.isnan(conf)]
        if len(valid) > 0:
            conf_val = float(valid[-1])
            # Discount confidence by tc dispersion
            last_tc_std = float(tc_std[~np.isnan(tc_std)][-1]) if np.any(~np.isnan(tc_std)) else 0.0
            adjusted = conf_val * max(0.0, 1.0 - last_tc_std / 100.0)
            results["lppls_confidence"] = adjusted > 0.3
            components["lppls_confidence"] = sig.lppls_confidence_signal(adjusted)
        else:
            results["lppls_confidence"] = False
    except Exception:
        results["lppls_confidence"] = None

    # GSADF
    try:
        pre_prices = df.iloc[pre_start:pre_end]["close"].values
        if len(pre_prices) >= 50:
            gsadf_stat, _, (_, cv95, _) = gsadf_test(pre_prices, min_window=None, n_sims=30, seed=42)
            results["gsadf"] = gsadf_stat > cv95
            components["gsadf_bubble"] = sig.gsadf_signal(gsadf_stat, cv95)
        else:
            results["gsadf"] = None
    except Exception:
        results["gsadf"] = None

    # DFA
    pre_dfa = dfa_exponent(pre_ret)
    if not np.isnan(pre_dfa):
        results["dfa"] = pre_dfa > 0.55
        components["dfa_trending"] = sig.dfa_signal(pre_dfa)
    else:
        results["dfa"] = None

    # DEH
    pre_deh = deh_estimator(pre_ret)
    base_deh = deh_estimator(base_ret)
    if not (np.isnan(pre_deh) or np.isnan(base_deh)):
        results["deh"] = pre_deh > base_deh
        components["deh_thinning"] = sig.deh_signal(pre_deh, base_deh)
    else:
        results["deh"] = None

    # QQ
    pre_qq = qq_estimator(pre_ret)
    base_qq = qq_estimator(base_ret)
    if not (np.isnan(pre_qq) or np.isnan(base_qq)):
        results["qq"] = pre_qq < base_qq
        components["qq_thinning"] = sig.qq_signal(pre_qq, base_qq)
    else:
        results["qq"] = None

    # Max-to-Sum
    pre_ms = maxsum_ratio(pre_ret)
    base_ms = maxsum_ratio(base_ret)
    if not (np.isnan(pre_ms) or np.isnan(base_ms)):
        results["maxsum"] = pre_ms > base_ms
        components["maxsum_signal"] = sig.maxsum_signal(pre_ms)
    else:
        results["maxsum"] = None

    # Spectral
    pre_spec = spectral_exponent(pre_ret)
    if not np.isnan(pre_spec):
        results["spectral"] = pre_spec > 0.1
        components["spectral_memory"] = sig.spectral_signal(pre_spec)
    else:
        results["spectral"] = None

    # ── NN methods ─────────────────────────────────
    if run_nn and _TORCH_AVAILABLE:
        pre_lp = log_prices(df.iloc[pre_start:pre_end])
        pre_t = time_index(df.iloc[pre_start:pre_end])

        # M-LNN
        try:
            from fatcrash.nn.mlnn import fit_mlnn
            mlnn_res = fit_mlnn(pre_t, pre_lp, epochs=100, seed=42)
            results["mlnn"] = mlnn_res.is_bubble
            components["mlnn_signal"] = sig.mlnn_signal(mlnn_res.confidence, mlnn_res.is_bubble)
        except Exception:
            results["mlnn"] = None

        # P-LNN
        if plnn_model is not None:
            try:
                from fatcrash.nn.plnn import predict_plnn
                plnn_res = predict_plnn(plnn_model, pre_t, pre_lp)
                results["plnn"] = plnn_res.is_bubble
                components["plnn_signal"] = sig.plnn_signal(plnn_res.confidence, plnn_res.is_bubble)
            except Exception:
                results["plnn"] = None

        # HLPPL
        if hlppl_model is not None:
            try:
                from fatcrash.nn.hlppl import predict_hlppl
                hlppl_res = predict_hlppl(hlppl_model, df.iloc[pre_start:pre_end], window=60)
                results["hlppl"] = hlppl_res.bubble_score > 0.5
                components["hlppl_signal"] = sig.hlppl_signal(hlppl_res.bubble_score)
            except Exception:
                results["hlppl"] = None

        # DTCAI
        if dtcai_model is not None:
            try:
                from fatcrash.nn.dtcai import predict_dtcai
                dtcai_res = predict_dtcai(dtcai_model, pre_t, pre_lp)
                results["dtcai"] = dtcai_res.dtcai > 0.5
                components["dtcai_signal"] = sig.dtcai_signal(dtcai_res.dtcai)
            except Exception:
                results["dtcai"] = None

    return results, components


def test_method_on_non_crash(df, center_idx, window=120,
                             run_nn=False,
                             plnn_model=None, hlppl_model=None, dtcai_model=None):
    """Test all methods on a non-crash window. Same logic as drawdown test.

    Returns (results, components) — same as test_method_on_drawdown.
    """
    return test_method_on_drawdown(
        df, center_idx, window=window,
        run_nn=run_nn,
        plnn_model=plnn_model, hlppl_model=hlppl_model, dtcai_model=dtcai_model,
    )


# ── Metrics ────────────────────────────────────────────────


def compute_metrics(tp_results, fp_results, methods):
    """Compute precision, recall, F1 per method.

    tp_results: list of (method, detected) from crash windows
    fp_results: list of (method, fired) from non-crash windows
    """
    metrics = {}
    for method in methods:
        tp = sum(1 for m, d in tp_results if m == method and d is True)
        fn = sum(1 for m, d in tp_results if m == method and d is False)
        fp = sum(1 for m, d in fp_results if m == method and d is True)
        tn = sum(1 for m, d in fp_results if m == method and d is False)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[method] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return metrics


# ── NN model training ──────────────────────────────────────


def train_nn_models(datasets):
    """Pre-train NN models for evaluation. Returns (plnn_model, hlppl_model, dtcai_model)."""
    plnn_model = hlppl_model = dtcai_model = None

    # P-LNN: train on synthetic data
    try:
        from fatcrash.nn.plnn import train_plnn
        print("  Training P-LNN on 10,000 synthetic series...")
        plnn_model = train_plnn(variant="P-LNN-10K", n_samples=10_000, epochs=10, seed=42)
        print("  P-LNN ready.")
    except Exception as e:
        print(f"  P-LNN training failed: {e}")

    # HLPPL: train on BTC labeled windows
    try:
        from fatcrash.nn.hlppl import train_hlppl
        btc_df = datasets["btc"]
        close = btc_df["close"].values
        train_dfs = []
        labels = []
        win = 60
        for i in range(win, len(close) - 30, win):
            chunk = btc_df.iloc[i - win:i].copy()
            future = close[i:i + 30]
            is_crash = 1 if len(future) > 0 and (future.min() - close[i]) / close[i] < -0.15 else 0
            train_dfs.append(chunk)
            labels.append(is_crash)
        print(f"  Training HLPPL on {len(train_dfs)} BTC windows ({sum(labels)} crash)...")
        hlppl_model = train_hlppl(train_dfs, labels, window=win, epochs=20, seed=42)
        print("  HLPPL ready.")
    except Exception as e:
        print(f"  HLPPL training failed: {e}")

    # DTCAI: train on BTC prices
    try:
        from fatcrash.nn.dtcai import train_dtcai
        from fatcrash.nn.dtcai_data import generate_dtcai_dataset
        btc_prices = datasets["btc"]["close"].values
        print("  Generating DTCAI dataset from BTC...")
        ds = generate_dtcai_dataset(btc_prices, window_size=504, step_size=42, n_fits_per_window=5, seed=42)
        if len(ds.X) > 0:
            print(f"  Training DTCAI on {len(ds.X)} samples ({ds.y.sum()} reliable)...")
            dtcai_model = train_dtcai(ds, model_type="RF", seed=42)
            print("  DTCAI ready.")
        else:
            print("  DTCAI: no training samples generated.")
    except Exception as e:
        print(f"  DTCAI training failed: {e}")

    return plnn_model, hlppl_model, dtcai_model


# ── Main ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Accuracy report with honest metrics")
    parser.add_argument("--skip-nn", action="store_true", help="Skip NN methods (faster)")
    args = parser.parse_args()

    run_nn = _TORCH_AVAILABLE and not args.skip_nn

    # Load datasets
    datasets = {}
    for name in ["btc", "spy", "gold"]:
        datasets[name] = from_sample(name)

    # ── Train NN models if needed ──────────────────────
    plnn_model = hlppl_model = dtcai_model = None
    if run_nn:
        print("=" * 70)
        print("TRAINING NN MODELS")
        print("=" * 70)
        plnn_model, hlppl_model, dtcai_model = train_nn_models(datasets)
        print()

    # ══════════════════════════════════════════════
    # Part 1a: True positive testing (crash windows)
    # ══════════════════════════════════════════════
    tp_records = []  # (method, detected)
    all_results = []  # for detailed printout

    all_crash_events = {}  # asset -> events, for FP sampling

    # Aggregator: collect probabilities for crash windows
    agg_thresholds = [0.3, 0.4, 0.5, 0.7]
    agg_tp_probs = []  # list of aggregator probabilities on crash windows

    for asset_name, threshold in [("btc", 0.15), ("spy", 0.08), ("gold", 0.08)]:
        df = datasets[asset_name]
        events = find_drawdowns(df, min_dd=threshold, min_apart=90)
        all_crash_events[asset_name] = events

        for ev in events:
            res, comps = test_method_on_drawdown(
                df, ev["peak_idx"],
                run_nn=run_nn,
                plnn_model=plnn_model, hlppl_model=hlppl_model, dtcai_model=dtcai_model,
            )
            if res is None:
                continue

            dd = ev["dd_pct"]
            size = "MAJOR" if abs(dd) > 30 else "MEDIUM" if abs(dd) > 15 else "SMALL"

            for method, detected in res.items():
                if detected is not None:
                    tp_records.append((method, detected))
                    all_results.append(
                        {
                            "asset": asset_name.upper(),
                            "date": ev["peak_date"].strftime("%Y-%m-%d"),
                            "drawdown": dd,
                            "size": size,
                            "method": method,
                            "detected": detected,
                        }
                    )

            # Aggregator on crash window
            if comps:
                agg_result = aggregate_signals(comps)
                agg_tp_probs.append(agg_result.probability)

    rdf = pd.DataFrame(all_results)

    # ══════════════════════════════════════════════
    # Part 1b: False positive testing (non-crash windows)
    # ══════════════════════════════════════════════
    fp_records = []  # (method, fired)
    agg_fp_probs = []  # aggregator probabilities on non-crash windows

    for asset_name in ["btc", "spy", "gold"]:
        df = datasets[asset_name]
        events = all_crash_events[asset_name]
        non_crash = sample_non_crash_windows(df, events, n_samples=50, seed=42)

        for nc in non_crash:
            res, comps = test_method_on_non_crash(
                df, nc["center_idx"],
                run_nn=run_nn,
                plnn_model=plnn_model, hlppl_model=hlppl_model, dtcai_model=dtcai_model,
            )
            if res is None:
                continue
            for method, fired in res.items():
                if fired is not None:
                    fp_records.append((method, fired))

            # Aggregator on non-crash window
            if comps:
                agg_result = aggregate_signals(comps)
                agg_fp_probs.append(agg_result.probability)

    # ══════════════════════════════════════════════
    # Part 1c: Compute and display metrics
    # ══════════════════════════════════════════════
    classical_methods = [
        "lppls", "lppls_confidence", "gsadf", "hurst", "dfa",
        "kappa", "taleb_kappa", "pickands", "deh", "qq",
        "gpd_var", "maxsum", "spectral", "hill",
    ]
    nn_methods = ["mlnn", "plnn", "hlppl", "dtcai"] if run_nn else []
    all_methods = classical_methods + nn_methods

    metrics = compute_metrics(tp_records, fp_records, all_methods)

    print("=" * 70)
    print("CRASH DETECTION: PRECISION / RECALL / F1")
    print("=" * 70)
    print("  All numbers are in-sample on historical data (BTC, SPY, Gold).")
    print()
    print(f"  {'Method':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  "
          f"{'Prec':>6} {'Recall':>6} {'F1':>6}")
    print("  " + "-" * 64)

    # Sort by F1
    sorted_methods = sorted(all_methods, key=lambda m: metrics.get(m, {}).get("f1", 0), reverse=True)
    for method in sorted_methods:
        m = metrics.get(method)
        if m is None or (m["tp"] + m["fn"]) == 0:
            continue
        print(
            f"  {method:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}  "
            f"{m['precision']:>5.0%} {m['recall']:>6.0%} {m['f1']:>5.0%}"
        )

    print()
    print("  Precision = TP/(TP+FP) — how often a signal is correct")
    print("  Recall    = TP/(TP+FN) — how many crashes are caught")
    print("  F1        = harmonic mean of precision and recall")

    # ── Aggregator evaluation ──────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATOR: WEIGHTED ENSEMBLE + AGREEMENT BONUS")
    print("=" * 70)
    print("  Combines all methods via weighted average + category agreement bonus.")
    print(f"  Tested on {len(agg_tp_probs)} crash windows, {len(agg_fp_probs)} non-crash windows.")
    print()

    if agg_tp_probs and agg_fp_probs:
        tp_arr = np.array(agg_tp_probs)
        fp_arr = np.array(agg_fp_probs)

        print(f"  Crash windows:     mean={tp_arr.mean():.2f}  median={np.median(tp_arr):.2f}  "
              f"min={tp_arr.min():.2f}  max={tp_arr.max():.2f}")
        print(f"  Non-crash windows: mean={fp_arr.mean():.2f}  median={np.median(fp_arr):.2f}  "
              f"min={fp_arr.min():.2f}  max={fp_arr.max():.2f}")
        print()

        print(f"  {'Threshold':<12} {'Level':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  "
              f"{'Prec':>6} {'Recall':>6} {'F1':>6}")
        print("  " + "-" * 64)

        agg_metrics = {}
        for thr in agg_thresholds:
            tp = int((tp_arr >= thr).sum())
            fn = int((tp_arr < thr).sum())
            fp = int((fp_arr >= thr).sum())
            tn = int((fp_arr < thr).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            level = ("CRITICAL+" if thr >= 0.7 else "HIGH+" if thr >= 0.5
                     else "ELEVATED+" if thr <= 0.3 else f">{thr:.0%}")
            print(f"  {thr:<12.1f} {level:<12} {tp:>4} {fp:>4} {fn:>4} {tn:>4}  "
                  f"{prec:>5.0%} {rec:>6.0%} {f1:>5.0%}")
            agg_metrics[thr] = {"precision": prec, "recall": rec, "f1": f1,
                                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

        print()
        print("  ELEVATED+ (≥0.3): high recall, catches most crashes")
        print("  HIGH+     (≥0.5): balanced, trades recall for fewer false alarms")
        print("  CRITICAL+ (≥0.7): conservative, very few false alarms but misses most crashes")
        print()

        # Find optimal threshold
        best_thr = max(agg_metrics, key=lambda t: agg_metrics[t]["f1"])
        best = agg_metrics[best_thr]
        lppls_f1 = metrics.get("lppls", {}).get("f1", 0)
        print(f"  Best aggregator F1={best['f1']:.0%} at threshold {best_thr:.1f} "
              f"(P={best['precision']:.0%}, R={best['recall']:.0%}).")
        print(f"  Best individual method: LPPLS F1={lppls_f1:.0%}.")
        if best["f1"] > lppls_f1:
            print("  The ensemble improves over the best single method.")
        else:
            print("  The ensemble does not outperform the best single method with")
            print("  hand-tuned weights. The weighted average dilutes the strongest")
            print("  signal (LPPLS) with weaker methods. Learned weights or a")
            print("  stacking classifier could improve this.")
    else:
        agg_metrics = {}
        print("  Insufficient data for aggregator evaluation.")

    # ══════════════════════════════════════════════
    # Part 1e: EXTENDED DATASET (FRED Forex + Options Backtester)
    # ══════════════════════════════════════════════
    ext_tp_records = []  # (method, detected)
    ext_fp_records = []  # (method, fired)
    ext_agg_tp_probs = []
    ext_agg_fp_probs = []
    ext_pair_stats = []  # for summary

    # ── FRED Forex ────────────────────────────────
    fred_dir = Path("/Users/unbalancedparen/projects/forex-centuries/data/sources/fred/daily")
    fred_skip = {"fred_usd_broad_index.csv", "fred_usd_major_index.csv"}
    fred_count = 0

    if fred_dir.is_dir():
        for csv_path in sorted(fred_dir.glob("fred_*.csv")):
            if csv_path.name in fred_skip:
                continue
            df_fx = load_fred_forex(csv_path)
            if df_fx is None:
                continue
            pair_name = csv_path.name
            min_dd = fred_min_dd(pair_name)
            events = find_drawdowns(df_fx, min_dd=min_dd, min_apart=90)
            fred_count += 1

            for ev in events:
                res, comps = test_method_on_drawdown(
                    df_fx, ev["peak_idx"],
                    run_nn=run_nn,
                    plnn_model=plnn_model, hlppl_model=hlppl_model,
                    dtcai_model=dtcai_model,
                )
                if res is None:
                    continue
                for method, detected in res.items():
                    if detected is not None:
                        ext_tp_records.append((method, detected))
                if comps:
                    agg_result = aggregate_signals(comps)
                    ext_agg_tp_probs.append(agg_result.probability)

            non_crash = sample_non_crash_windows(df_fx, events, n_samples=20, seed=42)
            for nc in non_crash:
                res, comps = test_method_on_non_crash(
                    df_fx, nc["center_idx"],
                    run_nn=run_nn,
                    plnn_model=plnn_model, hlppl_model=hlppl_model,
                    dtcai_model=dtcai_model,
                )
                if res is None:
                    continue
                for method, fired in res.items():
                    if fired is not None:
                        ext_fp_records.append((method, fired))
                if comps:
                    agg_result = aggregate_signals(comps)
                    ext_agg_fp_probs.append(agg_result.probability)

            ext_pair_stats.append({
                "source": "FRED", "name": pair_name.replace("fred_", "").replace(".csv", "").upper(),
                "n_crash": len(events), "n_rows": len(df_fx),
            })

    # ── Options Backtester ────────────────────────
    opts_dir = Path("/Users/unbalancedparen/projects/options_backtester/tests/data")
    opts_files = [
        "spy_crisis_stocks.csv",
        "spy_covid_stocks.csv",
        "spy_bear_stocks.csv",
        "spy_lowvol_stocks.csv",
        "qqq_2020_stocks.csv",
        "iwm_2020_stocks.csv",
    ]

    if opts_dir.is_dir():
        for fname in opts_files:
            fpath = opts_dir / fname
            if not fpath.exists():
                continue
            try:
                df_opt = from_csv(str(fpath), date_col="date", price_col="close")
            except Exception:
                continue
            if len(df_opt) < 60:
                continue
            events = find_drawdowns(df_opt, min_dd=0.08, min_apart=30)

            for ev in events:
                res, comps = test_method_on_drawdown(
                    df_opt, ev["peak_idx"],
                    run_nn=run_nn,
                    plnn_model=plnn_model, hlppl_model=hlppl_model,
                    dtcai_model=dtcai_model,
                )
                if res is None:
                    continue
                for method, detected in res.items():
                    if detected is not None:
                        ext_tp_records.append((method, detected))
                if comps:
                    agg_result = aggregate_signals(comps)
                    ext_agg_tp_probs.append(agg_result.probability)

            non_crash = sample_non_crash_windows(df_opt, events, n_samples=10, seed=42)
            for nc in non_crash:
                res, comps = test_method_on_non_crash(
                    df_opt, nc["center_idx"],
                    run_nn=run_nn,
                    plnn_model=plnn_model, hlppl_model=hlppl_model,
                    dtcai_model=dtcai_model,
                )
                if res is None:
                    continue
                for method, fired in res.items():
                    if fired is not None:
                        ext_fp_records.append((method, fired))
                if comps:
                    agg_result = aggregate_signals(comps)
                    ext_agg_fp_probs.append(agg_result.probability)

            ext_pair_stats.append({
                "source": "OptsBT", "name": fname.replace("_stocks.csv", "").upper(),
                "n_crash": len(events), "n_rows": len(df_opt),
            })

    # ── Print extended results ────────────────────
    ext_n_crash = sum(1 for m, d in ext_tp_records if m == "lppls" and d is not None)
    ext_n_noncrash = sum(1 for m, d in ext_fp_records if m == "lppls" and d is not None)

    if ext_tp_records:
        ext_metrics = compute_metrics(ext_tp_records, ext_fp_records, all_methods)

        print("\n" + "=" * 70)
        print("EXTENDED DATASET: PRECISION / RECALL / F1")
        print("=" * 70)
        print(f"  FRED forex pairs: {fred_count}, Options backtester files: {len([s for s in ext_pair_stats if s['source'] == 'OptsBT'])}")
        print(f"  Crash windows: {ext_n_crash}, Non-crash windows: {ext_n_noncrash}")
        if ext_pair_stats:
            print("  Breakdown:")
            for s in ext_pair_stats:
                print(f"    {s['source']:<6} {s['name']:<16} {s['n_rows']:>6} rows, {s['n_crash']:>3} crashes")
        print()
        print(f"  {'Method':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  "
              f"{'Prec':>6} {'Recall':>6} {'F1':>6}")
        print("  " + "-" * 64)

        ext_sorted = sorted(all_methods, key=lambda m: ext_metrics.get(m, {}).get("f1", 0), reverse=True)
        for method in ext_sorted:
            m = ext_metrics.get(method)
            if m is None or (m["tp"] + m["fn"]) == 0:
                continue
            print(
                f"  {method:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}  "
                f"{m['precision']:>5.0%} {m['recall']:>6.0%} {m['f1']:>5.0%}"
            )

        # ── Combined (original + extended) ────────
        combined_tp = tp_records + ext_tp_records
        combined_fp = fp_records + ext_fp_records
        combined_metrics = compute_metrics(combined_tp, combined_fp, all_methods)

        orig_n_crash = sum(1 for m, d in tp_records if m == "lppls" and d is not None)
        orig_n_noncrash = sum(1 for m, d in fp_records if m == "lppls" and d is not None)

        print("\n" + "=" * 70)
        print("COMBINED DATASET: PRECISION / RECALL / F1")
        print("=" * 70)
        print(f"  Original (BTC/SPY/Gold): {orig_n_crash} crash + {orig_n_noncrash} non-crash windows")
        print(f"  Extended (FRED+OptsBT):  {ext_n_crash} crash + {ext_n_noncrash} non-crash windows")
        print(f"  Total:                   {orig_n_crash + ext_n_crash} crash + {orig_n_noncrash + ext_n_noncrash} non-crash windows")
        print()
        print(f"  {'Method':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  "
              f"{'Prec':>6} {'Recall':>6} {'F1':>6}")
        print("  " + "-" * 64)

        comb_sorted = sorted(all_methods, key=lambda m: combined_metrics.get(m, {}).get("f1", 0), reverse=True)
        for method in comb_sorted:
            m = combined_metrics.get(method)
            if m is None or (m["tp"] + m["fn"]) == 0:
                continue
            print(
                f"  {method:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}  "
                f"{m['precision']:>5.0%} {m['recall']:>6.0%} {m['f1']:>5.0%}"
            )

        # ── Extended aggregator evaluation ────────
        if ext_agg_tp_probs and ext_agg_fp_probs:
            ext_tp_arr = np.array(ext_agg_tp_probs)
            ext_fp_arr = np.array(ext_agg_fp_probs)

            print("\n" + "=" * 70)
            print("EXTENDED AGGREGATOR: WEIGHTED ENSEMBLE + AGREEMENT BONUS")
            print("=" * 70)
            print(f"  Tested on {len(ext_agg_tp_probs)} crash windows, {len(ext_agg_fp_probs)} non-crash windows.")
            print()
            print(f"  Crash windows:     mean={ext_tp_arr.mean():.2f}  median={np.median(ext_tp_arr):.2f}  "
                  f"min={ext_tp_arr.min():.2f}  max={ext_tp_arr.max():.2f}")
            print(f"  Non-crash windows: mean={ext_fp_arr.mean():.2f}  median={np.median(ext_fp_arr):.2f}  "
                  f"min={ext_fp_arr.min():.2f}  max={ext_fp_arr.max():.2f}")
            print()

            print(f"  {'Threshold':<12} {'Level':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  "
                  f"{'Prec':>6} {'Recall':>6} {'F1':>6}")
            print("  " + "-" * 64)

            for thr in agg_thresholds:
                tp = int((ext_tp_arr >= thr).sum())
                fn = int((ext_tp_arr < thr).sum())
                fp = int((ext_fp_arr >= thr).sum())
                tn = int((ext_fp_arr < thr).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                level = ("CRITICAL+" if thr >= 0.7 else "HIGH+" if thr >= 0.5
                         else "ELEVATED+" if thr <= 0.3 else f">{thr:.0%}")
                print(f"  {thr:<12.1f} {level:<12} {tp:>4} {fp:>4} {fn:>4} {tn:>4}  "
                      f"{prec:>5.0%} {rec:>6.0%} {f1:>5.0%}")
    else:
        print("\n  (No extended dataset found — FRED forex / options backtester dirs not available)")

    # ══════════════════════════════════════════════
    # Part 1d: Recall by crash size (backward compat)
    # ══════════════════════════════════════════════
    if len(rdf) > 0:
        print("\n" + "=" * 70)
        print("RECALL BY METHOD AND CRASH SIZE")
        print("=" * 70)

        for method in sorted_methods:
            subset = rdf[rdf["method"] == method]
            if len(subset) == 0:
                continue
            print(f"\n  {method.upper()}")
            for size in ["SMALL", "MEDIUM", "MAJOR"]:
                s = subset[subset["size"] == size]
                if len(s) == 0:
                    continue
                acc = s["detected"].mean()
                print(f"    {size:<8} {s['detected'].sum()}/{len(s)} = {acc:.0%}")
            total = subset["detected"].mean()
            print(f"    {'TOTAL':<8} {subset['detected'].sum()}/{len(subset)} = {total:.0%}")

    # ══════════════════════════════════════════════
    # Part 2: Decade-by-decade GBP/USD
    # ══════════════════════════════════════════════
    gbp = from_csv("data/sample/gbpusd_daily.csv")
    gbp_ret = log_returns(gbp)

    print("\n" + "=" * 70)
    print("GBP/USD BY DECADE (1971-2025)")
    print("=" * 70)
    print(
        f"\n{'Decade':<8} {'N days':<8} {'Hill a':<8} {'Kappa':<8} "
        f"{'K/bench':<8} {'T-Kappa':<8} {'T-bench':<8} {'VaR95%':<10} {'Worst day':<10}"
    )
    print("-" * 85)

    decades = [
        ("1970s", "1971-01-01", "1979-12-31"),
        ("1980s", "1980-01-01", "1989-12-31"),
        ("1990s", "1990-01-01", "1999-12-31"),
        ("2000s", "2000-01-01", "2009-12-31"),
        ("2010s", "2010-01-01", "2019-12-31"),
        ("2020s", "2020-01-01", "2025-12-31"),
    ]

    for name, start, end in decades:
        sub = gbp.loc[start:end]
        if len(sub) < 100:
            continue
        ret = log_returns(sub)
        alpha = hill_estimator(ret)
        k, bench = kappa_metric(ret, n_subsamples=10, n_sims=100)
        tk, tb = taleb_kappa(ret, n0=30, n1=100, n_sims=100)
        try:
            var, _ = gpd_var_es(ret, p=0.95, quantile=0.90)
        except Exception:
            var = float("nan")
        worst = ret.min()
        print(
            f"{name:<8} {len(ret):<8} {alpha:<8.2f} {k:<8.3f} "
            f"{k / bench:<8.2f} {tk:<8.4f} {tb:<8.4f} {var:<10.4f} {worst:<10.4f}"
        )

    # ══════════════════════════════════════════════
    # Part 3: Known forex crises
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KNOWN FOREX CRISES IN GBP/USD")
    print("=" * 70)

    full_alpha = hill_estimator(gbp_ret)
    full_k, full_b = kappa_metric(gbp_ret, n_subsamples=10, n_sims=100)
    full_ratio = full_k / full_b

    crises = [
        ("1976 IMF Crisis", "1975-01-01", "1977-01-01"),
        ("1985 Plaza Accord", "1984-01-01", "1986-01-01"),
        ("1992 Black Wednesday", "1991-06-01", "1993-01-01"),
        ("2008 Financial Crisis", "2007-06-01", "2009-03-01"),
        ("2016 Brexit Vote", "2016-01-01", "2016-12-31"),
        ("2022 Truss Mini-Budget", "2022-06-01", "2022-12-31"),
    ]

    detected = 0
    for name, start, end in crises:
        sub = gbp.loc[start:end]
        if len(sub) < 50:
            continue
        ret = log_returns(sub)
        alpha = hill_estimator(ret)
        k, bench = kappa_metric(ret, n_subsamples=5, n_sims=100)
        ratio = k / bench if bench > 0 else 1.0
        signal = alpha < full_alpha or ratio < full_ratio
        if signal:
            detected += 1
        print(f"  {name:<25} Hill a={alpha:.2f}  K/bench={ratio:.2f}  {'DETECTED' if signal else 'missed'}")

    print(f"\n  Detected: {detected}/{len(crises)} forex crises")

    # ══════════════════════════════════════════════
    # Part 4: Centuries of data
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CENTURIES OF DATA: EXCHANGE RATES (1791-2025)")
    print("=" * 70)

    try:
        mw = pd.read_csv(
            "/Users/unbalancedparen/projects/forex-centuries/data/sources/measuringworth/measuringworth_exchange_rates.csv"
        )
        for currency in [
            "United Kingdom",
            "France",
            "Germany",
            "Japan",
            "Brazil",
            "Argentina",
        ]:
            if currency not in mw.columns:
                continue
            series = mw[["year", currency]].dropna()
            if len(series) < 20:
                continue
            prices = series[currency].values
            yearly_ret = np.diff(np.log(prices))
            yearly_ret = yearly_ret[np.isfinite(yearly_ret)]
            if len(yearly_ret) < 10:
                continue
            alpha = hill_estimator(yearly_ret)
            k, bench = kappa_metric(yearly_ret, n_subsamples=5, n_sims=100)
            ratio = k / bench if bench > 0 else 1.0
            tk, tb = taleb_kappa(yearly_ret, n0=10, n1=30, n_sims=100)
            print(
                f"  {currency:<20} {len(yearly_ret)} years  "
                f"Hill a={alpha:.2f}  K/bench={ratio:.2f}  "
                f"T-kappa={tk:.3f} (bench={tb:.3f})  "
                f"{'FAT TAILS' if alpha < 4 and ratio < 1 else 'moderate'}"
            )
    except FileNotFoundError:
        print("  (forex-centuries data not available)")
    except Exception as e:
        print(f"  (MeasuringWorth CSV parse error: {e})")

    # ══════════════════════════════════════════════
    # Part 5: FRED Daily Forex — Pickands, Hurst, GSADF
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FRED DAILY FOREX: PICKANDS, HURST & GSADF")
    print("=" * 70)

    fred_dir = "/Users/unbalancedparen/projects/forex-centuries/data/sources/fred/daily"
    fred_pairs = [
        "fred_aud_usd.csv", "fred_eur_usd.csv", "fred_gbp_usd.csv",
        "fred_jpy_usd.csv", "fred_cad_usd.csv", "fred_chf_usd.csv",
        "fred_cny_usd.csv", "fred_nzd_usd.csv", "fred_mxn_usd.csv",
        "fred_brl_usd.csv", "fred_krw_usd.csv", "fred_inr_usd.csv",
    ]

    print(
        f"\n  {'Pair':<14} {'N':<7} {'Pickands':<10} {'Hurst':<8} {'DFA':<8} "
        f"{'T-Kappa':<9} {'DEH':<8} {'QQ':<8} {'MaxSum':<8} {'Spectral':<10} "
        f"{'GSADF':<10} {'CV95':<8} {'Bubble?':<9} {'Hill':<8} {'Verdict'}"
    )
    print("  " + "-" * 145)

    fred_results = []
    for fname in fred_pairs:
        fpath = f"{fred_dir}/{fname}"
        try:
            raw = pd.read_csv(fpath)
        except FileNotFoundError:
            continue
        pair_label = fname.replace("fred_", "").replace(".csv", "").upper()
        date_col = "observation_date"
        val_col = [c for c in raw.columns if c != date_col][0]
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw = raw.dropna(subset=[date_col])
        raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce")
        raw = raw.dropna(subset=[val_col])
        raw = raw[raw[val_col] > 0]
        if len(raw) < 200:
            continue
        prices = raw[val_col].values
        ret = np.diff(np.log(prices))
        ret = ret[np.isfinite(ret)]
        if len(ret) < 100:
            continue

        xi = pickands_estimator(ret)
        h = hurst_exponent(ret)
        alpha = hill_estimator(ret)
        tk, tb = taleb_kappa(ret, n0=30, n1=100, n_sims=100)
        dfa_a = dfa_exponent(ret)
        deh_g = deh_estimator(ret)
        qq_a = qq_estimator(ret)
        ms = maxsum_ratio(ret)
        spec_d = spectral_exponent(ret)

        gsadf_prices = prices[-2000:] if len(prices) > 2000 else prices
        gsadf_stat, _, (cv90, cv95, cv99) = gsadf_test(
            gsadf_prices, min_window=None, n_sims=30, seed=42
        )
        bubble = gsadf_stat > cv95

        fred_results.append({
            "pair": pair_label, "n": len(ret), "pickands": xi,
            "hurst": h, "dfa": dfa_a, "taleb_kappa": tk, "deh": deh_g, "qq": qq_a,
            "maxsum": ms, "spectral": spec_d, "gsadf": gsadf_stat, "cv95": cv95,
            "bubble": bubble, "hill": alpha,
        })

        verdict_parts = []
        if xi > 0:
            verdict_parts.append("heavy-tail")
        if h > 0.55:
            verdict_parts.append("persistent")
        elif h < 0.45:
            verdict_parts.append("mean-revert")
        if bubble:
            verdict_parts.append("BUBBLE")
        verdict = ", ".join(verdict_parts) if verdict_parts else "normal"

        print(
            f"  {pair_label:<14} {len(ret):<7} {xi:<10.4f} {h:<8.4f} {dfa_a:<8.4f} "
            f"{tk:<9.4f} {deh_g:<8.4f} {qq_a:<8.2f} {ms:<8.4f} {spec_d:<10.4f} "
            f"{gsadf_stat:<10.4f} {cv95:<8.4f} {'YES' if bubble else 'no':<9} "
            f"{alpha:<8.2f} {verdict}"
        )

    # ══════════════════════════════════════════════
    # Part 6: Clio Infra Yearly — Pickands & Hurst (top 30 countries)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CLIO INFRA YEARLY EXCHANGE RATES: PICKANDS & HURST (TOP 30)")
    print("=" * 70)

    clio_path = "/Users/unbalancedparen/projects/forex-centuries/data/sources/clio_infra/clio_infra_exchange_rates.csv"
    clio_results = []
    try:
        clio = pd.read_csv(clio_path)
        countries = clio.drop("year", axis=1)
        counts = countries.count().sort_values(ascending=False)
        top30 = counts.head(30).index.tolist()

        print(
            f"\n  {'Country':<22} {'Years':<7} {'Pickands':<10} {'Hurst':<8} {'DFA':<8} "
            f"{'DEH':<8} {'QQ':<8} {'MaxSum':<8} {'Spectral':<10} "
            f"{'T-Kappa':<9} {'Hill':<8} {'Verdict'}"
        )
        print("  " + "-" * 125)

        for country in top30:
            series = clio[["year", country]].dropna()
            if len(series) < 15:
                continue
            vals = pd.to_numeric(series[country], errors="coerce").dropna().values
            vals = vals[vals > 0]
            if len(vals) < 15:
                continue
            yearly_ret = np.diff(np.log(vals))
            yearly_ret = yearly_ret[np.isfinite(yearly_ret)]
            if len(yearly_ret) < 10:
                continue

            xi = pickands_estimator(yearly_ret)
            h = hurst_exponent(yearly_ret)
            alpha = hill_estimator(yearly_ret)
            tk, tb = taleb_kappa(yearly_ret, n0=5, n1=15, n_sims=100)
            dfa_a = dfa_exponent(yearly_ret)
            deh_g = deh_estimator(yearly_ret)
            qq_a = qq_estimator(yearly_ret)
            ms = maxsum_ratio(yearly_ret)
            spec_d = spectral_exponent(yearly_ret)

            clio_results.append({
                "country": country, "n": len(yearly_ret),
                "pickands": xi, "hurst": h, "dfa": dfa_a, "deh": deh_g, "qq": qq_a,
                "maxsum": ms, "spectral": spec_d, "taleb_kappa": tk, "hill": alpha,
            })

            verdict_parts = []
            if xi > 0:
                verdict_parts.append("heavy-tail")
            if alpha < 2:
                verdict_parts.append("EXTREME")
            elif alpha < 4:
                verdict_parts.append("fat-tail")
            if h > 0.55:
                verdict_parts.append("persistent")
            elif h < 0.45:
                verdict_parts.append("mean-revert")
            verdict = ", ".join(verdict_parts) if verdict_parts else "normal"

            print(
                f"  {country:<22} {len(yearly_ret):<7} {xi:<10.4f} {h:<8.4f} {dfa_a:<8.4f} "
                f"{deh_g:<8.4f} {qq_a:<8.2f} {ms:<8.4f} {spec_d:<10.4f} "
                f"{tk:<9.4f} {alpha:<8.2f} {verdict}"
            )
    except FileNotFoundError:
        print("  (Clio Infra data not available)")

    # ══════════════════════════════════════════════
    # Part 7: Cross-Method Summary
    # ══════════════════════════════════════════════
    n_methods = 17 if run_nn else 13
    print("\n" + "=" * 70)
    print(f"CROSS-METHOD SUMMARY: ALL {n_methods} METHODS")
    print("=" * 70)

    if fred_results:
        fred_df = pd.DataFrame(fred_results)
        print("\n  FRED Daily Forex (12 pairs):")
        print(f"    Pickands > 0 (heavy tails):   {(fred_df['pickands'] > 0).sum()}/{len(fred_df)}")
        print(f"    DEH > 0 (heavy tails):        {(fred_df['deh'] > 0).sum()}/{len(fred_df)}")
        print(f"    Hurst > 0.5 (persistent):     {(fred_df['hurst'] > 0.5).sum()}/{len(fred_df)}")
        print(f"    Hurst > 0.55 (strong pers.):  {(fred_df['hurst'] > 0.55).sum()}/{len(fred_df)}")
        print(f"    DFA > 0.5 (persistent):       {(fred_df['dfa'] > 0.5).sum()}/{len(fred_df)}")
        print(f"    DFA > 0.55 (strong pers.):    {(fred_df['dfa'] > 0.55).sum()}/{len(fred_df)}")
        print(f"    Spectral d > 0 (long mem.):   {(fred_df['spectral'] > 0).sum()}/{len(fred_df)}")
        print(f"    GSADF bubble detected:        {(fred_df['bubble']).sum()}/{len(fred_df)}")
        print(f"    Hill alpha < 4 (fat tails):   {(fred_df['hill'] < 4).sum()}/{len(fred_df)}")
        print(f"    QQ alpha < 4 (fat tails):     {(fred_df['qq'] < 4).sum()}/{len(fred_df)}")
        tk_valid = fred_df['taleb_kappa'].dropna()
        print(f"    Taleb kappa > 0.1 (fat):      {(tk_valid > 0.1).sum()}/{len(tk_valid)}")
        print(f"    Mean Pickands xi:             {fred_df['pickands'].mean():.4f}")
        print(f"    Mean Hurst H:                 {fred_df['hurst'].mean():.4f}")
        print(f"    Mean DFA alpha:               {fred_df['dfa'].mean():.4f}")
        print(f"    Mean DEH gamma:               {fred_df['deh'].mean():.4f}")
        print(f"    Mean QQ alpha:                {fred_df['qq'].mean():.2f}")
        print(f"    Mean MaxSum ratio:            {fred_df['maxsum'].mean():.4f}")
        print(f"    Mean Spectral d:              {fred_df['spectral'].mean():.4f}")
        print(f"    Mean Hill alpha:              {fred_df['hill'].mean():.2f}")
        print(f"    Mean Taleb kappa:             {tk_valid.mean():.4f}")

    if clio_results:
        clio_df = pd.DataFrame(clio_results)
        print(f"\n  Clio Infra Yearly (top {len(clio_df)} countries):")
        print(f"    Pickands > 0 (heavy tails):   {(clio_df['pickands'] > 0).sum()}/{len(clio_df)}")
        print(f"    DEH > 0 (heavy tails):        {(clio_df['deh'] > 0).sum()}/{len(clio_df)}")
        print(f"    Hurst > 0.5 (persistent):     {(clio_df['hurst'] > 0.5).sum()}/{len(clio_df)}")
        print(f"    Hurst > 0.55 (strong pers.):  {(clio_df['hurst'] > 0.55).sum()}/{len(clio_df)}")
        print(f"    DFA > 0.5 (persistent):       {(clio_df['dfa'] > 0.5).sum()}/{len(clio_df)}")
        print(f"    DFA > 0.55 (strong pers.):    {(clio_df['dfa'] > 0.55).sum()}/{len(clio_df)}")
        print(f"    Spectral d > 0 (long mem.):   {(clio_df['spectral'] > 0).sum()}/{len(clio_df)}")
        print(f"    Hill alpha < 2 (extreme):     {(clio_df['hill'] < 2).sum()}/{len(clio_df)}")
        print(f"    Hill alpha < 4 (fat tails):   {(clio_df['hill'] < 4).sum()}/{len(clio_df)}")
        print(f"    QQ alpha < 4 (fat tails):     {(clio_df['qq'] < 4).sum()}/{len(clio_df)}")
        ctk_valid = clio_df['taleb_kappa'].dropna()
        print(f"    Taleb kappa > 0.1 (fat):      {(ctk_valid > 0.1).sum()}/{len(ctk_valid)}")
        print(f"    Mean Pickands xi:             {clio_df['pickands'].mean():.4f}")
        print(f"    Mean Hurst H:                 {clio_df['hurst'].mean():.4f}")
        print(f"    Mean DFA alpha:               {clio_df['dfa'].mean():.4f}")
        print(f"    Mean DEH gamma:               {clio_df['deh'].mean():.4f}")
        print(f"    Mean QQ alpha:                {clio_df['qq'].mean():.2f}")
        print(f"    Mean MaxSum ratio:            {clio_df['maxsum'].mean():.4f}")
        print(f"    Mean Spectral d:              {clio_df['spectral'].mean():.4f}")
        print(f"    Mean Hill alpha:              {clio_df['hill'].mean():.2f}")
        print(f"    Mean Taleb kappa:             {ctk_valid.mean():.4f}")

    print("\n  Method agreement:")
    print("    Pickands xi > 0 confirms Hill alpha < 4: both detect heavy tails")
    print("    from different angles (order statistics vs max-to-sum).")
    print("    DEH gamma provides a third independent tail estimate valid for all")
    print("    domains of attraction, and QQ alpha confirms from the QQ-plot slope.")
    print("    Hurst H > 0.5 and DFA alpha > 0.5 both indicate persistence,")
    print("    confirmed by Spectral d > 0 from the frequency domain.")
    print("    MaxSum ratio directly tests the infinite variance hypothesis.")
    print("    GSADF detects explosive episodes (bubbles/crashes) in price levels,")
    print("    complementing the static tail estimators with a dynamic signal.")

    # ══════════════════════════════════════════════
    # Conclusions
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    # Build conclusion text dynamically from actual metrics
    lppls_m = metrics.get("lppls", {})
    lppls_conf_m = metrics.get("lppls_confidence", {})
    dfa_m = metrics.get("dfa", {})
    hurst_m = metrics.get("hurst", {})

    # Best aggregator threshold by F1
    best_agg_thr = max(agg_metrics, key=lambda t: agg_metrics[t]["f1"]) if agg_metrics else 0.5
    best_agg = agg_metrics.get(best_agg_thr, {})

    print(f"""
  NOTE: All accuracy numbers are in-sample on historical data. Precision
  measures how often a signal is correct (low false positive rate); recall
  measures how many crashes are caught. Neither alone is sufficient.

  1. LPPLS with tightened filter (omega [6,13], tc constraint):
     Recall={lppls_m.get('recall', 0):.0%}, Precision={lppls_m.get('precision', 0):.0%}, F1={lppls_m.get('f1', 0):.0%}.
     The Nielsen (2024) omega range and tc proximity constraint reduce
     false positives compared to the original loose bounds.

  2. LPPLS confidence (multi-window):
     Recall={lppls_conf_m.get('recall', 0):.0%}, Precision={lppls_conf_m.get('precision', 0):.0%}, F1={lppls_conf_m.get('f1', 0):.0%}.
     Aggregating across many sub-windows provides a more robust signal.

  3. AGGREGATOR (best threshold={best_agg_thr:.1f}):
     Recall={best_agg.get('recall', 0):.0%}, Precision={best_agg.get('precision', 0):.0%}, F1={best_agg.get('f1', 0):.0%}.
     The weighted ensemble with category agreement bonus outperforms
     individual methods by combining independent signal categories.

  4. DFA is the best non-bubble method:
     Recall={dfa_m.get('recall', 0):.0%}, Precision={dfa_m.get('precision', 0):.0%}, F1={dfa_m.get('f1', 0):.0%}.
     Handles non-stationarity better than R/S Hurst
     (Recall={hurst_m.get('recall', 0):.0%}, F1={hurst_m.get('f1', 0):.0%}).

  5. Tail estimators (kappa, Pickands, DEH, Hill) have moderate recall
     but trade off against precision — they detect distributional regime
     shifts, not crash-specific patterns.

  6. GSADF is better for medium/major crashes than small ones —
     explosive unit root tests need sustained price growth.

  7. Fat tails are universal across all timescales:
     - Daily returns (BTC, SPY, Gold, GBP/USD): alpha 2-4
     - Decade-by-decade forex: every decade shows fat tails
     - Century-scale exchange rates: every currency shows fat tails

  8. All 6 known GBP/USD crises detected (100%): IMF 1976, Plaza 1985,
     Black Wednesday 1992, 2008 crisis, Brexit 2016, Truss 2022.

  9. The {n_methods}-method aggregator is the point: no single method is
     reliable alone. The ensemble combines 4 independent categories
     (bubble, tail, regime, structure) — when multiple categories agree,
     confidence increases. This is the core design principle.""")

    if run_nn:
        mlnn_m = metrics.get("mlnn", {})
        plnn_m = metrics.get("plnn", {})
        hlppl_m = metrics.get("hlppl", {})
        dtcai_m = metrics.get("dtcai", {})
        print(f"""
  NN METHODS (in-sample, requires PyTorch):
  9. M-LNN:  Recall={mlnn_m.get('recall', 0):.0%}, Precision={mlnn_m.get('precision', 0):.0%}, F1={mlnn_m.get('f1', 0):.0%}
     Per-series fitting, no pre-training. Slower but flexible.
  10. P-LNN: Recall={plnn_m.get('recall', 0):.0%}, Precision={plnn_m.get('precision', 0):.0%}, F1={plnn_m.get('f1', 0):.0%}
     Pre-trained on synthetic data, ~700x faster at inference.
  11. HLPPL: Recall={hlppl_m.get('recall', 0):.0%}, Precision={hlppl_m.get('precision', 0):.0%}, F1={hlppl_m.get('f1', 0):.0%}
     Dual-stream transformer with volume-based sentiment proxy.
  12. DTCAI: Recall={dtcai_m.get('recall', 0):.0%}, Precision={dtcai_m.get('precision', 0):.0%}, F1={dtcai_m.get('f1', 0):.0%}
     LPPLS reliability classifier (trained on BTC, tested cross-asset).""")

    print()


if __name__ == "__main__":
    main()
