"""DTCAI dataset generator — labeled LPPLS parameter sets from price series.

Runs LPPLS fits across rolling windows and labels each fit using the
Bree & Joseph crash criterion. The resulting dataset is used to train
the DTCAI reliability classifier.

Reference: Lee, G., Jeong, M., Park, T. & Ahn, K. (2025).
"More Than Ex-Post Fitting: LPPL and Its AI-Based Classification."
Humanities and Social Sciences Communications, 12, 236.

DISCLAIMER: This software is for academic research and educational purposes only.
It does not constitute financial advice. No warranty is provided regarding the
accuracy of predictions. Do not use for investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fatcrash._core import lppls_fit
from fatcrash.nn.crash_labels import CrashEvent, detect_crashes, label_lppls_params

FEATURE_NAMES = ["A", "B", "C", "tc", "phi", "omega", "beta"]


@dataclass
class DTCAIDataset:
    """Labeled LPPLS parameter dataset for DTCAI training."""

    X: npt.NDArray[np.float64]  # (n_samples, 7) — LPPLS params
    y: npt.NDArray[np.int64]  # (n_samples,) — binary labels
    feature_names: list[str]  # ["A", "B", "C", "tc", "phi", "omega", "beta"]


def _extract_features(
    fit_result: tuple[float, ...],
) -> npt.NDArray[np.float64]:
    """Extract 7 LPPLS features from a fit result.

    lppls_fit returns: (tc, m, omega, a, b, c1, c2, rss)
    Paper uses: (A, B, C, tc, phi, omega, beta)
    where C = sqrt(c1^2 + c2^2), phi = atan2(c2, c1), beta = m.
    """
    tc, m, omega, a, b, c1, c2, _rss = fit_result
    c = np.sqrt(c1**2 + c2**2)
    phi = np.arctan2(c2, c1)
    return np.array([a, b, c, tc, phi, omega, m], dtype=np.float64)


def generate_dtcai_dataset(
    prices: npt.NDArray[np.float64],
    window_size: int = 504,
    step_size: int = 21,
    n_fits_per_window: int = 10,
    crash_lookback: int = 262,
    crash_forward: int = 60,
    crash_threshold: float = 0.25,
    tc_tolerance: int = 10,
    seed: int = 42,
) -> DTCAIDataset:
    """Generate a labeled LPPLS parameter dataset from a price series.

    For each rolling window, runs multiple LPPLS fits with different seeds.
    Each fit is labeled reliable (1) or unreliable (0) based on whether
    the predicted tc is within tc_tolerance of an actual crash date.

    Args:
        prices: 1-D close price array.
        window_size: Rolling window length in trading days (default 504 = ~2 years).
        step_size: Step between windows in days (default 21 = ~1 month).
        n_fits_per_window: Number of LPPLS fits per window (default 10).
        crash_lookback: Lookback for crash detection (default 262).
        crash_forward: Forward window for crash detection (default 60).
        crash_threshold: Minimum decline for crash (default 0.25).
        tc_tolerance: Days tolerance for reliable label (default 10).
        seed: Base random seed.

    Returns:
        DTCAIDataset with features and labels.
    """
    n = len(prices)
    if n < window_size:
        return DTCAIDataset(
            X=np.empty((0, 7), dtype=np.float64),
            y=np.empty(0, dtype=np.int64),
            feature_names=list(FEATURE_NAMES),
        )

    # Detect crashes in the full series
    crash_events: list[CrashEvent] = detect_crashes(
        prices,
        lookback=crash_lookback,
        forward=crash_forward,
        threshold=crash_threshold,
    )
    crash_dates = [e.peak_date_idx for e in crash_events]

    log_prices = np.log(np.maximum(prices, 1e-10))

    features_list: list[npt.NDArray[np.float64]] = []
    labels_list: list[int] = []

    rng = np.random.default_rng(seed)

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window_times = np.arange(start, end, dtype=np.float64)
        window_log_prices = log_prices[start:end]

        tc_range = (float(end), float(end) + 0.2 * window_size)

        for j in range(n_fits_per_window):
            fit_seed = int(rng.integers(0, 2**31))
            try:
                result = lppls_fit(
                    window_times,
                    window_log_prices,
                    tc_range=tc_range,
                    seed=fit_seed,
                )
            except Exception:
                continue

            feats = _extract_features(result)

            # Skip if any NaN/inf in features
            if not np.all(np.isfinite(feats)):
                continue

            tc_pred = result[0]  # tc from fit
            label = label_lppls_params(tc_pred, crash_dates, tolerance=tc_tolerance)

            features_list.append(feats)
            labels_list.append(label)

    if not features_list:
        return DTCAIDataset(
            X=np.empty((0, 7), dtype=np.float64),
            y=np.empty(0, dtype=np.int64),
            feature_names=list(FEATURE_NAMES),
        )

    return DTCAIDataset(
        X=np.stack(features_list),
        y=np.array(labels_list, dtype=np.int64),
        feature_names=list(FEATURE_NAMES),
    )
