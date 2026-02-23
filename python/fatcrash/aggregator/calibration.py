"""Historical calibration of signal weights."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def calibrate_weights(
    signals_history: dict[str, npt.NDArray[np.float64]],
    crash_labels: npt.NDArray[np.bool_],
    method: str = "logistic",
) -> dict[str, float]:
    """Calibrate aggregation weights from historical data.

    Args:
        signals_history: dict mapping indicator name to time series of signals
        crash_labels: boolean array where True = crash occurred within horizon
        method: calibration method ("logistic" or "equal")

    Returns:
        dict mapping indicator name to calibrated weight
    """
    names = list(signals_history.keys())
    n = len(crash_labels)

    if method == "equal":
        weight = 1.0 / len(names)
        return {name: weight for name in names}

    if method == "logistic":
        from scipy.optimize import minimize

        # Build feature matrix
        X = np.column_stack([signals_history[name][:n] for name in names])
        y = crash_labels.astype(np.float64)

        # Remove NaN rows
        mask = ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        y = y[mask]

        if len(y) < 10:
            weight = 1.0 / len(names)
            return {name: weight for name in names}

        def neg_log_likelihood(w: npt.NDArray) -> float:
            logits = X @ w
            p = 1.0 / (1.0 + np.exp(-logits))
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        w0 = np.ones(len(names)) / len(names)
        result = minimize(neg_log_likelihood, w0, method="L-BFGS-B")
        raw_weights = np.abs(result.x)
        normalized = raw_weights / raw_weights.sum()

        return {name: float(w) for name, w in zip(names, normalized)}

    raise ValueError(f"Unknown calibration method: {method}")
