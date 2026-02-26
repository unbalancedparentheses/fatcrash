"""Crash labeling using Bree & Joseph (2013) criterion.

Detects price crashes defined as peaks followed by significant drawdowns.
Used by the DTCAI method (Lee, Jeong, Park & Ahn, 2025) to label LPPLS
parameter sets as reliable or unreliable.

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


@dataclass
class CrashEvent:
    """A detected crash event."""

    peak_date_idx: int  # index of price peak
    trough_date_idx: int  # index of lowest point within forward window
    drawdown: float  # magnitude of decline (positive, e.g. 0.30 = 30%)


def detect_crashes(
    prices: npt.NDArray[np.float64],
    lookback: int = 262,
    forward: int = 60,
    threshold: float = 0.25,
) -> list[CrashEvent]:
    """Detect crash events using the Bree & Joseph (2013) criterion.

    A crash is a price peak where:
    1. No higher close exists in the preceding `lookback` days.
    2. Price drops by more than `threshold` within the next `forward` days.

    Args:
        prices: 1-D array of (close) prices.
        lookback: Number of preceding days to check for higher close (~1 year).
        forward: Number of days after peak to measure drawdown.
        threshold: Minimum fractional decline to qualify as crash (e.g. 0.25 = 25%).

    Returns:
        List of CrashEvent sorted by peak index.
    """
    n = len(prices)
    if n < lookback + forward + 1:
        return []

    crashes: list[CrashEvent] = []

    for i in range(lookback, n - forward):
        peak_price = prices[i]

        # Condition 1: no higher close in preceding lookback days
        window_start = max(0, i - lookback)
        if np.max(prices[window_start:i]) >= peak_price:
            continue

        # Condition 2: price drops > threshold within next forward days
        forward_slice = prices[i + 1 : i + forward + 1]
        trough_price = np.min(forward_slice)
        drawdown = (peak_price - trough_price) / peak_price

        if drawdown > threshold:
            trough_offset = int(np.argmin(forward_slice))
            crashes.append(
                CrashEvent(
                    peak_date_idx=i,
                    trough_date_idx=i + 1 + trough_offset,
                    drawdown=drawdown,
                )
            )

    return crashes


def label_lppls_params(
    tc_predicted: float,
    crash_dates: list[int],
    tolerance: int = 10,
) -> int:
    """Label a single LPPLS fit as reliable (1) or unreliable (0).

    A fit is labeled reliable if its predicted critical time tc is within
    `tolerance` trading days of any actual crash date.

    Args:
        tc_predicted: Predicted critical time index from LPPLS fit.
        crash_dates: List of crash peak indices from detect_crashes.
        tolerance: Maximum distance in trading days to count as reliable.

    Returns:
        1 if reliable, 0 otherwise.
    """
    for crash_idx in crash_dates:
        if abs(tc_predicted - crash_idx) <= tolerance:
            return 1
    return 0
