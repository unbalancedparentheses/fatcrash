"""Tail analysis dashboard: Hill plot, kappa evolution, QQ plot."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def hill_plot(
    returns: npt.NDArray[np.float64],
    max_k: int | None = None,
    title: str = "Hill Plot",
) -> None:
    """Plot Hill estimator as function of k (order statistics used)."""
    import plotly.graph_objects as go
    from fatcrash._core import hill_estimator

    abs_returns = np.abs(returns[returns != 0])
    n = len(abs_returns)
    max_k = max_k or n // 2

    ks = list(range(5, max_k, max(1, max_k // 200)))
    alphas = [hill_estimator(returns, k=k) for k in ks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=alphas, mode="lines", name="Hill α"))
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="α=2 (infinite variance)")
    fig.add_hline(y=4.0, line_dash="dash", line_color="yellow", annotation_text="α=4 (thin tail boundary)")

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="k (order statistics)",
        yaxis_title="α (tail index)",
    )
    fig.show()


def kappa_evolution(
    dates: pd.DatetimeIndex,
    kappa_series: npt.NDArray[np.float64],
    benchmark: float,
    title: str = "Kappa Evolution",
) -> None:
    """Plot rolling kappa with Gaussian benchmark."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=kappa_series, name="κ", line=dict(color="cyan")))
    fig.add_hline(y=benchmark, line_dash="dash", line_color="yellow", annotation_text="Gaussian benchmark")

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Date",
        yaxis_title="κ",
    )
    fig.show()


def qq_plot(
    returns: npt.NDArray[np.float64],
    title: str = "QQ Plot vs Normal",
) -> None:
    """QQ plot comparing returns to normal distribution."""
    import plotly.graph_objects as go
    from scipy import stats

    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_returns, mode="markers", name="Data", marker=dict(size=2)))

    # 45-degree line
    min_val = min(theoretical.min(), sorted_returns.min())
    max_val = max(theoretical.max(), sorted_returns.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Normal",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
    )
    fig.show()
