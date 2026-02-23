"""EVT visualization: GPD tail fit, return level plots."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def gpd_tail_fit(
    returns: npt.NDArray[np.float64],
    sigma: float,
    xi: float,
    threshold: float,
    title: str = "GPD Tail Fit",
) -> None:
    """Plot empirical tail vs fitted GPD."""
    import plotly.graph_objects as go

    losses = -returns[returns < 0]
    exceedances = losses[losses > threshold] - threshold
    exceedances = np.sort(exceedances)

    n = len(exceedances)
    empirical_cdf = np.arange(1, n + 1) / (n + 1)
    empirical_survival = 1 - empirical_cdf

    # Fitted GPD survival
    if abs(xi) < 1e-8:
        fitted_survival = np.exp(-exceedances / sigma)
    else:
        fitted_survival = (1 + xi * exceedances / sigma) ** (-1 / xi)
        fitted_survival = np.clip(fitted_survival, 0, 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=exceedances, y=empirical_survival, mode="markers", name="Empirical", marker=dict(size=3))
    )
    fig.add_trace(
        go.Scatter(x=exceedances, y=fitted_survival, mode="lines", name="GPD fit", line=dict(color="red"))
    )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Exceedance",
        yaxis_title="Survival probability",
        yaxis_type="log",
    )
    fig.show()


def rolling_var_es_plot(
    dates: pd.DatetimeIndex,
    returns: npt.NDArray[np.float64],
    var: npt.NDArray[np.float64],
    es: npt.NDArray[np.float64],
    title: str = "Rolling VaR & ES",
) -> None:
    """Plot rolling VaR and ES with actual returns."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=-returns, name="Loss", line=dict(color="white", width=0.5)))
    fig.add_trace(go.Scatter(x=dates, y=var, name="VaR 99%", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=dates, y=es, name="ES 99%", line=dict(color="red")))

    # Mark exceedances
    mask = -returns > var
    if mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[mask],
                y=-returns[mask],
                mode="markers",
                name="VaR breach",
                marker=dict(color="red", size=5),
            )
        )

    fig.update_layout(template="plotly_dark", title=title, yaxis_title="Loss")
    fig.show()
