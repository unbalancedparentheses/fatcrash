"""Price + bubble confidence overlay charts."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def plot_price_with_confidence(
    df: pd.DataFrame,
    confidence: npt.NDArray[np.float64],
    title: str = "Price with LPPLS Confidence",
) -> None:
    """Plot price with LPPLS confidence overlay using plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
    )

    dates = df.index

    # Price
    fig.add_trace(
        go.Scatter(x=dates, y=df["close"], name="Price", line=dict(color="white")),
        row=1,
        col=1,
    )

    # Confidence heatmap as bar chart
    colors = [
        f"rgba({min(255, int(c * 510))}, {max(0, int(255 - c * 510))}, 0, 0.8)"
        for c in confidence[: len(dates)]
    ]
    fig.add_trace(
        go.Bar(x=dates, y=confidence[: len(dates)], name="LPPLS Confidence", marker_color=colors),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        showlegend=True,
        height=600,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", range=[0, 1], row=2, col=1)

    fig.show()


def plot_crash_signal(
    df: pd.DataFrame,
    probability: npt.NDArray[np.float64],
    title: str = "Crash Probability",
) -> None:
    """Plot price with aggregate crash probability."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
    )

    dates = df.index

    fig.add_trace(
        go.Scatter(x=dates, y=df["close"], name="Price", line=dict(color="white")),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=probability[: len(dates)],
            name="Crash Probability",
            fill="tozeroy",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    # Threshold lines
    for level, color in [(0.3, "yellow"), (0.5, "orange"), (0.7, "red")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, row=2, col=1)

    fig.update_layout(template="plotly_dark", title=title, height=600)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="P(crash)", range=[0, 1], row=2, col=1)

    fig.show()
