from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from smartmeter_dashboard.transformations import TIMESERIES_COLUMNS


def consumption_figure(frame: pd.DataFrame, *, title: str, interval: str) -> go.Figure:
    validate_timeseries_columns(frame)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["global_active_power_average"],
            mode="lines+markers" if interval == "day" or len(frame) <= 48 else "lines",
            name="Average active power",
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Average: %{y:.3f} kW<extra></extra>",
            line={"color": "#167D9A", "width": 2},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=frame["timestamp"],
            y=frame["reading_count"],
            name="Readings",
            opacity=0.28,
            marker_color="#F2A65A",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Readings: %{y:,}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title=title,
        template="plotly_dark",
        hovermode="x unified",
        margin={"l": 18, "r": 18, "t": 64, "b": 48},
        legend={"orientation": "h", "y": 1.12, "x": 0, "title_text": ""},
        bargap=0.12 if interval == "hour" else 0.28,
    )
    figure.update_xaxes(
        title_text="Time (UTC)",
        tickformat="%b %d\n%H:%M" if interval == "hour" else "%b %d",
        nticks=8 if interval == "hour" else 12,
        automargin=True,
    )
    figure.update_yaxes(title_text="Active power (kW)", secondary_y=False, automargin=True)
    figure.update_yaxes(
        title_text="Readings per bucket",
        secondary_y=True,
        rangemode="tozero",
        automargin=True,
    )
    return figure


def electrical_figure(
    frame: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    unit: str,
) -> go.Figure:
    validate_timeseries_columns(frame)
    if value_column not in frame.columns:
        raise ValueError(f"Time-series data is missing column: {value_column}")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame[value_column],
            mode="lines",
            name=title,
            connectgaps=False,
            line={"color": "#167D9A", "width": 2},
            hovertemplate=(f"%{{x|%Y-%m-%d %H:%M}}<br>{title}: %{{y:.3f}} {unit}<extra></extra>"),
        )
    )
    figure.update_layout(
        title=title,
        template="plotly_dark",
        hovermode="x unified",
        margin={"l": 18, "r": 18, "t": 52, "b": 48},
        yaxis_title=unit,
        xaxis_title="Time (UTC)",
    )
    figure.update_xaxes(tickformat="%b %d\n%H:%M", nticks=8, automargin=True)
    figure.update_yaxes(automargin=True)
    return figure


def validate_timeseries_columns(frame: pd.DataFrame) -> None:
    missing = set(TIMESERIES_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Time-series data is missing columns: {', '.join(sorted(missing))}")
