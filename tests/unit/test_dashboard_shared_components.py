from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smartmeter_dashboard.components.common import (
    render_chart,
    render_data_table,
    render_kpi_cards,
    render_status_indicator,
    render_visual_styles,
)


def test_kpi_cards_escape_content_and_preserve_long_values(monkeypatch):
    output = []
    monkeypatch.setattr(st, "markdown", lambda value, **_kwargs: output.append(value))
    long_value = "2026-11-01 08:35:30 UTC · 123456789012345.678 kW"

    render_kpi_cards([("<Meter>", long_value, "Full precision is preserved.")])

    markup = output[0]
    assert escape("<Meter>") in markup
    assert escape(long_value) in markup


def test_visual_styles_render_responsive_wrapping_kpi_cards(monkeypatch):
    output = []
    monkeypatch.setattr(st, "markdown", lambda value, **_kwargs: output.append(value))

    render_visual_styles()

    assert "repeat(auto-fit, minmax(175px, 1fr))" in output[0]
    assert "overflow-wrap: anywhere" in output[0]


def test_data_table_uses_responsive_width_and_wide_identifier_columns(monkeypatch):
    calls = []
    monkeypatch.setattr(st, "dataframe", lambda frame, **kwargs: calls.append((frame, kwargs)))

    render_data_table(pd.DataFrame({"run_id": ["long-run-id"], "status": ["completed"]}))

    kwargs = calls[0][1]
    assert kwargs["width"] == "stretch"
    assert kwargs["column_config"]["run_id"]["width"] == "large"
    assert kwargs["column_config"]["status"]["width"] == "medium"


def test_plotly_chart_uses_supported_stretch_width(monkeypatch):
    calls = []
    monkeypatch.setattr(st, "plotly_chart", lambda figure, **kwargs: calls.append(kwargs))

    render_chart(go.Figure())

    assert calls[0]["width"] == "stretch"
    assert "use_container_width" not in calls[0]


def test_status_indicators_distinguish_unavailable_from_normal(monkeypatch):
    messages = {"success": [], "error": [], "info": []}
    monkeypatch.setattr(st, "success", messages["success"].append)
    monkeypatch.setattr(st, "error", messages["error"].append)
    monkeypatch.setattr(st, "info", messages["info"].append)

    render_status_indicator("unavailable", label="Prediction")
    render_status_indicator("normal", label="Prediction")

    assert messages["error"] == ["Prediction: Unavailable"]
    assert messages["success"] == ["Prediction: Normal"]
