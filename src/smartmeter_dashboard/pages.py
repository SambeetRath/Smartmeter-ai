from __future__ import annotations

import streamlit as st

from smartmeter_dashboard.dashboard_pages import (
    anomaly_detection,
    data_quality,
    detection_runs,
    electrical_metrics,
    energy_analytics,
    overview,
)
from smartmeter_dashboard.state import DashboardFilters

PAGE_PLACEHOLDER_MESSAGES = {
    "Overview": "Overview analytics.",
    "Energy Analytics": "Server-aggregated power and reading trends.",
    "Electrical Metrics": "Server-aggregated electrical metric trends.",
    "Anomaly Detection": "Persisted anomaly predictions, including unavailable results.",
    "Detection Runs": "Detection-run history and processing counts.",
    "Data Quality": "Database-computed missing values and timestamp gaps.",
}


def render_page(
    page_name: str,
    filters: DashboardFilters,
    *,
    api_base_url: str = "http://127.0.0.1:8000",
) -> None:
    if page_name not in PAGE_PLACEHOLDER_MESSAGES:
        raise ValueError(f"Unknown dashboard page: {page_name}")

    st.header(page_name)
    if page_name == "Overview":
        overview.render(filters, api_base_url)
        return
    if page_name == "Energy Analytics":
        energy_analytics.render(filters, api_base_url)
        return
    if page_name == "Electrical Metrics":
        electrical_metrics.render(filters, api_base_url)
        return
    if page_name == "Anomaly Detection":
        anomaly_detection.render(filters, api_base_url)
        return
    if page_name == "Detection Runs":
        detection_runs.render(filters, api_base_url)
        return
    if page_name == "Data Quality":
        data_quality.render(filters, api_base_url)
        return

    if filters.date_range_error:
        st.warning("Update the date filter to continue.")
    elif filters.meter_identifier is None:
        st.info("Select a meter when one is available to configure dashboard filters.")
    else:
        st.info(PAGE_PLACEHOLDER_MESSAGES[page_name])
