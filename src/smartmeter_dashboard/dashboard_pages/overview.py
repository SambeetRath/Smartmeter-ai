from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from smartmeter_dashboard.analytics_data import (
    fetch_anomaly_summary,
    fetch_summary,
    fetch_timeseries,
)
from smartmeter_dashboard.api_client import (
    AnalyticsSummaryResponse,
    AnomalySummaryResponse,
    TimeSeriesResponse,
)
from smartmeter_dashboard.components.common import (
    render_chart,
    render_data_table,
    render_empty_state,
    render_error_state,
    render_kpi_cards,
    render_loading,
)
from smartmeter_dashboard.dashboard_pages.charts import consumption_figure
from smartmeter_dashboard.exceptions import DashboardAPIError
from smartmeter_dashboard.state import DashboardFilters
from smartmeter_dashboard.transformations import (
    missing_measurements_dataframe,
    timeseries_dataframe,
)


def _power(value: float | None) -> str:
    return f"{value:,.2f} kW" if value is not None else "—"


def render(
    filters: DashboardFilters,
    api_base_url: str,
    *,
    summary_loader: Callable[..., AnalyticsSummaryResponse] = fetch_summary,
    timeseries_loader: Callable[..., TimeSeriesResponse] = fetch_timeseries,
    anomaly_loader: Callable[..., AnomalySummaryResponse | None] = fetch_anomaly_summary,
) -> None:
    if filters.date_range_error:
        st.warning("Update the date filter to continue.")
        return
    if not filters.meter_identifier or not filters.start or not filters.end:
        render_empty_state("Select a meter and valid date range to view overview metrics.")
        return

    try:
        with render_loading("Loading meter summary and daily consumption..."):
            summary = summary_loader(
                api_base_url,
                filters.meter_identifier,
                filters.start,
                filters.end,
            )
            daily = timeseries_loader(
                api_base_url,
                filters.meter_identifier,
                filters.start,
                filters.end,
                "day",
            )
    except DashboardAPIError as error:
        render_error_state(f"Could not load overview analytics: {error}")
        return

    st.subheader("Consumption overview")
    latest_time = (
        f"Observed at {summary.latest_active_power_at:%Y-%m-%d %H:%M} UTC"
        if summary.latest_active_power_at is not None and summary.latest_active_power is not None
        else None
    )
    render_kpi_cards(
        [
            ("Total readings", f"{summary.reading_count:,}", None),
            ("Average active power", _power(summary.active_power_average), None),
            ("Peak active power", _power(summary.active_power_maximum), None),
            ("Latest active power", _power(summary.latest_active_power), latest_time),
        ]
    )

    if summary.reading_count == 0:
        render_empty_state("No readings were found for this meter and date range.")
    else:
        st.subheader("Missing measurements")
        missing = missing_measurements_dataframe(summary.missing_measurements)
        missing["missing_rate"] = missing["missing_rate"].map(
            lambda rate: f"{rate:.1%}" if pd.notna(rate) else "—"
        )
        render_data_table(missing)

    frame = timeseries_dataframe(daily)
    if frame.empty:
        render_empty_state("No daily consumption buckets are available for this period.")
    else:
        st.subheader("Daily consumption trend")
        render_chart(
            consumption_figure(frame, title="Daily active power and reading volume", interval="day")
        )

    try:
        anomaly = anomaly_loader(api_base_url, filters.meter_identifier)
    except DashboardAPIError as error:
        render_error_state(f"Could not load anomaly summary: {error}")
        return
    if anomaly is None:
        st.info("No completed anomaly-detection run is available for this meter yet.")
        return
    _render_anomaly_summary(anomaly)


def _render_anomaly_summary(summary: AnomalySummaryResponse) -> None:
    st.subheader(f"Latest completed detection run · {summary.algorithm}")
    rate = f"{summary.anomaly_rate:.1%}" if summary.anomaly_rate is not None else "—"
    render_kpi_cards(
        [
            ("Anomalies", f"{summary.anomaly_count:,}", None),
            ("Normal predictions", f"{summary.normal_count:,}", None),
            ("Unavailable predictions", f"{summary.unavailable_count:,}", None),
            ("Anomaly rate", rate, None),
        ]
    )
