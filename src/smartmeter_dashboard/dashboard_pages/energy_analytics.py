from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from smartmeter_dashboard.analytics_data import fetch_summary, fetch_timeseries
from smartmeter_dashboard.api_client import AnalyticsSummaryResponse, TimeSeriesResponse
from smartmeter_dashboard.components.common import (
    render_chart,
    render_empty_state,
    render_error_state,
    render_kpi_cards,
    render_loading,
)
from smartmeter_dashboard.dashboard_pages.charts import consumption_figure
from smartmeter_dashboard.exceptions import DashboardAPIError
from smartmeter_dashboard.state import DashboardFilters
from smartmeter_dashboard.transformations import timeseries_dataframe


def _power(value: float | None) -> str:
    return f"{value:,.2f} kW" if value is not None else "—"


def render(
    filters: DashboardFilters,
    api_base_url: str,
    *,
    summary_loader: Callable[..., AnalyticsSummaryResponse] = fetch_summary,
    timeseries_loader: Callable[..., TimeSeriesResponse] = fetch_timeseries,
) -> None:
    if filters.date_range_error:
        st.warning("Update the date filter to continue.")
        return
    if not filters.meter_identifier or not filters.start or not filters.end:
        render_empty_state("Select a meter and valid date range to view energy analytics.")
        return

    try:
        with render_loading("Loading server-aggregated energy analytics..."):
            summary = summary_loader(
                api_base_url,
                filters.meter_identifier,
                filters.start,
                filters.end,
            )
            hourly = timeseries_loader(
                api_base_url,
                filters.meter_identifier,
                filters.start,
                filters.end,
                "hour",
            )
            daily = timeseries_loader(
                api_base_url,
                filters.meter_identifier,
                filters.start,
                filters.end,
                "day",
            )
    except DashboardAPIError as error:
        render_error_state(f"Could not load energy analytics: {error}")
        return

    st.subheader("Selected-period summary")
    st.caption("Active power is reported in kW. It is not energy consumption in kWh.")
    render_kpi_cards(
        [
            ("Readings", f"{summary.reading_count:,}", None),
            ("Average active power", _power(summary.active_power_average), None),
            ("Peak active power", _power(summary.active_power_maximum), None),
        ]
    )

    hourly_frame = timeseries_dataframe(hourly)
    daily_frame = timeseries_dataframe(daily)
    if hourly_frame.empty and daily_frame.empty:
        render_empty_state("No aggregate buckets are available for the selected period.")
        return
    if not hourly_frame.empty:
        st.subheader("Hourly consumption")
        render_chart(
            consumption_figure(
                hourly_frame,
                title="Hourly average active power and reading volume",
                interval="hour",
            )
        )
    if not daily_frame.empty:
        st.subheader("Daily consumption")
        render_chart(
            consumption_figure(
                daily_frame,
                title="Daily average active power and reading volume",
                interval="day",
            )
        )
    if summary.reading_count == 0:
        st.caption("The selected period contains no readings.")
