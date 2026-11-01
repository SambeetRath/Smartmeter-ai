from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from smartmeter_dashboard.analytics_data import fetch_timeseries
from smartmeter_dashboard.api_client import TimeSeriesResponse
from smartmeter_dashboard.components.common import (
    render_chart,
    render_empty_state,
    render_error_state,
    render_loading,
)
from smartmeter_dashboard.dashboard_pages.charts import electrical_figure
from smartmeter_dashboard.exceptions import DashboardAPIError
from smartmeter_dashboard.state import DashboardFilters
from smartmeter_dashboard.transformations import (
    ELECTRICAL_METRICS,
    electrical_metrics_dataframe,
    timeseries_dataframe,
)


def render(
    filters: DashboardFilters,
    api_base_url: str,
    *,
    timeseries_loader: Callable[..., TimeSeriesResponse] = fetch_timeseries,
) -> None:
    if filters.date_range_error:
        st.warning("Update the date filter to continue.")
        return
    if not filters.meter_identifier or not filters.start or not filters.end:
        render_empty_state("Select a meter and valid date range to view electrical metrics.")
        return

    interval = st.selectbox(
        "Aggregation interval",
        options=("hour", "day"),
        format_func=lambda value: "Hourly" if value == "hour" else "Daily",
        key="electrical_metrics_interval",
    )
    try:
        with render_loading(f"Loading {interval} electrical aggregates..."):
            response = timeseries_loader(
                api_base_url,
                filters.meter_identifier,
                filters.start,
                filters.end,
                interval,
            )
    except DashboardAPIError as error:
        render_error_state(f"Could not load electrical metrics: {error}")
        return

    frame = timeseries_dataframe(response)
    if frame.empty:
        render_empty_state("No electrical metric buckets are available for this period.")
        return

    long_frame = electrical_metrics_dataframe(response)
    missing_counts = (
        long_frame.groupby("metric", sort=False)["value"]
        .apply(lambda values: int(values.isna().sum()))
        .to_dict()
    )
    st.caption(
        "Charts show server-aggregated values. Active and reactive power are reported in kW. "
        "Sub-metering values are average Wh per source reading; integrated kWh is not returned "
        "by this API. Missing aggregate values remain gaps and are not interpolated."
    )
    charts = st.columns(2)
    for index, (column, label, unit, description) in enumerate(ELECTRICAL_METRICS):
        with charts[index % 2]:
            missing = missing_counts[label]
            st.caption(f"{description} · {missing} bucket(s) without a value")
            render_chart(
                electrical_figure(frame, value_column=column, title=label, unit=unit),
            )
