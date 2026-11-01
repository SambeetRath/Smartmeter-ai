from __future__ import annotations

from datetime import date

import streamlit as st

from smartmeter_dashboard.api_client import MeterCoverage
from smartmeter_dashboard.state import (
    DATE_METER_STATE_KEY,
    DATE_RANGE_STATE_KEY,
    METER_STATE_KEY,
    DashboardFilters,
    default_date_range,
    validate_date_range,
)


def render_filter_panel(
    meters: list[MeterCoverage],
    *,
    today: date | None = None,
) -> DashboardFilters:
    st.sidebar.subheader("Filters")
    refresh_requested = st.sidebar.button("Refresh data", key="dashboard_refresh")
    if not meters:
        st.sidebar.info("No meter data is currently available.")
        return DashboardFilters(None, None, None, refresh_requested)

    identifiers = [meter.meter_identifier for meter in meters]
    if st.session_state.get(METER_STATE_KEY) not in identifiers:
        st.session_state[METER_STATE_KEY] = identifiers[0]
    meter_identifier = st.sidebar.selectbox(
        "Meter",
        options=identifiers,
        key=METER_STATE_KEY,
    )
    selected_meter = next(meter for meter in meters if meter.meter_identifier == meter_identifier)

    if st.session_state.get(DATE_METER_STATE_KEY) != meter_identifier:
        st.session_state[DATE_RANGE_STATE_KEY] = default_date_range(
            selected_meter,
            today=today,
        )
        st.session_state[DATE_METER_STATE_KEY] = meter_identifier

    selected_dates = st.sidebar.date_input(
        "Date range (UTC, inclusive)",
        key=DATE_RANGE_STATE_KEY,
        format="YYYY-MM-DD",
    )
    if (
        not isinstance(selected_dates, (tuple, list))
        or len(selected_dates) != 2
        or not all(isinstance(value, date) for value in selected_dates)
    ):
        st.sidebar.error("Select both a start date and an end date.")
        return DashboardFilters(
            meter_identifier,
            None,
            None,
            refresh_requested,
            "Select both a start date and an end date.",
        )

    start_date, end_date = selected_dates
    date_range_error = validate_date_range(start_date, end_date)
    if date_range_error:
        st.sidebar.error(date_range_error)

    return DashboardFilters(
        meter_identifier,
        start_date,
        end_date,
        refresh_requested,
        date_range_error,
    )
