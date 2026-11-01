from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from smartmeter_dashboard.api_client import AnomaliesPageResponse
from smartmeter_dashboard.components.common import (
    render_data_table,
    render_empty_state,
    render_error_state,
    render_kpi_cards,
    render_loading,
)
from smartmeter_dashboard.detection_data import fetch_anomalies
from smartmeter_dashboard.exceptions import DashboardAPIError
from smartmeter_dashboard.state import DashboardFilters
from smartmeter_dashboard.transformations import anomalies_dataframe

ALGORITHMS = (
    "global_iqr",
    "rolling_mean_std",
    "time_of_day",
    "isolation_forest",
)
PAGE_SIZE = 100
OFFSET_STATE_KEY = "anomaly_detection_offset"
FILTER_STATE_KEY = "anomaly_detection_filters"


def render(
    filters: DashboardFilters,
    api_base_url: str,
    *,
    anomalies_loader: Callable[..., AnomaliesPageResponse] = fetch_anomalies,
) -> None:
    if filters.date_range_error:
        st.warning("Update the date filter to continue.")
        return
    if not filters.meter_identifier or not filters.start or not filters.end:
        render_empty_state("Select a meter and valid date range to view detection results.")
        return

    controls = st.columns(3)
    prediction_status = controls[0].selectbox(
        "Prediction status",
        options=("anomaly", "normal", "unavailable"),
        format_func=lambda value: value.title(),
        key="anomaly_prediction_status",
    )
    algorithm_choice = controls[1].selectbox(
        "Algorithm",
        options=("All algorithms", *ALGORITHMS),
        key="anomaly_algorithm",
    )
    category = controls[2].text_input(
        "Category (optional)",
        key="anomaly_category",
        help="Filter by the exact category returned by the detector.",
    )
    current_filter = (
        filters.meter_identifier,
        filters.start,
        filters.end,
        prediction_status,
        algorithm_choice,
        category.strip(),
    )
    if st.session_state.get(FILTER_STATE_KEY) != current_filter:
        st.session_state[OFFSET_STATE_KEY] = 0
        st.session_state[FILTER_STATE_KEY] = current_filter
    offset = st.session_state.get(OFFSET_STATE_KEY, 0)
    algorithm = None if algorithm_choice == "All algorithms" else algorithm_choice

    try:
        with render_loading("Loading persisted detection predictions..."):
            result = anomalies_loader(
                api_base_url,
                meter_identifier=filters.meter_identifier,
                start=filters.start,
                end=filters.end,
                prediction_status=prediction_status,
                algorithm=algorithm,
                category=category.strip() or None,
                limit=PAGE_SIZE,
                offset=offset,
            )
    except DashboardAPIError as error:
        render_error_state(f"Could not load detection predictions: {error}")
        return

    render_kpi_cards(
        [
            ("Prediction status", prediction_status.title(), None),
            ("Rows on this page", f"{result.returned_count:,}", None),
            ("Page", f"{offset // PAGE_SIZE + 1:,}", None),
        ]
    )
    if prediction_status == "unavailable":
        st.info("These predictions were unavailable; they are not classified as normal.")

    frame = anomalies_dataframe(result)
    if frame.empty:
        render_empty_state(f"No {prediction_status} predictions match these filters.")
    else:
        frame["timestamp"] = frame["timestamp"].map(
            lambda value: (
                value.strftime("%Y-%m-%d %H:%M:%S UTC")
                if pd.notna(value)
                else "Timestamp unavailable"
            )
        )
        frame["anomaly_score"] = frame["anomaly_score"].map(
            lambda value: f"{value:.5g}" if pd.notna(value) else "Not provided"
        )
        frame["anomaly_category"] = frame["anomaly_category"].fillna("Not classified")
        render_data_table(frame)

    previous, next_page = st.columns(2)
    if previous.button("Previous page", disabled=offset == 0, key="anomaly_previous"):
        st.session_state[OFFSET_STATE_KEY] = max(0, offset - PAGE_SIZE)
        st.rerun()
    if next_page.button("Next page", disabled=not result.has_more, key="anomaly_next"):
        st.session_state[OFFSET_STATE_KEY] = result.next_offset or offset + PAGE_SIZE
        st.rerun()
