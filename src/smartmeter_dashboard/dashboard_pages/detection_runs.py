from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from smartmeter_dashboard.api_client import DetectionRunsPageResponse
from smartmeter_dashboard.components.common import (
    render_data_table,
    render_empty_state,
    render_error_state,
    render_kpi_cards,
    render_loading,
    render_status_indicator,
)
from smartmeter_dashboard.detection_data import fetch_detection_runs
from smartmeter_dashboard.exceptions import DashboardAPIError
from smartmeter_dashboard.state import DashboardFilters
from smartmeter_dashboard.transformations import detection_runs_dataframe

RUN_STATUSES = ("pending", "running", "completed", "failed")
ALGORITHMS = (
    "global_iqr",
    "rolling_mean_std",
    "time_of_day",
    "isolation_forest",
)
PAGE_SIZE = 100


def render(
    filters: DashboardFilters,
    api_base_url: str,
    *,
    runs_loader: Callable[..., DetectionRunsPageResponse] = fetch_detection_runs,
) -> None:
    if not filters.meter_identifier:
        render_empty_state("Select a meter to view detection runs.")
        return

    controls = st.columns(2)
    status_choice = controls[0].selectbox(
        "Run status",
        options=("All statuses", *RUN_STATUSES),
        key="detection_run_status",
    )
    algorithm_choice = controls[1].selectbox(
        "Algorithm",
        options=("All algorithms", *ALGORITHMS),
        key="detection_run_algorithm",
    )
    try:
        with render_loading("Loading detection-run history..."):
            result = runs_loader(
                api_base_url,
                meter_identifier=filters.meter_identifier,
                status=None if status_choice == "All statuses" else status_choice,
                algorithm=None if algorithm_choice == "All algorithms" else algorithm_choice,
                limit=PAGE_SIZE,
                offset=0,
            )
    except DashboardAPIError as error:
        render_error_state(f"Could not load detection runs: {error}")
        return

    frame = detection_runs_dataframe(result)
    st.caption(
        "Runs are listed by creation time for the selected meter, independently of the "
        "shared reading-date filter. Each run's prediction counts are scoped to that run; "
        "repeated runs may cover the same readings."
    )
    render_kpi_cards(
        [
            ("Runs in this page", f"{result.returned_count:,}", None),
            ("Completed", f"{int((frame['status'] == 'completed').sum()):,}", None),
            ("Failed", f"{int((frame['status'] == 'failed').sum()):,}", None),
        ]
    )
    if frame.empty:
        render_empty_state("No detection runs match these filters.")
        return

    display = frame[
        [
            "run_id",
            "algorithm",
            "status",
            "created_at",
            "started_at",
            "completed_at",
            "attempts",
            "readings_processed",
            "anomaly_count",
            "normal_count",
            "unavailable_count",
        ]
    ].copy()
    for column in ("created_at", "started_at", "completed_at"):
        missing_label = "Not started" if column == "started_at" else "Not completed"
        display[column] = display[column].map(
            lambda value, missing=missing_label: (
                value.strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(value) else missing
            )
        )
    render_data_table(display, height=440)

    selected_run_id = st.selectbox(
        "Run details",
        options=frame["run_id"].tolist(),
        format_func=lambda run_id: (
            f"{run_id} · {frame.loc[frame['run_id'] == run_id, 'algorithm'].iloc[0]}"
        ),
        key="selected_detection_run",
    )
    run = next(run for run in result.items if str(run.run_id) == str(selected_run_id))
    render_status_indicator(run.status, label="Selected run status")
    render_kpi_cards(
        [
            ("Algorithm", str(run.algorithm), None),
            ("Readings processed", f"{run.readings_processed:,}", None),
            ("Anomalies", f"{run.anomaly_count:,}", None),
            ("Normal predictions", f"{run.normal_count:,}", None),
            (
                "Unavailable predictions",
                f"{run.unavailable_count:,}",
                "Unavailable is distinct from a normal prediction.",
            ),
            ("Attempts", f"{run.attempts:,}", None),
        ]
    )
