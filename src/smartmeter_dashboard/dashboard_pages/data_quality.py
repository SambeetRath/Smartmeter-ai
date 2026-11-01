from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from smartmeter_dashboard.api_client import DataQualityResponse
from smartmeter_dashboard.components.common import (
    render_data_table,
    render_empty_state,
    render_error_state,
    render_kpi_cards,
    render_loading,
    render_status_indicator,
)
from smartmeter_dashboard.detection_data import fetch_data_quality
from smartmeter_dashboard.exceptions import DashboardAPIError
from smartmeter_dashboard.state import DashboardFilters
from smartmeter_dashboard.transformations import missing_measurements_dataframe


def render(
    filters: DashboardFilters,
    api_base_url: str,
    *,
    quality_loader: Callable[..., DataQualityResponse] = fetch_data_quality,
) -> None:
    if filters.date_range_error:
        st.warning("Update the date filter to continue.")
        return
    if not filters.meter_identifier or not filters.start or not filters.end:
        render_empty_state("Select a meter and valid date range to view data quality.")
        return

    try:
        with render_loading("Loading stored-data quality summary..."):
            result = quality_loader(
                api_base_url,
                meter_identifier=filters.meter_identifier,
                start=filters.start,
                end=filters.end,
            )
    except DashboardAPIError as error:
        render_error_state(f"Could not load data-quality summary: {error}")
        return

    gaps = result.timestamp_gaps
    if result.reading_count == 0:
        render_empty_state("No stored readings were found in this meter and date range.")
    render_kpi_cards(
        [
            ("Stored readings", f"{result.reading_count:,}", None),
            ("Observed timestamp gaps", f"{gaps.observed_gap_count:,}", None),
            (
                "Estimated missing intervals",
                f"{gaps.estimated_missing_intervals:,}",
                f"Based on an assumed {gaps.expected_interval_minutes}-minute cadence.",
            ),
        ]
    )
    st.subheader("Missing measurements")
    missing = missing_measurements_dataframe(result.missing_measurements)
    if missing.empty:
        render_empty_state("The API returned no per-measurement missing-value statistics.")
    else:
        missing["missing_rate"] = missing["missing_rate"].map(
            lambda rate: f"{rate:.1%}" if pd.notna(rate) else "Not available"
        )
        render_data_table(missing)

    st.subheader("Timestamp coverage")
    first = (
        gaps.first_observed_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if gaps.first_observed_at
        else "Not available"
    )
    last = (
        gaps.last_observed_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if gaps.last_observed_at
        else "Not available"
    )
    render_kpi_cards(
        [
            ("First observed reading", first, None),
            ("Last observed reading", last, None),
            (
                "Range boundary gaps",
                "Assessed" if gaps.boundary_gaps_assessed else "Not assessed",
                "The API does not infer gaps outside the first and last observed readings.",
            ),
        ]
    )
    st.caption(f"Stored-data metric scope: {result.database_metrics_scope}")

    st.subheader("Ingestion-time validation")
    if result.ingestion_validation.get("available") is True:
        render_status_indicator("available", label="Historical validation report")
        st.write(result.ingestion_validation.get("detail", "Available from API response."))
    else:
        render_status_indicator("unavailable", label="Historical validation report")
        st.info(
            result.ingestion_validation.get(
                "detail",
                "Ingestion-time validation findings were not stored and cannot be reconstructed.",
            )
        )
