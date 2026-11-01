from __future__ import annotations

from datetime import datetime

import streamlit as st

from smartmeter_dashboard.api_client import (
    AnomaliesPageResponse,
    DataQualityResponse,
    DetectionRunsPageResponse,
    SmartMeterAPIClient,
)
from smartmeter_dashboard.config import DashboardConfig

DETECTION_CACHE_TTL_SECONDS = 60


@st.cache_data(ttl=DETECTION_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_anomalies(
    api_base_url: str,
    *,
    meter_identifier: str,
    start: datetime,
    end: datetime,
    prediction_status: str,
    algorithm: str | None,
    category: str | None,
    limit: int,
    offset: int,
) -> AnomaliesPageResponse:
    client = SmartMeterAPIClient(DashboardConfig(api_base_url=api_base_url))
    try:
        return client.list_anomalies(
            meter_identifier=meter_identifier,
            start=start,
            end=end,
            prediction_status=prediction_status,
            algorithm=algorithm,
            category=category,
            limit=limit,
            offset=offset,
        )
    finally:
        client.close()


@st.cache_data(ttl=DETECTION_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_detection_runs(
    api_base_url: str,
    *,
    meter_identifier: str,
    created_start: datetime | None = None,
    created_end: datetime | None = None,
    status: str | None,
    algorithm: str | None,
    limit: int,
    offset: int,
) -> DetectionRunsPageResponse:
    client = SmartMeterAPIClient(DashboardConfig(api_base_url=api_base_url))
    try:
        return client.list_detection_runs(
            meter_identifier=meter_identifier,
            created_start=created_start,
            created_end=created_end,
            status=status,
            algorithm=algorithm,
            limit=limit,
            offset=offset,
        )
    finally:
        client.close()


@st.cache_data(ttl=DETECTION_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_data_quality(
    api_base_url: str,
    *,
    meter_identifier: str,
    start: datetime,
    end: datetime,
) -> DataQualityResponse:
    client = SmartMeterAPIClient(DashboardConfig(api_base_url=api_base_url))
    try:
        return client.analytics_quality(
            meter_identifier,
            start=start,
            end=end,
        )
    finally:
        client.close()


def clear_detection_cache() -> None:
    fetch_anomalies.clear()
    fetch_detection_runs.clear()
    fetch_data_quality.clear()
