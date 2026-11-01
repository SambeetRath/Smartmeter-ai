from __future__ import annotations

from datetime import datetime

import streamlit as st

from smartmeter_dashboard.api_client import (
    AnalyticsSummaryResponse,
    AnomalySummaryResponse,
    SmartMeterAPIClient,
    TimeSeriesResponse,
)
from smartmeter_dashboard.config import DashboardConfig
from smartmeter_dashboard.detection_data import clear_detection_cache
from smartmeter_dashboard.exceptions import APIHTTPError

ANALYTICS_CACHE_TTL_SECONDS = 60
NO_COMPLETED_DETECTION_RUN_DETAIL = "No completed detection run found."


@st.cache_data(ttl=ANALYTICS_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_summary(
    api_base_url: str,
    meter_identifier: str,
    start: datetime,
    end: datetime,
) -> AnalyticsSummaryResponse:
    client = SmartMeterAPIClient(DashboardConfig(api_base_url=api_base_url))
    try:
        return client.analytics_summary(meter_identifier, start=start, end=end)
    finally:
        client.close()


@st.cache_data(ttl=ANALYTICS_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_timeseries(
    api_base_url: str,
    meter_identifier: str,
    start: datetime,
    end: datetime,
    interval: str,
) -> TimeSeriesResponse:
    client = SmartMeterAPIClient(DashboardConfig(api_base_url=api_base_url))
    try:
        return client.analytics_timeseries(
            meter_identifier,
            start=start,
            end=end,
            interval=interval,
        )
    finally:
        client.close()


@st.cache_data(ttl=ANALYTICS_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_anomaly_summary(
    api_base_url: str,
    meter_identifier: str,
) -> AnomalySummaryResponse | None:
    client = SmartMeterAPIClient(DashboardConfig(api_base_url=api_base_url))
    try:
        try:
            return client.anomaly_summary(meter_identifier=meter_identifier)
        except APIHTTPError as error:
            if (
                error.status_code == 404
                and error.endpoint == "/anomalies/summary"
                and error.detail == NO_COMPLETED_DETECTION_RUN_DETAIL
            ):
                return None
            raise
    finally:
        client.close()


def clear_analytics_cache() -> None:
    fetch_summary.clear()
    fetch_timeseries.clear()
    fetch_anomaly_summary.clear()
    clear_detection_cache()
