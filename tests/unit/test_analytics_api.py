from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx
import pytest
from fastapi import FastAPI

from smartmeter.api.dependencies import get_db_session
from smartmeter.api.main import app
from smartmeter.services.analytics_service import AnalyticsService

UTC = timezone.utc
METER_IDENTIFIER = "analytics-unit-meter"


class AnalyticsState:
    def __init__(self) -> None:
        self.session = object()
        self.summary = {
            "reading_count": 2,
            "active_power_average": 2.0,
            "active_power_minimum": 1.0,
            "active_power_maximum": 3.0,
            "latest_active_power": 3.0,
            "latest_active_power_at": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            "missing_measurements": {
                name: {
                    "missing_count": int(name == "voltage"),
                    "missing_rate": 0.5 if name == "voltage" else 0.0,
                }
                for name in (
                    "global_active_power",
                    "global_reactive_power",
                    "voltage",
                    "global_intensity",
                    "sub_metering_1",
                    "sub_metering_2",
                    "sub_metering_3",
                )
            },
        }
        self.quality = {
            "reading_count": 2,
            "missing_measurements": self.summary["missing_measurements"],
            "timestamp_gaps": {
                "expected_interval_minutes": 1,
                "observed_gap_count": 0,
                "estimated_missing_intervals": 0,
                "first_observed_at": datetime(2024, 1, 1, tzinfo=UTC),
                "last_observed_at": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                "boundary_gaps_assessed": False,
            },
            "database_metrics_scope": "Computed from stored readings in the requested interval.",
        }
        self.buckets = [
            {
                "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
                "reading_count": 2,
                "global_active_power_average": 2.0,
                "global_active_power_minimum": 1.0,
                "global_active_power_maximum": 3.0,
                "global_reactive_power_average": 0.1,
                "voltage_average": 230.0,
                "global_intensity_average": 5.0,
                "sub_metering_1_average": 1.0,
                "sub_metering_2_average": 2.0,
                "sub_metering_3_average": 3.0,
            }
        ]
        self.calls = []


@pytest.fixture
def analytics_api(monkeypatch):
    state = AnalyticsState()

    class FakeAnalyticsService:
        def list_meters(self, session, *, limit, offset):
            assert session is state.session
            return [
                {
                    "meter_identifier": METER_IDENTIFIER,
                    "reading_count": 2,
                    "first_reading_at": datetime(2024, 1, 1, tzinfo=UTC),
                    "last_reading_at": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                }
            ][:limit]

        def summary(self, session, meter_identifier, *, start, end):
            state.calls.append(("summary", start, end))
            if meter_identifier != METER_IDENTIFIER:
                return None
            start_utc, end_utc = AnalyticsService.normalize_range(start, end)
            state.calls[-1] = ("summary", start_utc, end_utc)
            return state.summary

        def time_series(self, session, meter_identifier, *, start, end, interval):
            state.calls.append(("timeseries", interval))
            if meter_identifier != METER_IDENTIFIER:
                return None
            start_utc, end_utc = AnalyticsService.normalize_range(
                start, end, require_both=True, enforce_max_range=True
            )
            return start_utc, end_utc, state.buckets

        def quality(self, session, meter_identifier, *, start, end):
            state.calls.append(("quality", start, end))
            if meter_identifier != METER_IDENTIFIER:
                return None
            return state.quality

        @staticmethod
        def normalize_range(start, end):
            return AnalyticsService.normalize_range(start, end)

    monkeypatch.setattr(
        "smartmeter.api.routes.analytics.AnalyticsService",
        FakeAnalyticsService,
    )

    def override_session():
        yield state.session

    app.dependency_overrides[get_db_session] = override_session
    try:
        yield state
    finally:
        app.dependency_overrides.pop(get_db_session, None)


async def _get(application: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path)


def test_analytics_routes_are_registered():
    assert {
        "/meters",
        "/analytics/summary",
        "/analytics/timeseries",
        "/analytics/quality",
    }.issubset(app.openapi()["paths"])


def test_measurement_stats_rejects_impossible_missing_rate():
    from pydantic import ValidationError

    from smartmeter.api.analytics_schemas import MeasurementStats

    with pytest.raises(ValidationError):
        MeasurementStats(missing_count=2, missing_rate=1.5)


def test_meter_list_has_coverage_and_pagination(analytics_api):
    response = asyncio.run(_get(app, "/meters?limit=1"))

    assert response.status_code == 200
    assert response.json()["items"][0]["meter_identifier"] == METER_IDENTIFIER
    assert response.json()["items"][0]["reading_count"] == 2
    assert response.json()["returned_count"] == 1
    assert response.json()["has_more"] is False


def test_summary_returns_metrics_and_normalizes_date_filter(analytics_api):
    response = asyncio.run(
        _get(
            app,
            "/analytics/summary?"
            + urlencode(
                {
                    "meter_identifier": METER_IDENTIFIER,
                    "start": "2024-01-01T01:00:00+01:00",
                    "end": "2024-01-01T00:02:00Z",
                }
            ),
        )
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reading_count"] == 2
    assert payload["active_power_average"] == 2.0
    assert payload["missing_measurements"]["voltage"] == {
        "missing_count": 1,
        "missing_rate": 0.5,
    }
    assert analytics_api.calls[0][1:] == (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
    )


def test_summary_unknown_meter_and_invalid_range(analytics_api):
    unknown = asyncio.run(_get(app, "/analytics/summary?meter_identifier=missing"))
    invalid = asyncio.run(
        _get(
            app,
            "/analytics/summary?meter_identifier=analytics-unit-meter"
            "&start=2024-01-02T00%3A00%3A00Z&end=2024-01-01T00%3A00%3A00Z",
        )
    )

    assert unknown.status_code == 404
    assert invalid.status_code == 422


@pytest.mark.parametrize("interval", ["hour", "day"])
def test_timeseries_serializes_bounded_buckets(analytics_api, interval):
    response = asyncio.run(
        _get(
            app,
            "/analytics/timeseries?"
            + urlencode(
                {
                    "meter_identifier": METER_IDENTIFIER,
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-01-02T00:00:00Z",
                    "interval": interval,
                }
            ),
        )
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["interval"] == interval
    assert payload["returned_count"] == 1
    assert payload["buckets"][0]["reading_count"] == 2
    assert analytics_api.calls[-1] == ("timeseries", interval)


def test_timeseries_requires_aware_valid_bounded_range(analytics_api):
    missing_range = asyncio.run(
        _get(app, f"/analytics/timeseries?meter_identifier={METER_IDENTIFIER}")
    )
    naive_range = asyncio.run(
        _get(
            app,
            "/analytics/timeseries?meter_identifier=analytics-unit-meter"
            "&start=2024-01-01T00%3A00%3A00&end=2024-01-02T00%3A00%3A00Z",
        )
    )
    oversized_range = asyncio.run(
        _get(
            app,
            "/analytics/timeseries?meter_identifier=analytics-unit-meter"
            "&start=2024-01-01T00%3A00%3A00Z&end=2024-03-01T00%3A00%3A00Z",
        )
    )

    assert missing_range.status_code == 422
    assert naive_range.status_code == 422
    assert oversized_range.status_code == 422


def test_timeseries_unknown_meter_and_empty_results(analytics_api):
    unknown = asyncio.run(
        _get(
            app,
            "/analytics/timeseries?meter_identifier=missing"
            "&start=2024-01-01T00%3A00%3A00Z&end=2024-01-02T00%3A00%3A00Z",
        )
    )
    analytics_api.buckets = []
    empty = asyncio.run(
        _get(
            app,
            "/analytics/timeseries?meter_identifier=analytics-unit-meter"
            "&start=2024-01-01T00%3A00%3A00Z&end=2024-01-02T00%3A00%3A00Z",
        )
    )

    assert unknown.status_code == 404
    assert empty.status_code == 200
    assert empty.json()["buckets"] == []


def test_quality_labels_database_metrics_and_unavailable_ingestion_report(analytics_api):
    response = asyncio.run(_get(app, f"/analytics/quality?meter_identifier={METER_IDENTIFIER}"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["timestamp_gaps"]["expected_interval_minutes"] == 1
    assert payload["timestamp_gaps"]["boundary_gaps_assessed"] is False
    assert payload["ingestion_validation"]["available"] is False
    assert "cannot be reconstructed" in payload["ingestion_validation"]["detail"]


def test_quality_rejects_naive_boundary(analytics_api):
    response = asyncio.run(
        _get(
            app,
            "/analytics/quality?meter_identifier=analytics-unit-meter"
            "&start=2024-01-01T00%3A00%3A00",
        )
    )

    assert response.status_code == 422
