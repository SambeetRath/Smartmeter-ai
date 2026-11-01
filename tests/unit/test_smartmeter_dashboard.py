from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest
from pydantic import ValidationError

from smartmeter_dashboard.api_client import (
    AnalyticsSummaryResponse,
    APIConnectionError,
    APIHTTPError,
    APIRequestTimeout,
    APIResponseError,
    SmartMeterAPIClient,
)
from smartmeter_dashboard.config import (
    DEFAULT_API_BASE_URL,
    ConfigurationError,
    DashboardConfig,
)
from smartmeter_dashboard.transformations import (
    analytics_summary_dataframe,
    anomalies_dataframe,
    detection_runs_dataframe,
    meters_dataframe,
    missing_measurements_dataframe,
    quality_summary_dataframe,
    timeseries_dataframe,
)

UTC = timezone.utc
TESTS_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = TESTS_ROOT.parent / "src" / "smartmeter_dashboard"


def _client(handler) -> tuple[SmartMeterAPIClient, httpx.Client]:
    http_client = httpx.Client(
        base_url="http://api.example.test",
        transport=httpx.MockTransport(handler),
    )
    return SmartMeterAPIClient(DashboardConfig(), client=http_client), http_client


def _meter_page(items, *, limit=1000, offset=0, has_more=False, next_offset=None):
    return {
        "items": items,
        "limit": limit,
        "offset": offset,
        "returned_count": len(items),
        "has_more": has_more,
        "next_offset": next_offset,
    }


def _meter(identifier="test-meter"):
    return {
        "meter_identifier": identifier,
        "reading_count": 2,
        "first_reading_at": "2024-01-01T00:00:00Z",
        "last_reading_at": "2024-01-01T00:01:00Z",
    }


def test_config_uses_local_api_default(monkeypatch):
    monkeypatch.delenv("SMARTMETER_API_BASE_URL", raising=False)

    assert DashboardConfig.from_env().api_base_url == DEFAULT_API_BASE_URL


def test_config_normalizes_url_and_rejects_bad_or_credentialed_values(monkeypatch):
    monkeypatch.setenv("SMARTMETER_API_BASE_URL", "https://api.example.test/v1/")

    assert DashboardConfig.from_env().api_base_url == "https://api.example.test/v1"
    for value in (
        "",
        "ftp://api.example.test",
        "http://user:password@api.example.test",
        "http://api.example.test?token=secret",
        "http://api.example.test/#fragment",
        "http://api.example.test:invalid",
        "http://[invalid",
    ):
        with pytest.raises(ConfigurationError):
            DashboardConfig(api_base_url=value)


def test_health_and_paginated_meter_listing():
    seen = []

    def handler(request):
        seen.append((request.url.path, dict(request.url.params)))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "service": "smartmeter-ai"})
        offset = int(request.url.params["offset"])
        if offset == 0:
            return httpx.Response(
                200,
                json=_meter_page([_meter("meter one")], has_more=True, next_offset=1),
            )
        return httpx.Response(200, json=_meter_page([_meter("meter two")], offset=1))

    api, http_client = _client(handler)
    try:
        assert api.health().status == "ok"
        meters = api.list_meters(limit=1)
    finally:
        api.close()
        http_client.close()

    assert [meter.meter_identifier for meter in meters] == ["meter one", "meter two"]
    assert seen == [
        ("/health", {}),
        ("/meters", {"limit": "1", "offset": "0"}),
        ("/meters", {"limit": "1", "offset": "1"}),
    ]


def test_empty_meter_page_is_not_an_error():
    api, http_client = _client(lambda request: httpx.Response(200, json=_meter_page([])))
    try:
        assert api.list_meters() == []
    finally:
        http_client.close()


def test_http_client_uses_explicit_connection_and_read_timeouts():
    api = SmartMeterAPIClient(DashboardConfig())
    try:
        assert api._client.timeout.connect == 3.0
        assert api._client.timeout.read == 10.0
        assert api._client.timeout.write == 5.0
        assert api._client.timeout.pool == 3.0
    finally:
        api.close()


def test_meter_pagination_rejects_invalid_continuation():
    api, http_client = _client(
        lambda request: httpx.Response(
            200,
            json=_meter_page([_meter()], has_more=True, next_offset=0),
        )
    )
    try:
        with pytest.raises(APIResponseError, match="continuation offset"):
            api.list_meters(limit=1)
    finally:
        http_client.close()


def test_client_surfaces_http_error_without_credentials():
    api, http_client = _client(
        lambda request: httpx.Response(503, json={"detail": "Backend unavailable"})
    )
    try:
        with pytest.raises(APIHTTPError, match="HTTP 503") as error:
            api.health()
    finally:
        http_client.close()

    assert error.value.status_code == 503
    assert error.value.detail == "Backend unavailable"
    assert "Backend unavailable" not in str(error.value)


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (httpx.ConnectError("refused"), APIConnectionError),
        (httpx.ReadTimeout("slow"), APIRequestTimeout),
    ],
)
def test_client_surfaces_connection_errors(exception, expected):
    def handler(request):
        raise exception

    api, http_client = _client(handler)
    try:
        with pytest.raises(expected):
            api.health()
    finally:
        http_client.close()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, content=b"not-json"), "invalid JSON"),
        (httpx.Response(200, json={"status": "ok"}), "does not match its contract"),
        (httpx.Response(200, json=[]), "unexpected response shape"),
    ],
)
def test_client_rejects_invalid_json_and_response_shapes(response, message):
    api, http_client = _client(lambda request: response)
    try:
        with pytest.raises(APIResponseError, match=message):
            api.health()
    finally:
        http_client.close()


def test_client_analytics_contract_and_query_parameter_encoding():
    requests = []
    summary = {
        "meter_identifier": "meter & one",
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-02T00:00:00Z",
        "reading_count": 1,
        "active_power_average": 1.0,
        "active_power_minimum": 1.0,
        "active_power_maximum": 1.0,
        "latest_active_power": 1.0,
        "latest_active_power_at": "2024-01-01T00:00:00Z",
        "missing_measurements": {
            column: {"missing_count": 0, "missing_rate": 0.0}
            for column in (
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

    def handler(request):
        requests.append(request)
        if request.url.path == "/analytics/summary":
            return httpx.Response(200, json=summary)
        raise AssertionError(f"Unexpected URL {request.url.path}")

    api, http_client = _client(handler)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    try:
        result = api.analytics_summary("meter & one", start=start, end=end)
    finally:
        http_client.close()

    assert isinstance(result, AnalyticsSummaryResponse)
    assert result.reading_count == 1
    assert requests[0].url.params["meter_identifier"] == "meter & one"
    assert requests[0].url.params["start"] == start.isoformat()


def test_client_supports_timeseries_quality_and_detection_routes():
    requests = []

    def handler(request):
        requests.append(request)
        path = request.url.path
        if path == "/analytics/timeseries":
            return httpx.Response(
                200,
                json={
                    "meter_identifier": "m",
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-01-02T00:00:00Z",
                    "interval": "day",
                    "buckets": [],
                    "returned_count": 0,
                },
            )
        if path == "/analytics/quality":
            return httpx.Response(
                200,
                json={
                    "meter_identifier": "m",
                    "start": None,
                    "end": None,
                    "reading_count": 0,
                    "missing_measurements": {},
                    "timestamp_gaps": {
                        "expected_interval_minutes": 1,
                        "observed_gap_count": 0,
                        "estimated_missing_intervals": 0,
                        "first_observed_at": None,
                        "last_observed_at": None,
                        "boundary_gaps_assessed": False,
                    },
                    "database_metrics_scope": "db",
                    "ingestion_validation": {"available": False, "detail": "not stored"},
                },
            )
        if path == "/detection/runs":
            if request.method == "POST":
                return httpx.Response(
                    202,
                    json={
                        "run_id": "c0bfe534-57fc-4ed8-bd06-cbf482b47861",
                        "status": "pending",
                        "created_at": "2024-01-01T00:00:00Z",
                        "message": "queued",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "items": [],
                    "limit": 10,
                    "offset": 0,
                    "returned_count": 0,
                    "has_more": False,
                    "next_offset": None,
                },
            )
        if path.startswith("/detection/runs/"):
            return httpx.Response(200, json=_run())
        if path == "/anomalies":
            return httpx.Response(
                200,
                json={
                    "items": [],
                    "prediction_status": "anomaly",
                    "limit": 100,
                    "offset": 0,
                    "returned_count": 0,
                    "has_more": False,
                    "next_offset": None,
                },
            )
        if path == "/anomalies/summary":
            return httpx.Response(200, json=_anomaly_summary())
        raise AssertionError(f"Unexpected URL {path}")

    api, http_client = _client(handler)
    try:
        assert (
            api.analytics_timeseries(
                "m",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
                interval="day",
            ).buckets
            == []
        )
        assert api.analytics_quality("m").reading_count == 0
        assert api.list_detection_runs(limit=10).items == []
        assert api.get_detection_run("c0bfe534-57fc-4ed8-bd06-cbf482b47861").status == "completed"
        assert api.create_detection_run("m").status == "pending"
        assert api.list_anomalies(prediction_status="anomaly").items == []
        assert api.anomaly_summary(meter_identifier="m").anomaly_count == 1
    finally:
        http_client.close()

    assert {request.url.path for request in requests} == {
        "/analytics/timeseries",
        "/analytics/quality",
        "/detection/runs",
        "/detection/runs/c0bfe534-57fc-4ed8-bd06-cbf482b47861",
        "/anomalies",
        "/anomalies/summary",
    }


def _run():
    return {
        "run_id": "c0bfe534-57fc-4ed8-bd06-cbf482b47861",
        "meter_identifier": "m",
        "algorithm": "global_iqr",
        "status": "completed",
        "start": None,
        "end": None,
        "created_at": "2024-01-01T00:00:00Z",
        "started_at": None,
        "completed_at": "2024-01-01T00:01:00Z",
        "attempts": 1,
        "training_readings": 10,
        "readings_processed": 2,
        "anomaly_count": 1,
        "normal_count": 1,
        "unavailable_count": 0,
        "summary": None,
        "configuration": None,
        "error_message": None,
    }


def _anomaly_summary():
    return {
        "run_id": "c0bfe534-57fc-4ed8-bd06-cbf482b47861",
        "meter_identifier": "m",
        "algorithm": "global_iqr",
        "run_status": "completed",
        "readings_processed": 2,
        "anomaly_count": 1,
        "normal_count": 1,
        "unavailable_count": 0,
        "anomaly_rate": 0.5,
        "counts_by_category": {"high": 1},
        "counts_by_algorithm": {"global_iqr": 1},
    }


def test_transformations_produce_stable_empty_frames():
    assert list(meters_dataframe([]).columns) == [
        "meter_identifier",
        "reading_count",
        "first_reading_at",
        "last_reading_at",
    ]
    assert list(timeseries_dataframe([]).columns)[0] == "timestamp"
    assert list(anomalies_dataframe([]).columns)[0] == "run_id"
    assert list(detection_runs_dataframe([]).columns)[0] == "run_id"
    assert missing_measurements_dataframe({}).empty
    assert analytics_summary_dataframe(None).empty
    assert quality_summary_dataframe(None).empty


def test_transformations_preserve_api_metrics_and_datetime_types():
    meter_frame = meters_dataframe([_meter()])
    timeseries_frame = timeseries_dataframe(
        {
            "buckets": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "reading_count": 4,
                    "global_active_power_average": 1.25,
                }
            ]
        }
    )
    anomaly_frame = anomalies_dataframe(
        {
            "items": [
                {
                    "run_id": "c0bfe534-57fc-4ed8-bd06-cbf482b47861",
                    "reading_id": 1,
                    "meter_identifier": "m",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "prediction_status": "anomaly",
                    "algorithm": "global_iqr",
                    "anomaly_score": None,
                    "anomaly_category": None,
                }
            ]
        }
    )
    summary_frame = analytics_summary_dataframe(
        {
            "meter_identifier": "m",
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-02T00:00:00Z",
            "reading_count": 10,
            "active_power_average": 1.5,
            "active_power_minimum": 0.5,
            "active_power_maximum": 2.5,
            "latest_active_power": 2.0,
            "latest_active_power_at": "2024-01-01T23:59:00Z",
        }
    )
    quality_frame = quality_summary_dataframe(
        {
            "meter_identifier": "m",
            "reading_count": 10,
            "database_metrics_scope": "stored readings",
            "timestamp_gaps": {
                "expected_interval_minutes": 1,
                "observed_gap_count": 1,
                "estimated_missing_intervals": 2,
                "boundary_gaps_assessed": False,
            },
            "ingestion_validation": {"available": False},
        }
    )

    assert pd.api.types.is_datetime64_any_dtype(meter_frame["first_reading_at"])
    assert pd.api.types.is_datetime64_any_dtype(timeseries_frame["timestamp"])
    assert timeseries_frame.loc[0, "global_active_power_average"] == 1.25
    assert pd.api.types.is_datetime64_any_dtype(anomaly_frame["timestamp"])
    assert pd.isna(anomaly_frame.loc[0, "anomaly_score"])
    missing_frame = missing_measurements_dataframe(
        {
            "voltage": AnalyticsSummaryResponse.model_validate(
                {
                    "meter_identifier": "m",
                    "start": None,
                    "end": None,
                    "reading_count": 1,
                    "active_power_average": None,
                    "active_power_minimum": None,
                    "active_power_maximum": None,
                    "latest_active_power": None,
                    "latest_active_power_at": None,
                    "missing_measurements": {"voltage": {"missing_count": 2, "missing_rate": 0.25}},
                }
            ).missing_measurements["voltage"]
        }
    )
    assert missing_frame.loc[0, "missing_rate"] == 0.25
    assert pd.api.types.is_datetime64_any_dtype(summary_frame["latest_active_power_at"])
    assert summary_frame.loc[0, "active_power_average"] == 1.5
    assert quality_frame.loc[0, "estimated_missing_intervals"] == 2
    assert not quality_frame.loc[0, "ingestion_validation_available"]


def test_dashboard_source_never_reads_database_url_or_database_packages():
    for source_file in SOURCE_ROOT.rglob("*.py"):
        source = source_file.read_text(encoding="utf-8")
        assert "DATABASE_URL" not in source
        assert "smartmeter.database" not in source


def test_dashboard_modules_do_not_import_api_models():
    from smartmeter_dashboard import api_client

    assert api_client.__name__ == "smartmeter_dashboard.api_client"


def test_streamlit_shell_reports_empty_meter_list(monkeypatch):
    import streamlit as st

    from smartmeter_dashboard.app import main

    class FakeAPIClient:
        def __init__(self, _config):
            pass

        def health(self):
            return type("Health", (), {"service": "smartmeter-ai"})()

        def list_meters(self):
            return []

        def close(self):
            pass

    messages = []
    monkeypatch.setattr("smartmeter_dashboard.app.SmartMeterAPIClient", FakeAPIClient)
    monkeypatch.setattr(st, "set_page_config", lambda **_kwargs: None)
    monkeypatch.setattr(st, "title", lambda value: messages.append(value))
    monkeypatch.setattr(st, "caption", lambda value: messages.append(value))
    monkeypatch.setattr(st, "success", lambda value: messages.append(value))
    monkeypatch.setattr(st, "info", lambda value: messages.append(value))
    monkeypatch.setattr(st, "error", lambda value: messages.append(value))
    monkeypatch.setattr(st, "spinner", lambda _message: nullcontext())

    main()

    assert "SmartMeter AI" in messages
    assert any("No meters are registered" in message for message in messages)


def test_streamlit_shell_shows_selected_meter_coverage(monkeypatch):
    import streamlit as st

    import smartmeter_dashboard.app as app
    from smartmeter_dashboard.api_client import MeterCoverage

    messages = []
    state = {}

    class FakeSidebar:
        def subheader(self, _message):
            pass

        def button(self, _label, **_kwargs):
            return False

        def radio(self, _label, *, options, key, **_kwargs):
            state.setdefault(key, options[0])
            return state[key]

        def selectbox(self, _label, *, options, key, **_kwargs):
            state.setdefault(key, options[0])
            return state[key]

        def date_input(self, _label, *, key, **_kwargs):
            assert key in state
            return state[key]

        def success(self, message):
            messages.append(message)

        def error(self, message):
            messages.append(message)

        def info(self, message):
            messages.append(message)

    monkeypatch.setattr(st, "sidebar", FakeSidebar())
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(st, "set_page_config", lambda **_kwargs: None)
    monkeypatch.setattr(st, "title", lambda value: messages.append(value))
    monkeypatch.setattr(st, "caption", lambda value: messages.append(value))
    monkeypatch.setattr(st, "info", lambda value: messages.append(value))
    monkeypatch.setattr(st, "error", lambda value: messages.append(value))
    monkeypatch.setattr(st, "spinner", lambda _message: nullcontext())
    monkeypatch.setattr(app, "check_api", lambda _url: "smartmeter-ai")
    monkeypatch.setattr(app, "render_page", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app,
        "get_meters",
        lambda _url: [
            MeterCoverage(
                meter_identifier="demo-meter",
                reading_count=20,
                first_reading_at=datetime(2024, 1, 1, tzinfo=UTC),
                last_reading_at=datetime(2024, 1, 2, tzinfo=UTC),
            )
        ],
    )

    app.main()

    assert any("Meter: demo-meter" in message for message in messages)
    assert any("2024-01-01 to 2024-01-02" in message for message in messages)


def test_analytics_response_model_rejects_invalid_measurement_rates():
    with pytest.raises(ValidationError):
        AnalyticsSummaryResponse.model_validate(
            {
                "meter_identifier": "m",
                "start": None,
                "end": None,
                "reading_count": 1,
                "active_power_average": None,
                "active_power_minimum": None,
                "active_power_maximum": None,
                "latest_active_power": None,
                "latest_active_power_at": None,
                "missing_measurements": {"voltage": {"missing_count": 2, "missing_rate": 1.5}},
            }
        )
