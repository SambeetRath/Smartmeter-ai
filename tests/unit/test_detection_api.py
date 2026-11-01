from __future__ import annotations

import asyncio
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from smartmeter.api.dependencies import get_db_session
from smartmeter.api.main import app

UTC = timezone.utc
METER_IDENTIFIER = "synthetic-detection-test"


class FakeSession:
    def begin(self):
        return nullcontext(self)


class DetectionState:
    def __init__(self) -> None:
        self.session = FakeSession()
        self.meter = SimpleNamespace(id=42, meter_identifier=METER_IDENTIFIER)
        self.run = _run()
        self.completed_run: SimpleNamespace | None = self.run
        self.created_run: SimpleNamespace | None = None
        self.created_range: tuple[datetime | None, datetime | None] | None = None
        self.listed_runs: list[SimpleNamespace] = [self.run]
        self.reading_count = 10
        self.active_run_count = 0
        self.results = []


def _run(*, status: str = "completed") -> SimpleNamespace:
    run_id = uuid4()
    return SimpleNamespace(
        id=run_id,
        meter_id=42,
        meter=SimpleNamespace(meter_identifier=METER_IDENTIFIER),
        algorithm="global_iqr",
        status=status,
        start_timestamp=None,
        end_timestamp=None,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
        completed_at=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        attempts=1,
        training_readings=8,
        readings_processed=2,
        anomaly_count=1,
        normal_count=1,
        unavailable_count=0,
        summary={"result_rows": 2},
        configuration={"algorithm": "global_iqr"},
        error_message=None,
    )


@pytest.fixture
def detection_api(monkeypatch):
    state = DetectionState()

    class FakeMeterRepository:
        def get_by_identifier(self, session, meter_identifier):
            assert session is state.session
            return state.meter if meter_identifier == METER_IDENTIFIER else None

        def lock_by_identifier(self, session, meter_identifier):
            return self.get_by_identifier(session, meter_identifier)

    class FakeReadingRepository:
        def count_for_meter(self, session, meter_id, *, start=None, end=None):
            assert session is state.session
            assert meter_id == 42
            state.created_range = (start, end)
            return state.reading_count

    class FakeDetectionRunRepository:
        def count_active_for_meter(self, session, meter_id):
            assert session is state.session
            assert meter_id == 42
            return state.active_run_count

        def create_pending(self, session, *, meter_id, algorithm, start, end):
            assert session is state.session
            state.created_run = _run(status="pending")
            state.created_run.algorithm = algorithm
            state.created_run.start_timestamp = start
            state.created_run.end_timestamp = end
            return state.created_run

        def get_by_id(self, session, run_id):
            assert session is state.session
            return state.run if state.run.id == run_id else None

        def latest_completed(self, session, *, meter_identifier=None):
            assert session is state.session
            if meter_identifier not in (None, METER_IDENTIFIER):
                return None
            return state.completed_run

        def list_runs(self, session, **_kwargs):
            assert session is state.session
            return state.listed_runs

    class FakeAnomalyResultRepository:
        def list_results(self, session, **_kwargs):
            assert session is state.session
            return state.results

        def category_counts(self, session, run_id):
            assert session is state.session
            assert run_id == state.run.id
            return {"high_consumption": 1}

    monkeypatch.setattr(
        "smartmeter.api.routes.detections.MeterRepository",
        FakeMeterRepository,
    )
    monkeypatch.setattr(
        "smartmeter.api.routes.detections.ReadingRepository",
        FakeReadingRepository,
    )
    monkeypatch.setattr(
        "smartmeter.api.routes.detections.DetectionRunRepository",
        FakeDetectionRunRepository,
    )
    monkeypatch.setattr(
        "smartmeter.api.routes.detections.AnomalyResultRepository",
        FakeAnomalyResultRepository,
    )

    def override_session():
        yield state.session

    app.dependency_overrides[get_db_session] = override_session
    try:
        yield state
    finally:
        app.dependency_overrides.pop(get_db_session, None)


async def _request(
    application: FastAPI,
    method: str,
    path: str,
    *,
    payload: dict | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, json=payload)


def test_detection_routes_are_registered_without_database_access():
    schema = app.openapi()
    assert {
        "/detection/runs",
        "/detection/runs/{run_id}",
        "/anomalies",
        "/anomalies/summary",
    }.issubset(schema["paths"])


def test_create_detection_run_normalizes_time_range_and_returns_accepted(detection_api):
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/detection/runs",
            payload={
                "meter_identifier": METER_IDENTIFIER,
                "algorithm": "global_iqr",
                "start": "2024-01-01T01:00:00+01:00",
                "end": "2024-01-01T02:00:00+01:00",
            },
        )
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["run_id"] == str(detection_api.created_run.id)
    assert detection_api.created_range == (
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
    )


def test_create_detection_run_unknown_meter_returns_404(detection_api):
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/detection/runs",
            payload={"meter_identifier": "missing-meter"},
        )
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Meter not found."}
    assert detection_api.created_run is None


def test_create_detection_run_rejects_blank_meter_identifier(detection_api):
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/detection/runs",
            payload={"meter_identifier": "   "},
        )
    )

    assert response.status_code == 422


def test_create_detection_run_rejects_too_few_readings(detection_api):
    detection_api.reading_count = 2
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/detection/runs",
            payload={"meter_identifier": METER_IDENTIFIER},
        )
    )

    assert response.status_code == 422
    assert "At least three readings" in response.json()["detail"]
    assert detection_api.created_run is None


def test_create_detection_run_rejects_ranges_over_90_days(detection_api):
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/detection/runs",
            payload={
                "meter_identifier": METER_IDENTIFIER,
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-04-01T00:00:00Z",
            },
        )
    )

    assert response.status_code == 422
    assert "must not exceed 90 days" in str(response.json()["detail"])


def test_create_detection_run_rejects_more_than_maximum_readings(detection_api):
    detection_api.reading_count = 100_001

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/detection/runs",
            payload={"meter_identifier": METER_IDENTIFIER},
        )
    )

    assert response.status_code == 422
    assert "100,000 readings" in response.json()["detail"]
    assert detection_api.created_run is None


def test_create_detection_run_rejects_duplicate_active_run(detection_api):
    detection_api.active_run_count = 1

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/detection/runs",
            payload={"meter_identifier": METER_IDENTIFIER},
        )
    )

    assert response.status_code == 409
    assert "already pending or running" in response.json()["detail"]
    assert detection_api.created_run is None


def test_create_detection_run_rejects_naive_and_invalid_ranges(detection_api):
    naive = asyncio.run(
        _request(
            app,
            "POST",
            "/detection/runs",
            payload={
                "meter_identifier": METER_IDENTIFIER,
                "start": "2024-01-01T00:00:00",
            },
        )
    )
    invalid = asyncio.run(
        _request(
            app,
            "POST",
            "/detection/runs",
            payload={
                "meter_identifier": METER_IDENTIFIER,
                "start": "2024-01-02T00:00:00Z",
                "end": "2024-01-01T00:00:00Z",
            },
        )
    )

    assert naive.status_code == 422
    assert invalid.status_code == 422
    assert detection_api.created_run is None


def test_list_detection_runs_has_bounded_pagination(detection_api):
    detection_api.listed_runs = [_run(), _run()]

    response = asyncio.run(_request(app, "GET", "/detection/runs?limit=1&offset=0"))

    assert response.status_code == 200
    assert response.json()["returned_count"] == 1
    assert response.json()["has_more"] is True
    assert response.json()["next_offset"] == 1


def test_list_detection_runs_rejects_invalid_range_and_oversized_page(detection_api):
    invalid_range = asyncio.run(
        _request(
            app,
            "GET",
            "/detection/runs?created_start=2024-01-02T00%3A00%3A00Z"
            "&created_end=2024-01-01T00%3A00%3A00Z",
        )
    )
    oversized_page = asyncio.run(_request(app, "GET", "/detection/runs?limit=1001"))
    oversized_offset = asyncio.run(_request(app, "GET", "/detection/runs?offset=100001"))

    assert invalid_range.status_code == 422
    assert oversized_page.status_code == 422
    assert oversized_offset.status_code == 422


def test_get_detection_run_returns_run_metadata(detection_api):
    response = asyncio.run(_request(app, "GET", f"/detection/runs/{detection_api.run.id}"))

    assert response.status_code == 200
    assert response.json()["meter_identifier"] == METER_IDENTIFIER
    assert response.json()["readings_processed"] == 2


def test_get_detection_run_returns_404_for_unknown_run(detection_api):
    response = asyncio.run(_request(app, "GET", f"/detection/runs/{uuid4()}"))

    assert response.status_code == 404


def test_anomaly_summary_uses_single_run_counts(detection_api):
    response = asyncio.run(
        _request(app, "GET", f"/anomalies/summary?run_id={detection_api.run.id}")
    )

    assert response.status_code == 200
    assert response.json()["readings_processed"] == 2
    assert response.json()["anomaly_count"] == 1
    assert response.json()["anomaly_rate"] == 0.5
    assert response.json()["counts_by_algorithm"] == {"global_iqr": 1}


def test_anomaly_summary_returns_expected_404_when_no_completed_run_exists(detection_api):
    detection_api.completed_run = None

    response = asyncio.run(
        _request(
            app,
            "GET",
            f"/anomalies/summary?meter_identifier={METER_IDENTIFIER}",
        )
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "No completed detection run found."


def test_anomaly_summary_distinguishes_unknown_meter(detection_api):
    response = asyncio.run(
        _request(app, "GET", "/anomalies/summary?meter_identifier=unknown-meter")
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Meter not found."


def test_anomaly_summary_rejects_unfinished_run(detection_api):
    detection_api.run.status = "running"
    response = asyncio.run(
        _request(app, "GET", f"/anomalies/summary?run_id={detection_api.run.id}")
    )

    assert response.status_code == 409


def test_anomaly_list_serializes_missing_score_and_metadata_as_null(detection_api):
    run = detection_api.run
    reading = SimpleNamespace(
        id=7,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    detection_api.results = [
        (
            SimpleNamespace(
                prediction_status="anomaly",
                anomaly_score=None,
                anomaly_category=None,
                metadata_json=None,
            ),
            run,
            detection_api.meter,
            reading,
        )
    ]

    response = asyncio.run(_request(app, "GET", "/anomalies"))

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["anomaly_score"] is None
    assert item["anomaly_category"] is None
    assert item["metadata"] is None
