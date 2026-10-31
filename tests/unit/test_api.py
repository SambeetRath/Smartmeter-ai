from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.dialects import postgresql

from smartmeter.api.dependencies import get_db_session
from smartmeter.api.main import app, create_app
from smartmeter.database.repositories import ReadingRepository

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METER_IDENTIFIER = "synthetic-test-meter"
UTC = timezone.utc


class FakeMeterRepository:
    def __init__(self, meter: SimpleNamespace | None, expected_session) -> None:
        self.meter = meter
        self.expected_session = expected_session

    def get_by_identifier(self, session, meter_identifier):
        assert session is self.expected_session
        assert meter_identifier == METER_IDENTIFIER
        return self.meter


class FakeReadingRepository:
    def __init__(self, readings: list[SimpleNamespace], expected_session) -> None:
        self.readings = readings
        self.expected_session = expected_session
        self.call = None

    def get_for_meter(self, session, meter_id, **kwargs):
        assert session is self.expected_session
        assert meter_id == 42
        self.call = kwargs
        start = kwargs["start"]
        end = kwargs["end"]
        filtered = [
            reading
            for reading in self.readings
            if (start is None or reading.timestamp >= start)
            and (end is None or reading.timestamp < end)
        ]
        return filtered[kwargs["offset"] : kwargs["offset"] + kwargs["limit"]]


def _reading(minute: int, *, active_power: float | None = 1.25) -> SimpleNamespace:
    return SimpleNamespace(
        id=minute + 1,
        timestamp=datetime(2024, 1, 1, 0, minute, tzinfo=UTC),
        global_active_power=active_power,
        global_reactive_power=0.1,
        voltage=230.0,
        global_intensity=5.0,
        sub_metering_1=None,
        sub_metering_2=1.0,
        sub_metering_3=2.0,
    )


def _readings_url(**params: str | int) -> str:
    return f"/readings?{urlencode({'meter_identifier': METER_IDENTIFIER, **params})}"


@pytest.fixture
def api_reading_repositories(monkeypatch):
    session = object()
    meter_repository = FakeMeterRepository(
        SimpleNamespace(id=42, meter_identifier=METER_IDENTIFIER),
        session,
    )
    reading_repository = FakeReadingRepository(
        [_reading(minute) for minute in range(5)],
        session,
    )
    monkeypatch.setattr(
        "smartmeter.api.routes.readings.MeterRepository",
        lambda: meter_repository,
    )
    monkeypatch.setattr(
        "smartmeter.api.routes.readings.ReadingRepository",
        lambda: reading_repository,
    )

    def override_session():
        yield session

    app.dependency_overrides[get_db_session] = override_session
    try:
        yield session, meter_repository, reading_repository
    finally:
        app.dependency_overrides.pop(get_db_session, None)


async def _get(
    application: FastAPI,
    path: str,
    *,
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    transport = httpx.ASGITransport(
        app=application,
        raise_app_exceptions=raise_app_exceptions,
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path)


def test_application_imports_and_health_returns_expected_response():
    response = asyncio.run(_get(app, "/health"))

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "smartmeter-ai"}


def test_openapi_schema_is_available_and_describes_health_route():
    response = asyncio.run(_get(app, "/openapi.json"))

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "SmartMeter AI API"
    assert schema["info"]["version"] == "0.1.0"
    assert "/health" in schema["paths"]
    assert "/readings" in schema["paths"]


def test_readings_request_returns_ordered_numeric_items_and_pagination(api_reading_repositories):
    session, _, reading_repository = api_reading_repositories
    response = asyncio.run(
        _get(
            app,
            _readings_url(limit=2),
        )
    )

    assert response.status_code == 200
    assert response.json() == {
        "meter_identifier": METER_IDENTIFIER,
        "items": [
            {
                "id": 1,
                "timestamp": "2024-01-01T00:00:00Z",
                "global_active_power": 1.25,
                "global_reactive_power": 0.1,
                "voltage": 230.0,
                "global_intensity": 5.0,
                "sub_metering_1": None,
                "sub_metering_2": 1.0,
                "sub_metering_3": 2.0,
            },
            {
                "id": 2,
                "timestamp": "2024-01-01T00:01:00Z",
                "global_active_power": 1.25,
                "global_reactive_power": 0.1,
                "voltage": 230.0,
                "global_intensity": 5.0,
                "sub_metering_1": None,
                "sub_metering_2": 1.0,
                "sub_metering_3": 2.0,
            },
        ],
        "limit": 2,
        "offset": 0,
        "returned_count": 2,
        "has_more": True,
        "next_offset": 2,
    }
    assert reading_repository.call["limit"] == 3
    assert reading_repository.call["offset"] == 0
    assert session is not None


def test_readings_unknown_meter_returns_404(api_reading_repositories, monkeypatch):
    _, _, reading_repository = api_reading_repositories
    session, _, _ = api_reading_repositories
    missing_meter = FakeMeterRepository(None, session)
    monkeypatch.setattr("smartmeter.api.routes.readings.MeterRepository", lambda: missing_meter)

    response = asyncio.run(_get(app, f"/readings?meter_identifier={METER_IDENTIFIER}"))

    assert response.status_code == 404
    assert response.json() == {"detail": "Meter not found."}
    assert reading_repository.call is None


def test_readings_empty_result_is_success_with_empty_page(api_reading_repositories):
    _, _, reading_repository = api_reading_repositories
    reading_repository.readings = []

    response = asyncio.run(_get(app, f"/readings?meter_identifier={METER_IDENTIFIER}"))

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["returned_count"] == 0
    assert response.json()["has_more"] is False
    assert response.json()["next_offset"] is None


def test_readings_timestamp_filter_is_utc_normalized_and_half_open(api_reading_repositories):
    _, _, reading_repository = api_reading_repositories
    response = asyncio.run(
        _get(
            app,
            _readings_url(
                start="2024-01-01T01:01:00+01:00",
                end="2024-01-01T00:03:00Z",
            ),
        )
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == [2, 3]
    assert reading_repository.call["start"] == datetime(2024, 1, 1, 0, 1, tzinfo=UTC)
    assert reading_repository.call["end"] == datetime(2024, 1, 1, 0, 3, tzinfo=UTC)


@pytest.mark.parametrize(
    ("boundary", "expected_start", "expected_end", "expected_ids"),
    [
        ("start", "2024-01-01T00:02:00Z", None, [3, 4, 5]),
        ("end", None, "2024-01-01T00:02:00Z", [1, 2]),
    ],
)
def test_readings_supports_each_optional_boundary(
    api_reading_repositories,
    boundary,
    expected_start,
    expected_end,
    expected_ids,
):
    _, _, reading_repository = api_reading_repositories
    response = asyncio.run(_get(app, _readings_url(**{boundary: expected_start or expected_end})))

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == expected_ids
    expected = expected_start or expected_end
    timestamp = datetime.fromisoformat(expected.replace("Z", "+00:00"))
    assert reading_repository.call["start"] == (timestamp if expected_start is not None else None)
    assert reading_repository.call["end"] == (timestamp if expected_end is not None else None)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-01-01T00:02:00Z", "2024-01-01T00:02:00Z"),
        ("2024-01-01T00:03:00Z", "2024-01-01T00:02:00Z"),
    ],
)
def test_readings_rejects_invalid_timestamp_ranges(api_reading_repositories, start, end):
    _, _, reading_repository = api_reading_repositories
    response = asyncio.run(
        _get(
            app,
            _readings_url(start=start, end=end),
        )
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "start must be earlier than end."}
    assert reading_repository.call is None


def test_readings_rejects_naive_timestamp(api_reading_repositories):
    _, _, reading_repository = api_reading_repositories
    response = asyncio.run(
        _get(
            app,
            _readings_url(start="2024-01-01T00:00:00"),
        )
    )

    assert response.status_code == 422
    assert "must include a UTC offset" in response.json()["detail"]
    assert reading_repository.call is None


def test_readings_pagination_and_maximum_page_size(api_reading_repositories):
    _, _, reading_repository = api_reading_repositories
    response = asyncio.run(
        _get(
            app,
            _readings_url(limit=2, offset=1),
        )
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == [2, 3]
    assert response.json()["next_offset"] == 3
    assert reading_repository.call["offset"] == 1

    too_large = asyncio.run(_get(app, _readings_url(limit=1001)))
    assert too_large.status_code == 422


def test_readings_allows_maximum_limit_of_1000(api_reading_repositories):
    _, _, reading_repository = api_reading_repositories
    response = asyncio.run(_get(app, _readings_url(limit=1000)))

    assert response.status_code == 200
    assert reading_repository.call["limit"] == 1001


def test_import_does_not_require_database_configuration():
    environment = os.environ.copy()
    for variable in (
        "DATABASE_URL",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "SMARTMETER_TEST_DATABASE_URL",
    ):
        environment.pop(variable, None)
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH", "")) if part
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from smartmeter.api.main import app; "
                "assert app.title == 'SmartMeter AI API'; "
                "assert app.openapi()['paths'].get('/health') is not None"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        timeout=15,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_database_dependency_closes_session(monkeypatch):
    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(
        "smartmeter.api.dependencies.create_session_factory",
        lambda: lambda: session,
    )

    dependency = get_db_session()
    assert next(dependency) is session
    with pytest.raises(StopIteration):
        next(dependency)
    assert session.closed is True


def test_missing_database_configuration_stays_generic_in_http_response(monkeypatch):
    def missing_configuration():
        raise ValueError(
            "Set DATABASE_URL or all of POSTGRES_DB, POSTGRES_USER, and POSTGRES_PASSWORD."
        )

    monkeypatch.setattr(
        "smartmeter.api.dependencies.create_session_factory",
        missing_configuration,
    )
    response = asyncio.run(
        _get(
            create_app(),
            _readings_url(limit=1),
            raise_app_exceptions=False,
        )
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "DATABASE_URL" not in response.text


def test_reading_repository_applies_optional_half_open_timestamp_filters():
    class ScalarResult:
        def all(self):
            return []

    class StatementSession:
        statement = None

        def scalars(self, statement):
            self.statement = statement
            return ScalarResult()

    session = StatementSession()
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    repository = ReadingRepository()

    repository.get_for_meter(session, 42, start=start, limit=10)
    start_sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "meter_readings.timestamp >=" in start_sql
    assert "meter_readings.timestamp <" not in start_sql

    repository.get_for_meter(session, 42, end=end, limit=10)
    end_sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "meter_readings.timestamp <" in end_sql
    assert "meter_readings.timestamp >=" not in end_sql

    repository.get_for_meter(session, 42, start=start, end=end, limit=10)
    range_sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "meter_readings.timestamp >=" in range_sql
    assert "meter_readings.timestamp <" in range_sql


def test_unexpected_errors_return_generic_response_without_internal_details():
    test_app: FastAPI = create_app()

    def failing_endpoint() -> None:
        raise RuntimeError("private implementation detail")

    test_app.add_api_route("/_test-error", failing_endpoint)
    response = asyncio.run(_get(test_app, "/_test-error", raise_app_exceptions=False))

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "private implementation detail" not in response.text
