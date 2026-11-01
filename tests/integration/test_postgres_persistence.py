from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from alembic.config import Config
from fastapi import FastAPI
from sqlalchemy import inspect, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from alembic import command
from smartmeter.api.dependencies import get_db_session
from smartmeter.api.main import create_app
from smartmeter.database.connection import create_database_engine
from smartmeter.database.models import AnomalyResult, DetectionRun, Meter
from smartmeter.database.repositories import (
    AnalyticsRepository,
    AnomalyResultRepository,
    DetectionRunRepository,
    MeterRepository,
    ReadingRepository,
)
from smartmeter.services import detection_worker
from smartmeter.services.database_ingestion import ingest_parquet
from smartmeter.services.detection_worker import WorkerConfig, run_one_detection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATABASE_URL = os.getenv("SMARTMETER_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="Set SMARTMETER_TEST_DATABASE_URL to an isolated PostgreSQL *_test database.",
    ),
]


@pytest.fixture
def migrated_test_database(monkeypatch):
    assert TEST_DATABASE_URL is not None
    from sqlalchemy.engine import make_url

    parsed_url = make_url(TEST_DATABASE_URL)
    if not parsed_url.database or not parsed_url.database.endswith("_test"):
        pytest.fail(
            "SMARTMETER_TEST_DATABASE_URL must target a dedicated database ending in '_test'."
        )

    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    migration_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(migration_config, "head")

    engine = create_database_engine(TEST_DATABASE_URL)
    test_meter_prefixes = (
        "detection-e2e-",
        "detection-active-",
        "lease-boundary-",
        "claim-race-",
        "detection-failure-",
        "atomic-result-",
        "detection-integration-",
        "lease-integration-",
    )
    with engine.begin() as connection:
        connection.execute(
            update(DetectionRun)
            .where(
                DetectionRun.status.in_(("pending", "running")),
                DetectionRun.meter_id.in_(
                    select(Meter.id).where(
                        or_(
                            *(
                                Meter.meter_identifier.like(f"{prefix}%")
                                for prefix in test_meter_prefixes
                            )
                        )
                    )
                ),
            )
            .values(
                status="failed",
                completed_at=datetime.now(timezone.utc),
                lease_expires_at=None,
                lease_token=None,
                error_message="Recovered abandoned integration test run.",
            )
        )
    yield engine
    engine.dispose()


async def _api_request(
    application: FastAPI,
    method: str,
    path: str,
    *,
    payload: dict | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, json=payload)


def _synthetic_readings(start: datetime, count: int) -> list[dict]:
    readings = []
    for index in range(count):
        readings.append(
            {
                "timestamp": start + timedelta(minutes=index),
                "global_active_power": (100.0 if index == count - 2 else 1.0 + index / 10),
                "global_reactive_power": 0.1,
                "voltage": 230.0,
                "global_intensity": 5.0,
                "sub_metering_1": 0.0,
                "sub_metering_2": 1.0,
                "sub_metering_3": 2.0,
            }
        )
    readings[-1].update(
        {
            "global_active_power": None,
            "global_reactive_power": None,
            "voltage": None,
            "global_intensity": None,
            "sub_metering_1": None,
            "sub_metering_2": None,
            "sub_metering_3": None,
        }
    )
    return readings


def test_detection_api_worker_and_persisted_summary_end_to_end(migrated_test_database):
    session_factory = sessionmaker(
        bind=migrated_test_database,
        class_=Session,
        expire_on_commit=False,
    )
    meter_identifier = f"detection-e2e-{uuid.uuid4()}"
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=20)

    with session_factory.begin() as session:
        meter = MeterRepository().get_or_create(session, meter_identifier)
        ReadingRepository().insert_batch(
            session,
            meter.id,
            _synthetic_readings(start, 20),
        )

    application = create_app()

    def override_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    application.dependency_overrides[get_db_session] = override_session
    try:
        submitted = asyncio.run(
            _api_request(
                application,
                "POST",
                "/detection/runs",
                payload={
                    "meter_identifier": meter_identifier,
                    "algorithm": "global_iqr",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )
        )
        assert submitted.status_code == 202
        run_id = uuid.UUID(submitted.json()["run_id"])

        with session_factory() as session:
            pending = DetectionRunRepository().get_by_id(session, run_id)
            assert pending is not None
            assert pending.status == "pending"
            assert pending.attempts == 0

        assert (
            run_one_detection(
                session_factory=session_factory,
                config=WorkerConfig(lease_seconds=9, heartbeat_seconds=3),
            )
            == "completed"
        )

        retrieved = asyncio.run(_api_request(application, "GET", f"/detection/runs/{run_id}"))
        anomalies = asyncio.run(
            _api_request(application, "GET", f"/anomalies?run_id={run_id}&limit=10")
        )
        normals = asyncio.run(
            _api_request(
                application,
                "GET",
                f"/anomalies?run_id={run_id}&prediction_status=normal&limit=10",
            )
        )
        unavailable = asyncio.run(
            _api_request(
                application,
                "GET",
                f"/anomalies?run_id={run_id}&prediction_status=unavailable&limit=10",
            )
        )
        summary = asyncio.run(
            _api_request(application, "GET", f"/anomalies/summary?run_id={run_id}")
        )
    finally:
        application.dependency_overrides.clear()

    assert retrieved.status_code == 200
    assert retrieved.json()["status"] == "completed"
    assert retrieved.json()["attempts"] == 1
    assert retrieved.json()["summary"]["run_id"] == str(run_id)
    assert anomalies.status_code == normals.status_code == unavailable.status_code == 200
    assert summary.status_code == 200
    anomaly_items = anomalies.json()["items"]
    normal_items = normals.json()["items"]
    unavailable_items = unavailable.json()["items"]
    assert len(anomaly_items) == 1
    assert len(normal_items) == 4
    assert len(unavailable_items) == 1
    assert summary.json()["readings_processed"] == 6
    assert summary.json()["anomaly_count"] == 1
    assert summary.json()["normal_count"] == 4
    assert summary.json()["unavailable_count"] == 1
    assert (
        summary.json()["readings_processed"]
        == summary.json()["anomaly_count"]
        + summary.json()["normal_count"]
        + summary.json()["unavailable_count"]
    )
    with session_factory() as session:
        persisted_counts = {
            status: len(
                AnomalyResultRepository().list_results(
                    session,
                    meter_identifier=meter_identifier,
                    run_id=run_id,
                    start=None,
                    end=None,
                    prediction_status=status,
                    algorithm=None,
                    category=None,
                    limit=10,
                    offset=0,
                )
            )
            for status in ("anomaly", "normal", "unavailable")
        }
    assert persisted_counts == {"anomaly": 1, "normal": 4, "unavailable": 1}


def test_detection_api_serializes_active_job_submissions_per_meter(migrated_test_database):
    session_factory = sessionmaker(
        bind=migrated_test_database,
        class_=Session,
        expire_on_commit=False,
    )
    meter_identifier = f"detection-active-{uuid.uuid4()}"
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    with session_factory.begin() as session:
        meter = MeterRepository().get_or_create(session, meter_identifier)
        ReadingRepository().insert_batch(session, meter.id, _synthetic_readings(start, 4))

    application = create_app()

    def override_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    application.dependency_overrides[get_db_session] = override_session
    try:

        async def submit_concurrently():
            return await asyncio.gather(
                _api_request(
                    application,
                    "POST",
                    "/detection/runs",
                    payload={"meter_identifier": meter_identifier},
                ),
                _api_request(
                    application,
                    "POST",
                    "/detection/runs",
                    payload={"meter_identifier": meter_identifier},
                ),
            )

        first, second = asyncio.run(submit_concurrently())
    finally:
        application.dependency_overrides.clear()

    assert sorted((first.status_code, second.status_code)) == [202, 409]
    with session_factory() as session:
        meter = MeterRepository().get_by_identifier(session, meter_identifier)
        assert meter is not None
        assert (
            DetectionRunRepository().count_active_for_meter(
                session,
                meter.id,
            )
            == 1
        )
    with session_factory.begin() as session:
        cleaned = session.execute(
            update(DetectionRun)
            .where(
                DetectionRun.meter_id == meter.id,
                DetectionRun.status == "pending",
            )
            .values(
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error_message="Integration test cleanup.",
            )
        )
        assert cleaned.rowcount == 1


def test_expired_lease_cannot_be_renewed_or_finalize_a_run(migrated_test_database):
    session_factory = sessionmaker(
        bind=migrated_test_database,
        class_=Session,
        expire_on_commit=False,
    )
    repository = DetectionRunRepository()
    now = datetime.now(timezone.utc)

    with session_factory.begin() as session:
        meter = MeterRepository().get_or_create(session, f"lease-boundary-{uuid.uuid4()}")
        run = repository.create_pending(
            session,
            meter_id=meter.id,
            algorithm="global_iqr",
            start=None,
            end=None,
        )
        run_id = run.id
    with session_factory.begin() as session:
        claimed = repository.claim_next(session, lease_seconds=10, now=now)
        assert claimed is not None
        token = claimed.lease_token
    expired_at = now + timedelta(seconds=11)

    with session_factory.begin() as session:
        assert not repository.heartbeat(
            session,
            run_id,
            token,
            lease_seconds=10,
            now=expired_at,
        )
        assert not repository.mark_failed(
            session,
            run_id,
            token,
            error_message="stale worker",
            now=expired_at,
        )
        assert not repository.mark_completed(
            session,
            run_id,
            token,
            results=[],
            training_readings=1,
            readings_processed=0,
            anomaly_count=0,
            normal_count=0,
            unavailable_count=0,
            summary={},
            configuration={},
            now=expired_at,
        )
        current = repository.get_by_id(session, run_id)
        assert current is not None
        assert current.status == "running"
        assert current.lease_token == token
    with session_factory.begin() as session:
        reclaimed = repository.claim_next(
            session,
            lease_seconds=10,
            now=expired_at + timedelta(seconds=1),
        )
        assert reclaimed is not None
        assert reclaimed.id == run_id
        assert reclaimed.lease_token != token
        assert repository.mark_failed(
            session,
            run_id,
            reclaimed.lease_token,
            error_message="integration test cleanup",
            now=expired_at + timedelta(seconds=2),
        )


def test_concurrent_claim_skips_a_run_already_locked_by_another_worker(
    migrated_test_database,
):
    session_factory = sessionmaker(
        bind=migrated_test_database,
        class_=Session,
        expire_on_commit=False,
    )
    repository = DetectionRunRepository()

    with session_factory.begin() as session:
        meter = MeterRepository().get_or_create(session, f"claim-race-{uuid.uuid4()}")
        run = repository.create_pending(
            session,
            meter_id=meter.id,
            algorithm="global_iqr",
            start=None,
            end=None,
        )
        run_id = run.id

    with session_factory() as first_session:
        with first_session.begin():
            first_claim = repository.claim_next(
                first_session,
                lease_seconds=60,
            )
            assert first_claim is not None
            assert first_claim.id == run_id
            with session_factory.begin() as second_session:
                second_claim = repository.claim_next(
                    second_session,
                    lease_seconds=60,
                )
                assert second_claim is None
            assert repository.mark_failed(
                first_session,
                run_id,
                first_claim.lease_token,
                error_message="integration test cleanup",
            )


def test_failed_detection_does_not_leave_persisted_results(
    migrated_test_database,
    monkeypatch,
):
    session_factory = sessionmaker(
        bind=migrated_test_database,
        class_=Session,
        expire_on_commit=False,
    )
    meter_identifier = f"detection-failure-{uuid.uuid4()}"
    timestamp = datetime(2024, 3, 1, tzinfo=timezone.utc)
    with session_factory.begin() as session:
        meter = MeterRepository().get_or_create(session, meter_identifier)
        ReadingRepository().insert_batch(
            session,
            meter.id,
            _synthetic_readings(timestamp, 4),
        )
        run = DetectionRunRepository().create_pending(
            session,
            meter_id=meter.id,
            algorithm="global_iqr",
            start=None,
            end=None,
        )
        run_id = run.id

    def fail_processing(*_args, **_kwargs):
        raise RuntimeError("private failure detail")

    monkeypatch.setattr(detection_worker, "process_detection_run", fail_processing)
    assert (
        run_one_detection(
            session_factory=session_factory,
            config=WorkerConfig(lease_seconds=9, heartbeat_seconds=3),
        )
        == "failed"
    )

    with session_factory() as session:
        failed = DetectionRunRepository().get_by_id(session, run_id)
        result_count = (
            session.query(AnomalyResult).filter(AnomalyResult.detection_run_id == run_id).count()
        )

    assert failed is not None
    assert failed.status == "failed"
    assert failed.readings_processed == 0
    assert failed.error_message is not None
    assert "private failure detail" not in failed.error_message
    assert result_count == 0


def test_result_insert_failure_rolls_back_completion_and_all_rows(migrated_test_database):
    session_factory = sessionmaker(
        bind=migrated_test_database,
        class_=Session,
        expire_on_commit=False,
    )
    repository = DetectionRunRepository()
    with session_factory.begin() as session:
        meter = MeterRepository().get_or_create(session, f"atomic-result-{uuid.uuid4()}")
        run = repository.create_pending(
            session,
            meter_id=meter.id,
            algorithm="global_iqr",
            start=None,
            end=None,
        )
        run_id = run.id
    with session_factory.begin() as session:
        claimed = repository.claim_next(session, lease_seconds=60)
        assert claimed is not None
        assert claimed.id == run_id
        token = claimed.lease_token

    with pytest.raises(IntegrityError):
        with session_factory.begin() as session:
            repository.mark_completed(
                session,
                run_id,
                token,
                results=[
                    {
                        "detection_run_id": run_id,
                        "reading_id": 2_147_483_647,
                        "prediction_status": "normal",
                        "anomaly_score": None,
                        "anomaly_category": None,
                        "metadata_json": {},
                    }
                ],
                training_readings=1,
                readings_processed=1,
                anomaly_count=0,
                normal_count=1,
                unavailable_count=0,
                summary={},
                configuration={},
            )

    with session_factory.begin() as session:
        persisted = repository.get_by_id(session, run_id)
        result_count = (
            session.query(AnomalyResult).filter(AnomalyResult.detection_run_id == run_id).count()
        )
        assert persisted is not None
        assert persisted.status == "running"
        assert persisted.readings_processed == 0
        assert result_count == 0
        assert repository.mark_failed(
            session,
            run_id,
            token,
            error_message="integration test cleanup",
        )


def test_migration_repositories_constraints_and_transaction_rollback(migrated_test_database):
    engine = migrated_test_database
    table_names = set(inspect(engine).get_table_names())
    assert {
        "meters",
        "meter_readings",
        "detection_runs",
        "anomaly_results",
        "alembic_version",
    }.issubset(table_names)

    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    meters = MeterRepository()
    readings = ReadingRepository()
    meter_identifier = f"integration-{uuid.uuid4()}"
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    with session_factory.begin() as session:
        meter = meters.get_or_create(
            session,
            meter_identifier,
            location="Integration test only",
        )
        meter_id = meter.id

    reading_payload = {
        "timestamp": timestamp,
        "global_active_power": 1.25,
        "global_reactive_power": None,
        "voltage": 230.0,
        "global_intensity": 5.0,
        "sub_metering_1": None,
        "sub_metering_2": 1.0,
        "sub_metering_3": 2.0,
    }
    with session_factory.begin() as session:
        assert readings.insert_batch(session, meter_id, [reading_payload]) == 1
    with session_factory.begin() as session:
        assert readings.insert_batch(session, meter_id, [reading_payload]) == 0
        assert readings.exists(session, meter_id, timestamp)
        persisted = readings.get_in_range(
            session,
            meter_id,
            timestamp,
            timestamp.replace(year=2025),
        )
        summary = readings.summary_for_meter(session, meter_id)

    assert len(persisted) == 1
    assert persisted[0].global_reactive_power is None
    assert summary.reading_count == 1
    assert summary.active_power_average == 1.25

    with pytest.raises(IntegrityError):
        with session_factory.begin() as session:
            readings.insert_batch(session, -1, [reading_payload])

    with session_factory.begin() as session:
        assert readings.count_for_meter(session, meter_id) == 1


def test_detection_worker_completes_run_and_persists_all_predictions(migrated_test_database):
    engine = migrated_test_database
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    meter_identifier = f"detection-integration-{uuid.uuid4()}"
    timestamps = [datetime(2024, 1, 1, 0, minute, tzinfo=timezone.utc) for minute in range(10)]

    with session_factory.begin() as session:
        meter = MeterRepository().get_or_create(
            session,
            meter_identifier,
            location="Integration test only",
        )
        meter_id = meter.id
        readings = [
            {
                "timestamp": timestamp,
                "global_active_power": 100.0 if index == 8 else 1.0 + index / 10,
                "global_reactive_power": 0.1 + index / 100,
                "voltage": 230.0 + index / 10,
                "global_intensity": 5.0 + index / 10,
                "sub_metering_1": index / 10,
                "sub_metering_2": 1.0 + index / 10,
                "sub_metering_3": 2.0 + index / 10,
            }
            for index, timestamp in enumerate(timestamps)
        ]
        assert ReadingRepository().insert_batch(session, meter_id, readings) == len(readings)
        run = DetectionRunRepository().create_pending(
            session,
            meter_id=meter_id,
            algorithm="global_iqr",
            start=None,
            end=None,
        )
        run_id = run.id

    assert (
        run_one_detection(
            session_factory=session_factory,
            config=WorkerConfig(poll_seconds=0.1, lease_seconds=9, heartbeat_seconds=3),
        )
        == "completed"
    )

    with session_factory() as session:
        completed = DetectionRunRepository().get_by_id(session, run_id)
        persisted = list(
            session.query(AnomalyResult)
            .filter(AnomalyResult.detection_run_id == run_id)
            .order_by(AnomalyResult.reading_id)
        )

    assert completed is not None
    assert completed.status == "completed"
    assert completed.attempts == 1
    assert completed.readings_processed == len(persisted) == 3
    assert completed.anomaly_count == 1
    assert completed.normal_count == 2
    assert (
        completed.readings_processed
        == completed.anomaly_count + completed.normal_count + completed.unavailable_count
    )


def test_detection_run_reclaims_expired_leases_and_fails_after_max_attempts(
    migrated_test_database,
):
    engine = migrated_test_database
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    meter_identifier = f"lease-integration-{uuid.uuid4()}"
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    repository = DetectionRunRepository()

    with session_factory.begin() as session:
        meter = MeterRepository().get_or_create(session, meter_identifier)
        run = repository.create_pending(
            session,
            meter_id=meter.id,
            algorithm="global_iqr",
            start=None,
            end=None,
        )
        run_id = run.id

    with session_factory.begin() as session:
        first_claim = repository.claim_next(session, lease_seconds=10, now=now)
        assert first_claim is not None
        first_token = first_claim.lease_token
    with session_factory.begin() as session:
        second_claim = repository.claim_next(
            session,
            lease_seconds=10,
            now=now.replace(second=11),
        )
        assert second_claim is not None
        assert second_claim.lease_token != first_token
        assert second_claim.attempts == 2
    with session_factory.begin() as session:
        third_claim = repository.claim_next(
            session,
            lease_seconds=10,
            now=now.replace(second=22),
        )
        assert third_claim is not None
        assert third_claim.attempts == 3
    with session_factory.begin() as session:
        assert (
            repository.fail_exhausted(
                session,
                maximum_attempts=3,
                now=now.replace(second=33),
            )
            == 1
        )
        assert (
            repository.claim_next(
                session,
                lease_seconds=10,
                now=now.replace(second=34),
            )
            is None
        )
        failed = repository.get_by_id(session, run_id)

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_message == "Worker lease expired after the maximum number of attempts."


def test_parquet_ingestion_is_repeatable_in_postgres(migrated_test_database, tmp_path):
    import pandas as pd

    engine = migrated_test_database
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    source_path = tmp_path / "synthetic_meter.parquet"
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:01:00"]),
            "global_active_power": [1.25, None],
            "global_reactive_power": [0.1, 0.2],
            "voltage": [230.0, 231.0],
            "global_intensity": [5.0, 5.1],
            "sub_metering_1": [None, 0.0],
            "sub_metering_2": [1.0, 2.0],
            "sub_metering_3": [2.0, 3.0],
        }
    )
    frame.to_parquet(source_path, engine="pyarrow", index=False)
    meter_identifier = f"parquet-integration-{uuid.uuid4()}"
    ingestion_options = {
        "parquet_path": source_path,
        "meter_identifier": meter_identifier,
        "batch_size": 1,
        "session_factory": session_factory,
    }

    first_run = ingest_parquet(**ingestion_options)
    second_run = ingest_parquet(**ingestion_options)

    assert first_run.records_read == 2
    assert first_run.records_inserted == 2
    assert first_run.records_skipped == 0
    assert second_run.records_inserted == 0
    assert second_run.records_skipped == 2

    with session_factory.begin() as session:
        meter = MeterRepository().get_by_identifier(session, meter_identifier)
        assert meter is not None
        persisted = ReadingRepository().get_for_meter(session, meter.id)
    assert len(persisted) == 2
    assert persisted[1].global_active_power is None


def test_analytics_api_aggregates_stored_readings_in_postgresql(migrated_test_database):
    session_factory = sessionmaker(
        bind=migrated_test_database,
        class_=Session,
        expire_on_commit=False,
    )
    meter_identifier = f"analytics-integration-{uuid.uuid4()}"
    start = datetime(2024, 3, 1, tzinfo=timezone.utc)
    readings = [
        {
            "timestamp": start,
            "global_active_power": 1.0,
            "global_reactive_power": 0.1,
            "voltage": 230.0,
            "global_intensity": 5.0,
            "sub_metering_1": 1.0,
            "sub_metering_2": 2.0,
            "sub_metering_3": 3.0,
        },
        {
            "timestamp": start + timedelta(minutes=1),
            "global_active_power": 3.0,
            "global_reactive_power": None,
            "voltage": 231.0,
            "global_intensity": 6.0,
            "sub_metering_1": 1.0,
            "sub_metering_2": 2.0,
            "sub_metering_3": 3.0,
        },
        {
            "timestamp": start + timedelta(minutes=4),
            "global_active_power": None,
            "global_reactive_power": 0.3,
            "voltage": None,
            "global_intensity": 7.0,
            "sub_metering_1": 1.0,
            "sub_metering_2": 2.0,
            "sub_metering_3": 3.0,
        },
        {
            "timestamp": start + timedelta(minutes=5),
            "global_active_power": 5.0,
            "global_reactive_power": 0.5,
            "voltage": 233.0,
            "global_intensity": 8.0,
            "sub_metering_1": 1.0,
            "sub_metering_2": 2.0,
            "sub_metering_3": 3.0,
        },
    ]
    with session_factory.begin() as session:
        meter = MeterRepository().get_or_create(session, meter_identifier)
        assert ReadingRepository().insert_batch(session, meter.id, readings) == 4
        summary = AnalyticsRepository().summary_for_meter(
            session,
            meter.id,
            start=start,
            end=start + timedelta(minutes=6),
        )
    assert summary["reading_count"] == 4
    assert summary["active_power_average"] == 3.0
    assert summary["active_power_minimum"] == 1.0
    assert summary["active_power_maximum"] == 5.0
    assert summary["latest_active_power"] == 5.0
    assert summary["latest_active_power_at"] == start + timedelta(minutes=5)
    assert summary["missing_measurements"]["global_active_power"]["missing_count"] == 1
    assert summary["missing_measurements"]["global_active_power"]["missing_rate"] == 0.25
    assert summary["missing_measurements"]["global_reactive_power"]["missing_count"] == 1
    assert summary["missing_measurements"]["voltage"]["missing_count"] == 1

    application = create_app()

    def override_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    application.dependency_overrides[get_db_session] = override_session
    try:
        meters = asyncio.run(_api_request(application, "GET", "/meters?limit=1000"))
        summary_response = asyncio.run(
            _api_request(
                application,
                "GET",
                f"/analytics/summary?meter_identifier={meter_identifier}"
                "&start=2024-03-01T01%3A00%3A00%2B01%3A00"
                "&end=2024-03-01T00%3A06%3A00Z",
            )
        )
        hourly = asyncio.run(
            _api_request(
                application,
                "GET",
                f"/analytics/timeseries?meter_identifier={meter_identifier}"
                "&start=2024-03-01T00%3A00%3A00Z"
                "&end=2024-03-01T01%3A00%3A00Z&interval=hour",
            )
        )
        daily = asyncio.run(
            _api_request(
                application,
                "GET",
                f"/analytics/timeseries?meter_identifier={meter_identifier}"
                "&start=2024-03-01T00%3A00%3A00Z"
                "&end=2024-03-02T00%3A00%3A00Z&interval=day",
            )
        )
        quality = asyncio.run(
            _api_request(
                application,
                "GET",
                f"/analytics/quality?meter_identifier={meter_identifier}"
                "&start=2024-03-01T00%3A00%3A00Z"
                "&end=2024-03-01T00%3A06%3A00Z",
            )
        )
        invalid_range = asyncio.run(
            _api_request(
                application,
                "GET",
                f"/analytics/quality?meter_identifier={meter_identifier}"
                "&start=2024-03-01T00%3A06%3A00Z"
                "&end=2024-03-01T00%3A00%3A00Z",
            )
        )
        unknown = asyncio.run(
            _api_request(
                application,
                "GET",
                "/analytics/summary?meter_identifier=analytics-unknown-meter",
            )
        )
    finally:
        application.dependency_overrides.clear()

    assert meters.status_code == 200
    meter_item = next(
        item for item in meters.json()["items"] if item["meter_identifier"] == meter_identifier
    )
    assert meter_item["reading_count"] == 4
    assert meter_item["first_reading_at"] == "2024-03-01T00:00:00Z"
    assert meter_item["last_reading_at"] == "2024-03-01T00:05:00Z"
    assert summary_response.status_code == 200
    assert summary_response.json()["reading_count"] == 4
    assert hourly.status_code == 200
    assert hourly.json()["returned_count"] == 1
    assert hourly.json()["buckets"][0]["reading_count"] == 4
    assert hourly.json()["buckets"][0]["global_active_power_average"] == 3.0
    assert daily.status_code == 200
    assert daily.json()["returned_count"] == 1
    assert quality.status_code == 200
    quality_body = quality.json()
    assert quality_body["timestamp_gaps"]["observed_gap_count"] == 1
    assert quality_body["timestamp_gaps"]["estimated_missing_intervals"] == 2
    assert quality_body["timestamp_gaps"]["boundary_gaps_assessed"] is False
    assert quality_body["ingestion_validation"]["available"] is False
    assert invalid_range.status_code == 422
    assert unknown.status_code == 404
