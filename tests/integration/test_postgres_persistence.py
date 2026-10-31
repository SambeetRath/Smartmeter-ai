from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from alembic import command
from smartmeter.database.connection import create_database_engine
from smartmeter.database.repositories import MeterRepository, ReadingRepository
from smartmeter.services.database_ingestion import ingest_parquet

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
    yield engine
    engine.dispose()


def test_migration_repositories_constraints_and_transaction_rollback(migrated_test_database):
    engine = migrated_test_database
    table_names = set(inspect(engine).get_table_names())
    assert {"meters", "meter_readings", "alembic_version"}.issubset(table_names)

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
