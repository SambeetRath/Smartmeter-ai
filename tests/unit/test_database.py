from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import UniqueConstraint

from smartmeter.database.config import get_database_url, get_ingestion_batch_size
from smartmeter.database.models import Meter, MeterReading
from smartmeter.services.database_ingestion import (
    READING_COLUMNS,
    ingest_parquet,
    normalize_reading_batch,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_valid_household.csv"


def test_database_url_is_required_and_postgresql_only():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        get_database_url({})
    with pytest.raises(ValueError, match="postgresql\\+psycopg"):
        get_database_url({"DATABASE_URL": "sqlite:///not_allowed.db"})
    assert get_database_url(
        {"DATABASE_URL": "postgresql+psycopg://meter:secret@localhost/smartmeter"}
    ).startswith("postgresql+psycopg://")


def test_database_url_can_be_built_from_compose_environment_safely():
    url = get_database_url(
        {
            "POSTGRES_DB": "meter_db",
            "POSTGRES_USER": "meter_user",
            "POSTGRES_PASSWORD": "safe@password",
            "POSTGRES_PORT": "5432",
            "SMARTMETER_DB_HOST": "postgres",
        }
    )

    assert url.startswith("postgresql+psycopg://")
    assert "safe%40password" in url
    assert "@postgres:5432/meter_db" in url


def test_ingestion_batch_size_must_be_positive():
    assert get_ingestion_batch_size({"SMARTMETER_INGESTION_BATCH_SIZE": "12"}) == 12
    with pytest.raises(ValueError, match="positive integer"):
        get_ingestion_batch_size({"SMARTMETER_INGESTION_BATCH_SIZE": "0"})
    with pytest.raises(ValueError, match="positive integer"):
        get_ingestion_batch_size({"SMARTMETER_INGESTION_BATCH_SIZE": "many"})


def test_models_define_meter_timestamp_uniqueness_and_restrict_delete():
    unique_constraints = [
        constraint
        for constraint in MeterReading.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    ]
    assert [column.name for column in unique_constraints[0].columns] == [
        "meter_id",
        "timestamp",
    ]
    meter_foreign_key = next(iter(MeterReading.__table__.foreign_keys))
    assert meter_foreign_key.ondelete == "RESTRICT"
    assert Meter.__table__.c.meter_identifier.unique
    assert MeterReading.__table__.c.timestamp.type.timezone


def test_normalize_batch_retains_nulls_uses_utc_and_counts_invalid_rows():
    rows = {
        "timestamp": [
            datetime(2024, 1, 1),
            datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
            pd.NaT,
            datetime(2024, 1, 1, 0, 3),
        ],
        **{column: [1.25, None, 2.0, float("inf")] for column in READING_COLUMNS},
    }
    rows["global_active_power"][3] = 4.0
    frame = pd.DataFrame(rows)

    readings, rejected = normalize_reading_batch(frame)

    assert rejected == 2
    assert len(readings) == 2
    assert readings[0]["timestamp"].tzinfo == timezone.utc
    assert readings[1]["global_active_power"] is None


class _FakeSessionFactory:
    def begin(self):
        class Transaction:
            def __enter__(self):
                return object()

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        return Transaction()


class _FakeMeterRepository:
    def get_or_create(self, session, meter_identifier, *, location=None):
        return SimpleNamespace(id=7)


class _IdempotentFakeReadingRepository:
    def __init__(self):
        self.timestamps = set()

    def insert_batch(self, session, meter_id, readings):
        inserted = 0
        for reading in readings:
            timestamp = reading["timestamp"]
            if timestamp not in self.timestamps:
                self.timestamps.add(timestamp)
                inserted += 1
        return inserted


def test_parquet_ingestion_is_batched_idempotent_and_preserves_nulls(tmp_path):
    source = pd.read_csv(FIXTURE_PATH, sep=";")
    source = source.rename(columns={column: column.lower() for column in source.columns})
    source["timestamp"] = pd.to_datetime(
        source["date"] + " " + source["time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    source.loc[1, "global_active_power"] = float("nan")
    parquet_path = tmp_path / "synthetic.parquet"
    source.to_parquet(parquet_path, engine="pyarrow", index=False)
    readings = _IdempotentFakeReadingRepository()
    options = {
        "parquet_path": parquet_path,
        "meter_identifier": "synthetic_test_meter",
        "batch_size": 2,
        "session_factory": _FakeSessionFactory(),
        "meter_repository": _FakeMeterRepository(),
        "reading_repository": readings,
    }

    first_run = ingest_parquet(**options)
    second_run = ingest_parquet(**options)

    assert first_run.records_read == len(source)
    assert first_run.records_inserted == len(source)
    assert first_run.records_skipped == 0
    assert first_run.records_rejected == 0
    assert second_run.records_inserted == 0
    assert second_run.records_skipped == len(source)
    assert len(readings.timestamps) == len(source)
