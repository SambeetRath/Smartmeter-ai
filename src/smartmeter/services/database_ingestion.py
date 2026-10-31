from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as parquet

from smartmeter.config import DEFAULT_PROCESSED_DATASET_PATH, PROJECT_ROOT
from smartmeter.database.config import get_ingestion_batch_size
from smartmeter.database.repositories import MeterRepository, ReadingRepository
from smartmeter.database.session import SessionFactory, create_session_factory
from smartmeter.logging_config import configure_logging

DEFAULT_METER_IDENTIFIER = "uci_household_1"
METER_LOCATION = (
    "Single household in the UCI Individual Household Electric Power Consumption dataset"
)
READING_COLUMNS = (
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
)

logger = logging.getLogger("smartmeter.database_ingestion")


@dataclass(frozen=True)
class IngestionResult:
    """Counters from one Parquet-to-PostgreSQL ingestion run."""

    meter_identifier: str
    records_read: int
    records_inserted: int
    records_skipped: int
    records_rejected: int


def _utc_datetime(value: Any) -> datetime | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        timestamp = value.to_pydatetime()
    elif isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = pd.Timestamp(value).to_pydatetime()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def normalize_reading_batch(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], int]:
    """Normalize one bounded batch; retain NULL measurements and count invalid rows."""
    records: list[dict[str, Any]] = []
    rejected = 0
    for row in frame.to_dict(orient="records"):
        try:
            timestamp = _utc_datetime(row.get("timestamp"))
        except (OverflowError, TypeError, ValueError):
            rejected += 1
            continue
        if timestamp is None:
            rejected += 1
            continue

        payload: dict[str, Any] = {"timestamp": timestamp}
        invalid_row = False
        for column in READING_COLUMNS:
            value = row.get(column)
            if pd.isna(value):
                payload[column] = None
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                invalid_row = True
                break
            if not math.isfinite(numeric_value) or numeric_value < 0:
                invalid_row = True
                break
            payload[column] = numeric_value

        if invalid_row:
            rejected += 1
        else:
            records.append(payload)
    return records, rejected


def _parquet_batches(
    parquet_path: Path,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Processed Parquet dataset was not found: {parquet_path}")
    parquet_file = parquet.ParquetFile(parquet_path)
    required_columns = {"timestamp", *READING_COLUMNS}
    missing_columns = required_columns.difference(parquet_file.schema_arrow.names)
    if missing_columns:
        raise ValueError(
            "Processed Parquet dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["timestamp", *READING_COLUMNS],
    ):
        yield batch.to_pandas()


def ingest_parquet(
    parquet_path: str | Path = DEFAULT_PROCESSED_DATASET_PATH,
    *,
    meter_identifier: str = DEFAULT_METER_IDENTIFIER,
    batch_size: int | None = None,
    session_factory: SessionFactory | None = None,
    meter_repository: MeterRepository | None = None,
    reading_repository: ReadingRepository | None = None,
) -> IngestionResult:
    """Ingest Parquet batches into PostgreSQL with per-batch transactions and idempotency."""
    source_path = Path(parquet_path).expanduser()
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    selected_batch_size = batch_size or get_ingestion_batch_size()
    if selected_batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not meter_identifier.strip():
        raise ValueError("meter_identifier must not be empty.")

    factory = session_factory or create_session_factory()
    meters = meter_repository or MeterRepository()
    readings = reading_repository or ReadingRepository()

    with factory.begin() as session:
        meter_id = meters.get_or_create(
            session,
            meter_identifier,
            location=METER_LOCATION,
        ).id

    records_read = 0
    records_inserted = 0
    records_skipped = 0
    records_rejected = 0

    try:
        for frame in _parquet_batches(source_path, selected_batch_size):
            records_read += len(frame)
            batch_records, rejected = normalize_reading_batch(frame)
            records_rejected += rejected
            if not batch_records:
                continue
            with factory.begin() as session:
                inserted = readings.insert_batch(session, meter_id, batch_records)
            records_inserted += inserted
            records_skipped += len(batch_records) - inserted
    except Exception:
        logger.exception(
            "Parquet ingestion failed: read=%s inserted=%s skipped=%s rejected=%s. "
            "Previously committed batches are safe to retry.",
            records_read,
            records_inserted,
            records_skipped,
            records_rejected,
        )
        raise

    result = IngestionResult(
        meter_identifier=meter_identifier,
        records_read=records_read,
        records_inserted=records_inserted,
        records_skipped=records_skipped,
        records_rejected=records_rejected,
    )
    logger.info(
        "Database ingestion completed for meter %s: read=%s inserted=%s skipped=%s rejected=%s",
        result.meter_identifier,
        result.records_read,
        result.records_inserted,
        result.records_skipped,
        result.records_rejected,
    )
    return result


def main() -> None:
    """Run the configured Parquet ingestion command."""
    configure_logging()
    parquet_path = Path(os.getenv("SMARTMETER_PARQUET_PATH", str(DEFAULT_PROCESSED_DATASET_PATH)))
    meter_identifier = os.getenv("SMARTMETER_METER_IDENTIFIER", DEFAULT_METER_IDENTIFIER)
    ingest_parquet(
        parquet_path,
        meter_identifier=meter_identifier,
    )


if __name__ == "__main__":
    main()
