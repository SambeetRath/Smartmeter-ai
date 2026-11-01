from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from smartmeter.database.models import MeterReading

MAX_PAGE_SIZE = 10_000


@dataclass(frozen=True)
class ReadingSummary:
    """Basic aggregate statistics for a meter's readings."""

    reading_count: int
    active_power_average: float | None
    active_power_minimum: float | None
    active_power_maximum: float | None


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Reading date-time boundaries must include a timezone.")
    return value.astimezone(timezone.utc)


def _validate_page(limit: int, offset: int) -> None:
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}.")
    if offset < 0:
        raise ValueError("offset must not be negative.")


class ReadingRepository:
    """Parameterized read and batch-write operations for meter readings."""

    def insert_batch(
        self,
        session: Session,
        meter_id: int,
        readings: list[dict[str, Any]],
    ) -> int:
        """Insert a batch, ignoring already-ingested meter/timestamp pairs."""
        if not readings:
            return 0
        rows = [{**reading, "meter_id": meter_id} for reading in readings]
        statement = (
            insert(MeterReading)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_meter_readings_meter_timestamp")
            .returning(MeterReading.id)
        )
        return len(session.scalars(statement).all())

    def exists(self, session: Session, meter_id: int, timestamp: datetime) -> bool:
        """Check whether a meter already has a reading at the given UTC instant."""
        timestamp_utc = _utc_datetime(timestamp)
        return (
            session.scalar(
                select(
                    select(MeterReading.id)
                    .where(
                        MeterReading.meter_id == meter_id,
                        MeterReading.timestamp == timestamp_utc,
                    )
                    .exists()
                )
            )
            is True
        )

    def get_for_meter(
        self,
        session: Session,
        meter_id: int,
        *,
        limit: int = 1_000,
        offset: int = 0,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[MeterReading]:
        """Return an ordered, bounded page of readings, optionally within [start, end)."""
        _validate_page(limit, offset)
        start_utc = _utc_datetime(start) if start is not None else None
        end_utc = _utc_datetime(end) if end is not None else None
        if start_utc is not None and end_utc is not None and start_utc >= end_utc:
            raise ValueError("start must be earlier than end.")

        statement = select(MeterReading).where(MeterReading.meter_id == meter_id)
        if start_utc is not None:
            statement = statement.where(MeterReading.timestamp >= start_utc)
        if end_utc is not None:
            statement = statement.where(MeterReading.timestamp < end_utc)
        statement = statement.order_by(MeterReading.timestamp).limit(limit).offset(offset)
        return list(session.scalars(statement).all())

    def get_in_range(
        self,
        session: Session,
        meter_id: int,
        start: datetime,
        end: datetime,
        *,
        limit: int = 1_000,
        offset: int = 0,
    ) -> list[MeterReading]:
        """Return readings in the half-open UTC interval [start, end)."""
        _validate_page(limit, offset)
        start_utc = _utc_datetime(start)
        end_utc = _utc_datetime(end)
        if start_utc >= end_utc:
            raise ValueError("start must be earlier than end.")
        return self.get_for_meter(
            session,
            meter_id,
            limit=limit,
            offset=offset,
            start=start_utc,
            end=end_utc,
        )

    def count_for_meter(
        self,
        session: Session,
        meter_id: int,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """Count a meter's readings, optionally within the half-open interval [start, end)."""
        start_utc = _utc_datetime(start) if start is not None else None
        end_utc = _utc_datetime(end) if end is not None else None
        if start_utc is not None and end_utc is not None and start_utc >= end_utc:
            raise ValueError("start must be earlier than end.")

        statement = select(func.count(MeterReading.id)).where(MeterReading.meter_id == meter_id)
        if start_utc is not None:
            statement = statement.where(MeterReading.timestamp >= start_utc)
        if end_utc is not None:
            statement = statement.where(MeterReading.timestamp < end_utc)
        return int(session.scalar(statement) or 0)

    def summary_for_meter(self, session: Session, meter_id: int) -> ReadingSummary:
        """Return count and active-power aggregate statistics for a meter."""
        row = session.execute(
            select(
                func.count(MeterReading.id),
                func.avg(MeterReading.global_active_power),
                func.min(MeterReading.global_active_power),
                func.max(MeterReading.global_active_power),
            ).where(MeterReading.meter_id == meter_id)
        ).one()
        return ReadingSummary(
            reading_count=int(row[0]),
            active_power_average=float(row[1]) if row[1] is not None else None,
            active_power_minimum=float(row[2]) if row[2] is not None else None,
            active_power_maximum=float(row[3]) if row[3] is not None else None,
        )
