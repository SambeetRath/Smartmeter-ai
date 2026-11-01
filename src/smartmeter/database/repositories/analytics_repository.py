from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from smartmeter.database.models import Meter, MeterReading

MEASUREMENT_COLUMNS = (
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
)

MAX_PAGE_SIZE = 1_000


def _as_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _as_int(value: Any) -> int:
    return int(value or 0)


class AnalyticsRepository:
    """Read-only SQL aggregation queries for dashboard analytics."""

    def list_meters(self, session: Session, *, limit: int, offset: int) -> list[dict[str, Any]]:
        if not 1 <= limit <= MAX_PAGE_SIZE + 1 or offset < 0:
            raise ValueError("Invalid meter page.")
        statement = (
            select(
                Meter.meter_identifier,
                func.count(MeterReading.id).label("reading_count"),
                func.min(MeterReading.timestamp).label("first_reading_at"),
                func.max(MeterReading.timestamp).label("last_reading_at"),
            )
            .outerjoin(MeterReading, MeterReading.meter_id == Meter.id)
            .group_by(Meter.id, Meter.meter_identifier)
            .order_by(Meter.meter_identifier)
            .limit(limit)
            .offset(offset)
        )
        rows = session.execute(statement).mappings().all()
        return [
            {
                "meter_identifier": row["meter_identifier"],
                "reading_count": _as_int(row["reading_count"]),
                "first_reading_at": row["first_reading_at"],
                "last_reading_at": row["last_reading_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _scope(
        statement,
        meter_id: int,
        *,
        start: datetime | None,
        end: datetime | None,
    ):
        statement = statement.where(MeterReading.meter_id == meter_id)
        if start is not None:
            statement = statement.where(MeterReading.timestamp >= start)
        if end is not None:
            statement = statement.where(MeterReading.timestamp < end)
        return statement

    def _measurement_stats(self, total, *, prefix: str = "") -> dict[str, Any]:
        stats = {}
        for column_name in MEASUREMENT_COLUMNS:
            missing_count = _as_int(total[f"{prefix}{column_name}_missing_count"])
            reading_count = _as_int(total[f"{prefix}reading_count"])
            stats[column_name] = {
                "missing_count": missing_count,
                "missing_rate": missing_count / reading_count if reading_count else None,
            }
        return stats

    def _summary_row(
        self,
        session: Session,
        meter_id: int,
        *,
        start: datetime | None,
        end: datetime | None,
    ):
        aggregate_columns = [
            func.count(MeterReading.id).label("reading_count"),
            func.avg(MeterReading.global_active_power).label("active_power_average"),
            func.min(MeterReading.global_active_power).label("active_power_minimum"),
            func.max(MeterReading.global_active_power).label("active_power_maximum"),
        ]
        aggregate_columns.extend(
            func.count(case((getattr(MeterReading, name).is_(None), 1))).label(
                f"{name}_missing_count"
            )
            for name in MEASUREMENT_COLUMNS
        )
        latest_scope = self._scope(
            select(MeterReading.global_active_power).where(
                MeterReading.global_active_power.is_not(None)
            ),
            meter_id,
            start=start,
            end=end,
        )
        latest_active_power = (
            latest_scope.order_by(MeterReading.timestamp.desc()).limit(1).scalar_subquery()
        )
        latest_at_scope = self._scope(
            select(MeterReading.timestamp).where(MeterReading.global_active_power.is_not(None)),
            meter_id,
            start=start,
            end=end,
        )
        latest_active_power_at = (
            latest_at_scope.order_by(MeterReading.timestamp.desc()).limit(1).scalar_subquery()
        )
        aggregate_columns.extend(
            (
                latest_active_power.label("latest_active_power"),
                latest_active_power_at.label("latest_active_power_at"),
            )
        )
        statement = self._scope(
            select(*aggregate_columns),
            meter_id,
            start=start,
            end=end,
        )
        return session.execute(statement).mappings().one()

    def summary_for_meter(
        self,
        session: Session,
        meter_id: int,
        *,
        start: datetime | None,
        end: datetime | None,
    ) -> dict[str, Any]:
        row = self._summary_row(session, meter_id, start=start, end=end)
        reading_count = _as_int(row["reading_count"])
        return {
            "reading_count": reading_count,
            "active_power_average": _as_float(row["active_power_average"]),
            "active_power_minimum": _as_float(row["active_power_minimum"]),
            "active_power_maximum": _as_float(row["active_power_maximum"]),
            "latest_active_power": _as_float(row["latest_active_power"]),
            "latest_active_power_at": row["latest_active_power_at"],
            "missing_measurements": self._measurement_stats(row),
        }

    def measurement_stats_for_meter(
        self,
        session: Session,
        meter_id: int,
        *,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        """Aggregate each persisted measurement directly over the selected readings."""
        aggregate_columns = [func.count(MeterReading.id).label("reading_count")]
        for column_name in MEASUREMENT_COLUMNS:
            column = getattr(MeterReading, column_name)
            aggregate_columns.extend(
                (
                    func.count(column).label(f"{column_name}_value_count"),
                    func.avg(column).label(f"{column_name}_average"),
                    func.min(column).label(f"{column_name}_minimum"),
                    func.max(column).label(f"{column_name}_maximum"),
                    func.count(case((column.is_(None), 1))).label(f"{column_name}_missing_count"),
                )
            )
        statement = self._scope(
            select(*aggregate_columns),
            meter_id,
            start=start,
            end=end,
        )
        row = session.execute(statement).mappings().one()
        reading_count = _as_int(row["reading_count"])
        measurements = {}
        for column_name in MEASUREMENT_COLUMNS:
            missing_count = _as_int(row[f"{column_name}_missing_count"])
            measurements[column_name] = {
                "value_count": _as_int(row[f"{column_name}_value_count"]),
                "average": _as_float(row[f"{column_name}_average"]),
                "minimum": _as_float(row[f"{column_name}_minimum"]),
                "maximum": _as_float(row[f"{column_name}_maximum"]),
                "missing_count": missing_count,
                "missing_rate": missing_count / reading_count if reading_count else None,
            }
        return {
            "reading_count": reading_count,
            "measurements": measurements,
            "aggregation_scope": "Direct SQL aggregates over stored readings in [start, end).",
        }

    def aggregate_time_series(
        self,
        session: Session,
        meter_id: int,
        *,
        start: datetime,
        end: datetime,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if interval not in {"hour", "day"}:
            raise ValueError("interval must be 'hour' or 'day'.")
        if not 1 <= limit <= 751:
            raise ValueError("Invalid time-series result limit.")
        bucket = func.date_trunc(interval, MeterReading.timestamp).label("timestamp")
        aggregate_columns = [
            bucket,
            func.count(MeterReading.id).label("reading_count"),
            func.avg(MeterReading.global_active_power).label("global_active_power_average"),
            func.min(MeterReading.global_active_power).label("global_active_power_minimum"),
            func.max(MeterReading.global_active_power).label("global_active_power_maximum"),
        ]
        aggregate_columns.extend(
            func.avg(getattr(MeterReading, column_name)).label(f"{column_name}_average")
            for column_name in MEASUREMENT_COLUMNS[1:]
        )
        statement = (
            self._scope(
                select(*aggregate_columns),
                meter_id,
                start=start,
                end=end,
            )
            .group_by(bucket)
            .order_by(bucket)
            .limit(limit)
        )
        rows = session.execute(statement).mappings().all()
        return [
            {
                "timestamp": row["timestamp"],
                "reading_count": _as_int(row["reading_count"]),
                **{
                    key: _as_float(row[key])
                    for key in (
                        "global_active_power_average",
                        "global_active_power_minimum",
                        "global_active_power_maximum",
                        "global_reactive_power_average",
                        "voltage_average",
                        "global_intensity_average",
                        "sub_metering_1_average",
                        "sub_metering_2_average",
                        "sub_metering_3_average",
                    )
                },
            }
            for row in rows
        ]

    def quality_for_meter(
        self,
        session: Session,
        meter_id: int,
        *,
        start: datetime | None,
        end: datetime | None,
    ) -> dict[str, Any]:
        summary = self._summary_row(session, meter_id, start=start, end=end)
        reading_count = _as_int(summary["reading_count"])

        scoped_readings = self._scope(
            select(
                MeterReading.timestamp.label("timestamp"),
                func.lag(MeterReading.timestamp)
                .over(order_by=MeterReading.timestamp)
                .label("previous_timestamp"),
            ),
            meter_id,
            start=start,
            end=end,
        ).cte("scoped_readings")
        gap_minutes = (
            func.extract(
                "epoch",
                scoped_readings.c.timestamp - scoped_readings.c.previous_timestamp,
            )
            / 60
        )
        is_gap = gap_minutes > 1
        gap_statement = select(
            func.count(case((is_gap, 1))).label("observed_gap_count"),
            func.coalesce(
                func.sum(case((is_gap, func.ceil(gap_minutes) - 1), else_=0)),
                0,
            ).label("estimated_missing_intervals"),
            func.min(scoped_readings.c.timestamp).label("first_observed_at"),
            func.max(scoped_readings.c.timestamp).label("last_observed_at"),
        )
        gaps = session.execute(gap_statement).mappings().one()
        return {
            "reading_count": reading_count,
            "missing_measurements": self._measurement_stats(summary),
            "timestamp_gaps": {
                "expected_interval_minutes": 1,
                "observed_gap_count": _as_int(gaps["observed_gap_count"]),
                "estimated_missing_intervals": _as_int(gaps["estimated_missing_intervals"]),
                "first_observed_at": gaps["first_observed_at"],
                "last_observed_at": gaps["last_observed_at"],
                "boundary_gaps_assessed": False,
            },
            "database_metrics_scope": "Computed from stored readings in the requested interval.",
        }
