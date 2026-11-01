from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from smartmeter.database.repositories import AnalyticsRepository, MeterRepository

MAX_ANALYTICS_RANGE = timedelta(days=31)
MAX_TIMESERIES_BUCKETS = 750


class AnalyticsService:
    """Validate analytics query boundaries and delegate SQL aggregates."""

    def __init__(self) -> None:
        self._meters = MeterRepository()
        self._analytics = AnalyticsRepository()

    @staticmethod
    def normalize_range(
        start: datetime | None,
        end: datetime | None,
        *,
        require_both: bool = False,
        enforce_max_range: bool = False,
    ) -> tuple[datetime | None, datetime | None]:
        if require_both and (start is None or end is None):
            raise ValueError("Both start and end are required.")
        for value, name in ((start, "start"), (end, "end")):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must include a UTC offset.")
        start_utc = start.astimezone(timezone.utc) if start is not None else None
        end_utc = end.astimezone(timezone.utc) if end is not None else None
        if start_utc is not None and end_utc is not None:
            if start_utc >= end_utc:
                raise ValueError("start must be earlier than end.")
            if enforce_max_range and end_utc - start_utc > MAX_ANALYTICS_RANGE:
                raise ValueError(
                    f"Analytics range must not exceed {MAX_ANALYTICS_RANGE.days} days."
                )
        return start_utc, end_utc

    def _meter_id(self, session: Session, meter_identifier: str) -> int | None:
        meter = self._meters.get_by_identifier(session, meter_identifier)
        return meter.id if meter is not None else None

    def list_meters(
        self,
        session: Session,
        *,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        return self._analytics.list_meters(session, limit=limit + 1, offset=offset)

    def summary(
        self,
        session: Session,
        meter_identifier: str,
        *,
        start: datetime | None,
        end: datetime | None,
    ) -> dict[str, Any] | None:
        start_utc, end_utc = self.normalize_range(start, end)
        meter_id = self._meter_id(session, meter_identifier)
        if meter_id is None:
            return None
        return self._analytics.summary_for_meter(
            session,
            meter_id,
            start=start_utc,
            end=end_utc,
        )

    def electrical_metrics(
        self,
        session: Session,
        meter_identifier: str,
        *,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any] | None:
        start_utc, end_utc = self.normalize_range(
            start,
            end,
            require_both=True,
            enforce_max_range=True,
        )
        assert start_utc is not None and end_utc is not None
        meter_id = self._meter_id(session, meter_identifier)
        if meter_id is None:
            return None
        return self._analytics.measurement_stats_for_meter(
            session,
            meter_id,
            start=start_utc,
            end=end_utc,
        )

    def compare_periods(
        self,
        session: Session,
        meter_identifier: str,
        *,
        period_a_start: datetime,
        period_a_end: datetime,
        period_b_start: datetime,
        period_b_end: datetime,
    ) -> dict[str, Any] | None:
        period_a_start_utc, period_a_end_utc = self.normalize_range(
            period_a_start,
            period_a_end,
            require_both=True,
            enforce_max_range=True,
        )
        period_b_start_utc, period_b_end_utc = self.normalize_range(
            period_b_start,
            period_b_end,
            require_both=True,
            enforce_max_range=True,
        )
        assert (
            period_a_start_utc is not None
            and period_a_end_utc is not None
            and period_b_start_utc is not None
            and period_b_end_utc is not None
        )
        if not (period_a_end_utc <= period_b_start_utc or period_b_end_utc <= period_a_start_utc):
            raise ValueError("Comparison periods must not overlap.")

        meter_id = self._meter_id(session, meter_identifier)
        if meter_id is None:
            return None

        period_a = self._analytics.summary_for_meter(
            session,
            meter_id,
            start=period_a_start_utc,
            end=period_a_end_utc,
        )
        period_b = self._analytics.summary_for_meter(
            session,
            meter_id,
            start=period_b_start_utc,
            end=period_b_end_utc,
        )
        metrics = (
            "reading_count",
            "active_power_average",
            "active_power_minimum",
            "active_power_maximum",
        )
        differences = {
            name: (
                period_b[name] - period_a[name]
                if period_a[name] is not None and period_b[name] is not None
                else None
            )
            for name in metrics
        }
        percentage_changes = {
            name: (
                (period_b[name] - period_a[name]) / period_a[name] * 100
                if period_a[name] not in (None, 0) and period_b[name] is not None
                else None
            )
            for name in metrics
        }
        return {
            "meter_identifier": meter_identifier,
            "period_a": {
                "start": period_a_start_utc,
                "end": period_a_end_utc,
                **{name: period_a[name] for name in metrics},
            },
            "period_b": {
                "start": period_b_start_utc,
                "end": period_b_end_utc,
                **{name: period_b[name] for name in metrics},
            },
            "difference_period_b_minus_period_a": differences,
            "percentage_change_from_period_a": percentage_changes,
            "percentage_change_definition": (
                "(period_b - period_a) / period_a * 100; null when period_a is zero "
                "or either value is unavailable."
            ),
        }

    def time_series(
        self,
        session: Session,
        meter_identifier: str,
        *,
        start: datetime | None,
        end: datetime | None,
        interval: str,
    ) -> tuple[datetime, datetime, list[dict[str, Any]]] | None:
        start_utc, end_utc = self.normalize_range(
            start,
            end,
            require_both=True,
            enforce_max_range=True,
        )
        if interval not in {"hour", "day"}:
            raise ValueError("interval must be 'hour' or 'day'.")
        meter_id = self._meter_id(session, meter_identifier)
        if meter_id is None:
            return None
        buckets = self._analytics.aggregate_time_series(
            session,
            meter_id,
            start=start_utc,
            end=end_utc,
            interval=interval,
            limit=MAX_TIMESERIES_BUCKETS + 1,
        )
        if len(buckets) > MAX_TIMESERIES_BUCKETS:
            raise ValueError(
                f"Time series exceeds the maximum of {MAX_TIMESERIES_BUCKETS} buckets."
            )
        return start_utc, end_utc, buckets

    def quality(
        self,
        session: Session,
        meter_identifier: str,
        *,
        start: datetime | None,
        end: datetime | None,
    ) -> dict[str, Any] | None:
        start_utc, end_utc = self.normalize_range(start, end)
        meter_id = self._meter_id(session, meter_identifier)
        if meter_id is None:
            return None
        return self._analytics.quality_for_meter(
            session,
            meter_id,
            start=start_utc,
            end=end_utc,
        )
