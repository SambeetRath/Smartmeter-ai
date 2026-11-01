from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from smartmeter_dashboard.api_client import MeterCoverage

DEFAULT_FILTER_DAYS = 7
MAX_FILTER_DAYS = 31
PAGE_STATE_KEY = "dashboard_page"
METER_STATE_KEY = "dashboard_selected_meter"
DATE_RANGE_STATE_KEY = "dashboard_date_range"
DATE_METER_STATE_KEY = "dashboard_date_meter"


@dataclass(frozen=True)
class DashboardFilters:
    meter_identifier: str | None
    start_date: date | None
    end_date: date | None
    refresh_requested: bool = False
    date_range_error: str | None = None

    @property
    def start(self) -> datetime | None:
        if self.start_date is None:
            return None
        return datetime.combine(self.start_date, time.min, tzinfo=timezone.utc)

    @property
    def end(self) -> datetime | None:
        if self.end_date is None:
            return None
        return datetime.combine(
            self.end_date + timedelta(days=1),
            time.min,
            tzinfo=timezone.utc,
        )


def default_date_range(
    meter: MeterCoverage,
    *,
    today: date | None = None,
) -> tuple[date, date]:
    """Choose up to seven recent covered days for a meter."""
    if meter.last_reading_at is None:
        end = today or datetime.now(timezone.utc).date()
        return end - timedelta(days=DEFAULT_FILTER_DAYS - 1), end

    end = meter.last_reading_at.date()
    earliest = meter.first_reading_at.date() if meter.first_reading_at else end
    if earliest > end:
        return end, end
    start = max(earliest, end - timedelta(days=DEFAULT_FILTER_DAYS - 1))
    return start, end


def validate_date_range(start: date, end: date) -> str | None:
    if end < start:
        return "The end date must be on or after the start date."
    if (end - start).days + 1 > MAX_FILTER_DAYS:
        return f"Choose a date range of no more than {MAX_FILTER_DAYS} days."
    return None
