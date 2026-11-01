from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from smartmeter.database.base import Base
from smartmeter.database.models import Meter, MeterReading
from smartmeter.services.analytics_service import AnalyticsService

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
END = START + timedelta(days=1)


class FakeMeters:
    def get_by_identifier(self, _session, _identifier):
        return SimpleNamespace(id=7)


class FakeAnalyticsRepository:
    def __init__(self, summaries):
        self.summaries = iter(summaries)
        self.calls = []

    def summary_for_meter(self, session, meter_id, *, start, end):
        self.calls.append((session, meter_id, start, end))
        return next(self.summaries)


def _period_summary(*, count, average, minimum, maximum):
    return {
        "reading_count": count,
        "active_power_average": average,
        "active_power_minimum": minimum,
        "active_power_maximum": maximum,
    }


def _service(summaries):
    service = AnalyticsService()
    service._meters = FakeMeters()
    service._analytics = FakeAnalyticsRepository(summaries)
    return service


def test_compare_periods_uses_direct_summary_for_each_half_open_period():
    session = object()
    service = _service(
        [
            _period_summary(count=10, average=2.0, minimum=1.0, maximum=4.0),
            _period_summary(count=20, average=3.0, minimum=1.5, maximum=6.0),
        ]
    )

    result = service.compare_periods(
        session,
        "meter-a",
        period_a_start=START,
        period_a_end=END,
        period_b_start=END,
        period_b_end=END + timedelta(days=1),
    )

    assert result is not None
    assert result["period_a"]["reading_count"] == 10
    assert result["period_b"]["reading_count"] == 20
    assert result["difference_period_b_minus_period_a"]["active_power_average"] == 1.0
    assert result["percentage_change_from_period_a"]["active_power_average"] == 50.0
    assert result["difference_period_b_minus_period_a"]["reading_count"] == 10
    assert service._analytics.calls == [
        (session, 7, START, END),
        (session, 7, END, END + timedelta(days=1)),
    ]


def test_compare_periods_returns_null_percentage_for_zero_or_missing_baseline():
    service = _service(
        [
            _period_summary(count=0, average=0.0, minimum=None, maximum=None),
            _period_summary(count=5, average=3.0, minimum=2.0, maximum=4.0),
        ]
    )

    result = service.compare_periods(
        object(),
        "meter-a",
        period_a_start=START,
        period_a_end=END,
        period_b_start=END,
        period_b_end=END + timedelta(days=1),
    )

    assert result is not None
    assert result["difference_period_b_minus_period_a"]["active_power_average"] == 3.0
    assert result["percentage_change_from_period_a"]["active_power_average"] is None
    assert result["difference_period_b_minus_period_a"]["active_power_minimum"] is None
    assert result["percentage_change_from_period_a"]["active_power_minimum"] is None
    assert result["percentage_change_from_period_a"]["reading_count"] is None


@pytest.mark.parametrize(
    ("period_a_start", "period_a_end", "period_b_start", "period_b_end"),
    [
        (START, START, END, END + timedelta(days=1)),
        (START, END, START + timedelta(hours=12), END + timedelta(days=1)),
        (START, START + timedelta(days=32), END + timedelta(days=40), END + timedelta(days=41)),
    ],
)
def test_compare_periods_rejects_invalid_overlap_and_excessive_ranges(
    period_a_start,
    period_a_end,
    period_b_start,
    period_b_end,
):
    service = _service([])

    with pytest.raises(ValueError):
        service.compare_periods(
            object(),
            "meter-a",
            period_a_start=period_a_start,
            period_a_end=period_a_end,
            period_b_start=period_b_start,
            period_b_end=period_b_end,
        )

    assert service._analytics.calls == []


def _insert_sqlite_fixture(readings):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[Meter.__table__, MeterReading.__table__])
    session = Session(engine)
    meter = Meter(meter_identifier="comparison-fixture")
    session.add(meter)
    session.flush()
    session.add_all(
        [
            MeterReading(
                meter_id=meter.id,
                timestamp=timestamp,
                global_active_power=(
                    Decimal(str(active_power)) if active_power is not None else None
                ),
                voltage=Decimal(str(voltage)) if voltage is not None else None,
            )
            for timestamp, active_power, voltage in readings
        ]
    )
    session.flush()
    return engine, session


def test_sqlite_fixture_verifies_direct_aggregates_null_counts_and_half_open_comparison():
    period_a_start = START
    period_a_end = START + timedelta(hours=1)
    period_b_end = START + timedelta(hours=2)
    engine, session = _insert_sqlite_fixture(
        [
            (START, 0, 220),
            (START + timedelta(minutes=30), 10, None),
            (period_a_end, 10, 240),
            (START + timedelta(minutes=90), 30, 260),
            (period_b_end, 999, 999),
        ]
    )
    try:
        service = AnalyticsService()

        metrics = service.electrical_metrics(
            session,
            "comparison-fixture",
            start=period_a_start,
            end=period_a_end,
        )
        comparison = service.compare_periods(
            session,
            "comparison-fixture",
            period_a_start=period_a_start,
            period_a_end=period_a_end,
            period_b_start=period_a_end,
            period_b_end=period_b_end,
        )

        assert metrics is not None
        assert metrics["reading_count"] == 2
        assert metrics["measurements"]["global_active_power"] == {
            "value_count": 2,
            "average": 5.0,
            "minimum": 0.0,
            "maximum": 10.0,
            "missing_count": 0,
            "missing_rate": 0.0,
        }
        assert metrics["measurements"]["voltage"] == {
            "value_count": 1,
            "average": 220.0,
            "minimum": 220.0,
            "maximum": 220.0,
            "missing_count": 1,
            "missing_rate": 0.5,
        }
        assert comparison is not None
        assert comparison["period_a"]["reading_count"] == 2
        assert comparison["period_a"]["active_power_average"] == 5.0
        assert comparison["period_b"]["reading_count"] == 2
        assert comparison["period_b"]["active_power_average"] == 20.0
        assert comparison["difference_period_b_minus_period_a"]["active_power_average"] == 15.0
        assert comparison["percentage_change_from_period_a"]["active_power_average"] == 300.0
        assert comparison["period_b"]["active_power_maximum"] == 30.0
    finally:
        session.close()
        engine.dispose()


def test_sqlite_fixture_verifies_empty_period_and_zero_baseline_percentage():
    period_a_start = START
    period_a_end = START + timedelta(hours=1)
    period_b_start = START + timedelta(hours=2)
    period_b_end = START + timedelta(hours=3)
    engine, session = _insert_sqlite_fixture(
        [
            (period_a_start, 0, None),
            (period_b_start, 5, 230),
        ]
    )
    try:
        service = AnalyticsService()
        empty_period = service.electrical_metrics(
            session,
            "comparison-fixture",
            start=period_a_end,
            end=period_b_start,
        )
        comparison = service.compare_periods(
            session,
            "comparison-fixture",
            period_a_start=period_a_start,
            period_a_end=period_a_end,
            period_b_start=period_b_start,
            period_b_end=period_b_end,
        )

        assert empty_period is not None
        assert empty_period["reading_count"] == 0
        assert empty_period["measurements"]["voltage"]["average"] is None
        assert empty_period["measurements"]["voltage"]["missing_rate"] is None
        assert comparison is not None
        assert comparison["period_a"]["active_power_average"] == 0.0
        assert comparison["period_b"]["active_power_average"] == 5.0
        assert comparison["difference_period_b_minus_period_a"]["active_power_average"] == 5.0
        assert comparison["percentage_change_from_period_a"]["active_power_average"] is None
    finally:
        session.close()
        engine.dispose()
