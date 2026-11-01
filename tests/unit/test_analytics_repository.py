from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.dialects import postgresql

from smartmeter.database.repositories.analytics_repository import (
    MEASUREMENT_COLUMNS,
    AnalyticsRepository,
)

UTC = timezone.utc


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows

    def one(self):
        if isinstance(self.rows, dict):
            return self.rows
        return self.rows[0]


class CaptureSession:
    def __init__(self, *rows):
        self.rows = list(rows)
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        return FakeResult(self.rows.pop(0))


def _compile(statement) -> tuple[str, dict]:
    compiled = statement.compile(dialect=postgresql.dialect())
    return str(compiled), compiled.params


def _empty_summary():
    return {
        "reading_count": 0,
        "active_power_average": None,
        "active_power_minimum": None,
        "active_power_maximum": None,
        "latest_active_power": None,
        "latest_active_power_at": None,
        **{
            f"{column}_missing_count": 0
            for column in (
                "global_active_power",
                "global_reactive_power",
                "voltage",
                "global_intensity",
                "sub_metering_1",
                "sub_metering_2",
                "sub_metering_3",
            )
        },
    }


def test_summary_uses_single_sql_aggregate_with_half_open_filters():
    session = CaptureSession(_empty_summary())

    result = AnalyticsRepository().summary_for_meter(
        session,
        7,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 2, tzinfo=UTC),
    )

    sql, params = _compile(session.statements[0])
    assert result["reading_count"] == 0
    assert len(result["missing_measurements"]) == 7
    assert result["missing_measurements"]["voltage"]["missing_rate"] is None
    assert "count(meter_readings.id)" in sql.lower()
    assert "meter_readings.timestamp >=" in sql.lower()
    assert "meter_readings.timestamp <" in sql.lower()
    assert "global_active_power_missing_count" in sql
    assert "latest_active_power" in sql
    assert 7 in params.values()


def test_meter_coverage_query_uses_outer_join_and_bounded_page():
    session = CaptureSession(
        [
            {
                "meter_identifier": "meter-a",
                "reading_count": 0,
                "first_reading_at": None,
                "last_reading_at": None,
            }
        ]
    )

    result = AnalyticsRepository().list_meters(session, limit=10, offset=20)

    sql, _ = _compile(session.statements[0])
    assert result[0]["reading_count"] == 0
    assert "LEFT OUTER JOIN meter_readings" in sql
    assert "LIMIT" in sql
    assert "OFFSET" in sql


def test_time_series_aggregates_in_postgres_with_buckets_and_limits():
    session = CaptureSession([])

    result = AnalyticsRepository().aggregate_time_series(
        session,
        7,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 2, tzinfo=UTC),
        interval="hour",
        limit=751,
    )

    sql, params = _compile(session.statements[0])
    assert result == []
    assert "date_trunc" in sql.lower()
    assert "group by date_trunc" in sql.lower()
    assert "avg(meter_readings.voltage)" in sql.lower()
    assert "meter_readings.timestamp >=" in sql.lower()
    assert "meter_readings.timestamp <" in sql.lower()
    assert 7 in params.values()
    assert 751 in params.values()


def test_quality_query_uses_window_lag_and_never_reads_raw_rows():
    session = CaptureSession(
        _empty_summary(),
        {
            "observed_gap_count": 1,
            "estimated_missing_intervals": 2,
            "first_observed_at": datetime(2024, 1, 1, tzinfo=UTC),
            "last_observed_at": datetime(2024, 1, 1, 0, 3, tzinfo=UTC),
        },
    )

    result = AnalyticsRepository().quality_for_meter(session, 7, start=None, end=None)
    summary_sql, _ = _compile(session.statements[0])
    gap_sql, _ = _compile(session.statements[1])

    assert result["timestamp_gaps"]["observed_gap_count"] == 1
    assert result["timestamp_gaps"]["estimated_missing_intervals"] == 2
    assert "count(meter_readings.id)" in summary_sql.lower()
    assert "lag(meter_readings.timestamp)" in gap_sql.lower()
    assert "extract(epoch" in gap_sql.lower()
    assert "scoped_readings" in gap_sql


def test_electrical_metrics_aggregate_each_underlying_measurement_directly():
    row = {"reading_count": 4}
    for column in MEASUREMENT_COLUMNS:
        row.update(
            {
                f"{column}_value_count": 3,
                f"{column}_average": 2.5,
                f"{column}_minimum": 1.0,
                f"{column}_maximum": 4.0,
                f"{column}_missing_count": 1,
            }
        )
    session = CaptureSession(row)

    result = AnalyticsRepository().measurement_stats_for_meter(
        session,
        7,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 2, tzinfo=UTC),
    )

    sql, params = _compile(session.statements[0])
    assert result["reading_count"] == 4
    assert result["measurements"]["voltage"] == {
        "value_count": 3,
        "average": 2.5,
        "minimum": 1.0,
        "maximum": 4.0,
        "missing_count": 1,
        "missing_rate": 0.25,
    }
    for column in MEASUREMENT_COLUMNS:
        assert f"avg(meter_readings.{column})" in sql.lower()
        assert f"min(meter_readings.{column})" in sql.lower()
        assert f"max(meter_readings.{column})" in sql.lower()
    assert "date_trunc" not in sql.lower()
    assert "meter_readings.timestamp >=" in sql.lower()
    assert "meter_readings.timestamp <" in sql.lower()
    assert 7 in params.values()
