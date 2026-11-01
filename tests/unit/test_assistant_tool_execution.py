from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy.exc import SQLAlchemyError

from smartmeter.assistant.tools import AnalyticsToolExecutor, ToolExecutionError

UTC = timezone.utc
START = datetime(2010, 1, 1, tzinfo=UTC)
END = START + timedelta(days=1)
METER = "uci_household_1"
RUN_ID = UUID("5402306a-5285-424a-9fab-c2556146b3ef")


class SessionContext:
    def __init__(self) -> None:
        self.closed = False
        self.session = object()

    def __enter__(self):
        return self.session

    def __exit__(self, *_exc_info):
        self.closed = True
        return False


class FakeAnalytics:
    def __init__(self) -> None:
        self.calls = []
        self.summary_result = {
            "reading_count": 4,
            "active_power_average": 2.5,
            "active_power_minimum": 1.0,
            "active_power_maximum": 4.0,
            "latest_active_power": 4.0,
            "latest_active_power_at": END,
            "missing_measurements": {},
        }
        self.timeseries_result = (
            START,
            END,
            [
                {
                    "timestamp": START,
                    "reading_count": 4,
                    "global_active_power_average": 2.5,
                }
            ],
        )
        self.electrical_result = {
            "reading_count": 4,
            "measurements": {"voltage": {"average": 230.0}},
            "aggregation_scope": "Direct SQL aggregates.",
        }
        self.quality_result = {
            "reading_count": 4,
            "missing_measurements": {},
            "timestamp_gaps": {
                "expected_interval_minutes": 1,
                "observed_gap_count": 0,
                "estimated_missing_intervals": 0,
                "first_observed_at": START,
                "last_observed_at": END,
                "boundary_gaps_assessed": False,
            },
            "database_metrics_scope": "Stored readings only.",
        }
        self.comparison_result = {
            "meter_identifier": METER,
            "period_a": {"start": START, "end": END},
            "period_b": {"start": END, "end": END + timedelta(days=1)},
            "difference_period_b_minus_period_a": {"active_power_average": 1.0},
            "percentage_change_from_period_a": {"active_power_average": 50.0},
        }
        self.missing_meter = False

    def summary(self, session, meter_identifier, *, start, end):
        self.calls.append(("summary", session, meter_identifier, start, end))
        return None if self.missing_meter else self.summary_result

    def time_series(self, session, meter_identifier, *, start, end, interval):
        self.calls.append(("time_series", session, meter_identifier, start, end, interval))
        return None if self.missing_meter else self.timeseries_result

    def electrical_metrics(self, session, meter_identifier, *, start, end):
        self.calls.append(("electrical_metrics", session, meter_identifier, start, end))
        return None if self.missing_meter else self.electrical_result

    def quality(self, session, meter_identifier, *, start, end):
        self.calls.append(("quality", session, meter_identifier, start, end))
        return None if self.missing_meter else self.quality_result

    def compare_periods(self, session, meter_identifier, **periods):
        self.calls.append(("compare_periods", session, meter_identifier, periods))
        return None if self.missing_meter else self.comparison_result


class FakeMeters:
    def __init__(self) -> None:
        self.meter = SimpleNamespace(id=1, meter_identifier=METER)

    def get_by_identifier(self, _session, _identifier):
        return self.meter


class FakeDetectionRuns:
    def __init__(self) -> None:
        self.run = SimpleNamespace(
            id=RUN_ID,
            meter=SimpleNamespace(meter_identifier=METER),
            algorithm="global_iqr",
            status="completed",
            start_timestamp=START,
            end_timestamp=END,
            created_at=END,
            started_at=END,
            completed_at=END,
            training_readings=3,
            readings_processed=4,
            anomaly_count=1,
            normal_count=3,
            unavailable_count=0,
            configuration={"internal": "must not be returned"},
            error_message="must not be returned",
        )
        self.latest = self.run

    def get_by_id(self, _session, run_id):
        return self.run if run_id == self.run.id else None

    def latest_completed(self, _session, *, meter_identifier):
        assert meter_identifier == METER
        return self.latest


class FakeAnomalies:
    def category_counts(self, _session, run_id):
        assert run_id == RUN_ID
        return {"spike": 1}


@pytest.fixture
def executor_bundle():
    context = SessionContext()
    analytics = FakeAnalytics()
    meters = FakeMeters()
    runs = FakeDetectionRuns()
    executor = AnalyticsToolExecutor(
        session_factory=lambda: context,
        analytics_service=analytics,
        meter_repository=meters,
        detection_repository=runs,
        anomaly_repository=FakeAnomalies(),
    )
    return executor, context, analytics, meters, runs


def _range_args() -> dict[str, str]:
    return {
        "meter_identifier": METER,
        "start": START.isoformat(),
        "end": END.isoformat(),
    }


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_key"),
    [
        ("get_energy_summary", _range_args(), "active_power_average"),
        (
            "get_energy_timeseries",
            {**_range_args(), "interval": "hour"},
            "buckets",
        ),
        ("get_electrical_metrics", _range_args(), "measurements"),
        ("get_data_quality", _range_args(), "ingestion_validation"),
        (
            "get_anomaly_summary",
            {"meter_identifier": METER},
            "counts_by_category",
        ),
        ("get_detection_run_details", {"run_id": str(RUN_ID)}, "training_readings"),
        (
            "compare_energy_periods",
            {
                "meter_identifier": METER,
                "period_a_start": START.isoformat(),
                "period_a_end": END.isoformat(),
                "period_b_start": END.isoformat(),
                "period_b_end": (END + timedelta(days=1)).isoformat(),
            },
            "percentage_change_from_period_a",
        ),
    ],
)
def test_each_tool_executes_and_returns_json_serializable_result(
    executor_bundle,
    tool_name: str,
    arguments: dict[str, str],
    expected_key: str,
):
    executor, context, _analytics, _meters, _runs = executor_bundle

    result = executor.execute(tool_name, arguments)

    assert expected_key in result
    json.dumps(result, allow_nan=False)
    assert context.closed


def test_summary_uses_bounded_validated_half_open_range(executor_bundle):
    executor, _context, analytics, _meters, _runs = executor_bundle

    result = executor.execute("get_energy_summary", _range_args())

    name, _session, meter, start, end = analytics.calls[0]
    assert name == "summary"
    assert meter == METER
    assert start == START
    assert end == END
    assert result["start"] == START.isoformat()
    assert result["end"] == END.isoformat()


def test_empty_timeseries_is_a_valid_empty_result(executor_bundle):
    executor, _context, analytics, _meters, _runs = executor_bundle
    analytics.timeseries_result = (START, END, [])

    result = executor.execute(
        "get_energy_timeseries",
        {**_range_args(), "interval": "day"},
    )

    assert result["buckets"] == []
    assert result["returned_count"] == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"meter_identifier": METER, "start": "2010-01-01T00:00:00", "end": END.isoformat()},
        {
            "meter_identifier": METER,
            "start": END.isoformat(),
            "end": START.isoformat(),
        },
        {
            "meter_identifier": METER,
            "start": START.isoformat(),
            "end": (START + timedelta(days=32)).isoformat(),
        },
    ],
)
def test_invalid_or_excessive_arguments_fail_before_opening_a_session(arguments):
    factory_calls = []
    executor = AnalyticsToolExecutor(
        session_factory=lambda: factory_calls.append(True),
        analytics_service=FakeAnalytics(),
    )

    with pytest.raises(ToolExecutionError) as error:
        executor.execute("get_energy_summary", arguments)

    assert error.value.code == "invalid_arguments"
    assert factory_calls == []


def test_unknown_tool_fails_before_opening_a_session():
    factory_calls = []
    executor = AnalyticsToolExecutor(
        session_factory=lambda: factory_calls.append(True),
        analytics_service=FakeAnalytics(),
    )

    with pytest.raises(ToolExecutionError) as error:
        executor.execute("execute_sql", {"query": "SELECT 1"})

    assert error.value.code == "unknown_tool"
    assert factory_calls == []


def test_non_string_tool_name_fails_safely_before_opening_a_session():
    factory_calls = []
    executor = AnalyticsToolExecutor(
        session_factory=lambda: factory_calls.append(True),
        analytics_service=FakeAnalytics(),
    )

    with pytest.raises(ToolExecutionError) as error:
        executor.execute([], {})  # type: ignore[arg-type]

    assert error.value.code == "unknown_tool"
    assert factory_calls == []


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_energy_summary", _range_args()),
        ("get_energy_timeseries", {**_range_args(), "interval": "hour"}),
        ("get_electrical_metrics", _range_args()),
        ("get_data_quality", _range_args()),
        (
            "compare_energy_periods",
            {
                "meter_identifier": METER,
                "period_a_start": START.isoformat(),
                "period_a_end": END.isoformat(),
                "period_b_start": END.isoformat(),
                "period_b_end": (END + timedelta(days=1)).isoformat(),
            },
        ),
    ],
)
def test_unknown_meter_is_not_reported_as_empty_data(executor_bundle, tool_name, arguments):
    executor, _context, analytics, _meters, _runs = executor_bundle
    analytics.missing_meter = True

    with pytest.raises(ToolExecutionError) as error:
        executor.execute(tool_name, arguments)

    assert error.value.code == "meter_not_found"


def test_anomaly_summary_distinguishes_unknown_meter_and_no_completed_run(executor_bundle):
    executor, _context, _analytics, meters, runs = executor_bundle
    meters.meter = None

    with pytest.raises(ToolExecutionError) as unknown_meter:
        executor.execute("get_anomaly_summary", {"meter_identifier": METER})
    assert unknown_meter.value.code == "meter_not_found"

    meters.meter = SimpleNamespace(id=1, meter_identifier=METER)
    runs.latest = None
    with pytest.raises(ToolExecutionError) as no_run:
        executor.execute("get_anomaly_summary", {"meter_identifier": METER})
    assert no_run.value.code == "no_completed_run"


def test_anomaly_summary_distinguishes_unknown_and_noncompleted_runs(executor_bundle):
    executor, _context, _analytics, _meters, runs = executor_bundle

    with pytest.raises(ToolExecutionError) as unknown_run:
        executor.execute("get_anomaly_summary", {"run_id": str(UUID(int=0))})
    assert unknown_run.value.code == "not_found"

    runs.run.status = "running"
    with pytest.raises(ToolExecutionError) as noncompleted:
        executor.execute("get_anomaly_summary", {"run_id": str(RUN_ID)})
    assert noncompleted.value.code == "run_not_completed"


def test_anomaly_summary_returns_reconciled_run_projection(executor_bundle):
    executor, _context, _analytics, _meters, _runs = executor_bundle

    result = executor.execute("get_anomaly_summary", {"run_id": str(RUN_ID)})

    assert result["anomaly_count"] == 1
    assert result["normal_count"] == 3
    assert result["unavailable_count"] == 0
    assert result["anomaly_rate"] == 0.25
    assert result["counts_by_category"] == {"spike": 1}


def test_run_details_omit_configuration_and_failure_internals(executor_bundle):
    executor, _context, _analytics, _meters, _runs = executor_bundle

    result = executor.execute("get_detection_run_details", {"run_id": str(RUN_ID)})

    assert result["status"] == "completed"
    assert result["readings_processed"] == 4
    assert "configuration" not in result
    assert "error_message" not in result
    assert "attempts" not in result


def test_unknown_run_details_fails_as_not_found(executor_bundle):
    executor, _context, _analytics, _meters, _runs = executor_bundle

    with pytest.raises(ToolExecutionError) as error:
        executor.execute("get_detection_run_details", {"run_id": str(UUID(int=0))})

    assert error.value.code == "not_found"


def test_period_comparison_returns_direct_service_result_and_zero_safe_changes(executor_bundle):
    executor, _context, analytics, _meters, _runs = executor_bundle
    analytics.comparison_result = {
        "meter_identifier": METER,
        "period_a": {"active_power_average": 0.0},
        "period_b": {"active_power_average": 5.0},
        "difference_period_b_minus_period_a": {"active_power_average": 5.0},
        "percentage_change_from_period_a": {"active_power_average": None},
    }
    arguments = {
        "meter_identifier": METER,
        "period_a_start": START.isoformat(),
        "period_a_end": END.isoformat(),
        "period_b_start": END.isoformat(),
        "period_b_end": (END + timedelta(days=1)).isoformat(),
    }

    result = executor.execute("compare_energy_periods", arguments)

    assert result["difference_period_b_minus_period_a"]["active_power_average"] == 5.0
    assert result["percentage_change_from_period_a"]["active_power_average"] is None
    assert analytics.calls[0][0] == "compare_periods"


def test_database_exception_is_safely_reported_and_session_is_closed(executor_bundle):
    executor, context, analytics, _meters, _runs = executor_bundle

    def fail(*_args, **_kwargs):
        raise SQLAlchemyError("sensitive statement parameters")

    analytics.summary = fail

    with pytest.raises(ToolExecutionError) as error:
        executor.execute("get_energy_summary", _range_args())

    assert error.value.code == "data_access_error"
    assert "sensitive" not in str(error.value)
    assert context.closed


def test_unexpected_error_is_sanitized_and_session_is_closed(executor_bundle):
    executor, context, analytics, _meters, _runs = executor_bundle

    def fail(*_args, **_kwargs):
        raise RuntimeError("unexpected failure")

    analytics.summary = fail

    with pytest.raises(ToolExecutionError) as error:
        executor.execute("get_energy_summary", _range_args())

    assert error.value.code == "internal_error"
    assert "unexpected failure" not in str(error.value)
    assert context.closed


def test_unserializable_or_nonfinite_results_are_rejected(executor_bundle):
    executor, _context, analytics, _meters, _runs = executor_bundle
    analytics.summary_result = {"invalid": float("nan")}

    with pytest.raises(ToolExecutionError) as error:
        executor.execute("get_energy_summary", _range_args())

    assert error.value.code == "invalid_result"
