from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from smartmeter.assistant.models import ToolDefinition
from smartmeter.assistant.tools import (
    APPROVED_TOOL_NAMES,
    DEFAULT_TOOL_REGISTRY,
    TOOL_SPECS,
    AnomalySummaryInput,
    CompareEnergyPeriodsInput,
    DataQualityInput,
    DetectionRunDetailsInput,
    DuplicateToolError,
    ElectricalMetricsInput,
    EnergySummaryInput,
    EnergyTimeseriesInput,
    ToolRegistry,
    UnknownToolError,
)

START = datetime(2010, 1, 1, tzinfo=timezone.utc)
END = START + timedelta(days=1)
METER = "uci_household_1"
RUN_ID = UUID("5402306a-5285-424a-9fab-c2556146b3ef")

VALID_ARGUMENTS: dict[str, dict[str, object]] = {
    "get_energy_summary": {
        "meter_identifier": METER,
        "start": START.isoformat(),
        "end": END.isoformat(),
    },
    "get_energy_timeseries": {
        "meter_identifier": METER,
        "start": START.isoformat(),
        "end": END.isoformat(),
        "interval": "hour",
    },
    "get_electrical_metrics": {
        "meter_identifier": METER,
        "start": START.isoformat(),
        "end": END.isoformat(),
    },
    "get_data_quality": {
        "meter_identifier": METER,
        "start": START.isoformat(),
        "end": END.isoformat(),
    },
    "get_anomaly_summary": {"meter_identifier": METER},
    "get_detection_run_details": {"run_id": str(RUN_ID)},
    "compare_energy_periods": {
        "meter_identifier": METER,
        "period_a_start": START.isoformat(),
        "period_a_end": END.isoformat(),
        "period_b_start": END.isoformat(),
        "period_b_end": (END + timedelta(days=1)).isoformat(),
    },
}

INPUT_MODELS = {
    "get_energy_summary": EnergySummaryInput,
    "get_energy_timeseries": EnergyTimeseriesInput,
    "get_electrical_metrics": ElectricalMetricsInput,
    "get_data_quality": DataQualityInput,
    "get_anomaly_summary": AnomalySummaryInput,
    "get_detection_run_details": DetectionRunDetailsInput,
    "compare_energy_periods": CompareEnergyPeriodsInput,
}


def test_registry_contains_seven_unique_approved_tools_and_definitions():
    definitions = DEFAULT_TOOL_REGISTRY.list_definitions()

    assert len(definitions) == 7
    assert tuple(definition.name for definition in definitions) == APPROVED_TOOL_NAMES
    assert len(APPROVED_TOOL_NAMES) == len(set(APPROVED_TOOL_NAMES))
    assert set(APPROVED_TOOL_NAMES) == set(VALID_ARGUMENTS)
    assert all(isinstance(definition, ToolDefinition) for definition in definitions)
    assert all(definition.description for definition in definitions)
    assert all(definition.parameters["additionalProperties"] is False for definition in definitions)


@pytest.mark.parametrize("tool_name", tuple(VALID_ARGUMENTS))
def test_registry_parses_valid_arguments_into_registered_typed_model(tool_name: str):
    parsed = DEFAULT_TOOL_REGISTRY.validate_arguments(tool_name, VALID_ARGUMENTS[tool_name])

    assert type(parsed) is INPUT_MODELS[tool_name]


@pytest.mark.parametrize("tool_name", tuple(VALID_ARGUMENTS))
def test_each_tool_rejects_missing_required_arguments(tool_name: str):
    with pytest.raises(ValidationError):
        DEFAULT_TOOL_REGISTRY.validate_arguments(tool_name, {})


@pytest.mark.parametrize("tool_name", tuple(VALID_ARGUMENTS))
def test_each_tool_rejects_unexpected_arguments(tool_name: str):
    arguments = {
        **VALID_ARGUMENTS[tool_name],
        "untrusted_handler": "os.system",
    }

    with pytest.raises(ValidationError):
        DEFAULT_TOOL_REGISTRY.validate_arguments(tool_name, arguments)


def test_registry_rejects_duplicate_registration():
    with pytest.raises(DuplicateToolError, match="Duplicate analytics tool name"):
        ToolRegistry((TOOL_SPECS[0], TOOL_SPECS[0]))


def test_registry_uses_exact_name_lookup_and_rejects_unknown_names():
    assert DEFAULT_TOOL_REGISTRY.get_definition("get_energy_summary").name == "get_energy_summary"
    with pytest.raises(UnknownToolError, match="Unknown analytics tool name"):
        DEFAULT_TOOL_REGISTRY.get_tool_spec("GET_ENERGY_SUMMARY")
    with pytest.raises(UnknownToolError):
        DEFAULT_TOOL_REGISTRY.validate_arguments("execute_sql", {"query": "SELECT 1"})


@pytest.mark.parametrize("meter_identifier", ["", " ", "x" * 101])
def test_invalid_meter_identifiers_are_rejected(meter_identifier: str):
    with pytest.raises(ValidationError):
        EnergySummaryInput(
            meter_identifier=meter_identifier,
            start=START,
            end=END,
        )


def test_energy_date_ranges_require_timezone_and_normalize_to_utc():
    with pytest.raises(ValidationError, match="UTC offset"):
        EnergySummaryInput(
            meter_identifier=METER,
            start=datetime(2010, 1, 1),
            end=END,
        )

    local_start = datetime.fromisoformat("2010-01-01T03:00:00+03:00")
    parsed = EnergySummaryInput(meter_identifier=METER, start=local_start, end=END)

    assert parsed.start == START
    assert parsed.start.tzinfo is timezone.utc


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (END, START),
        (START, START),
        (START, START + timedelta(days=32)),
    ],
)
def test_energy_summary_rejects_invalid_or_excessive_ranges(start: datetime, end: datetime):
    with pytest.raises(ValidationError):
        EnergySummaryInput(meter_identifier=METER, start=start, end=end)


def test_timeseries_rejects_unsupported_interval():
    with pytest.raises(ValidationError):
        EnergyTimeseriesInput(
            meter_identifier=METER,
            start=START,
            end=END,
            interval="minute",
        )


def test_anomaly_summary_requires_meter_or_run_and_accepts_either_or_both():
    with pytest.raises(ValidationError, match="meter_identifier, run_id"):
        AnomalySummaryInput()

    assert AnomalySummaryInput(meter_identifier=METER).meter_identifier == METER
    assert AnomalySummaryInput(run_id=RUN_ID).run_id == RUN_ID
    both = AnomalySummaryInput(meter_identifier=METER, run_id=RUN_ID)
    assert both.meter_identifier == METER


def test_detection_details_validates_run_id_as_uuid():
    assert DetectionRunDetailsInput(run_id=RUN_ID).run_id == RUN_ID
    with pytest.raises(ValidationError):
        DetectionRunDetailsInput(run_id="not-a-uuid")


def test_period_comparison_normalizes_utc_and_accepts_adjacent_periods():
    parsed = CompareEnergyPeriodsInput(
        meter_identifier=METER,
        period_a_start=START,
        period_a_end=END,
        period_b_start=END,
        period_b_end=END + timedelta(days=1),
    )

    assert parsed.period_a_start.tzinfo is timezone.utc
    assert parsed.period_a_end == parsed.period_b_start


@pytest.mark.parametrize(
    ("period_a_start", "period_a_end", "period_b_start", "period_b_end"),
    [
        (START, START, END, END + timedelta(days=1)),
        (START, END, START + timedelta(hours=12), END + timedelta(days=1)),
        (START, START + timedelta(days=32), END + timedelta(days=40), END + timedelta(days=41)),
    ],
)
def test_period_comparison_rejects_invalid_overlapping_or_excessive_periods(
    period_a_start: datetime,
    period_a_end: datetime,
    period_b_start: datetime,
    period_b_end: datetime,
):
    with pytest.raises(ValidationError):
        CompareEnergyPeriodsInput(
            meter_identifier=METER,
            period_a_start=period_a_start,
            period_a_end=period_a_end,
            period_b_start=period_b_start,
            period_b_end=period_b_end,
        )


@pytest.mark.parametrize(
    "forbidden_argument",
    [
        {"sql": "DROP TABLE meter_readings"},
        {"python_expression": "__import__('os').system('whoami')"},
        {"handler": "smartmeter.services.detection_service.process_detection_run"},
    ],
)
def test_tool_arguments_cannot_select_execution_or_mutation_capabilities(
    forbidden_argument: dict[str, str],
):
    arguments = {**VALID_ARGUMENTS["get_energy_summary"], **forbidden_argument}

    with pytest.raises(ValidationError):
        DEFAULT_TOOL_REGISTRY.validate_arguments("get_energy_summary", arguments)

    assert not {
        "execute_sql",
        "execute_python",
        "submit_detection_run",
        "create_meter",
        "delete_meter",
    }.intersection(APPROVED_TOOL_NAMES)
    assert all(not hasattr(spec, "handler") for spec in TOOL_SPECS)


def test_returned_tool_definitions_cannot_mutate_registry_schemas():
    definition = DEFAULT_TOOL_REGISTRY.get_definition("get_energy_summary")
    definition.parameters["properties"].clear()

    next_definition = DEFAULT_TOOL_REGISTRY.get_definition("get_energy_summary")
    assert "meter_identifier" in next_definition.parameters["properties"]
