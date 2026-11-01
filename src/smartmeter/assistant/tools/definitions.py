from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from smartmeter.assistant.models import ToolDefinition
from smartmeter.assistant.tools.schemas import (
    AnomalySummaryInput,
    CompareEnergyPeriodsInput,
    DataQualityInput,
    DetectionRunDetailsInput,
    ElectricalMetricsInput,
    EnergySummaryInput,
    EnergyTimeseriesInput,
)


@dataclass(frozen=True)
class ToolSpec[InputModelT: BaseModel]:
    """Provider-neutral tool metadata paired with its local Pydantic input schema."""

    name: str
    description: str
    input_model: type[InputModelT]

    def definition(self) -> ToolDefinition:
        """Create a fresh provider-neutral definition and JSON Schema."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(mode="validation"),
        )

    def validate_arguments(self, arguments: object) -> InputModelT:
        """Parse untrusted arguments into the registered typed input model."""
        return self.input_model.model_validate(arguments)


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_energy_summary",
        description=(
            "Read a bounded aggregate energy summary for one meter. Timestamps are UTC and "
            "the range is half-open [start, end), with a maximum span of 31 days."
        ),
        input_model=EnergySummaryInput,
    ),
    ToolSpec(
        name="get_energy_timeseries",
        description=(
            "Read hourly or daily energy trends for one meter. Timestamps are UTC and "
            "the half-open range is limited to 31 days and at most 750 buckets."
        ),
        input_model=EnergyTimeseriesInput,
    ),
    ToolSpec(
        name="get_electrical_metrics",
        description=(
            "Read direct range-wide averages, minima, maxima, non-null counts, and missing-value "
            "statistics for the seven persisted electrical measurements."
        ),
        input_model=ElectricalMetricsInput,
    ),
    ToolSpec(
        name="get_data_quality",
        description=(
            "Read database-observed missing measurements and timestamp gaps. Gap estimates "
            "assume one-minute sampling, do not assess boundary gaps, and cannot reconstruct "
            "ingestion-time validation findings."
        ),
        input_model=DataQualityInput,
    ),
    ToolSpec(
        name="get_anomaly_summary",
        description=(
            "Read counts for the latest completed detection run for a meter or for a specified "
            "run. No anomaly categories or run status may be inferred."
        ),
        input_model=AnomalySummaryInput,
    ),
    ToolSpec(
        name="get_detection_run_details",
        description=(
            "Read safe, user-relevant details for one detection run. Internal configuration "
            "and worker/lease data are not part of the tool result."
        ),
        input_model=DetectionRunDetailsInput,
    ),
    ToolSpec(
        name="compare_energy_periods",
        description=(
            "Compare direct aggregate energy statistics for two non-overlapping half-open "
            "periods on one meter. Each period is limited to 31 days. Returns period B minus "
            "period A differences and null percentage changes when a baseline is zero or missing."
        ),
        input_model=CompareEnergyPeriodsInput,
    ),
)
