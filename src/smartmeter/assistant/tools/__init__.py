"""Provider-independent, read-only analytics tool schemas and registry."""

from smartmeter.assistant.tools.definitions import TOOL_SPECS, ToolSpec
from smartmeter.assistant.tools.execution import AnalyticsToolExecutor, ToolExecutionError
from smartmeter.assistant.tools.registry import (
    APPROVED_TOOL_NAMES,
    DEFAULT_TOOL_REGISTRY,
    DuplicateToolError,
    ToolRegistry,
    UnknownToolError,
)
from smartmeter.assistant.tools.schemas import (
    AnomalySummaryInput,
    CompareEnergyPeriodsInput,
    DataQualityInput,
    DetectionRunDetailsInput,
    ElectricalMetricsInput,
    EnergySummaryInput,
    EnergyTimeseriesInput,
)

__all__ = [
    "APPROVED_TOOL_NAMES",
    "AnalyticsToolExecutor",
    "AnomalySummaryInput",
    "CompareEnergyPeriodsInput",
    "DEFAULT_TOOL_REGISTRY",
    "DataQualityInput",
    "DetectionRunDetailsInput",
    "DuplicateToolError",
    "ElectricalMetricsInput",
    "EnergySummaryInput",
    "EnergyTimeseriesInput",
    "TOOL_SPECS",
    "ToolRegistry",
    "ToolExecutionError",
    "ToolSpec",
    "UnknownToolError",
]
