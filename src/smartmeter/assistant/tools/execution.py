from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from smartmeter.assistant.tools.registry import DEFAULT_TOOL_REGISTRY, UnknownToolError
from smartmeter.assistant.tools.schemas import (
    AnomalySummaryInput,
    CompareEnergyPeriodsInput,
    DataQualityInput,
    DetectionRunDetailsInput,
    ElectricalMetricsInput,
    EnergySummaryInput,
    EnergyTimeseriesInput,
)
from smartmeter.database.models import DetectionRun
from smartmeter.database.repositories import (
    AnomalyResultRepository,
    DetectionRunRepository,
    MeterRepository,
)
from smartmeter.database.session import SessionFactory, create_session_factory
from smartmeter.services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

SessionContextFactory = Callable[[], AbstractContextManager[Session]]


class ToolExecutionError(RuntimeError):
    """Safe, structured failure returned by the trusted analytics tool layer."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AnalyticsToolExecutor:
    """Execute only the fixed analytics allowlist using trusted application services.

    Raw argument dictionaries are validated against the approved registry before a session
    is opened. Dispatch is a static name-to-method match; tool arguments never select code.
    Each invocation owns one context-managed synchronous read session.
    """

    def __init__(
        self,
        *,
        session_factory: SessionFactory | SessionContextFactory | None = None,
        analytics_service: AnalyticsService | None = None,
        meter_repository: MeterRepository | None = None,
        detection_repository: DetectionRunRepository | None = None,
        anomaly_repository: AnomalyResultRepository | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._analytics = analytics_service if analytics_service is not None else AnalyticsService()
        self._meters = meter_repository if meter_repository is not None else MeterRepository()
        self._detection_runs = (
            detection_repository if detection_repository is not None else DetectionRunRepository()
        )
        self._anomalies = (
            anomaly_repository if anomaly_repository is not None else AnomalyResultRepository()
        )

    def execute(self, tool_name: str, arguments: object) -> dict[str, Any]:
        """Validate and execute one exact registered tool; results are JSON-compatible."""
        if not isinstance(tool_name, str):
            raise ToolExecutionError("unknown_tool", "Unknown analytics tool.")
        try:
            tool_spec = DEFAULT_TOOL_REGISTRY.get_tool_spec(tool_name)
        except UnknownToolError:
            raise ToolExecutionError("unknown_tool", "Unknown analytics tool.") from None

        try:
            parsed = tool_spec.validate_arguments(arguments)
        except ValidationError:
            raise ToolExecutionError(
                "invalid_arguments",
                "Tool arguments do not match the approved input schema.",
            ) from None

        try:
            factory = (
                self._session_factory
                if self._session_factory is not None
                else create_session_factory()
            )
            with factory() as session:
                result = self._dispatch(tool_name, parsed, session)
        except ToolExecutionError:
            raise
        except SQLAlchemyError:
            logger.error("Analytics tool %s failed during a database operation.", tool_name)
            raise ToolExecutionError(
                "data_access_error",
                "The requested analytics data could not be retrieved.",
            ) from None
        except Exception as error:
            logger.error(
                "Analytics tool %s failed unexpectedly (%s).",
                tool_name,
                type(error).__name__,
            )
            raise ToolExecutionError(
                "internal_error",
                "The analytics request could not be completed.",
            ) from None

        try:
            encoded = jsonable_encoder(result)
            if not isinstance(encoded, dict):
                raise TypeError("Tool result must be a JSON object.")
            json.dumps(encoded, allow_nan=False)
        except (TypeError, ValueError):
            raise ToolExecutionError(
                "invalid_result",
                "The analytics result could not be safely serialized.",
            ) from None
        return encoded

    def _dispatch(
        self,
        tool_name: str,
        arguments: BaseModel,
        session: Session,
    ) -> dict[str, Any]:
        if tool_name == "get_energy_summary" and isinstance(arguments, EnergySummaryInput):
            return self._energy_summary(session, arguments)
        if tool_name == "get_energy_timeseries" and isinstance(arguments, EnergyTimeseriesInput):
            return self._energy_timeseries(session, arguments)
        if tool_name == "get_electrical_metrics" and isinstance(arguments, ElectricalMetricsInput):
            return self._electrical_metrics(session, arguments)
        if tool_name == "get_data_quality" and isinstance(arguments, DataQualityInput):
            return self._data_quality(session, arguments)
        if tool_name == "get_anomaly_summary" and isinstance(arguments, AnomalySummaryInput):
            return self._anomaly_summary(session, arguments)
        if tool_name == "get_detection_run_details" and isinstance(
            arguments, DetectionRunDetailsInput
        ):
            return self._detection_run_details(session, arguments)
        if tool_name == "compare_energy_periods" and isinstance(
            arguments, CompareEnergyPeriodsInput
        ):
            return self._compare_energy_periods(session, arguments)
        raise ToolExecutionError("unknown_tool", "Unknown analytics tool.")

    @staticmethod
    def _not_found(message: str) -> ToolExecutionError:
        return ToolExecutionError("not_found", message)

    def _energy_summary(self, session: Session, arguments: EnergySummaryInput) -> dict[str, Any]:
        result = self._analytics.summary(
            session,
            arguments.meter_identifier,
            start=arguments.start,
            end=arguments.end,
        )
        if result is None:
            raise ToolExecutionError("meter_not_found", "Meter not found.")
        return {
            "meter_identifier": arguments.meter_identifier,
            "start": arguments.start,
            "end": arguments.end,
            **result,
        }

    def _energy_timeseries(
        self,
        session: Session,
        arguments: EnergyTimeseriesInput,
    ) -> dict[str, Any]:
        result = self._analytics.time_series(
            session,
            arguments.meter_identifier,
            start=arguments.start,
            end=arguments.end,
            interval=arguments.interval,
        )
        if result is None:
            raise ToolExecutionError("meter_not_found", "Meter not found.")
        start, end, buckets = result
        return {
            "meter_identifier": arguments.meter_identifier,
            "start": start,
            "end": end,
            "interval": arguments.interval,
            "buckets": buckets,
            "returned_count": len(buckets),
        }

    def _electrical_metrics(
        self,
        session: Session,
        arguments: ElectricalMetricsInput,
    ) -> dict[str, Any]:
        result = self._analytics.electrical_metrics(
            session,
            arguments.meter_identifier,
            start=arguments.start,
            end=arguments.end,
        )
        if result is None:
            raise ToolExecutionError("meter_not_found", "Meter not found.")
        return {
            "meter_identifier": arguments.meter_identifier,
            "start": arguments.start,
            "end": arguments.end,
            **result,
        }

    def _data_quality(self, session: Session, arguments: DataQualityInput) -> dict[str, Any]:
        result = self._analytics.quality(
            session,
            arguments.meter_identifier,
            start=arguments.start,
            end=arguments.end,
        )
        if result is None:
            raise ToolExecutionError("meter_not_found", "Meter not found.")
        return {
            "meter_identifier": arguments.meter_identifier,
            "start": arguments.start,
            "end": arguments.end,
            **result,
            "ingestion_validation": {
                "available": False,
                "detail": (
                    "Ingestion-time validation findings are not stored per reading and cannot "
                    "be reconstructed from the database."
                ),
            },
        }

    def _anomaly_summary(
        self,
        session: Session,
        arguments: AnomalySummaryInput,
    ) -> dict[str, Any]:
        if arguments.run_id is not None:
            run = self._detection_runs.get_by_id(session, arguments.run_id)
            if run is None:
                raise self._not_found("Detection run not found.")
            meter_identifier = run.meter.meter_identifier
            if (
                arguments.meter_identifier is not None
                and meter_identifier != arguments.meter_identifier
            ):
                raise self._not_found("Detection run not found for this meter.")
        else:
            if arguments.meter_identifier is None:
                raise ToolExecutionError(
                    "invalid_arguments",
                    "Provide a meter identifier or a detection run ID.",
                )
            if self._meters.get_by_identifier(session, arguments.meter_identifier) is None:
                raise ToolExecutionError("meter_not_found", "Meter not found.")
            run = self._detection_runs.latest_completed(
                session,
                meter_identifier=arguments.meter_identifier,
            )
            if run is None:
                raise ToolExecutionError(
                    "no_completed_run",
                    "No completed detection run exists for this meter.",
                )
            meter_identifier = arguments.meter_identifier

        if run.status != "completed":
            raise ToolExecutionError(
                "run_not_completed",
                f"Anomaly summary is unavailable because the detection run is {run.status}.",
            )
        return self._run_summary(session, run, meter_identifier)

    def _run_summary(
        self,
        session: Session,
        run: DetectionRun,
        meter_identifier: str,
    ) -> dict[str, Any]:
        counts_by_category = self._anomalies.category_counts(session, run.id)
        return {
            "run_id": run.id,
            "meter_identifier": meter_identifier,
            "algorithm": run.algorithm,
            "run_status": run.status,
            "readings_processed": run.readings_processed,
            "anomaly_count": run.anomaly_count,
            "normal_count": run.normal_count,
            "unavailable_count": run.unavailable_count,
            "anomaly_rate": (
                run.anomaly_count / run.readings_processed if run.readings_processed else None
            ),
            "counts_by_category": counts_by_category,
            "counts_by_algorithm": {run.algorithm: run.anomaly_count},
        }

    def _detection_run_details(
        self,
        session: Session,
        arguments: DetectionRunDetailsInput,
    ) -> dict[str, Any]:
        run = self._detection_runs.get_by_id(session, arguments.run_id)
        if run is None:
            raise self._not_found("Detection run not found.")
        return {
            "run_id": run.id,
            "meter_identifier": run.meter.meter_identifier,
            "algorithm": run.algorithm,
            "status": run.status,
            "start": run.start_timestamp,
            "end": run.end_timestamp,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "training_readings": run.training_readings,
            "readings_processed": run.readings_processed,
            "anomaly_count": run.anomaly_count,
            "normal_count": run.normal_count,
            "unavailable_count": run.unavailable_count,
        }

    def _compare_energy_periods(
        self,
        session: Session,
        arguments: CompareEnergyPeriodsInput,
    ) -> dict[str, Any]:
        result = self._analytics.compare_periods(
            session,
            arguments.meter_identifier,
            period_a_start=arguments.period_a_start,
            period_a_end=arguments.period_a_end,
            period_b_start=arguments.period_b_start,
            period_b_end=arguments.period_b_end,
        )
        if result is None:
            raise ToolExecutionError("meter_not_found", "Meter not found.")
        return result
