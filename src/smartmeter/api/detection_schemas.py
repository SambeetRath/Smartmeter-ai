from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from smartmeter.services.detection_limits import MAX_DETECTION_RANGE


class DetectionAlgorithm(StrEnum):
    GLOBAL_IQR = "global_iqr"
    ROLLING_MEAN_STD = "rolling_mean_std"
    TIME_OF_DAY = "time_of_day"
    ISOLATION_FOREST = "isolation_forest"


class DetectionRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictionStatus(StrEnum):
    NORMAL = "normal"
    ANOMALY = "anomaly"
    UNAVAILABLE = "unavailable"


class DetectionRunCreateRequest(BaseModel):
    """Request one durable, chronological detector run."""

    meter_identifier: Annotated[
        str,
        StringConstraints(min_length=1, max_length=100, strip_whitespace=True),
    ]
    algorithm: DetectionAlgorithm = Field(
        default=DetectionAlgorithm.GLOBAL_IQR,
        description="One supported detector to run against the chronological held-out split.",
    )
    start: datetime | None = Field(
        default=None,
        description="Inclusive ISO-8601 timestamp with UTC offset.",
    )
    end: datetime | None = Field(
        default=None,
        description="Exclusive ISO-8601 timestamp with UTC offset.",
    )

    @model_validator(mode="after")
    def normalize_range(self) -> DetectionRunCreateRequest:
        for field_name in ("start", "end"):
            value = getattr(self, field_name)
            if value is not None:
                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(f"{field_name} must include a UTC offset.")
                setattr(self, field_name, value.astimezone(timezone.utc))
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("start must be earlier than end.")
        if (
            self.start is not None
            and self.end is not None
            and self.end - self.start > MAX_DETECTION_RANGE
        ):
            raise ValueError(f"Detection range must not exceed {MAX_DETECTION_RANGE.days} days.")
        return self


class DetectionRunAcceptedResponse(BaseModel):
    run_id: UUID
    status: DetectionRunStatus
    created_at: datetime
    message: str


class DetectionRunResponse(BaseModel):
    """Run metadata, persisted prediction counts, and safe failure details."""

    model_config = ConfigDict(from_attributes=True)

    run_id: UUID
    meter_identifier: str
    algorithm: DetectionAlgorithm
    status: DetectionRunStatus
    start: datetime | None
    end: datetime | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    attempts: int
    training_readings: int
    readings_processed: int
    anomaly_count: int
    normal_count: int
    unavailable_count: int
    summary: dict[str, Any] | None
    configuration: dict[str, Any] | None
    error_message: str | None


class DetectionRunsPageResponse(BaseModel):
    items: list[DetectionRunResponse]
    limit: int
    offset: int
    returned_count: int
    has_more: bool
    next_offset: int | None


class AnomalyResponse(BaseModel):
    run_id: UUID
    reading_id: int
    meter_identifier: str
    timestamp: datetime
    prediction_status: PredictionStatus
    algorithm: DetectionAlgorithm
    anomaly_score: float | None
    anomaly_category: str | None
    metadata: dict[str, Any] | None


class AnomaliesPageResponse(BaseModel):
    items: list[AnomalyResponse]
    prediction_status: PredictionStatus
    limit: int
    offset: int
    returned_count: int
    has_more: bool
    next_offset: int | None


class AnomalySummaryResponse(BaseModel):
    run_id: UUID
    meter_identifier: str
    algorithm: DetectionAlgorithm
    run_status: DetectionRunStatus
    readings_processed: int
    anomaly_count: int
    normal_count: int
    unavailable_count: int
    anomaly_rate: float | None
    counts_by_category: dict[str, int]
    counts_by_algorithm: dict[str, int]
