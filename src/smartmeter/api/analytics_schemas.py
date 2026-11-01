from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MeasurementStats(BaseModel):
    missing_count: int
    missing_rate: float | None = Field(ge=0, le=1)


class MeterCoverage(BaseModel):
    meter_identifier: str
    reading_count: int
    first_reading_at: datetime | None
    last_reading_at: datetime | None


class MetersPageResponse(BaseModel):
    items: list[MeterCoverage]
    limit: int
    offset: int
    returned_count: int
    has_more: bool
    next_offset: int | None


class AnalyticsSummaryResponse(BaseModel):
    meter_identifier: str
    start: datetime | None
    end: datetime | None
    reading_count: int
    active_power_average: float | None
    active_power_minimum: float | None
    active_power_maximum: float | None
    latest_active_power: float | None
    latest_active_power_at: datetime | None
    missing_measurements: dict[str, MeasurementStats]


class TimeSeriesInterval(StrEnum):
    HOUR = "hour"
    DAY = "day"


class TimeSeriesBucket(BaseModel):
    timestamp: datetime
    reading_count: int
    global_active_power_average: float | None
    global_active_power_minimum: float | None
    global_active_power_maximum: float | None
    global_reactive_power_average: float | None
    voltage_average: float | None
    global_intensity_average: float | None
    sub_metering_1_average: float | None
    sub_metering_2_average: float | None
    sub_metering_3_average: float | None


class TimeSeriesResponse(BaseModel):
    meter_identifier: str
    start: datetime
    end: datetime
    interval: TimeSeriesInterval
    buckets: list[TimeSeriesBucket]
    returned_count: int


class TimestampGapSummary(BaseModel):
    expected_interval_minutes: int
    observed_gap_count: int
    estimated_missing_intervals: int
    first_observed_at: datetime | None
    last_observed_at: datetime | None
    boundary_gaps_assessed: bool


class IngestionValidationAvailability(BaseModel):
    available: bool
    detail: str


class DataQualityResponse(BaseModel):
    meter_identifier: str
    start: datetime | None
    end: datetime | None
    reading_count: int
    missing_measurements: dict[str, MeasurementStats]
    timestamp_gaps: TimestampGapSummary
    database_metrics_scope: str
    ingestion_validation: IngestionValidationAvailability
