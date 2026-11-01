from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

from smartmeter.services.analytics_service import MAX_ANALYTICS_RANGE

MeterIdentifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=100, strip_whitespace=True),
]


class StrictToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MeterDateRangeInput(StrictToolInput):
    """A required, bounded meter query over the half-open interval [start, end)."""

    meter_identifier: MeterIdentifier
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamps must include a UTC offset.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_date_range(self) -> MeterDateRangeInput:
        if self.start >= self.end:
            raise ValueError("start must be earlier than end.")
        if self.end - self.start > MAX_ANALYTICS_RANGE:
            raise ValueError(f"Analytics range must not exceed {MAX_ANALYTICS_RANGE.days} days.")
        return self


class EnergySummaryInput(MeterDateRangeInput):
    """Arguments for a bounded energy summary."""


class EnergyTimeseriesInput(MeterDateRangeInput):
    """Arguments for an hourly or daily energy time series."""

    interval: Literal["hour", "day"]


class ElectricalMetricsInput(MeterDateRangeInput):
    """Arguments for supported measurements over a bounded range."""


class DataQualityInput(MeterDateRangeInput):
    """Arguments for database-observed data-quality statistics."""


class AnomalySummaryInput(StrictToolInput):
    """Arguments for one meter's latest completed run or a specified run."""

    meter_identifier: MeterIdentifier | None = None
    run_id: UUID | None = None

    @model_validator(mode="after")
    def require_meter_or_run(self) -> AnomalySummaryInput:
        if self.meter_identifier is None and self.run_id is None:
            raise ValueError("Provide meter_identifier, run_id, or both.")
        return self


class DetectionRunDetailsInput(StrictToolInput):
    """Arguments for retrieving one detection run's safe details."""

    run_id: UUID


class CompareEnergyPeriodsInput(StrictToolInput):
    """Arguments for comparing two non-overlapping half-open meter intervals."""

    meter_identifier: MeterIdentifier
    period_a_start: datetime
    period_a_end: datetime
    period_b_start: datetime
    period_b_end: datetime

    @field_validator(
        "period_a_start",
        "period_a_end",
        "period_b_start",
        "period_b_end",
    )
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamps must include a UTC offset.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_periods(self) -> CompareEnergyPeriodsInput:
        periods = (
            (self.period_a_start, self.period_a_end, "period A"),
            (self.period_b_start, self.period_b_end, "period B"),
        )
        for start, end, label in periods:
            if start >= end:
                raise ValueError(f"{label} start must be earlier than its end.")
            if end - start > MAX_ANALYTICS_RANGE:
                raise ValueError(f"{label} must not exceed {MAX_ANALYTICS_RANGE.days} days.")
        if not (
            self.period_a_end <= self.period_b_start or self.period_b_end <= self.period_a_start
        ):
            raise ValueError("Comparison periods must not overlap.")
        return self
