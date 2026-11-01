from __future__ import annotations

from datetime import datetime
from typing import Any, TypeVar
from uuid import UUID

import httpx
from pydantic import BaseModel, Field, ValidationError

from smartmeter_dashboard.config import DashboardConfig
from smartmeter_dashboard.exceptions import (
    APIConnectionError,
    APIHTTPError,
    APIRequestTimeout,
    APIResponseError,
)

T = TypeVar("T", bound=BaseModel)

CONNECT_TIMEOUT_SECONDS = 3.0
READ_TIMEOUT_SECONDS = 10.0
WRITE_TIMEOUT_SECONDS = 5.0
POOL_TIMEOUT_SECONDS = 3.0
DEFAULT_PAGE_SIZE = 1_000
MAX_METER_RECORDS = 101_000
MAX_PAGE_OFFSET = 100_000


class HealthResponse(BaseModel):
    status: str
    service: str


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


class MeasurementStats(BaseModel):
    missing_count: int
    missing_rate: float | None = Field(ge=0, le=1)


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
    interval: str
    buckets: list[TimeSeriesBucket]
    returned_count: int


class TimestampGapSummary(BaseModel):
    expected_interval_minutes: int
    observed_gap_count: int
    estimated_missing_intervals: int
    first_observed_at: datetime | None
    last_observed_at: datetime | None
    boundary_gaps_assessed: bool


class DataQualityResponse(BaseModel):
    meter_identifier: str
    start: datetime | None
    end: datetime | None
    reading_count: int
    missing_measurements: dict[str, MeasurementStats]
    timestamp_gaps: TimestampGapSummary
    database_metrics_scope: str
    ingestion_validation: dict[str, Any]


class DetectionRunResponse(BaseModel):
    run_id: UUID
    meter_identifier: str
    algorithm: str
    status: str
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


class DetectionRunAcceptedResponse(BaseModel):
    run_id: UUID
    status: str
    created_at: datetime
    message: str


class AnomalyResponse(BaseModel):
    run_id: UUID
    reading_id: int
    meter_identifier: str
    timestamp: datetime
    prediction_status: str
    algorithm: str
    anomaly_score: float | None
    anomaly_category: str | None
    metadata: dict[str, Any] | None


class AnomaliesPageResponse(BaseModel):
    items: list[AnomalyResponse]
    prediction_status: str
    limit: int
    offset: int
    returned_count: int
    has_more: bool
    next_offset: int | None


class AnomalySummaryResponse(BaseModel):
    run_id: UUID
    meter_identifier: str
    algorithm: str
    run_status: str
    readings_processed: int
    anomaly_count: int
    normal_count: int
    unavailable_count: int
    anomaly_rate: float | None
    counts_by_category: dict[str, int]
    counts_by_algorithm: dict[str, int]


class SmartMeterAPIClient:
    """Typed HTTP client for the SmartMeter AI API; it never accesses the database."""

    def __init__(
        self,
        config: DashboardConfig | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config or DashboardConfig.from_env()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self._config.api_base_url,
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SECONDS,
                read=READ_TIMEOUT_SECONDS,
                write=WRITE_TIMEOUT_SECONDS,
                pool=POOL_TIMEOUT_SECONDS,
            ),
            follow_redirects=False,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SmartMeterAPIClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _timestamp(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    def _request_model(self, endpoint: str, model: type[T], **kwargs: Any) -> T:
        try:
            response = self._client.get(endpoint, **kwargs)
        except httpx.TimeoutException as error:
            raise APIRequestTimeout(
                f"Request to {endpoint} exceeded its configured timeout."
            ) from error
        except httpx.RequestError as error:
            raise APIConnectionError(
                f"Could not connect to the API endpoint {endpoint}."
            ) from error
        if not response.is_success:
            detail = f"HTTP {response.status_code}"
            try:
                body = response.json()
                if isinstance(body, dict) and isinstance(body.get("detail"), str):
                    detail = body["detail"][:300]
            except ValueError:
                pass
            raise APIHTTPError(response.status_code, endpoint, detail)
        try:
            payload = response.json()
        except ValueError as error:
            raise APIResponseError(f"API endpoint {endpoint} returned invalid JSON.") from error
        if not isinstance(payload, dict):
            raise APIResponseError(
                f"API endpoint {endpoint} returned an unexpected response shape."
            )
        try:
            return model.model_validate(payload)
        except ValidationError as error:
            raise APIResponseError(
                f"API endpoint {endpoint} returned data that does not match its contract."
            ) from error

    def health(self) -> HealthResponse:
        return self._request_model("/health", HealthResponse)

    def list_meters(self, *, limit: int = DEFAULT_PAGE_SIZE) -> list[MeterCoverage]:
        if not 1 <= limit <= DEFAULT_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {DEFAULT_PAGE_SIZE}.")
        items: list[MeterCoverage] = []
        offset = 0
        while True:
            page = self._request_model(
                "/meters",
                MetersPageResponse,
                params={"limit": limit, "offset": offset},
            )
            items.extend(page.items)
            if len(items) > MAX_METER_RECORDS:
                raise APIResponseError(
                    f"Meter listing exceeded the client safety cap of {MAX_METER_RECORDS}."
                )
            if not page.has_more:
                return items
            if page.next_offset is None or page.next_offset <= offset:
                raise APIResponseError("Meter pagination returned an invalid continuation offset.")
            if page.next_offset > MAX_PAGE_OFFSET:
                raise APIResponseError(
                    f"Meter pagination exceeded the API offset cap of {MAX_PAGE_OFFSET}."
                )
            offset = page.next_offset

    def analytics_summary(
        self,
        meter_identifier: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> AnalyticsSummaryResponse:
        return self._request_model(
            "/analytics/summary",
            AnalyticsSummaryResponse,
            params={
                "meter_identifier": meter_identifier,
                "start": self._timestamp(start),
                "end": self._timestamp(end),
            },
        )

    def analytics_timeseries(
        self,
        meter_identifier: str,
        *,
        start: datetime,
        end: datetime,
        interval: str = "hour",
    ) -> TimeSeriesResponse:
        return self._request_model(
            "/analytics/timeseries",
            TimeSeriesResponse,
            params={
                "meter_identifier": meter_identifier,
                "start": self._timestamp(start),
                "end": self._timestamp(end),
                "interval": interval,
            },
        )

    def analytics_quality(
        self,
        meter_identifier: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> DataQualityResponse:
        return self._request_model(
            "/analytics/quality",
            DataQualityResponse,
            params={
                "meter_identifier": meter_identifier,
                "start": self._timestamp(start),
                "end": self._timestamp(end),
            },
        )

    def list_detection_runs(self, **filters: Any) -> DetectionRunsPageResponse:
        return self._request_model(
            "/detection/runs",
            DetectionRunsPageResponse,
            params={
                key: self._timestamp(value) if isinstance(value, datetime) else value
                for key, value in filters.items()
                if value is not None
            },
        )

    def get_detection_run(self, run_id: UUID | str) -> DetectionRunResponse:
        return self._request_model(f"/detection/runs/{run_id}", DetectionRunResponse)

    def create_detection_run(
        self,
        meter_identifier: str,
        *,
        algorithm: str = "global_iqr",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> DetectionRunAcceptedResponse:
        try:
            response = self._client.post(
                "/detection/runs",
                json={
                    "meter_identifier": meter_identifier,
                    "algorithm": algorithm,
                    "start": self._timestamp(start),
                    "end": self._timestamp(end),
                },
            )
        except httpx.TimeoutException as error:
            raise APIRequestTimeout("Detection-run submission timed out.") from error
        except httpx.RequestError as error:
            raise APIConnectionError("Could not submit a detection run to the API.") from error
        if not response.is_success:
            detail = f"HTTP {response.status_code}"
            try:
                body = response.json()
                if isinstance(body, dict) and isinstance(body.get("detail"), str):
                    detail = body["detail"][:300]
            except ValueError:
                pass
            raise APIHTTPError(response.status_code, "/detection/runs", detail)
        try:
            return DetectionRunAcceptedResponse.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise APIResponseError(
                "API returned an invalid detection-run submission response."
            ) from error

    def list_anomalies(self, **filters: Any) -> AnomaliesPageResponse:
        params = {
            key: self._timestamp(value)
            if isinstance(value, datetime)
            else str(value)
            if isinstance(value, UUID)
            else value
            for key, value in filters.items()
            if value is not None
        }
        return self._request_model("/anomalies", AnomaliesPageResponse, params=params)

    def anomaly_summary(
        self,
        *,
        run_id: UUID | str | None = None,
        meter_identifier: str | None = None,
    ) -> AnomalySummaryResponse:
        return self._request_model(
            "/anomalies/summary",
            AnomalySummaryResponse,
            params={
                key: value
                for key, value in {
                    "run_id": str(run_id) if run_id is not None else None,
                    "meter_identifier": meter_identifier,
                }.items()
                if value is not None
            },
        )
