from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from smartmeter.api.analytics_schemas import (
    AnalyticsSummaryResponse,
    DataQualityResponse,
    MetersPageResponse,
    TimeSeriesInterval,
    TimeSeriesResponse,
)
from smartmeter.api.dependencies import get_db_session
from smartmeter.api.limits import MAX_PAGE_OFFSET
from smartmeter.services.analytics_service import AnalyticsService

router = APIRouter()


def _page_metadata(items: list[dict[str, Any]], limit: int, offset: int) -> dict[str, Any]:
    has_more = len(items) > limit
    returned_count = min(len(items), limit)
    return {
        "limit": limit,
        "offset": offset,
        "returned_count": returned_count,
        "has_more": has_more,
        "next_offset": offset + returned_count if has_more else None,
    }


def _invalid_query(error: ValueError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(error))


@router.get("/meters", response_model=MetersPageResponse, tags=["meters"])
def list_meters(
    session: Annotated[Session, Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=1_000)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_PAGE_OFFSET)] = 0,
) -> MetersPageResponse:
    meters = AnalyticsService().list_meters(session, limit=limit, offset=offset)
    page = meters[:limit]
    return MetersPageResponse(
        items=page,
        **_page_metadata(meters, limit, offset),
    )


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse, tags=["analytics"])
def get_analytics_summary(
    meter_identifier: Annotated[str, Query(min_length=1, max_length=100)],
    session: Annotated[Session, Depends(get_db_session)],
    start: Annotated[
        datetime | None, Query(description="Inclusive ISO-8601 timestamp with UTC offset")
    ] = None,
    end: Annotated[
        datetime | None, Query(description="Exclusive ISO-8601 timestamp with UTC offset")
    ] = None,
) -> AnalyticsSummaryResponse:
    service = AnalyticsService()
    try:
        result = service.summary(
            session,
            meter_identifier,
            start=start,
            end=end,
        )
    except ValueError as error:
        raise _invalid_query(error) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Meter not found.")
    return AnalyticsSummaryResponse(
        meter_identifier=meter_identifier,
        start=service.normalize_range(start, end)[0],
        end=service.normalize_range(start, end)[1],
        **result,
    )


@router.get("/analytics/timeseries", response_model=TimeSeriesResponse, tags=["analytics"])
def get_analytics_time_series(
    meter_identifier: Annotated[str, Query(min_length=1, max_length=100)],
    start: Annotated[datetime, Query(description="Inclusive ISO-8601 timestamp with UTC offset")],
    end: Annotated[datetime, Query(description="Exclusive ISO-8601 timestamp with UTC offset")],
    session: Annotated[Session, Depends(get_db_session)],
    interval: TimeSeriesInterval = TimeSeriesInterval.HOUR,
) -> TimeSeriesResponse:
    try:
        result = AnalyticsService().time_series(
            session,
            meter_identifier,
            start=start,
            end=end,
            interval=interval.value,
        )
    except ValueError as error:
        raise _invalid_query(error) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Meter not found.")
    start_utc, end_utc, buckets = result
    return TimeSeriesResponse(
        meter_identifier=meter_identifier,
        start=start_utc,
        end=end_utc,
        interval=interval,
        buckets=buckets,
        returned_count=len(buckets),
    )


@router.get("/analytics/quality", response_model=DataQualityResponse, tags=["analytics"])
def get_data_quality(
    meter_identifier: Annotated[str, Query(min_length=1, max_length=100)],
    session: Annotated[Session, Depends(get_db_session)],
    start: Annotated[
        datetime | None, Query(description="Inclusive ISO-8601 timestamp with UTC offset")
    ] = None,
    end: Annotated[
        datetime | None, Query(description="Exclusive ISO-8601 timestamp with UTC offset")
    ] = None,
) -> DataQualityResponse:
    service = AnalyticsService()
    try:
        result = service.quality(
            session,
            meter_identifier,
            start=start,
            end=end,
        )
        normalized = service.normalize_range(start, end)
    except ValueError as error:
        raise _invalid_query(error) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Meter not found.")
    return DataQualityResponse(
        meter_identifier=meter_identifier,
        start=normalized[0],
        end=normalized[1],
        ingestion_validation={
            "available": False,
            "detail": (
                "The ingestion-time validation report is not stored per reading and cannot be "
                "reconstructed from the database."
            ),
        },
        **result,
    )
