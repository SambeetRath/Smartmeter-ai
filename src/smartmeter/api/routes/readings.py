from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from smartmeter.api.dependencies import get_db_session
from smartmeter.api.limits import MAX_PAGE_OFFSET
from smartmeter.api.schemas import ReadingsPageResponse
from smartmeter.database.repositories import MeterRepository, ReadingRepository

router = APIRouter()


def _utc_boundary(value: datetime | None, name: str) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(
            status_code=422,
            detail=f"{name} must include a UTC offset (ISO-8601).",
        )
    return value.astimezone(timezone.utc)


@router.get(
    "/readings",
    response_model=ReadingsPageResponse,
    tags=["readings"],
)
def get_readings(
    meter_identifier: Annotated[str, Query(min_length=1, max_length=100)],
    session: Annotated[Session, Depends(get_db_session)],
    start: Annotated[
        datetime | None, Query(description="Inclusive ISO-8601 timestamp with offset")
    ] = None,
    end: Annotated[
        datetime | None, Query(description="Exclusive ISO-8601 timestamp with offset")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=1_000)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_PAGE_OFFSET)] = 0,
) -> ReadingsPageResponse:
    """Return one bounded, timestamp-ordered page of readings for a meter.

    Timestamp boundaries must include an ISO-8601 UTC offset. Values are
    normalized to UTC, and the optional range follows [start, end) semantics.
    """
    start_utc = _utc_boundary(start, "start")
    end_utc = _utc_boundary(end, "end")
    if start_utc is not None and end_utc is not None and start_utc >= end_utc:
        raise HTTPException(status_code=422, detail="start must be earlier than end.")

    meter = MeterRepository().get_by_identifier(session, meter_identifier)
    if meter is None:
        raise HTTPException(status_code=404, detail="Meter not found.")

    readings = ReadingRepository().get_for_meter(
        session,
        meter.id,
        limit=limit + 1,
        offset=offset,
        start=start_utc,
        end=end_utc,
    )
    has_more = len(readings) > limit
    page = readings[:limit]
    return ReadingsPageResponse(
        meter_identifier=meter.meter_identifier,
        items=page,
        limit=limit,
        offset=offset,
        returned_count=len(page),
        has_more=has_more,
        next_offset=offset + len(page) if has_more else None,
    )
