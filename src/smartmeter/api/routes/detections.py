from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from smartmeter.api.dependencies import get_db_session
from smartmeter.api.detection_schemas import (
    AnomaliesPageResponse,
    AnomalyResponse,
    AnomalySummaryResponse,
    DetectionAlgorithm,
    DetectionRunAcceptedResponse,
    DetectionRunCreateRequest,
    DetectionRunResponse,
    DetectionRunsPageResponse,
    DetectionRunStatus,
    PredictionStatus,
)
from smartmeter.api.limits import MAX_PAGE_OFFSET
from smartmeter.database.models import DetectionRun
from smartmeter.database.repositories import (
    AnomalyResultRepository,
    DetectionRunRepository,
    MeterRepository,
    ReadingRepository,
)
from smartmeter.services.detection_limits import (
    MAX_ACTIVE_DETECTION_RUNS_PER_METER,
    MAX_DETECTION_READINGS,
)

router = APIRouter(tags=["detection"])
DbSession = Annotated[Session, Depends(get_db_session)]


def _utc_range(
    start: datetime | None,
    end: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    for value, name in ((start, "start"), (end, "end")):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise HTTPException(status_code=422, detail=f"{name} must include a UTC offset.")
    start_utc = start.astimezone(timezone.utc) if start is not None else None
    end_utc = end.astimezone(timezone.utc) if end is not None else None
    if start_utc is not None and end_utc is not None and start_utc >= end_utc:
        raise HTTPException(status_code=422, detail="start must be earlier than end.")
    return start_utc, end_utc


def _run_response(run: DetectionRun, meter_identifier: str) -> DetectionRunResponse:
    return DetectionRunResponse(
        run_id=run.id,
        meter_identifier=meter_identifier,
        algorithm=run.algorithm,
        status=run.status,
        start=run.start_timestamp,
        end=run.end_timestamp,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        attempts=run.attempts,
        training_readings=run.training_readings,
        readings_processed=run.readings_processed,
        anomaly_count=run.anomaly_count,
        normal_count=run.normal_count,
        unavailable_count=run.unavailable_count,
        summary=run.summary,
        configuration=run.configuration,
        error_message=run.error_message,
    )


def _page_metadata(items: list, limit: int, offset: int) -> dict[str, int | bool | None]:
    has_more = len(items) > limit
    returned_count = min(len(items), limit)
    return {
        "limit": limit,
        "offset": offset,
        "returned_count": returned_count,
        "has_more": has_more,
        "next_offset": offset + returned_count if has_more else None,
    }


@router.post(
    "/detection/runs",
    response_model=DetectionRunAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue an anomaly detection run",
    description=(
        "Create a durable pending run. A separate detection worker must be running to process "
        "the job. Optional timestamps define the half-open UTC interval [start, end)."
    ),
)
def create_detection_run(
    request: DetectionRunCreateRequest,
    session: DbSession,
) -> DetectionRunAcceptedResponse:
    with session.begin():
        meter = MeterRepository().lock_by_identifier(session, request.meter_identifier)
        if meter is None:
            raise HTTPException(status_code=404, detail="Meter not found.")
        active_runs = DetectionRunRepository().count_active_for_meter(session, meter.id)
        if active_runs >= MAX_ACTIVE_DETECTION_RUNS_PER_METER:
            raise HTTPException(
                status_code=409,
                detail="A detection run is already pending or running for this meter.",
            )
        count = ReadingRepository().count_for_meter(
            session,
            meter.id,
            start=request.start,
            end=request.end,
        )
        if count < 3:
            raise HTTPException(
                status_code=422,
                detail="At least three readings are required in the requested range.",
            )
        if count > MAX_DETECTION_READINGS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Detection range exceeds the maximum of {MAX_DETECTION_READINGS:,} readings."
                ),
            )
        run = DetectionRunRepository().create_pending(
            session,
            meter_id=meter.id,
            algorithm=request.algorithm.value,
            start=request.start,
            end=request.end,
        )
        response = DetectionRunAcceptedResponse(
            run_id=run.id,
            status=DetectionRunStatus.PENDING,
            created_at=run.created_at,
            message="Detection run queued. A detection worker will process it.",
        )
    return response


@router.get(
    "/detection/runs",
    response_model=DetectionRunsPageResponse,
    summary="List detection runs",
    description="List runs ordered by creation time descending; date filters apply to created_at.",
)
def list_detection_runs(
    session: DbSession,
    meter_identifier: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    run_status: Annotated[DetectionRunStatus | None, Query(alias="status")] = None,
    algorithm: DetectionAlgorithm | None = None,
    created_start: Annotated[
        datetime | None, Query(description="Inclusive created_at timestamp with UTC offset")
    ] = None,
    created_end: Annotated[
        datetime | None, Query(description="Exclusive created_at timestamp with UTC offset")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=1_000)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_PAGE_OFFSET)] = 0,
) -> DetectionRunsPageResponse:
    start_utc, end_utc = _utc_range(created_start, created_end)
    if meter_identifier is not None:
        meter = MeterRepository().get_by_identifier(session, meter_identifier)
        if meter is None:
            raise HTTPException(status_code=404, detail="Meter not found.")
    runs = DetectionRunRepository().list_runs(
        session,
        meter_identifier=meter_identifier,
        status=run_status.value if run_status else None,
        algorithm=algorithm.value if algorithm else None,
        created_start=start_utc,
        created_end=end_utc,
        limit=limit + 1,
        offset=offset,
    )
    page = runs[:limit]
    return DetectionRunsPageResponse(
        items=[_run_response(run, run.meter.meter_identifier) for run in page],
        **_page_metadata(runs, limit, offset),
    )


@router.get(
    "/detection/runs/{run_id}",
    response_model=DetectionRunResponse,
    summary="Get detection run status and results summary",
)
def get_detection_run(run_id: UUID, session: DbSession) -> DetectionRunResponse:
    run = DetectionRunRepository().get_by_id(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Detection run not found.")
    return _run_response(run, run.meter.meter_identifier)


@router.get(
    "/anomalies",
    response_model=AnomaliesPageResponse,
    summary="List persisted detection predictions",
    description=(
        "Defaults to prediction_status=anomaly. Unavailable predictions are excluded unless "
        "explicitly requested."
    ),
)
def list_anomalies(
    session: DbSession,
    meter_identifier: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    run_id: UUID | None = None,
    start: Annotated[
        datetime | None, Query(description="Inclusive reading timestamp with UTC offset")
    ] = None,
    end: Annotated[
        datetime | None, Query(description="Exclusive reading timestamp with UTC offset")
    ] = None,
    prediction_status: PredictionStatus = PredictionStatus.ANOMALY,
    algorithm: DetectionAlgorithm | None = None,
    category: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    limit: Annotated[int, Query(ge=1, le=1_000)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_PAGE_OFFSET)] = 0,
) -> AnomaliesPageResponse:
    start_utc, end_utc = _utc_range(start, end)
    if (
        meter_identifier is not None
        and MeterRepository().get_by_identifier(session, meter_identifier) is None
    ):
        raise HTTPException(status_code=404, detail="Meter not found.")
    if run_id is not None and DetectionRunRepository().get_by_id(session, run_id) is None:
        raise HTTPException(status_code=404, detail="Detection run not found.")
    results = AnomalyResultRepository().list_results(
        session,
        meter_identifier=meter_identifier,
        run_id=run_id,
        start=start_utc,
        end=end_utc,
        prediction_status=prediction_status.value,
        algorithm=algorithm.value if algorithm else None,
        category=category,
        limit=limit + 1,
        offset=offset,
    )
    page = results[:limit]
    items = [
        AnomalyResponse(
            run_id=run.id,
            reading_id=reading.id,
            meter_identifier=meter.meter_identifier,
            timestamp=reading.timestamp,
            prediction_status=result.prediction_status,
            algorithm=run.algorithm,
            anomaly_score=result.anomaly_score,
            anomaly_category=result.anomaly_category,
            metadata=result.metadata_json,
        )
        for result, run, meter, reading in page
    ]
    return AnomaliesPageResponse(
        items=items,
        prediction_status=prediction_status,
        **_page_metadata(results, limit, offset),
    )


@router.get(
    "/anomalies/summary",
    response_model=AnomalySummaryResponse,
    summary="Summarize one completed detection run",
    description=(
        "When run_id is omitted, summarizes the latest completed run, optionally for a meter. "
        "Metrics are scoped to one run to avoid double-counting readings across repeated runs."
    ),
)
def get_anomaly_summary(
    session: DbSession,
    run_id: UUID | None = None,
    meter_identifier: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
) -> AnomalySummaryResponse:
    run_repository = DetectionRunRepository()
    if run_id is not None:
        run = run_repository.get_by_id(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Detection run not found.")
        if meter_identifier is not None and run.meter.meter_identifier != meter_identifier:
            raise HTTPException(status_code=404, detail="Detection run not found for this meter.")
    else:
        if (
            meter_identifier is not None
            and MeterRepository().get_by_identifier(session, meter_identifier) is None
        ):
            raise HTTPException(status_code=404, detail="Meter not found.")
        run = run_repository.latest_completed(session, meter_identifier=meter_identifier)
        if run is None:
            raise HTTPException(status_code=404, detail="No completed detection run found.")
    if run.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Anomaly summary is available only after the run has completed.",
        )

    return AnomalySummaryResponse(
        run_id=run.id,
        meter_identifier=run.meter.meter_identifier,
        algorithm=run.algorithm,
        run_status=run.status,
        readings_processed=run.readings_processed,
        anomaly_count=run.anomaly_count,
        normal_count=run.normal_count,
        unavailable_count=run.unavailable_count,
        anomaly_rate=(
            run.anomaly_count / run.readings_processed if run.readings_processed else None
        ),
        counts_by_category=AnomalyResultRepository().category_counts(session, run.id),
        counts_by_algorithm={run.algorithm: run.anomaly_count},
    )
