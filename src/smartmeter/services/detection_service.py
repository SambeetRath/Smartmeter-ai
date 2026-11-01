from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pandas as pd

from smartmeter.anomaly_detection.pipeline import (
    PipelineConfig,
    PipelineRunResult,
    load_postgresql_dataset,
    run_anomaly_pipeline,
)
from smartmeter.database.repositories import DetectionRunRepository
from smartmeter.database.session import SessionFactory, create_session_factory
from smartmeter.services.detection_limits import MAX_DETECTION_READINGS

logger = logging.getLogger(__name__)


class LostDetectionLeaseError(RuntimeError):
    """Raised when a worker no longer owns the run it was processing."""


def _utc_timestamp_key(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Detection pipeline returned a timestamp without a UTC offset.")
    return timestamp.tz_convert("UTC")


def process_detection_run(
    run_id: UUID,
    lease_token: UUID,
    *,
    session_factory: SessionFactory | None = None,
    pipeline_runner: Callable[..., PipelineRunResult] | None = None,
) -> None:
    """Run one claimed detection job and atomically persist all held-out predictions."""
    factory = session_factory or create_session_factory()
    runner = pipeline_runner or run_anomaly_pipeline
    with factory() as session:
        run = DetectionRunRepository().get_by_id(session, run_id)
        if run is None or run.status != "running" or run.lease_token != lease_token:
            raise LostDetectionLeaseError(f"Worker no longer owns detection run {run_id}.")
        meter_identifier = run.meter.meter_identifier
        algorithm = run.algorithm
        start = run.start_timestamp
        end = run.end_timestamp

    frame = load_postgresql_dataset(
        meter_identifier,
        session_factory=factory,
        start=start,
        end=end,
        include_reading_ids=True,
        max_rows=MAX_DETECTION_READINGS,
    )
    reading_ids = {
        _utc_timestamp_key(timestamp): int(reading_id)
        for reading_id, timestamp in zip(frame["id"], frame["timestamp"], strict=True)
    }

    result = runner(
        frame.drop(columns=["id"]),
        config=PipelineConfig(
            source="postgresql",
            meter_identifier=meter_identifier,
            detector_names=(algorithm,),
            include_normal_predictions=True,
        ),
        meter_identifier=meter_identifier,
    )
    detector_summary = result.summary["detectors"][algorithm]
    readings_processed = int(detector_summary["evaluated_observations"])
    anomaly_count = int(detector_summary["anomaly_predictions"])
    normal_count = int(detector_summary["normal_predictions_omitted_from_csv"])
    unavailable_count = int(detector_summary["unavailable_predictions"])
    if readings_processed != anomaly_count + normal_count + unavailable_count:
        raise ValueError("Detector output counts do not reconcile with evaluated readings.")
    if len(result.results) != readings_processed:
        raise ValueError("Complete prediction output does not match the evaluated reading count.")

    rows: list[dict[str, Any]] = []
    for prediction in result.results.itertuples(index=False):
        prediction_timestamp = _utc_timestamp_key(prediction.timestamp)
        try:
            reading_id = reading_ids[prediction_timestamp]
        except KeyError as exc:
            raise ValueError(
                "Detector output contains a timestamp that was not loaded from PostgreSQL."
            ) from exc
        if prediction.result_kind not in ("normal", "anomaly", "unavailable"):
            raise ValueError(f"Unsupported detector result status: {prediction.result_kind}")
        score = prediction.anomaly_score
        rows.append(
            {
                "detection_run_id": run_id,
                "reading_id": reading_id,
                "prediction_status": prediction.result_kind,
                "anomaly_score": (
                    float(score) if score is not None and math.isfinite(float(score)) else None
                ),
                "anomaly_category": prediction.anomaly_category,
                "metadata_json": json.loads(prediction.detector_metadata),
            }
        )

    summary = {
        **result.summary,
        "run_id": str(run_id),
        "readings_processed_definition": (
            "Number of chronological held-out readings receiving a prediction; training "
            "readings are reported separately."
        ),
        "detector_counts": detector_summary,
    }
    configuration = {
        **result.summary["configuration"],
        "requested_start": start.isoformat() if start else None,
        "requested_end": end.isoformat() if end else None,
        "prediction_persistence": "normal, anomaly, and unavailable held-out predictions",
    }

    with factory.begin() as session:
        completed = DetectionRunRepository().mark_completed(
            session,
            run_id,
            lease_token,
            results=rows,
            training_readings=int(result.summary["training_rows"]),
            readings_processed=readings_processed,
            anomaly_count=anomaly_count,
            normal_count=normal_count,
            unavailable_count=unavailable_count,
            summary=summary,
            configuration=configuration,
        )
        if not completed:
            raise LostDetectionLeaseError(f"Worker no longer owns detection run {run_id}.")
    logger.info(
        "Detection run %s completed: algorithm=%s, processed=%d, anomalies=%d, "
        "normal=%d, unavailable=%d.",
        run_id,
        algorithm,
        readings_processed,
        anomaly_count,
        normal_count,
        unavailable_count,
    )
