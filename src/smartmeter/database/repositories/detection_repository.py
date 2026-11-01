from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from smartmeter.database.models import AnomalyResult, DetectionRun, Meter, MeterReading

DETECTION_ALGORITHMS = (
    "global_iqr",
    "rolling_mean_std",
    "time_of_day",
    "isolation_forest",
)
RUN_STATUSES = ("pending", "running", "completed", "failed")
PREDICTION_STATUSES = ("normal", "anomaly", "unavailable")


class DetectionRunRepository:
    """Persistence and durable queue operations for detection runs."""

    def count_active_for_meter(self, session: Session, meter_id: int) -> int:
        """Count pending and running jobs for one meter."""
        return int(
            session.scalar(
                select(func.count(DetectionRun.id)).where(
                    DetectionRun.meter_id == meter_id,
                    DetectionRun.status.in_(("pending", "running")),
                )
            )
            or 0
        )

    def create_pending(
        self,
        session: Session,
        *,
        meter_id: int,
        algorithm: str,
        start: datetime | None,
        end: datetime | None,
    ) -> DetectionRun:
        run = DetectionRun(
            meter_id=meter_id,
            algorithm=algorithm,
            status="pending",
            start_timestamp=start,
            end_timestamp=end,
            configuration={"algorithm": algorithm},
        )
        session.add(run)
        session.flush()
        return run

    def get_by_id(self, session: Session, run_id: UUID) -> DetectionRun | None:
        return session.get(DetectionRun, run_id)

    def latest_completed(
        self,
        session: Session,
        *,
        meter_identifier: str | None = None,
    ) -> DetectionRun | None:
        statement = (
            select(DetectionRun)
            .join(Meter, Meter.id == DetectionRun.meter_id)
            .where(DetectionRun.status == "completed")
            .order_by(DetectionRun.completed_at.desc(), DetectionRun.id.desc())
            .limit(1)
        )
        if meter_identifier is not None:
            statement = statement.where(Meter.meter_identifier == meter_identifier)
        return session.scalar(statement)

    def list_runs(
        self,
        session: Session,
        *,
        meter_identifier: str | None,
        status: str | None,
        algorithm: str | None,
        created_start: datetime | None,
        created_end: datetime | None,
        limit: int,
        offset: int,
    ) -> list[DetectionRun]:
        statement = (
            select(DetectionRun)
            .options(selectinload(DetectionRun.meter))
            .join(Meter, Meter.id == DetectionRun.meter_id)
        )
        if meter_identifier is not None:
            statement = statement.where(Meter.meter_identifier == meter_identifier)
        if status is not None:
            statement = statement.where(DetectionRun.status == status)
        if algorithm is not None:
            statement = statement.where(DetectionRun.algorithm == algorithm)
        if created_start is not None:
            statement = statement.where(DetectionRun.created_at >= created_start)
        if created_end is not None:
            statement = statement.where(DetectionRun.created_at < created_end)
        statement = (
            statement.order_by(DetectionRun.created_at.desc(), DetectionRun.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(session.scalars(statement).all())

    def claim_next(
        self,
        session: Session,
        *,
        lease_seconds: int,
        now: datetime | None = None,
        maximum_attempts: int = 3,
    ) -> DetectionRun | None:
        """Claim one queued or expired run, skipping rows locked by other workers."""
        current_time = now or datetime.now(timezone.utc)
        statement = (
            select(DetectionRun)
            .where(
                or_(
                    DetectionRun.status == "pending",
                    and_(
                        DetectionRun.status == "running",
                        DetectionRun.lease_expires_at <= current_time,
                        DetectionRun.attempts < maximum_attempts,
                    ),
                )
            )
            .order_by(DetectionRun.created_at, DetectionRun.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        run = session.scalar(statement)
        if run is None:
            return None
        run.status = "running"
        run.started_at = run.started_at or current_time
        run.heartbeat_at = current_time
        run.lease_expires_at = current_time + timedelta(seconds=lease_seconds)
        run.lease_token = uuid4()
        run.attempts += 1
        run.error_message = None
        session.flush()
        return run

    def heartbeat(
        self,
        session: Session,
        run_id: UUID,
        lease_token: UUID,
        *,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        current_time = now or datetime.now(timezone.utc)
        result = session.execute(
            update(DetectionRun)
            .where(
                DetectionRun.id == run_id,
                DetectionRun.status == "running",
                DetectionRun.lease_token == lease_token,
                DetectionRun.lease_expires_at > current_time,
            )
            .values(
                heartbeat_at=current_time,
                lease_expires_at=current_time + timedelta(seconds=lease_seconds),
            )
        )
        return result.rowcount == 1

    def mark_failed(
        self,
        session: Session,
        run_id: UUID,
        lease_token: UUID,
        *,
        error_message: str,
        now: datetime | None = None,
    ) -> bool:
        current_time = now or datetime.now(timezone.utc)
        result = session.execute(
            update(DetectionRun)
            .where(
                DetectionRun.id == run_id,
                DetectionRun.status == "running",
                DetectionRun.lease_token == lease_token,
                DetectionRun.lease_expires_at > current_time,
            )
            .values(
                status="failed",
                completed_at=current_time,
                lease_expires_at=None,
                lease_token=None,
                error_message=error_message[:512],
            )
        )
        return result.rowcount == 1

    def fail_exhausted(
        self,
        session: Session,
        *,
        maximum_attempts: int,
        now: datetime | None = None,
    ) -> int:
        current_time = now or datetime.now(timezone.utc)
        result = session.execute(
            update(DetectionRun)
            .where(
                DetectionRun.status == "running",
                DetectionRun.lease_expires_at <= current_time,
                DetectionRun.attempts >= maximum_attempts,
            )
            .values(
                status="failed",
                completed_at=current_time,
                lease_expires_at=None,
                lease_token=None,
                error_message="Worker lease expired after the maximum number of attempts.",
            )
        )
        return int(result.rowcount or 0)

    def mark_completed(
        self,
        session: Session,
        run_id: UUID,
        lease_token: UUID,
        *,
        results: list[dict[str, Any]],
        training_readings: int,
        readings_processed: int,
        anomaly_count: int,
        normal_count: int,
        unavailable_count: int,
        summary: dict[str, Any],
        configuration: dict[str, Any],
        now: datetime | None = None,
        batch_size: int = 2_000,
    ) -> bool:
        if readings_processed != anomaly_count + normal_count + unavailable_count:
            raise ValueError("Prediction counts must reconcile with readings_processed.")
        if not 1 <= batch_size <= 10_000:
            raise ValueError("batch_size must be between 1 and 10000.")

        current_time = now or datetime.now(timezone.utc)
        updated = session.execute(
            update(DetectionRun)
            .where(
                DetectionRun.id == run_id,
                DetectionRun.status == "running",
                DetectionRun.lease_token == lease_token,
                DetectionRun.lease_expires_at > current_time,
            )
            .values(
                status="completed",
                completed_at=current_time,
                heartbeat_at=current_time,
                lease_expires_at=None,
                lease_token=None,
                training_readings=training_readings,
                readings_processed=readings_processed,
                anomaly_count=anomaly_count,
                normal_count=normal_count,
                unavailable_count=unavailable_count,
                summary=summary,
                configuration=configuration,
                error_message=None,
            )
        )
        if updated.rowcount != 1:
            return False

        insert_statement = AnomalyResult.__table__.insert()
        for start in range(0, len(results), batch_size):
            session.execute(insert_statement, results[start : start + batch_size])
        return True


class AnomalyResultRepository:
    """Filtered, paginated anomaly-result queries and run-level aggregates."""

    def list_results(
        self,
        session: Session,
        *,
        meter_identifier: str | None,
        run_id: UUID | None,
        start: datetime | None,
        end: datetime | None,
        prediction_status: str,
        algorithm: str | None,
        category: str | None,
        limit: int,
        offset: int,
    ) -> list[tuple[AnomalyResult, DetectionRun, Meter, MeterReading]]:
        statement = (
            select(AnomalyResult, DetectionRun, Meter, MeterReading)
            .join(DetectionRun, DetectionRun.id == AnomalyResult.detection_run_id)
            .join(Meter, Meter.id == DetectionRun.meter_id)
            .join(MeterReading, MeterReading.id == AnomalyResult.reading_id)
            .where(AnomalyResult.prediction_status == prediction_status)
        )
        if meter_identifier is not None:
            statement = statement.where(Meter.meter_identifier == meter_identifier)
        if run_id is not None:
            statement = statement.where(AnomalyResult.detection_run_id == run_id)
        if start is not None:
            statement = statement.where(MeterReading.timestamp >= start)
        if end is not None:
            statement = statement.where(MeterReading.timestamp < end)
        if algorithm is not None:
            statement = statement.where(DetectionRun.algorithm == algorithm)
        if category is not None:
            statement = statement.where(AnomalyResult.anomaly_category == category)
        statement = (
            statement.order_by(MeterReading.timestamp, AnomalyResult.id).limit(limit).offset(offset)
        )
        return list(session.execute(statement).tuples().all())

    def category_counts(self, session: Session, run_id: UUID) -> dict[str, int]:
        rows = session.execute(
            select(AnomalyResult.anomaly_category, func.count(AnomalyResult.id))
            .where(
                AnomalyResult.detection_run_id == run_id,
                AnomalyResult.prediction_status == "anomaly",
            )
            .group_by(AnomalyResult.anomaly_category)
        ).all()
        return {
            category if category is not None else "unspecified": int(count)
            for category, count in rows
        }
