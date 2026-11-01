from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from smartmeter.database.base import Base

READING_NUMERIC = Numeric(12, 6)
JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


class Meter(Base):
    """A distinct physical or logical smart meter."""

    __tablename__ = "meters"

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    meter_identifier: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    readings: Mapped[list[MeterReading]] = relationship(
        back_populates="meter",
        passive_deletes="all",
    )


class MeterReading(Base):
    """One timestamped reading from a meter; NULL measurements are retained."""

    __tablename__ = "meter_readings"
    __table_args__ = (
        UniqueConstraint(
            "meter_id",
            "timestamp",
            name="uq_meter_readings_meter_timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    meter_id: Mapped[int] = mapped_column(
        ForeignKey("meters.id", ondelete="RESTRICT"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    global_active_power: Mapped[Decimal | None] = mapped_column(READING_NUMERIC, nullable=True)
    global_reactive_power: Mapped[Decimal | None] = mapped_column(READING_NUMERIC, nullable=True)
    voltage: Mapped[Decimal | None] = mapped_column(READING_NUMERIC, nullable=True)
    global_intensity: Mapped[Decimal | None] = mapped_column(READING_NUMERIC, nullable=True)
    sub_metering_1: Mapped[Decimal | None] = mapped_column(READING_NUMERIC, nullable=True)
    sub_metering_2: Mapped[Decimal | None] = mapped_column(READING_NUMERIC, nullable=True)
    sub_metering_3: Mapped[Decimal | None] = mapped_column(READING_NUMERIC, nullable=True)

    meter: Mapped[Meter] = relationship(back_populates="readings")


class DetectionRun(Base):
    """A durable request to run one detector for one meter."""

    __tablename__ = "detection_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="valid_status",
        ),
        CheckConstraint(
            "algorithm IN ('global_iqr', 'rolling_mean_std', 'time_of_day', 'isolation_forest')",
            name="valid_algorithm",
        ),
        CheckConstraint(
            "readings_processed = anomaly_count + normal_count + unavailable_count",
            name="prediction_counts_reconcile",
        ),
        CheckConstraint(
            "start_timestamp IS NULL OR end_timestamp IS NULL OR start_timestamp < end_timestamp",
            name="valid_requested_range",
        ),
        Index("ix_detection_runs_status_lease_expires_at", "status", "lease_expires_at"),
        Index("ix_detection_runs_meter_id_created_at", "meter_id", "created_at"),
        Index("ix_detection_runs_algorithm_created_at", "algorithm", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    meter_id: Mapped[int] = mapped_column(
        ForeignKey("meters.id", ondelete="RESTRICT"),
        nullable=False,
    )
    algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    start_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[UUID | None] = mapped_column()
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    training_readings: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    readings_processed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    anomaly_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    normal_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    unavailable_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    summary: Mapped[dict | None] = mapped_column(JSON_DOCUMENT)
    configuration: Mapped[dict | None] = mapped_column(JSON_DOCUMENT)
    error_message: Mapped[str | None] = mapped_column(String(512))

    meter: Mapped[Meter] = relationship()
    results: Mapped[list[AnomalyResult]] = relationship(
        back_populates="detection_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class AnomalyResult(Base):
    """One prediction result for one reading within a detection run."""

    __tablename__ = "anomaly_results"
    __table_args__ = (
        UniqueConstraint(
            "detection_run_id",
            "reading_id",
            name="uq_anomaly_results_run_reading",
        ),
        CheckConstraint(
            "prediction_status IN ('normal', 'anomaly', 'unavailable')",
            name="valid_prediction_status",
        ),
        Index("ix_anomaly_results_run_status", "detection_run_id", "prediction_status"),
        Index("ix_anomaly_results_reading_id", "reading_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    detection_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("detection_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    reading_id: Mapped[int] = mapped_column(
        ForeignKey("meter_readings.id", ondelete="CASCADE"),
        nullable=False,
    )
    prediction_status: Mapped[str] = mapped_column(String(16), nullable=False)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    anomaly_category: Mapped[str | None] = mapped_column(String(100))
    metadata_json: Mapped[dict | None] = mapped_column(JSON_DOCUMENT)

    detection_run: Mapped[DetectionRun] = relationship(back_populates="results")
    reading: Mapped[MeterReading] = relationship()
