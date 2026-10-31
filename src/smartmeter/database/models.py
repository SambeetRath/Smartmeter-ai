from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from smartmeter.database.base import Base

READING_NUMERIC = Numeric(12, 6)


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
