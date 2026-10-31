from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from smartmeter.database.models import Meter


class MeterRepository:
    """Database operations for meter metadata."""

    def get_or_create(
        self,
        session: Session,
        meter_identifier: str,
        *,
        location: str | None = None,
    ) -> Meter:
        """Return a meter, creating it safely when concurrent callers race."""
        if not meter_identifier.strip():
            raise ValueError("meter_identifier must not be empty.")

        statement = (
            insert(Meter)
            .values(meter_identifier=meter_identifier, location=location)
            .on_conflict_do_nothing(index_elements=[Meter.meter_identifier])
        )
        session.execute(statement)
        meter = session.scalar(select(Meter).where(Meter.meter_identifier == meter_identifier))
        if meter is None:
            raise RuntimeError("Meter insert completed but the meter could not be retrieved.")
        return meter

    def get_by_identifier(self, session: Session, meter_identifier: str) -> Meter | None:
        """Find a meter by its unique identifier."""
        return session.scalar(select(Meter).where(Meter.meter_identifier == meter_identifier))
