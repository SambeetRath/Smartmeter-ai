from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ReadingResponse(BaseModel):
    """A meter reading with nullable measurement values."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    global_active_power: float | None
    global_reactive_power: float | None
    voltage: float | None
    global_intensity: float | None
    sub_metering_1: float | None
    sub_metering_2: float | None
    sub_metering_3: float | None


class ReadingsPageResponse(BaseModel):
    """A bounded page of readings and its continuation information."""

    meter_identifier: str
    items: list[ReadingResponse]
    limit: int
    offset: int
    returned_count: int
    has_more: bool
    next_offset: int | None
