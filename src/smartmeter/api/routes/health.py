from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["smartmeter-ai"]


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Return process liveness without depending on external services."""
    return HealthResponse(status="ok", service="smartmeter-ai")
