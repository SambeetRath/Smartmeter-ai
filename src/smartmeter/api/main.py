import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from smartmeter.api.routes.health import router as health_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create the API application without initializing database connections."""
    application = FastAPI(
        title="SmartMeter AI API",
        version="0.1.0",
        description="HTTP API for SmartMeter AI energy analytics.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    application.include_router(health_router)

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled API error for %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return application


app = create_app()
