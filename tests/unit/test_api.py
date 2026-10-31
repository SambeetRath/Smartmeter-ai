from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI

from smartmeter.api.main import app, create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def _get(
    application: FastAPI,
    path: str,
    *,
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    transport = httpx.ASGITransport(
        app=application,
        raise_app_exceptions=raise_app_exceptions,
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path)


def test_application_imports_and_health_returns_expected_response():
    response = asyncio.run(_get(app, "/health"))

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "smartmeter-ai"}


def test_openapi_schema_is_available_and_describes_health_route():
    response = asyncio.run(_get(app, "/openapi.json"))

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "SmartMeter AI API"
    assert schema["info"]["version"] == "0.1.0"
    assert "/health" in schema["paths"]


def test_import_does_not_require_database_configuration():
    environment = os.environ.copy()
    for variable in (
        "DATABASE_URL",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "SMARTMETER_TEST_DATABASE_URL",
    ):
        environment.pop(variable, None)
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH", "")) if part
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from smartmeter.api.main import app; "
                "assert app.title == 'SmartMeter AI API'; "
                "assert app.openapi()['paths'].get('/health') is not None"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        timeout=15,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_unexpected_errors_return_generic_response_without_internal_details():
    test_app: FastAPI = create_app()

    def failing_endpoint() -> None:
        raise RuntimeError("private implementation detail")

    test_app.add_api_route("/_test-error", failing_endpoint)
    response = asyncio.run(_get(test_app, "/_test-error", raise_app_exceptions=False))

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "private implementation detail" not in response.text
