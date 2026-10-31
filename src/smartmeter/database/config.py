from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

DEFAULT_INGESTION_BATCH_SIZE = 5_000


def get_database_url(environ: Mapping[str, str] | None = None) -> str:
    """Return a validated PostgreSQL+psycopg URL without logging credentials."""
    variables = os.environ if environ is None else environ
    database_url = variables.get("DATABASE_URL", "").strip()
    if not database_url:
        required = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
        missing = [key for key in required if not variables.get(key)]
        if missing:
            raise ValueError(
                "Set DATABASE_URL or all of POSTGRES_DB, POSTGRES_USER, and POSTGRES_PASSWORD."
            )
        try:
            port = int(variables.get("POSTGRES_PORT", "5432"))
        except ValueError as exc:
            raise ValueError("POSTGRES_PORT must be a valid port number.") from exc
        if not 1 <= port <= 65535:
            raise ValueError("POSTGRES_PORT must be between 1 and 65535.")
        database_url = URL.create(
            "postgresql+psycopg",
            username=variables["POSTGRES_USER"],
            password=variables["POSTGRES_PASSWORD"],
            host=variables.get("SMARTMETER_DB_HOST", "localhost"),
            port=port,
            database=variables["POSTGRES_DB"],
        ).render_as_string(hide_password=False)

    try:
        parsed_url = make_url(database_url)
    except ArgumentError as exc:
        raise ValueError("DATABASE_URL is not a valid SQLAlchemy URL.") from exc

    if parsed_url.drivername != "postgresql+psycopg":
        raise ValueError("DATABASE_URL must use the postgresql+psycopg driver.")
    if not parsed_url.database:
        raise ValueError("DATABASE_URL must include a database name.")
    return database_url


def get_ingestion_batch_size(environ: Mapping[str, str] | None = None) -> int:
    """Return a positive configured database-ingestion batch size."""
    variables = os.environ if environ is None else environ
    value = variables.get("SMARTMETER_INGESTION_BATCH_SIZE", str(DEFAULT_INGESTION_BATCH_SIZE))
    try:
        batch_size = int(value)
    except ValueError as exc:
        raise ValueError("SMARTMETER_INGESTION_BATCH_SIZE must be a positive integer.") from exc
    if batch_size <= 0:
        raise ValueError("SMARTMETER_INGESTION_BATCH_SIZE must be a positive integer.")
    return batch_size
