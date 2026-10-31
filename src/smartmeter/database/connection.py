from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url

from smartmeter.database.config import get_database_url


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create a pooled PostgreSQL engine with liveness checks and UTC sessions."""
    url = database_url or get_database_url()
    if make_url(url).drivername != "postgresql+psycopg":
        raise ValueError("Database engine requires the postgresql+psycopg driver.")
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        connect_args={"options": "-c timezone=UTC", "connect_timeout": 5},
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Get the process-wide engine, initialized only when first requested."""
    return create_database_engine()
