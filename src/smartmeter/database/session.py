from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeAlias

from sqlalchemy.orm import Session, sessionmaker

from smartmeter.database.connection import get_engine

SessionFactory: TypeAlias = sessionmaker[Session]


def create_session_factory(engine=None) -> SessionFactory:
    """Create a session factory bound to the provided or configured engine."""
    return sessionmaker(
        bind=engine or get_engine(),
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def session_scope(
    session_factory: SessionFactory | None = None,
) -> Iterator[Session]:
    """Yield a session in one transaction and reliably commit or roll it back."""
    factory = session_factory or create_session_factory()
    with factory.begin() as session:
        yield session
