from collections.abc import Iterator

from sqlalchemy.orm import Session

from smartmeter.database.session import create_session_factory


def get_db_session() -> Iterator[Session]:
    """Yield one lazily connected database session for the current request."""
    session = create_session_factory()()
    try:
        yield session
    finally:
        session.close()
