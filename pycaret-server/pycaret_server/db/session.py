"""SQLAlchemy engine + session factory + FastAPI dependency."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from pycaret_server.config import get_settings

settings = get_settings()

# SQLite-specific kwargs: allow cross-thread access for FastAPI worker threads.
# For other drivers no special kwargs; SQLAlchemy handles pool sizing by default.
_connect_args: dict = {}
if settings.database_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args=_connect_args,
    # Pool options are defaults for now; tune when we hit scale.
    pool_pre_ping=True,
    future=True,
)

session_factory = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def get_session() -> Session:
    """Return a new session. Caller is responsible for close/commit.

    Prefer `get_db` as a FastAPI dependency for request-scoped sessions."""
    return session_factory()


def get_db() -> Generator[Session]:
    """FastAPI dependency: yields a session, closes it after the request."""
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
