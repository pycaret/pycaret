"""SQLAlchemy engine + session factory + FastAPI dependency.

Phase 3: this module is dual-backend-aware. SQLite stays the zero-config
dev default (``PYCARET_DATABASE_URL=sqlite:///./pycaret.db``); production
deploys point at Postgres via ``postgresql+psycopg://user:pass@host/db``.

The driver is selected purely by the URL — the rest of the application
code is engine-agnostic. The only divergence is connection-pool tuning
and SQLite's cross-thread relaxation, both handled here.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from pycaret_server.config import get_settings

settings = get_settings()

# Per-driver kwargs.
#
#   - SQLite: ``check_same_thread=False`` lets FastAPI's threadpool reuse
#     one connection. We also disable the default StaticPool sizing —
#     the SQLAlchemy default is fine.
#   - Postgres: default pool (5 + 10 overflow) is enough for a single
#     backend instance + a handful of workers. ``pool_pre_ping`` catches
#     stale connections that survived a Postgres restart.
#
# Anything heavier (PgBouncer, read replicas, statement timeouts) ships
# in Phase 13's enterprise-polish slice.
_connect_args: dict = {}
_pool_kwargs: dict = {"pool_pre_ping": True}
_db_url = settings.database_url

if _db_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}
elif _db_url.startswith(("postgresql", "postgres")):
    _pool_kwargs.update(
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,  # drop idle connections after 30 min
    )

engine = create_engine(
    _db_url,
    echo=settings.debug,
    connect_args=_connect_args,
    future=True,
    **_pool_kwargs,
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
