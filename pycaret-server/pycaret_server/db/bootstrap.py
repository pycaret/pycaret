"""Schema bootstrap — the bridge between SQLite dev and Alembic-managed prod.

`ensure_schema(engine, dev_auto_migrate=True)`:

- If ``alembic_version`` table already exists, assume the operator is running
  migrations themselves → no-op.
- Else if ``dev_auto_migrate`` is True (defaults to SQLite dev), call
  ``alembic upgrade head`` programmatically so local dev is one-command.
- Else fail loudly so a misconfigured production deploy can't silently run
  against an out-of-date schema.

This sits in `db/` (not `app.py`) so CLI + tests can call it without going
through the FastAPI app factory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

_log = logging.getLogger(__name__)

# alembic.ini lives at the package root (../../alembic.ini relative to this file).
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def ensure_schema(engine: Engine, *, dev_auto_migrate: bool = True) -> None:
    """Bring the connected DB up to the latest Alembic revision when needed.

    Parameters
    ----------
    engine
        SQLAlchemy Engine pointing at the target DB.
    dev_auto_migrate
        When True, blank databases get `alembic upgrade head` run automatically.
        Only safe for single-process dev; prod should set this False and run
        migrations explicitly (`pycaret-server migrate` or `alembic upgrade head`).
    """
    # Already-migrated DB? Respect whatever the operator has.
    insp = inspect(engine)
    has_alembic = insp.has_table("alembic_version")
    has_any_user_table = insp.has_table("users")

    if has_alembic:
        _log.debug("alembic_version table present; leaving schema alone")
        return

    if has_any_user_table and not has_alembic:
        _log.warning(
            "Database has tables but no alembic_version — legacy create_all schema. "
            "Stamping as baseline. Future migrations will work from here."
        )
        _run_alembic("stamp", "head", url=str(engine.url))
        return

    if not dev_auto_migrate:
        raise RuntimeError(
            "Database is empty and dev_auto_migrate=False. "
            "Run `pycaret-server migrate` or `alembic upgrade head` before starting the server."
        )

    _log.info("Empty database — applying baseline migration")
    _run_alembic("upgrade", "head", url=str(engine.url))


def _run_alembic(*argv: str, url: str) -> None:
    """Invoke Alembic programmatically so we can share the live engine's URL."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)
    cmd, *rest = argv
    getattr(command, cmd)(cfg, *rest)
