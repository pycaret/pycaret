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

from sqlalchemy import inspect, text
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
        # alembic_version is present. In dev, bring the schema up to head so
        # newer migrations land without operator action. In prod we leave it
        # alone — explicit ``alembic upgrade head`` is the deploy contract.
        if dev_auto_migrate:
            try:
                _run_alembic("upgrade", "head", url=str(engine.url))
            except Exception as exc:  # noqa: BLE001
                _log.warning("dev auto-migrate failed (continuing): %s", exc)
        else:
            _log.debug("alembic_version table present; leaving schema alone")
        return

    if has_any_user_table and not has_alembic:
        # Pre-Alembic legacy DB. Detect which revision matches the current
        # schema and stamp there. If the detected revision is behind head and
        # we're in dev, also run upgrade head so the missing tables get
        # created.
        revision, is_head = _detect_revision(insp)
        if is_head:
            _log.info(
                "Legacy DB schema already matches alembic head — stamping head."
            )
            _run_alembic("stamp", "head", url=str(engine.url))
            return
        _log.warning(
            "Legacy DB without alembic_version. Detected schema revision %r — "
            "stamping there and upgrading to head.",
            revision,
        )
        _run_alembic("stamp", revision, url=str(engine.url))
        if dev_auto_migrate:
            _run_alembic("upgrade", "head", url=str(engine.url))
        return

    if not dev_auto_migrate:
        raise RuntimeError(
            "Database is empty and dev_auto_migrate=False. "
            "Run `pycaret-server migrate` or `alembic upgrade head` before starting the server."
        )

    _log.info("Empty database — applying baseline migration")
    _run_alembic("upgrade", "head", url=str(engine.url))


def _detect_revision(insp) -> tuple[str, bool]:
    """Identify which migration matches the current set of tables/columns.

    Walks revisions from newest → oldest and returns the first revision whose
    distinguishing tables are all present. Falls back to baseline if nothing
    past baseline is detected.

    Returns ``(revision_id, is_head)`` — ``is_head`` is True only when the
    newest fingerprint matched, so the caller can skip a redundant
    ``upgrade head`` in that case.
    """
    # Each entry: (revision_id, [tables that must all exist for this revision]).
    # Order matters — newest first. The check returns the latest match.
    REVISION_FINGERPRINTS: list[tuple[str, list[str]]] = [
        # session 24: scheduled_jobs / webhook_subscriptions / experiment_templates
        ("b2c3d4e5f6a7", ["scheduled_jobs", "webhook_subscriptions", "experiment_templates"]),
        # session 22: prediction_logs / trials / model_library
        ("a1b2c3d4e5f6", ["prediction_logs", "trials", "model_library"]),
        # session 21: audit_logs / drift_reports
        ("0cd9d5ea2e17", ["audit_logs", "drift_reports"]),
        # session 17: llm_provider_settings / llm_consultations
        ("d582b350c276", ["llm_provider_settings", "llm_consultations"]),
    ]
    for idx, (rev, required) in enumerate(REVISION_FINGERPRINTS):
        if all(insp.has_table(t) for t in required):
            return rev, idx == 0
    # Default: pre-LLM baseline.
    return "9f9b7c770df0", False


def _run_alembic(*argv: str, url: str) -> None:
    """Invoke Alembic programmatically so we can share the live engine's URL.

    Alembic's ``script_location`` is resolved against the process CWD when
    loaded from an ``.ini``. In CI the CWD is the repo root, not the server
    package dir, so we substitute an absolute path here. This lets the
    bootstrap work from any working directory (pytest, uvicorn, CLI).
    """
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.set_main_option(
        "script_location",
        str(_ALEMBIC_INI.parent / "pycaret_server" / "migrations"),
    )
    cmd, *rest = argv
    getattr(command, cmd)(cfg, *rest)
