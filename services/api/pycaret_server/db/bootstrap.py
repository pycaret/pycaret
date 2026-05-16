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
    # Sessions 25 / 26 / 27 / Phase 0 added columns to ``trials`` (no new
    # table). Distinguish them from session 24 by inspecting the column set on
    # ``trials``. Newest-first so we stamp at the highest applicable rev.
    if insp.has_table("trials"):
        try:
            trial_cols = {c["name"] for c in insp.get_columns("trials")}
        except Exception:  # noqa: BLE001
            trial_cols = set()
        try:
            run_cols = (
                {c["name"] for c in insp.get_columns("runs")}
                if insp.has_table("runs")
                else set()
            )
        except Exception:  # noqa: BLE001
            run_cols = set()
        has_25 = {"stored_path", "params"}.issubset(trial_cols)
        has_26 = "notes" in trial_cols
        has_27 = "kind" in trial_cols and "parent_trial_ids" in trial_cols
        # Phase 0 fingerprint: trials gained ``experiment_id`` /
        # ``created_by_action_id``, the artifact columns moved to ``runs``
        # (so ``runs.trial_id`` and ``runs.sequence`` exist), and the legacy
        # ``trials.run_id`` / ``trials.rank`` / ``trials.is_best`` columns are
        # gone.
        has_phase0_trials = (
            "experiment_id" in trial_cols
            and "created_by_action_id" in trial_cols
            and "run_id" not in trial_cols
            and "rank" not in trial_cols
            and "is_best" not in trial_cols
        )
        has_phase0_runs = {"trial_id", "sequence", "metrics"}.issubset(run_cols)
        # Phase 0-v2 reverted the model: Trial owns artifact + metrics +
        # status; Run loses trial_id/sequence/etc. Distinguishing
        # fingerprints: trials has ``run_id`` AND ``status``; runs no
        # longer has ``trial_id``.
        has_phase0_v2_trials = (
            "experiment_id" in trial_cols
            and "run_id" in trial_cols
            and "status" in trial_cols
            and "metrics" in trial_cols
        )
        has_phase0_v2_runs = "trial_id" not in run_cols
        scheduled = insp.has_table("scheduled_jobs")
        has_phase1_jobs = insp.has_table("jobs")
        # Phase 4-12 cut: registry + governance + monitoring tables all
        # land together. A single fingerprint match means we're at head.
        has_phase_4_12 = (
            insp.has_table("registered_models")
            and insp.has_table("registered_model_versions")
            and insp.has_table("connections")
            and insp.has_table("datasets")
            and insp.has_table("lineage")
            and insp.has_table("git_repositories")
            and insp.has_table("alert_rules")
            and insp.has_table("metric_points")
            and insp.has_table("approval_workflows")
            and insp.has_table("secrets")
        )
        # Phase 8 + 11: notebook runtime + statistical computing.
        has_phase_8_11 = (
            insp.has_table("notebooks")
            and insp.has_table("notebook_sessions")
            and insp.has_table("analyses")
        )
        if (
            has_phase0_v2_trials
            and has_phase0_v2_runs
            and scheduled
            and has_phase1_jobs
            and has_phase_4_12
            and has_phase_8_11
        ):
            return "d4e5f6a8b9c0", True  # head — Phase 0-v2 revert
        if (
            has_phase0_trials
            and has_phase0_runs
            and scheduled
            and has_phase1_jobs
            and has_phase_4_12
            and has_phase_8_11
        ):
            return "c3d4e5f6a8b9", False  # Phases 8/11, needs upgrade to v2
        if (
            has_phase0_trials
            and has_phase0_runs
            and scheduled
            and has_phase1_jobs
            and has_phase_4_12
        ):
            return "b2c3d4e5f6a8", False  # Phases 4/5/7/10/12, needs upgrade
        if has_phase0_trials and has_phase0_runs and scheduled and has_phase1_jobs:
            return "a1b2c3d4e5f7", False  # Phase 1, needs upgrade
        if has_phase0_trials and has_phase0_runs and scheduled:
            return "f0a1b2c3d4e5", False  # Phase 0, needs upgrade
        if has_27 and has_26 and has_25 and scheduled:
            return "e5f6a7b8c9d0", False  # session 27, needs upgrade to Phase 0
        if has_26 and has_25 and scheduled:
            return "d4e5f6a7b8c9", False  # session 26, needs upgrade
        if has_25 and scheduled:
            return "c3d4e5f6a7b8", False  # session 25, needs upgrade

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
    for rev, required in REVISION_FINGERPRINTS:
        if all(insp.has_table(t) for t in required):
            return rev, False  # not head — head is d4e5f6a8b9c0 (Phase 0-v2 revert)
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
