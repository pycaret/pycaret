"""In-process background scheduler for periodic platform jobs.

Backed by ``apscheduler.schedulers.background.BackgroundScheduler`` — runs
inside the FastAPI process. No Redis / Celery required for v1; if you need
multi-worker concurrency, swap to a queue-backed orchestrator at the
``RunOrchestrator`` boundary (the schedule definitions stay the same).

Two job kinds today:

- ``drift_monitor``  — periodically snapshots prediction-log distributions
  vs. each deployment's baseline and writes a ``DriftReport`` row when
  drift is detected.
- ``retrain``        — re-runs a configured experiment on its current data
  source so the deployed model stays fresh.

Schedules live in the ``scheduled_jobs`` table; the scheduler hydrates
from that table at startup. Additions / deletions go through the
``ScheduledJob`` SQLAlchemy model + the API routes.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from pycaret_server.db import ScheduledJob, get_session

_log = logging.getLogger(__name__)

# Single in-process scheduler. Lazily created on first ``get_scheduler()``.
_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()


def get_scheduler() -> BackgroundScheduler:
    """Return the process-singleton APScheduler instance, started lazily."""
    global _scheduler
    with _lock:
        if _scheduler is None:
            _scheduler = BackgroundScheduler(timezone="UTC")
            _scheduler.start(paused=False)
            _log.info("Scheduler started.")
        return _scheduler


def shutdown_scheduler() -> None:
    """For test fixtures + reload mode."""
    global _scheduler
    with _lock:
        if _scheduler is not None:
            try:
                _scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001 — never block shutdown on the scheduler
                pass
            _scheduler = None


# ----------------------------------------------------------------- registry


def register_jobs_from_db() -> int:
    """Hydrate the scheduler from ``ScheduledJob`` rows. Returns count loaded."""
    sched = get_scheduler()
    with get_session() as s:
        rows = s.query(ScheduledJob).filter(ScheduledJob.enabled.is_(True)).all()
        n = 0
        for row in rows:
            try:
                _add_to_scheduler(sched, row)
                n += 1
            except Exception as exc:  # noqa: BLE001
                _log.exception("failed to schedule job %s: %s", row.id, exc)
        return n


def schedule_job(job: ScheduledJob) -> None:
    """Add or replace a job in the live scheduler."""
    sched = get_scheduler()
    try:
        sched.remove_job(job.id)
    except Exception:  # noqa: BLE001 — job not yet scheduled
        pass
    if job.enabled:
        _add_to_scheduler(sched, job)


def unschedule_job(job_id: str) -> None:
    sched = get_scheduler()
    try:
        sched.remove_job(job_id)
    except Exception:  # noqa: BLE001 — already gone
        pass


def _add_to_scheduler(sched: BackgroundScheduler, row: ScheduledJob) -> None:
    """Translate a ``ScheduledJob`` row into an APScheduler trigger."""
    handler = JOB_HANDLERS.get(row.kind)
    if handler is None:
        raise ValueError(f"unknown scheduled-job kind: {row.kind!r}")

    trigger: Any
    schedule = row.schedule or {}
    if schedule.get("interval_seconds"):
        trigger = IntervalTrigger(seconds=int(schedule["interval_seconds"]))
    elif schedule.get("cron"):
        trigger = CronTrigger.from_crontab(schedule["cron"], timezone="UTC")
    else:
        raise ValueError(
            f"scheduled job {row.id} has no interval_seconds or cron in schedule"
        )

    sched.add_job(
        handler,
        trigger=trigger,
        id=row.id,
        kwargs={"job_id": row.id},
        replace_existing=True,
        misfire_grace_time=300,
    )


# ----------------------------------------------------------- job handlers


def run_drift_monitor(job_id: str) -> None:
    """Snapshot drift for the deployment configured on this job.

    Reads the most recent ``PredictionLog`` rows for the deployment's
    workspace, computes a simple distribution-shift score per numeric
    feature, and writes a ``DriftReport`` row. UI surfaces it in
    ``/workspaces/{ws}/drift`` automatically.
    """
    from pycaret_server.drift_monitor import run_drift_check

    with get_session() as s:
        job = s.get(ScheduledJob, job_id)
        if job is None or not job.enabled:
            return
        deployment_id = (job.target_id or "").strip()
        if not deployment_id:
            _log.warning("drift_monitor job %s has no target_id; skipping", job_id)
            return
        try:
            run_drift_check(s, deployment_id, job.created_by)
            job.last_run_at = datetime.now(UTC)
            job.last_status = "ok"
            job.last_error = None
        except Exception as exc:  # noqa: BLE001
            _log.exception("drift_monitor job %s failed", job_id)
            job.last_run_at = datetime.now(UTC)
            job.last_status = "error"
            job.last_error = f"{type(exc).__name__}: {exc}"
        s.commit()


def run_retrain(job_id: str) -> None:
    """Re-run a configured experiment using the currently-attached data source."""
    from pycaret_server.api.schemas import RunCreate
    from pycaret_server.db import Experiment
    from pycaret_server.runs.dispatch import dispatch_run

    with get_session() as s:
        job = s.get(ScheduledJob, job_id)
        if job is None or not job.enabled:
            return
        experiment_id = (job.target_id or "").strip()
        if not experiment_id:
            _log.warning("retrain job %s has no target_id; skipping", job_id)
            return
        exp = s.get(Experiment, experiment_id)
        if exp is None:
            _log.warning("retrain job %s: experiment %s missing", job_id, experiment_id)
            job.last_status = "error"
            job.last_error = "experiment not found"
            job.last_run_at = datetime.now(UTC)
            s.commit()
            return

        spec = job.spec or {}
        body = RunCreate(
            plan=spec.get("plan", "compare"),
            model_id=spec.get("model_id"),
            plan_params=spec.get("plan_params"),
            sklearn_dataset=spec.get("sklearn_dataset"),
            data_source_id=spec.get("data_source_id"),
        )
        try:
            run = dispatch_run(s, exp, body, user_id=job.created_by)
            job.last_run_at = datetime.now(UTC)
            job.last_status = "ok"
            job.last_run_run_id = run.id
            job.last_error = None
        except Exception as exc:  # noqa: BLE001
            _log.exception("retrain job %s failed", job_id)
            job.last_run_at = datetime.now(UTC)
            job.last_status = "error"
            job.last_error = f"{type(exc).__name__}: {exc}"
        s.commit()


JOB_HANDLERS = {
    "drift_monitor": run_drift_monitor,
    "retrain": run_retrain,
}
