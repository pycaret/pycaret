"""Phase 14: queue + worker admin routes.

Per-class queue depth + active worker visibility, so an admin page can
render "how many jobs waiting on the gpu queue / how many workers
listening".

Endpoints:

- ``GET  /admin/queues``           queue depth + recent throughput
- ``GET  /admin/workers``          active worker heartbeats
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pycaret_server.auth import CurrentUser
from pycaret_server.db import Job, get_db

router = APIRouter(tags=["admin-queues"], prefix="/admin")


@router.get("/queues")
def list_queues(
    user: CurrentUser,  # noqa: ARG001 — auth gate, ignored otherwise
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Per-queue counts across statuses + recent throughput.

    ``recent_succeeded`` counts Jobs that finished in the last hour.
    Lets an admin tell at a glance whether a GPU pool is alive without
    SSHing the worker hosts.
    """
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=1)
    rows = db.execute(
        select(Job.queue, Job.status, func.count(Job.id))
        .group_by(Job.queue, Job.status)
    ).all()
    queues: dict[str, dict[str, int]] = {}
    for queue, st, cnt in rows:
        queues.setdefault(queue or "default", {})[st] = cnt
    recent = db.execute(
        select(Job.queue, func.count(Job.id))
        .where(Job.status == "succeeded", Job.finished_at >= cutoff)
        .group_by(Job.queue)
    ).all()
    recent_map = {q: c for q, c in recent}
    return {
        "queues": [
            {
                "name": name,
                "queued": counts.get("queued", 0),
                "running": counts.get("running", 0),
                "succeeded": counts.get("succeeded", 0),
                "failed": counts.get("failed", 0),
                "cancelled": counts.get("cancelled", 0),
                "recent_throughput_1h": recent_map.get(name, 0),
            }
            for name, counts in sorted(queues.items())
        ],
        "as_of": now.isoformat(),
    }


@router.get("/workers")
def list_workers(
    user: CurrentUser,  # noqa: ARG001 — auth gate
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Workers currently holding a Job lock.

    We don't have a ``workers`` table — derive from ``Job.locked_by``
    grouped by id. A worker that finished its job and went idle drops
    off this list; for a real heartbeat surface, the worker would push
    a periodic row to a small ``worker_heartbeats`` table (a future
    cut).
    """
    rows = db.execute(
        select(Job.locked_by, func.count(Job.id), func.max(Job.locked_at))
        .where(Job.status == "running", Job.locked_by.is_not(None))
        .group_by(Job.locked_by)
    ).all()
    return {
        "workers": [
            {
                "worker_id": worker_id,
                "running_jobs": int(cnt),
                "last_lock_at": last.isoformat() if last else None,
            }
            for worker_id, cnt, last in rows
        ]
    }


@router.get("/system")
def system_info(
    user: CurrentUser,  # noqa: ARG001 — auth gate
) -> dict:
    """Process-local capability probe.

    Surfaces what the *API process* can see — GPU devices, Redis
    reachability, runs backend mode — so an admin page can render
    "GPU: Tesla T4 ×1, Redis: ✓, backend: redis". For per-worker
    inventory you'd push a row into the future ``worker_heartbeats``
    table; for now this is the best-available single-shot view.
    """
    from pycaret_server.config import get_settings
    from pycaret_server.runtime import detect_gpus

    settings = get_settings()
    inv = detect_gpus()
    redis_ok = False
    redis_error: str | None = None
    if settings.runs_backend == "redis":
        try:
            from pycaret_server.runs.queue_redis import is_healthy

            redis_ok = bool(is_healthy(settings.redis_url))
        except Exception as exc:  # noqa: BLE001
            redis_error = str(exc)
    return {
        "runs_backend": settings.runs_backend,
        "redis": {
            "url": settings.redis_url if settings.runs_backend == "redis" else None,
            "healthy": redis_ok,
            "error": redis_error,
        },
        "gpu": {
            "available": inv.available,
            "count": inv.count,
            "devices": list(inv.devices),
            "source": inv.source,
            "error": inv.error,
        },
        "worker_queues": [
            q.strip()
            for q in (settings.worker_queues or "default").split(",")
            if q.strip()
        ],
    }
