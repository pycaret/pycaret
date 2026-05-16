"""Phase 1 Redis-backed Job queue adapter.

Thin shim around ``redis-py``. The schema is:

- One Redis LIST per queue name (``pycaret:queue:default``,
  ``pycaret:queue:gpu``, etc.). LPUSH on enqueue, BLPOP on dequeue.
- One Redis HASH per Job id (``pycaret:job:<id>``) — used only for the
  visibility timeout / heartbeat extension. The authoritative state
  lives in the Postgres ``jobs`` table.

The worker entrypoint (``pycaret-worker serve``) BLPOPs from its queues
and looks the Job up in the DB. If a worker dies mid-job, the next
worker that polls finds the Job in ``running`` with a stale
``locked_at`` and re-claims it.

``redis-py`` is an optional dependency — kept out of the main install
so SQLite-only dev paths don't pull it in. ``pip install pycaret-server[redis]``
brings it in for prod.
"""

from __future__ import annotations

from typing import Any


def _client(redis_url: str) -> Any:
    """Return a connected redis.Redis. Late import keeps ``redis`` optional."""
    import redis  # type: ignore[import-not-found]

    return redis.Redis.from_url(redis_url, decode_responses=True)


def _queue_key(queue: str) -> str:
    return f"pycaret:queue:{queue}"


def enqueue_job(job_id: str, *, queue: str = "default", redis_url: str) -> None:
    """LPUSH a job id onto its queue. Workers BLPOP from the right side."""
    client = _client(redis_url)
    client.lpush(_queue_key(queue), job_id)


def dequeue_job(
    queues: list[str], *, redis_url: str, timeout: float = 5.0
) -> str | None:
    """BRPOP across ``queues``; return the job id or None on timeout.

    Order matters: when multiple queues have work, the first one in the
    list wins. Phase 14's worker class routing uses this — a GPU worker
    lists ``[gpu, default]`` and only steals from default when its own
    queue is empty.
    """
    client = _client(redis_url)
    keys = [_queue_key(q) for q in queues]
    res = client.brpop(keys, timeout=timeout)  # type: ignore[arg-type]
    if res is None:
        return None
    _key, job_id = res
    return job_id


def heartbeat(job_id: str, *, ttl_seconds: int = 60, redis_url: str) -> None:
    """Refresh a Job's visibility timeout. Called periodically by workers
    so a long-running job isn't reclaimed mid-flight.

    Stored as a key with TTL so a crashed worker's heartbeat naturally
    expires and the next worker can take over.
    """
    client = _client(redis_url)
    client.set(f"pycaret:job:{job_id}:heartbeat", "1", ex=ttl_seconds)


def is_healthy(redis_url: str) -> bool:
    """Quick liveness check used by the ``doctor`` CLI command."""
    try:
        client = _client(redis_url)
        return bool(client.ping())
    except Exception:  # noqa: BLE001
        return False
