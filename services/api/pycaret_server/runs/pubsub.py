"""Phase 6 cross-process pub/sub for run events.

In Phase 0/1 the event broker (``runs/broker.py``) fans events within
a single Python process via ``asyncio.Queue``. That stops working when
the worker runs in a separate container (Phase 1 ``runs_backend=redis``):
the worker fits a model, emits events to its in-process broker, but the
WebSocket clients are connected to the API process and never see them.

This module bridges the two with a Redis pub/sub channel per run.

- **Worker side**: ``publish_to_redis(run_id, event)`` after the existing
  in-process publish. The worker doesn't know if anyone's listening —
  it just forwards.
- **API side**: ``run_subscriber_loop(run_id)`` (started lazily by the
  WebSocket handler) subscribes to the Redis channel and re-publishes
  every message back into the local broker so existing WebSocket
  clients pick it up.

This keeps the protocol stable across both backends — the WebSocket
client never has to care whether the worker is local or remote.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from pycaret_server.config import get_settings
from pycaret_server.runs.broker import event_broker

_log = logging.getLogger(__name__)


def _channel(run_id: str) -> str:
    return f"pycaret:events:run:{run_id}"


def publish_to_redis(run_id: str, event: dict[str, Any]) -> None:
    """Publish ``event`` to the run's Redis channel. Best-effort.

    Called from the worker right after the in-process publish.
    Failures are swallowed — losing a fan-out event is preferable to
    failing the training run.
    """
    settings = get_settings()
    if settings.runs_backend != "redis":
        return
    try:
        from pycaret_server.runs import queue_redis  # late import

        client = queue_redis._client(settings.redis_url)
        client.publish(_channel(run_id), json.dumps(event, default=str))
    except Exception:  # noqa: BLE001
        _log.debug("redis fan-out failed", exc_info=True)


# Tracks which run subscriptions are already running on the API side so
# we don't start one per WebSocket client.
_subscribed: set[str] = set()
_subscribed_lock = threading.Lock()


def ensure_subscribed(run_id: str) -> None:
    """Start a background task that bridges Redis → in-process broker.

    Idempotent — only the first call per run_id spins up a task. The
    task self-terminates when the run reaches a terminal state (the
    worker publishes a sentinel ``run.closed`` event, which closes the
    local broker via existing ``close_run`` flow).
    """
    settings = get_settings()
    if settings.runs_backend != "redis":
        return
    with _subscribed_lock:
        if run_id in _subscribed:
            return
        _subscribed.add(run_id)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        with _subscribed_lock:
            _subscribed.discard(run_id)
        return
    loop.create_task(_subscriber(run_id))


async def _subscriber(run_id: str) -> None:
    """Read Redis → publish to local broker → exit on terminal event."""
    settings = get_settings()
    try:
        from pycaret_server.runs import queue_redis

        client = queue_redis._client(settings.redis_url)
        pubsub = client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(_channel(run_id))
        for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            try:
                event = json.loads(msg["data"])
            except Exception:  # noqa: BLE001
                continue
            event_broker.publish(run_id, event)
            if event.get("kind") in ("run.succeeded", "run.failed", "run.cancelled"):
                event_broker.close_run(run_id)
                break
    except Exception:  # noqa: BLE001
        _log.debug("redis subscriber for %s ended", run_id, exc_info=True)
    finally:
        with _subscribed_lock:
            _subscribed.discard(run_id)
