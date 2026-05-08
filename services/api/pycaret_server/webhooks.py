"""Outgoing webhook delivery.

Fires HTTP POSTs to ``WebhookSubscription.url`` rows whose ``event_types``
include the dispatched event AND whose ``filters`` are a subset of the
event payload.

Each delivery includes:

  * ``X-PyCaret-Event``      — the event type (e.g. ``run.succeeded``)
  * ``X-PyCaret-Signature``  — HMAC-SHA256 of the body, hex-encoded, using
                               the row's encrypted secret (decrypted at fire
                               time). Subscribers verify by recomputing.
  * ``X-PyCaret-Delivery``   — UUID per delivery, idempotency key.

Delivery is best-effort and synchronous (small platform). Failures update
``last_status_code`` / ``last_error`` on the subscription row but do NOT
raise — webhook delivery failures must never break the originating action.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import uuid
from datetime import UTC, datetime

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.crypto import decrypt as _decrypt_secret
from pycaret_server.db import WebhookSubscription, get_session

_log = logging.getLogger(__name__)
_DEFAULT_TIMEOUT = 5.0


def fire_event(event_type: str, payload: dict) -> int:
    """Synchronously dispatch ``event_type`` to all matching subscriptions.

    Returns the number of subscriptions notified. Never raises on per-row
    delivery failures.
    """
    workspace_id = (payload or {}).get("workspace_id")
    if not workspace_id:
        return 0

    notified = 0
    with get_session() as s:
        rows = s.scalars(
            select(WebhookSubscription).where(
                WebhookSubscription.workspace_id == workspace_id,
                WebhookSubscription.enabled.is_(True),
            )
        ).all()
        for row in rows:
            if event_type not in (row.event_types or []):
                continue
            if not _filters_match(row.filters, payload):
                continue
            _deliver(s, row, event_type, payload)
            notified += 1
        s.commit()
    return notified


def fire_event_async(event_type: str, payload: dict) -> threading.Thread:
    """Fire-and-forget wrapper that schedules delivery on a daemon thread.

    Use this from request handlers so a slow webhook target doesn't pin
    the user's response.
    """
    t = threading.Thread(
        target=_safe_fire,
        args=(event_type, payload),
        name=f"webhook-{event_type}",
        daemon=True,
    )
    t.start()
    return t


def _safe_fire(event_type: str, payload: dict) -> None:
    try:
        fire_event(event_type, payload)
    except Exception:  # noqa: BLE001
        _log.exception("webhook fire_event(%s) failed", event_type)


def _filters_match(filters: dict | None, payload: dict) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        if payload.get(k) != v:
            return False
    return True


def _deliver(
    s: Session,
    row: WebhookSubscription,
    event_type: str,
    payload: dict,
) -> None:
    body = json.dumps(
        {
            "event": event_type,
            "delivery_id": str(uuid.uuid4()),
            "fired_at": datetime.now(UTC).isoformat(),
            "data": payload,
        },
        default=str,
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "X-PyCaret-Event": event_type,
        "X-PyCaret-Delivery": str(uuid.uuid4()),
        "User-Agent": "PyCaret-Server/0.1",
    }
    if row.secret_encrypted:
        try:
            secret = _decrypt_secret(row.secret_encrypted).encode("utf-8")
        except Exception:  # noqa: BLE001
            secret = row.secret_encrypted.encode("utf-8")  # legacy plaintext
        sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
        headers["X-PyCaret-Signature"] = sig

    try:
        resp = requests.post(
            row.url, data=body, headers=headers, timeout=_DEFAULT_TIMEOUT
        )
        row.last_status_code = resp.status_code
        row.last_error = None if resp.ok else f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        row.last_status_code = None
        row.last_error = f"{type(exc).__name__}: {exc}"
    row.last_fired_at = datetime.now(UTC)
