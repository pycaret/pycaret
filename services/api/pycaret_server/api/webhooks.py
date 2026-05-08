"""Webhook subscription CRUD.

Routes (under ``/api/v1``):

  GET    /workspaces/{ws_id}/webhooks
  POST   /workspaces/{ws_id}/webhooks
  GET    /webhooks/{id}
  PATCH  /webhooks/{id}
  DELETE /webhooks/{id}
  POST   /webhooks/{id}/test

Admin-only writes; member can read.

Supported event types (v1):
  - ``run.succeeded`` / ``run.failed`` / ``run.cancelled``
  - ``deployment.created`` / ``deployment.deleted`` / ``deployment.rollback``
  - ``drift.alert``
  - ``schedule.failed``
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access, _require_admin
from pycaret_server.auth import CurrentUser
from pycaret_server.crypto import encrypt as _encrypt_secret
from pycaret_server.db import WebhookSubscription, get_db
from pycaret_server.webhooks import fire_event

router = APIRouter(tags=["webhooks"])

VALID_EVENT_TYPES = {
    "run.succeeded",
    "run.failed",
    "run.cancelled",
    "deployment.created",
    "deployment.deleted",
    "deployment.rollback",
    "drift.alert",
    "schedule.failed",
}


class WebhookCreate(BaseModel):
    url: str
    event_types: list[str]
    secret: str | None = Field(default=None, min_length=8)
    filters: dict | None = None
    enabled: bool = True


class WebhookPatch(BaseModel):
    url: str | None = None
    event_types: list[str] | None = None
    secret: str | None = None
    filters: dict | None = None
    enabled: bool | None = None


class WebhookRead(BaseModel):
    id: str
    workspace_id: str
    url: str
    event_types: list[str]
    has_secret: bool
    filters: dict | None
    enabled: bool
    last_fired_at: str | None
    last_status_code: int | None
    last_error: str | None


def _serialise(row: WebhookSubscription) -> WebhookRead:
    return WebhookRead(
        id=row.id,
        workspace_id=row.workspace_id,
        url=row.url,
        event_types=list(row.event_types or []),
        has_secret=bool(row.secret_encrypted),
        filters=dict(row.filters) if row.filters else None,
        enabled=row.enabled,
        last_fired_at=row.last_fired_at.isoformat() if row.last_fired_at else None,
        last_status_code=row.last_status_code,
        last_error=row.last_error,
    )


def _validate_events(events: list[str]) -> None:
    invalid = [e for e in events if e not in VALID_EVENT_TYPES]
    if invalid:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown event types: {invalid}; valid: {sorted(VALID_EVENT_TYPES)}",
        )


@router.get("/workspaces/{workspace_id}/webhooks")
def list_webhooks(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(WebhookSubscription)
        .where(WebhookSubscription.workspace_id == workspace_id)
        .order_by(WebhookSubscription.created_at.desc())
    ).all()
    return {"items": [_serialise(r).model_dump() for r in rows]}


@router.post(
    "/workspaces/{workspace_id}/webhooks",
    status_code=status.HTTP_201_CREATED,
)
def create_webhook(
    workspace_id: str,
    payload: WebhookCreate,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> WebhookRead:
    _require_admin(user, db, workspace_id)
    if not payload.event_types:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "event_types must be non-empty"
        )
    _validate_events(payload.event_types)

    row = WebhookSubscription(
        workspace_id=workspace_id,
        url=payload.url,
        event_types=list(payload.event_types),
        secret_encrypted=_encrypt_secret(payload.secret) if payload.secret else None,
        filters=dict(payload.filters) if payload.filters else None,
        enabled=payload.enabled,
        created_by=user.id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _serialise(row)


@router.get("/webhooks/{webhook_id}")
def get_webhook(
    webhook_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> WebhookRead:
    row = db.get(WebhookSubscription, webhook_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "webhook not found")
    _require_access(user, db, row.workspace_id)
    return _serialise(row)


@router.patch("/webhooks/{webhook_id}")
def patch_webhook(
    webhook_id: str,
    payload: WebhookPatch,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> WebhookRead:
    row = db.get(WebhookSubscription, webhook_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "webhook not found")
    _require_admin(user, db, row.workspace_id)

    if payload.url is not None:
        row.url = payload.url
    if payload.event_types is not None:
        _validate_events(payload.event_types)
        row.event_types = list(payload.event_types)
    if payload.secret is not None:
        row.secret_encrypted = _encrypt_secret(payload.secret) if payload.secret else None
    if payload.filters is not None:
        row.filters = dict(payload.filters) if payload.filters else None
    if payload.enabled is not None:
        row.enabled = payload.enabled

    db.commit()
    db.refresh(row)
    return _serialise(row)


@router.delete(
    "/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_webhook(
    webhook_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    row = db.get(WebhookSubscription, webhook_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "webhook not found")
    _require_admin(user, db, row.workspace_id)
    db.delete(row)
    db.commit()


@router.post("/webhooks/{webhook_id}/test")
def test_webhook(
    webhook_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Fire a synthetic ``run.succeeded`` event to this subscription only."""
    row = db.get(WebhookSubscription, webhook_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "webhook not found")
    _require_admin(user, db, row.workspace_id)

    # Build a single-target event by limiting filters to this row's id.
    payload = {
        "workspace_id": row.workspace_id,
        "test": True,
        "_target_webhook_id": row.id,
    }
    # Direct fire without going through fire_event's filter loop — we want
    # to deliver to *this row* even if it doesn't list the synthetic event.
    from pycaret_server.webhooks import _deliver

    _deliver(db, row, "test.ping", payload)
    db.commit()
    db.refresh(row)
    return _serialise(row).model_dump()
