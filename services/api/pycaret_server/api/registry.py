"""Phase 7: Model registry routes.

Endpoints:

- ``GET    /workspaces/{ws}/registered-models``
- ``POST   /workspaces/{ws}/registered-models``         create
- ``GET    /registered-models/{id}``
- ``GET    /registered-models/{id}/versions``
- ``POST   /registered-models/{id}/versions``           promote a Run → new version
- ``POST   /registered-models/{id}/versions/{v}/promote``  → set status
- ``POST   /registered-models/{id}/versions/{v}/rollback`` → swap current_version
- ``DELETE /registered-models/{id}``

Promotion writes a new RegisteredModelVersion row (immutable bytes
pointer) and bumps the parent's ``current_version_id``. Rollback just
flips ``current_version_id`` back to an older row — no retraining.

Deployments reference the version via the FK added in the Phase 4-12
migration. Existing deployments keep working via the legacy
``pipeline_id`` column until each install migrates.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import (
    RegisteredModel,
    RegisteredModelVersion,
    Run,
    Trial,
    get_db,
)

router = APIRouter(tags=["registry"])


# ─────────────────────────────────────────────── serialisers


def _serialise_model(m: RegisteredModel) -> dict:
    return {
        "id": m.id,
        "workspace_id": m.workspace_id,
        "project_id": m.project_id,
        "name": m.name,
        "description": m.description,
        "current_version_id": m.current_version_id,
        "tags": list(m.tags or []),
        "owner_user_id": m.owner_user_id,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "created_by": m.created_by,
    }


def _serialise_version(v: RegisteredModelVersion) -> dict:
    return {
        "id": v.id,
        "registered_model_id": v.registered_model_id,
        "version": v.version,
        "run_id": v.run_id,
        "trial_id": v.trial_id,
        "stored_path": v.stored_path,
        "sha256": v.sha256,
        "size_bytes": v.size_bytes,
        "params": dict(v.params or {}),
        "metrics": dict(v.metrics or {}),
        "status": v.status,
        "promoted_by": v.promoted_by,
        "promoted_at": v.promoted_at.isoformat() if v.promoted_at else None,
        "notes": v.notes,
        "created_at": v.created_at.isoformat() if v.created_at else None,
    }


def _model_access(model_id: str, user, db: Session) -> RegisteredModel:
    m = db.get(RegisteredModel, model_id)
    if m is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "registered model not found")
    _require_access(user, db, m.workspace_id)
    return m


# ─────────────────────────────────────────────── list / create / get


@router.get("/workspaces/{workspace_id}/registered-models")
def list_models(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(RegisteredModel)
        .where(RegisteredModel.workspace_id == workspace_id)
        .order_by(RegisteredModel.created_at.desc())
    ).all()
    return [_serialise_model(m) for m in rows]


@router.post(
    "/workspaces/{workspace_id}/registered-models",
    status_code=status.HTTP_201_CREATED,
)
def create_model(
    workspace_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Body: ``{name, description?, project_id?, tags?}``."""
    _require_access(user, db, workspace_id)
    name = payload.get("name")
    if not name or not isinstance(name, str):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name is required")
    if db.scalar(
        select(RegisteredModel).where(
            RegisteredModel.workspace_id == workspace_id,
            RegisteredModel.name == name,
        )
    ) is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"registered model {name!r} already exists in workspace",
        )
    m = RegisteredModel(
        workspace_id=workspace_id,
        project_id=payload.get("project_id"),
        name=name,
        description=payload.get("description"),
        tags=list(payload.get("tags") or []),
        owner_user_id=payload.get("owner_user_id") or user.id,
        created_by=user.id,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return _serialise_model(m)


@router.get("/registered-models/{model_id}")
def get_model(
    model_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    m = _model_access(model_id, user, db)
    return _serialise_model(m)


@router.delete(
    "/registered-models/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_model(
    model_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete a registered model. Refuses if any version is in
    ``production`` status — un-promote first."""
    m = _model_access(model_id, user, db)
    has_prod = db.scalar(
        select(RegisteredModelVersion).where(
            RegisteredModelVersion.registered_model_id == model_id,
            RegisteredModelVersion.status == "production",
        )
    )
    if has_prod is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "registered model has a version in production; archive it first",
        )
    db.delete(m)
    db.commit()
    return None


# ─────────────────────────────────────────────── versions


@router.get("/registered-models/{model_id}/versions")
def list_versions(
    model_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _model_access(model_id, user, db)
    rows = db.scalars(
        select(RegisteredModelVersion)
        .where(RegisteredModelVersion.registered_model_id == model_id)
        .order_by(RegisteredModelVersion.version.desc())
    ).all()
    return [_serialise_version(v) for v in rows]


#
# NOTE — session-56: the old POST /registered-models/{model_id}/versions
# endpoint was removed. RegisteredModelVersion rows are now created
# automatically by the unified `POST /runs/{run_id}/trials/{trial_id}/promote`
# (see runs.py:promote_trial). The old endpoint was always-broken anyway —
# it read attributes off ``Run`` that live on ``Trial`` — and had zero test
# coverage, so removing it is forward-only.
#


@router.post(
    "/registered-models/{model_id}/versions/{version_id}/promote"
)
def set_version_status(
    model_id: str,
    version_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Flip a version's status (``staging`` → ``production`` → ``archived``).

    Body: ``{status, set_current?}``.
    """
    m = _model_access(model_id, user, db)
    v = db.get(RegisteredModelVersion, version_id)
    if v is None or v.registered_model_id != model_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "version not found")
    new_status = payload.get("status")
    if new_status not in ("staging", "production", "archived"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid status")
    # Only one production version per model at a time — archive any
    # existing prod version when a new one moves in.
    if new_status == "production":
        existing = db.scalars(
            select(RegisteredModelVersion).where(
                RegisteredModelVersion.registered_model_id == model_id,
                RegisteredModelVersion.status == "production",
                RegisteredModelVersion.id != version_id,
            )
        ).all()
        for old in existing:
            old.status = "archived"
    v.status = new_status
    v.promoted_at = datetime.now(UTC)
    v.promoted_by = user.id
    if payload.get("set_current", new_status == "production"):
        m.current_version_id = v.id
    db.commit()
    db.refresh(v)
    return _serialise_version(v)


@router.post(
    "/registered-models/{model_id}/versions/{version_id}/request-promotion",
    status_code=status.HTTP_201_CREATED,
)
def request_promotion(
    model_id: str,
    version_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Open an :class:`ApprovalWorkflow` to promote a version to production.

    The requester is auto-signed as the first approver. When
    ``required_approvals=1`` (the default) the workflow flips straight
    to ``approved`` and the UI can immediately offer the **Execute**
    button. For higher signature counts the workflow stays pending
    until other approvers sign off via the inbox.

    Body (all optional):
      * ``required_approvals`` — N signatures needed. Default 1.
      * ``comment`` — free-form note attached to the requester's signature.
    """
    from pycaret_server.db import ApprovalWorkflow

    m = _model_access(model_id, user, db)
    v = db.get(RegisteredModelVersion, version_id)
    if v is None or v.registered_model_id != model_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "version not found")
    if v.status == "production":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "version is already in production — nothing to approve",
        )

    required = int((payload or {}).get("required_approvals") or 1)
    comment = (payload or {}).get("comment")

    # Auto-sign the requester as approval #1.
    signatures = [
        {
            "user_id": user.id,
            "approved_at": datetime.now(UTC).isoformat(),
            "comment": comment,
        }
    ]
    initial_status = "approved" if len(signatures) >= required else "pending"

    wf = ApprovalWorkflow(
        workspace_id=m.workspace_id,
        target_kind="registered_model_version",
        target_id=version_id,
        action="promote_to_production",
        status=initial_status,
        required_approvals=required,
        approvals=signatures,
        request_payload={
            "registered_model_id": model_id,
            "version": v.version,
            "current_status": v.status,
            "model_id": v.run_id and v.trial_id,
        },
        requested_by=user.id,
    )
    db.add(wf)
    db.commit()
    db.refresh(wf)
    return {
        "id": wf.id,
        "status": wf.status,
        "required_approvals": wf.required_approvals,
        "approvals": list(wf.approvals or []),
        "target_kind": wf.target_kind,
        "target_id": wf.target_id,
        "action": wf.action,
    }


@router.post(
    "/registered-models/{model_id}/versions/{version_id}/rollback"
)
def rollback_to_version(
    model_id: str,
    version_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Flip the parent model's ``current_version_id`` back to this version.

    No retraining; no new bytes. The version becomes ``production`` and
    any prior production version is moved to ``archived``.
    """
    m = _model_access(model_id, user, db)
    v = db.get(RegisteredModelVersion, version_id)
    if v is None or v.registered_model_id != model_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "version not found")
    # Mirror set_version_status semantics: only one prod version.
    others = db.scalars(
        select(RegisteredModelVersion).where(
            RegisteredModelVersion.registered_model_id == model_id,
            RegisteredModelVersion.status == "production",
            RegisteredModelVersion.id != version_id,
        )
    ).all()
    for old in others:
        old.status = "archived"
    v.status = "production"
    v.promoted_at = datetime.now(UTC)
    v.promoted_by = user.id
    m.current_version_id = v.id
    db.commit()
    db.refresh(v)
    return _serialise_version(v)
