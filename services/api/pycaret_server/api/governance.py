"""Phase 12: governance + approval workflow routes.

Endpoints:

- ``GET    /workspaces/{ws}/approvals``                 list
- ``POST   /workspaces/{ws}/approvals``                 open a new request
- ``POST   /approvals/{id}/approve``                    sign off
- ``POST   /approvals/{id}/reject``                     deny
- ``POST   /approvals/{id}/execute``                    run the gated action

The approval lifecycle is intentionally narrow for v1:

1. A user (or backend, on behalf of a user) opens an Approval row with
   the target+action+payload they want gated.
2. Approvers ``POST .../approve`` until ``len(approvals) >= required``.
3. The original requester (or any admin) calls ``execute`` — backend
   dispatches the action and marks the row ``executed``.

Phase 7's "promote to production" is the canonical use-case; the model
is open enough to extend to deployment delete, project archive, etc.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import ApprovalWorkflow, get_db

router = APIRouter(tags=["governance"])


def _serialise(w: ApprovalWorkflow) -> dict:
    return {
        "id": w.id,
        "workspace_id": w.workspace_id,
        "target_kind": w.target_kind,
        "target_id": w.target_id,
        "action": w.action,
        "status": w.status,
        "required_approvals": w.required_approvals,
        "approvals": list(w.approvals or []),
        "request_payload": w.request_payload,
        "requested_by": w.requested_by,
        "created_at": w.created_at.isoformat() if w.created_at else None,
        "updated_at": w.updated_at.isoformat() if w.updated_at else None,
    }


@router.get("/workspaces/{workspace_id}/approvals")
def list_approvals(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    status_filter: str | None = None,
) -> list[dict]:
    _require_access(user, db, workspace_id)
    q = (
        select(ApprovalWorkflow)
        .where(ApprovalWorkflow.workspace_id == workspace_id)
        .order_by(ApprovalWorkflow.created_at.desc())
    )
    if status_filter:
        q = q.where(ApprovalWorkflow.status == status_filter)
    rows = db.scalars(q).all()
    return [_serialise(w) for w in rows]


@router.post(
    "/workspaces/{workspace_id}/approvals",
    status_code=status.HTTP_201_CREATED,
)
def open_approval(
    workspace_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Body: ``{target_kind, target_id?, action, required_approvals?, request_payload?}``."""
    _require_access(user, db, workspace_id)
    target_kind = payload.get("target_kind")
    action = payload.get("action")
    if not target_kind or not action:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "target_kind + action required",
        )
    w = ApprovalWorkflow(
        workspace_id=workspace_id,
        target_kind=str(target_kind),
        target_id=payload.get("target_id"),
        action=str(action),
        status="pending",
        required_approvals=int(payload.get("required_approvals") or 1),
        approvals=[],
        request_payload=payload.get("request_payload"),
        requested_by=user.id,
    )
    db.add(w)
    db.commit()
    db.refresh(w)
    return _serialise(w)


def _approval_access(approval_id: str, user, db: Session) -> ApprovalWorkflow:
    w = db.get(ApprovalWorkflow, approval_id)
    if w is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "approval not found")
    _require_access(user, db, w.workspace_id)
    return w


@router.post("/approvals/{approval_id}/approve")
def approve(
    approval_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Add the calling user to the approvals list.

    If the row already has enough signatures, status flips to
    ``approved`` (still needs an explicit ``execute`` to take effect).
    """
    w = _approval_access(approval_id, user, db)
    if w.status not in ("pending",):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"approval is {w.status!r}, can't approve",
        )
    existing = list(w.approvals or [])
    if any((a or {}).get("user_id") == user.id for a in existing):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "user has already approved"
        )
    existing.append(
        {
            "user_id": user.id,
            "approved_at": datetime.now(UTC).isoformat(),
            "comment": (payload or {}).get("comment"),
        }
    )
    w.approvals = existing
    if len(existing) >= int(w.required_approvals or 1):
        w.status = "approved"
    db.commit()
    db.refresh(w)
    return _serialise(w)


@router.post("/approvals/{approval_id}/reject")
def reject(
    approval_id: str,
    payload: dict[str, Any],
    user: CurrentUser,  # noqa: ARG001 — used by _approval_access
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    w = _approval_access(approval_id, user, db)
    if w.status not in ("pending", "approved"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"approval is {w.status!r}, can't reject",
        )
    w.status = "rejected"
    existing = list(w.approvals or [])
    existing.append(
        {
            "user_id": user.id,
            "rejected_at": datetime.now(UTC).isoformat(),
            "comment": (payload or {}).get("comment"),
        }
    )
    w.approvals = existing
    db.commit()
    db.refresh(w)
    return _serialise(w)


@router.post("/approvals/{approval_id}/execute")
def execute(
    approval_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Run the gated action. The dispatcher is a thin registry — each
    (target_kind, action) wires up to a callback.

    v1 supports ``registered_model_version.promote_to_production`` —
    flips the version to ``production`` via the registry endpoint's
    logic. Other actions register themselves over time.
    """
    w = _approval_access(approval_id, user, db)
    if w.status != "approved":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"approval is {w.status!r}, must be ``approved`` first",
        )
    try:
        _dispatch_action(db, w, user.id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"action failed: {type(exc).__name__}: {exc}",
        ) from exc
    w.status = "executed"
    db.commit()
    db.refresh(w)
    return _serialise(w)


def _dispatch_action(db: Session, w: ApprovalWorkflow, executor_user_id: str) -> None:
    """Tiny dispatch table for governed actions.

    Add new entries here as Phase 12 grows. Each branch's contract is
    "raise on failure, return on success" — the wrapping endpoint
    converts exceptions to HTTP errors.
    """
    from pycaret_server.db import RegisteredModel, RegisteredModelVersion

    if w.target_kind == "registered_model_version" and w.action == "promote_to_production":
        v = db.get(RegisteredModelVersion, w.target_id) if w.target_id else None
        if v is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, "target version not found"
            )
        m = db.get(RegisteredModel, v.registered_model_id)
        if m is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, "target model not found"
            )
        # Archive any current production version, then flip.
        others = db.scalars(
            select(RegisteredModelVersion).where(
                RegisteredModelVersion.registered_model_id == m.id,
                RegisteredModelVersion.status == "production",
                RegisteredModelVersion.id != v.id,
            )
        ).all()
        for old in others:
            old.status = "archived"
        v.status = "production"
        v.promoted_by = executor_user_id
        v.promoted_at = datetime.now(UTC)
        m.current_version_id = v.id
        return
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        f"no handler for (target_kind={w.target_kind!r}, action={w.action!r})",
    )
