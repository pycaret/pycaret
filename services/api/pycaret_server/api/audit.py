"""Audit-log viewer routes.

Two surfaces:

- ``GET /admin/audit-logs``                  — installation-wide, superuser only.
- ``GET /workspaces/{workspace_id}/audit-logs`` — workspace-scoped, workspace
  admin or superuser.

Both support pagination (``limit``, ``offset``) + filters
(``action``, ``user_id``, ``target_type``, ``target_id``, ``since``, ``until``).

Reads are not themselves audited (that would be infinite recursion once the
admin opens the viewer). The middleware that *writes* rows lives in
``pycaret_server.audit``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_admin
from pycaret_server.auth import CurrentUser
from pycaret_server.auth.deps import require_admin as superuser_required
from pycaret_server.db import AuditLog, Workspace, get_db

router = APIRouter(tags=["audit-logs"])


class AuditLogRead(BaseModel):
    id: str
    workspace_id: str | None
    user_id: str | None
    action: str
    method: str
    path: str
    target_type: str | None
    target_id: str | None
    status_code: int | None
    payload: dict | None
    ip_address: str | None
    user_agent: str | None
    created_at: datetime


def _serialise(row: AuditLog) -> AuditLogRead:
    return AuditLogRead(
        id=row.id,
        workspace_id=row.workspace_id,
        user_id=row.user_id,
        action=row.action,
        method=row.method,
        path=row.path,
        target_type=row.target_type,
        target_id=row.target_id,
        status_code=row.status_code,
        payload=dict(row.payload) if row.payload else None,
        ip_address=row.ip_address,
        user_agent=row.user_agent,
        created_at=row.created_at,
    )


def _apply_filters(
    stmt,
    *,
    action: str | None,
    user_id: str | None,
    target_type: str | None,
    target_id: str | None,
    since: datetime | None,
    until: datetime | None,
):
    """Append optional filter clauses."""
    conds = []
    if action:
        conds.append(AuditLog.action == action)
    if user_id:
        conds.append(AuditLog.user_id == user_id)
    if target_type:
        conds.append(AuditLog.target_type == target_type)
    if target_id:
        conds.append(AuditLog.target_id == target_id)
    if since:
        conds.append(AuditLog.created_at >= since)
    if until:
        conds.append(AuditLog.created_at <= until)
    return stmt.where(and_(*conds)) if conds else stmt


@router.get(
    "/admin/audit-logs",
    response_model=list[AuditLogRead],
    dependencies=[Depends(superuser_required)],
)
def list_audit_logs_admin(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    action: str | None = None,
    user_id: str | None = None,
    workspace_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[AuditLogRead]:
    """Installation-wide audit log, superuser only."""
    stmt = select(AuditLog)
    stmt = _apply_filters(
        stmt,
        action=action,
        user_id=user_id,
        target_type=target_type,
        target_id=target_id,
        since=since,
        until=until,
    )
    if workspace_id:
        stmt = stmt.where(AuditLog.workspace_id == workspace_id)
    stmt = stmt.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
    rows = db.scalars(stmt).all()
    return [_serialise(r) for r in rows]


@router.get(
    "/workspaces/{workspace_id}/audit-logs",
    response_model=list[AuditLogRead],
)
def list_audit_logs_for_workspace(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    action: str | None = None,
    user_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[AuditLogRead]:
    """Workspace-scoped audit log, workspace admin or superuser."""
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_admin(user, db, workspace_id)

    stmt = select(AuditLog).where(AuditLog.workspace_id == workspace_id)
    stmt = _apply_filters(
        stmt,
        action=action,
        user_id=user_id,
        target_type=target_type,
        target_id=target_id,
        since=since,
        until=until,
    )
    stmt = stmt.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
    rows = db.scalars(stmt).all()
    return [_serialise(r) for r in rows]
