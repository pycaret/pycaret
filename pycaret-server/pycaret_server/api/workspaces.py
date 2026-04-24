"""Workspace CRUD.

For v1, workspace visibility is determined by membership — users only see
workspaces where they have a ``WorkspaceMember`` row. Admins (superuser) see
all workspaces.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import WorkspaceCreate, WorkspaceResponse
from pycaret_server.auth import CurrentUser
from pycaret_server.db import Workspace, WorkspaceMember, get_db

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


def _serialize(ws: Workspace) -> WorkspaceResponse:
    return WorkspaceResponse(
        id=ws.id,
        name=ws.name,
        description=ws.description,
        created_at=ws.created_at,
        created_by=ws.created_by,
    )


@router.get("", response_model=list[WorkspaceResponse])
def list_workspaces(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[WorkspaceResponse]:
    """List workspaces the current user has access to."""
    if user.is_superuser:
        q = select(Workspace)
    else:
        q = (
            select(Workspace)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
            .where(WorkspaceMember.user_id == user.id)
        )
    return [_serialize(w) for w in db.scalars(q).all()]


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(
    payload: WorkspaceCreate,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> WorkspaceResponse:
    """Create a new workspace. The creator becomes its admin automatically."""
    if db.scalar(select(Workspace).where(Workspace.name == payload.name)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"workspace {payload.name!r} already exists")
    ws = Workspace(
        name=payload.name,
        description=payload.description,
        created_by=user.id,
    )
    db.add(ws)
    db.flush()
    db.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="admin"))
    db.commit()
    db.refresh(ws)
    return _serialize(ws)


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> WorkspaceResponse:
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, ws.id)
    return _serialize(ws)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workspace(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_admin(user, db, ws.id)
    db.delete(ws)
    db.commit()


# ---------------------------------------------------------------- helpers


def _require_access(user, db: Session, workspace_id: str) -> WorkspaceMember | None:
    """Raise 403 unless user is superuser or a member of the workspace.

    Returns the membership row when present (handy for role checks).
    """
    if user.is_superuser:
        return None
    m = db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user.id,
        )
    )
    if m is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not a member of this workspace")
    return m


def _require_admin(user, db: Session, workspace_id: str) -> None:
    """Raise 403 unless user is superuser OR workspace admin."""
    if user.is_superuser:
        return
    m = _require_access(user, db, workspace_id)
    if m is None or m.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "workspace admin required")
