"""Project CRUD (nested under a workspace)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import ProjectCreate, ProjectResponse
from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import Project, Workspace, get_db

router = APIRouter(prefix="/workspaces/{workspace_id}/projects", tags=["projects"])


def _serialize(p: Project) -> ProjectResponse:
    return ProjectResponse(
        id=p.id,
        workspace_id=p.workspace_id,
        name=p.name,
        description=p.description,
        tags=list(p.tags or []),
        created_at=p.created_at,
        created_by=p.created_by,
    )


@router.get("", response_model=list[ProjectResponse])
def list_projects(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[ProjectResponse]:
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, workspace_id)
    projects = db.scalars(select(Project).where(Project.workspace_id == workspace_id)).all()
    return [_serialize(p) for p in projects]


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    workspace_id: str,
    payload: ProjectCreate,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ProjectResponse:
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, workspace_id)

    if (
        db.scalar(
            select(Project).where(
                Project.workspace_id == workspace_id,
                Project.name == payload.name,
            )
        )
        is not None
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, f"project {payload.name!r} already exists")

    p = Project(
        workspace_id=workspace_id,
        name=payload.name,
        description=payload.description,
        tags=payload.tags,
        created_by=user.id,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return _serialize(p)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    workspace_id: str,
    project_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ProjectResponse:
    _require_access(user, db, workspace_id)
    p = db.get(Project, project_id)
    if p is None or p.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    return _serialize(p)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    workspace_id: str,
    project_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    _require_access(user, db, workspace_id)
    p = db.get(Project, project_id)
    if p is None or p.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    db.delete(p)
    db.commit()
