"""Experiment template CRUD.

A template captures a known-good ``setup_params`` + plan defaults that a
user can pick from on the New-Experiment screen instead of filling out the
dynamic form from scratch.

Routes:

  GET    /workspaces/{ws_id}/experiment-templates
  POST   /workspaces/{ws_id}/experiment-templates
  GET    /experiment-templates/{id}
  PATCH  /experiment-templates/{id}
  DELETE /experiment-templates/{id}

Workspace member can read; admin to write.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access, _require_admin
from pycaret_server.auth import CurrentUser
from pycaret_server.db import ExperimentTemplate, get_db

router = APIRouter(tags=["experiment-templates"])


class TemplateCreate(BaseModel):
    name: str
    description: str | None = None
    task: str
    setup_params: dict[str, Any]
    plan_params: dict[str, Any] | None = None


class TemplatePatch(BaseModel):
    name: str | None = None
    description: str | None = None
    setup_params: dict[str, Any] | None = None
    plan_params: dict[str, Any] | None = None


class TemplateRead(BaseModel):
    id: str
    workspace_id: str
    name: str
    description: str | None
    task: str
    setup_params: dict[str, Any]
    plan_params: dict[str, Any] | None
    created_at: str | None
    updated_at: str | None


def _serialise(row: ExperimentTemplate) -> TemplateRead:
    return TemplateRead(
        id=row.id,
        workspace_id=row.workspace_id,
        name=row.name,
        description=row.description,
        task=row.task,
        setup_params=dict(row.setup_params or {}),
        plan_params=dict(row.plan_params) if row.plan_params else None,
        created_at=row.created_at.isoformat() if row.created_at else None,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
    )


@router.get("/workspaces/{workspace_id}/experiment-templates")
def list_templates(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    task: str | None = None,
) -> dict:
    _require_access(user, db, workspace_id)
    stmt = select(ExperimentTemplate).where(
        ExperimentTemplate.workspace_id == workspace_id
    )
    if task:
        stmt = stmt.where(ExperimentTemplate.task == task)
    stmt = stmt.order_by(ExperimentTemplate.created_at.desc())
    return {"items": [_serialise(r).model_dump() for r in db.scalars(stmt).all()]}


@router.post(
    "/workspaces/{workspace_id}/experiment-templates",
    status_code=status.HTTP_201_CREATED,
)
def create_template(
    workspace_id: str,
    payload: TemplateCreate,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> TemplateRead:
    _require_admin(user, db, workspace_id)
    row = ExperimentTemplate(
        workspace_id=workspace_id,
        name=payload.name,
        description=payload.description,
        task=payload.task,
        setup_params=payload.setup_params,
        plan_params=payload.plan_params,
        created_by=user.id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _serialise(row)


@router.get("/experiment-templates/{template_id}")
def get_template(
    template_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> TemplateRead:
    row = db.get(ExperimentTemplate, template_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "template not found")
    _require_access(user, db, row.workspace_id)
    return _serialise(row)


@router.patch("/experiment-templates/{template_id}")
def patch_template(
    template_id: str,
    payload: TemplatePatch,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> TemplateRead:
    row = db.get(ExperimentTemplate, template_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "template not found")
    _require_admin(user, db, row.workspace_id)
    if payload.name is not None:
        row.name = payload.name
    if payload.description is not None:
        row.description = payload.description
    if payload.setup_params is not None:
        row.setup_params = payload.setup_params
    if payload.plan_params is not None:
        row.plan_params = payload.plan_params
    db.commit()
    db.refresh(row)
    return _serialise(row)


@router.delete(
    "/experiment-templates/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_template(
    template_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    row = db.get(ExperimentTemplate, template_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "template not found")
    _require_admin(user, db, row.workspace_id)
    db.delete(row)
    db.commit()
