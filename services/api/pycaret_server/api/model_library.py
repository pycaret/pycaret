"""Workspace-scoped Model Library: editable view of the engine model registry.

Routes (all under ``/api/v1``):

  GET    /workspaces/{ws_id}/model-library?task=classification
       Lazy-syncs from the engine on first read of a (workspace, task) pair,
       then serves DB rows. Pass ``?task=`` to scope; omit to return every
       row across every task.
  PATCH  /workspaces/{ws_id}/model-library/{id}
       Toggle ``enabled`` and/or set ``custom_params``. Admin-only.
  POST   /workspaces/{ws_id}/model-library/sync?task=...
       Re-sync a task's library from the current engine version: adds new
       model_ids that don't exist in DB, leaves existing rows alone (so user
       customisations survive engine upgrades). Admin-only.

v1 enforcement: this is a UI-driven catalog. ``compare_models`` and friends
do not yet filter by ``enabled``; respect for the toggle ships in V2 along
with engine support for a custom registry argument.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pycaret.api import list_models
from pycaret.core.tasks import TaskType
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access, _require_admin
from pycaret_server.auth import CurrentUser
from pycaret_server.db import ModelLibrary, get_db

router = APIRouter(tags=["model-library"])


_VALID_TASKS = {t.value for t in TaskType}


# ----------------------------------------------------------------- schemas


class ModelLibraryRowRead(BaseModel):
    id: str
    workspace_id: str
    task_type: str
    model_id: str
    name: str
    enabled: bool
    custom_params: dict | None
    created_at: str | None
    updated_at: str | None


class ModelLibraryPatch(BaseModel):
    enabled: bool | None = None
    custom_params: dict | None = Field(
        default=None,
        description=(
            "Override params used when this model is instantiated. "
            "Pass `{}` to clear; omit to leave unchanged."
        ),
    )


# ----------------------------------------------------------------- helpers


def _serialise(row: ModelLibrary) -> ModelLibraryRowRead:
    return ModelLibraryRowRead(
        id=row.id,
        workspace_id=row.workspace_id,
        task_type=row.task_type,
        model_id=row.model_id,
        name=row.name,
        enabled=row.enabled,
        custom_params=dict(row.custom_params) if row.custom_params else None,
        created_at=row.created_at.isoformat() if row.created_at else None,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
    )


def _sync_task(
    db: Session, workspace_id: str, task_type: str, user_id: str | None
) -> None:
    """Insert ModelLibrary rows for any engine model_id missing in this workspace."""
    try:
        engine_cards = list_models(task_type)
    except Exception:
        return  # task not supported (e.g. legacy task names) — silently no-op.

    existing = set(
        db.scalars(
            select(ModelLibrary.model_id).where(
                ModelLibrary.workspace_id == workspace_id,
                ModelLibrary.task_type == task_type,
            )
        ).all()
    )

    for card in engine_cards:
        if card.id in existing:
            continue
        db.add(
            ModelLibrary(
                workspace_id=workspace_id,
                task_type=task_type,
                model_id=card.id,
                name=card.name,
                enabled=True,
                custom_params=None,
                created_by=user_id,
            )
        )


# ----------------------------------------------------------------- routes


@router.get("/workspaces/{workspace_id}/model-library")
def list_library(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    task: str | None = None,
) -> dict:
    """List the workspace's model library, lazily seeding from the engine on first read."""
    _require_access(user, db, workspace_id)

    tasks_to_serve = [task] if task else sorted(_VALID_TASKS)
    if task and task not in _VALID_TASKS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown task {task!r}; valid: {sorted(_VALID_TASKS)}",
        )

    # Lazy seed: any task with zero rows for this workspace gets populated.
    seeded_any = False
    for t in tasks_to_serve:
        already = db.scalar(
            select(ModelLibrary.id)
            .where(
                ModelLibrary.workspace_id == workspace_id,
                ModelLibrary.task_type == t,
            )
            .limit(1)
        )
        if already is None:
            _sync_task(db, workspace_id, t, user.id)
            seeded_any = True
    if seeded_any:
        db.commit()

    stmt = select(ModelLibrary).where(ModelLibrary.workspace_id == workspace_id)
    if task:
        stmt = stmt.where(ModelLibrary.task_type == task)
    stmt = stmt.order_by(ModelLibrary.task_type.asc(), ModelLibrary.model_id.asc())
    rows = db.scalars(stmt).all()

    return {
        "workspace_id": workspace_id,
        "items": [_serialise(r).model_dump() for r in rows],
    }


@router.patch("/workspaces/{workspace_id}/model-library/{row_id}")
def patch_library(
    workspace_id: str,
    row_id: str,
    payload: ModelLibraryPatch,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ModelLibraryRowRead:
    """Update enabled flag and/or custom_params for a single library row.

    Admin-only.
    """
    _require_admin(user, db, workspace_id)
    row = db.get(ModelLibrary, row_id)
    if row is None or row.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model library row not found")

    if payload.enabled is not None:
        row.enabled = payload.enabled
    if payload.custom_params is not None:
        row.custom_params = dict(payload.custom_params) if payload.custom_params else None

    db.commit()
    db.refresh(row)
    return _serialise(row)


@router.post("/workspaces/{workspace_id}/model-library/sync")
def sync_library(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    task: str | None = None,
) -> dict:
    """Re-sync the engine registry for a task (or all tasks) into this workspace.

    Adds any new model_ids the engine has gained since last sync; preserves
    existing rows so admin customisations survive engine upgrades. Admin-only.
    """
    _require_admin(user, db, workspace_id)
    if task and task not in _VALID_TASKS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown task {task!r}; valid: {sorted(_VALID_TASKS)}",
        )
    tasks = [task] if task else sorted(_VALID_TASKS)
    before = db.scalar(
        select(ModelLibrary.id)
        .where(ModelLibrary.workspace_id == workspace_id)
        .limit(1)
    )
    for t in tasks:
        _sync_task(db, workspace_id, t, user.id)
    db.commit()
    after_count = db.scalar(
        select(ModelLibrary.id)
        .where(ModelLibrary.workspace_id == workspace_id)
        .order_by(ModelLibrary.created_at.desc())
        .limit(1)
    )
    return {
        "workspace_id": workspace_id,
        "synced_tasks": tasks,
        "had_existing_rows": before is not None,
        "latest_row_id": after_count,
    }
