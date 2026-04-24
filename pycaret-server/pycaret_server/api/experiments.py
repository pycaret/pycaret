"""Experiment CRUD (nested under a project)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pycaret.core.tasks import TaskType
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import ExperimentCreate, ExperimentResponse
from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import Experiment, Project, get_db

router = APIRouter(prefix="/projects/{project_id}/experiments", tags=["experiments"])


def _serialize(e: Experiment) -> ExperimentResponse:
    return ExperimentResponse(
        id=e.id,
        project_id=e.project_id,
        name=e.name,
        task=e.task,
        target=e.target,
        setup_params=dict(e.setup_params or {}),
        data_source_id=e.data_source_id,
        created_at=e.created_at,
        created_by=e.created_by,
    )


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    project_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[ExperimentResponse]:
    p = db.get(Project, project_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    _require_access(user, db, p.workspace_id)
    items = db.scalars(select(Experiment).where(Experiment.project_id == project_id)).all()
    return [_serialize(e) for e in items]


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(
    project_id: str,
    payload: ExperimentCreate,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ExperimentResponse:
    p = db.get(Project, project_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    _require_access(user, db, p.workspace_id)

    # Validate task
    try:
        TaskType(payload.task)
    except ValueError as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"invalid task {payload.task!r}; must be one of {[t.value for t in TaskType]}",
        ) from e

    if (
        db.scalar(
            select(Experiment).where(
                Experiment.project_id == project_id,
                Experiment.name == payload.name,
            )
        )
        is not None
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"experiment {payload.name!r} already exists in project"
        )

    e = Experiment(
        project_id=project_id,
        name=payload.name,
        task=payload.task,
        target=payload.target,
        setup_params=dict(payload.setup_params),
        data_source_id=payload.data_source_id,
        created_by=user.id,
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return _serialize(e)


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    project_id: str,
    experiment_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ExperimentResponse:
    e = db.get(Experiment, experiment_id)
    if e is None or e.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "experiment not found")
    p = db.get(Project, project_id)
    _require_access(user, db, p.workspace_id)
    return _serialize(e)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(
    project_id: str,
    experiment_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    e = db.get(Experiment, experiment_id)
    if e is None or e.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "experiment not found")
    p = db.get(Project, project_id)
    _require_access(user, db, p.workspace_id)
    db.delete(e)
    db.commit()
