"""Experiment CRUD (nested under a project)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pycaret.core.tasks import TaskType
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import ExperimentCreate, ExperimentResponse, RunStats
from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import Experiment, Project, Run, get_db

router = APIRouter(prefix="/projects/{project_id}/experiments", tags=["experiments"])


def _serialize(e: Experiment, stats: RunStats | None = None) -> ExperimentResponse:
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
        run_stats=stats or RunStats(),
    )


def _run_stats_for_experiment(db: Session, experiment_id: str) -> RunStats:
    """Aggregate runs for a single experiment — used by the detail endpoints."""
    counts = dict(
        db.execute(
            select(Run.status, func.count(Run.id))
            .where(Run.experiment_id == experiment_id)
            .group_by(Run.status)
        ).all()
    )
    latest = db.execute(
        select(Run.status, Run.finished_at, Run.created_at)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.created_at.desc())
        .limit(1)
    ).first()
    return RunStats(
        total=sum(counts.values()),
        succeeded=counts.get("succeeded", 0),
        failed=counts.get("failed", 0),
        cancelled=counts.get("cancelled", 0),
        running=counts.get("running", 0),
        queued=counts.get("queued", 0),
        last_status=latest[0] if latest else None,
        last_finished_at=(latest[1] if latest else None),
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
    items = db.scalars(
        select(Experiment).where(Experiment.project_id == project_id)
    ).all()
    if not items:
        return []

    exp_ids = [e.id for e in items]

    # One GROUP BY query for all (experiment_id, status) counts — single
    # round-trip regardless of experiment count.
    counts: dict[str, dict[str, int]] = {}
    for exp_id, status_val, cnt in db.execute(
        select(Run.experiment_id, Run.status, func.count(Run.id))
        .where(Run.experiment_id.in_(exp_ids))
        .group_by(Run.experiment_id, Run.status)
    ).all():
        counts.setdefault(exp_id, {})[status_val] = cnt

    # Most-recent run per experiment for the "last status" chip. SQLAlchemy
    # doesn't have a portable window-function helper that's nice on SQLite,
    # so we just fetch every run id + created_at for these experiments and
    # walk it client-side — cheap, since runs-per-experiment is bounded by
    # what a single user submits.
    latest: dict[str, tuple] = {}
    for exp_id, run_status, finished_at, created_at in db.execute(
        select(Run.experiment_id, Run.status, Run.finished_at, Run.created_at)
        .where(Run.experiment_id.in_(exp_ids))
        .order_by(Run.experiment_id, Run.created_at.desc())
    ).all():
        if exp_id not in latest:
            latest[exp_id] = (run_status, finished_at)

    def stats_for(exp_id: str) -> RunStats:
        c = counts.get(exp_id, {})
        last = latest.get(exp_id)
        return RunStats(
            total=sum(c.values()),
            succeeded=c.get("succeeded", 0),
            failed=c.get("failed", 0),
            cancelled=c.get("cancelled", 0),
            running=c.get("running", 0),
            queued=c.get("queued", 0),
            last_status=last[0] if last else None,
            last_finished_at=last[1] if last else None,
        )

    return [_serialize(e, stats_for(e.id)) for e in items]


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
    return _serialize(e, _run_stats_for_experiment(db, experiment_id))


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
