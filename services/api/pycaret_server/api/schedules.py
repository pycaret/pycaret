"""Scheduled-job CRUD.

Routes (under ``/api/v1``):

  GET    /workspaces/{ws_id}/schedules
  POST   /workspaces/{ws_id}/schedules
  GET    /schedules/{id}
  PATCH  /schedules/{id}    (toggle enabled, update cron, update spec)
  DELETE /schedules/{id}
  POST   /schedules/{id}/run-now    (fire the handler immediately, out-of-band)

Two ``kind`` values supported in v1:

  * ``"drift_monitor"`` — schedule.target_id = deployment_id
  * ``"retrain"``       — schedule.target_id = experiment_id; spec carries
                          run-creation params (data_source_id, plan, model_id)

Admin-only writes; any workspace member can read.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access, _require_admin
from pycaret_server.auth import CurrentUser
from pycaret_server.db import Deployment, Experiment, Project, ScheduledJob, get_db
from pycaret_server.scheduler import schedule_job, unschedule_job

router = APIRouter(tags=["schedules"])


_VALID_KINDS = {"drift_monitor", "retrain"}


class ScheduleCreate(BaseModel):
    kind: str
    target_id: str
    schedule: dict[str, Any] = Field(
        description=(
            'One of: {"interval_seconds": int} or {"cron": "0 */6 * * *"} (UTC).'
        )
    )
    spec: dict[str, Any] | None = None
    enabled: bool = True


class SchedulePatch(BaseModel):
    schedule: dict[str, Any] | None = None
    spec: dict[str, Any] | None = None
    enabled: bool | None = None


class ScheduleRead(BaseModel):
    id: str
    workspace_id: str
    kind: str
    target_id: str | None
    schedule: dict[str, Any]
    spec: dict[str, Any] | None
    enabled: bool
    last_run_at: str | None
    last_status: str | None
    last_error: str | None
    last_run_run_id: str | None


def _serialise(row: ScheduledJob) -> ScheduleRead:
    return ScheduleRead(
        id=row.id,
        workspace_id=row.workspace_id,
        kind=row.kind,
        target_id=row.target_id,
        schedule=dict(row.schedule or {}),
        spec=dict(row.spec) if row.spec else None,
        enabled=row.enabled,
        last_run_at=row.last_run_at.isoformat() if row.last_run_at else None,
        last_status=row.last_status,
        last_error=row.last_error,
        last_run_run_id=row.last_run_run_id,
    )


def _validate_target(
    db: Session, kind: str, target_id: str, workspace_id: str
) -> None:
    if kind == "drift_monitor":
        d = db.get(Deployment, target_id)
        if d is None or d.workspace_id != workspace_id:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"deployment {target_id!r} not in workspace {workspace_id!r}",
            )
    elif kind == "retrain":
        e = db.get(Experiment, target_id)
        if e is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"experiment {target_id!r} not found",
            )
        proj = db.get(Project, e.project_id)
        if proj is None or proj.workspace_id != workspace_id:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"experiment {target_id!r} not in workspace {workspace_id!r}",
            )
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown kind {kind!r}; must be one of {sorted(_VALID_KINDS)}",
        )


def _validate_schedule(schedule: dict[str, Any]) -> None:
    if not isinstance(schedule, dict):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "schedule must be an object"
        )
    if "interval_seconds" in schedule:
        try:
            n = int(schedule["interval_seconds"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "schedule.interval_seconds must be an integer",
            ) from exc
        if n < 30:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "schedule.interval_seconds must be >= 30",
            )
    elif "cron" in schedule:
        if not isinstance(schedule["cron"], str) or not schedule["cron"].strip():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "schedule.cron must be a non-empty crontab string",
            )
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "schedule must contain interval_seconds or cron",
        )


@router.get("/workspaces/{workspace_id}/schedules")
def list_schedules(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(ScheduledJob)
        .where(ScheduledJob.workspace_id == workspace_id)
        .order_by(ScheduledJob.created_at.desc())
    ).all()
    return {"items": [_serialise(r).model_dump() for r in rows]}


@router.post(
    "/workspaces/{workspace_id}/schedules",
    status_code=status.HTTP_201_CREATED,
)
def create_schedule(
    workspace_id: str,
    payload: ScheduleCreate,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ScheduleRead:
    _require_admin(user, db, workspace_id)
    if payload.kind not in _VALID_KINDS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown kind {payload.kind!r}; must be one of {sorted(_VALID_KINDS)}",
        )
    _validate_target(db, payload.kind, payload.target_id, workspace_id)
    _validate_schedule(payload.schedule)

    row = ScheduledJob(
        workspace_id=workspace_id,
        kind=payload.kind,
        target_id=payload.target_id,
        schedule=payload.schedule,
        spec=payload.spec,
        enabled=payload.enabled,
        created_by=user.id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    schedule_job(row)
    return _serialise(row)


@router.get("/schedules/{job_id}")
def get_schedule(
    job_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ScheduleRead:
    row = db.get(ScheduledJob, job_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "schedule not found")
    _require_access(user, db, row.workspace_id)
    return _serialise(row)


@router.patch("/schedules/{job_id}")
def patch_schedule(
    job_id: str,
    payload: SchedulePatch,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ScheduleRead:
    row = db.get(ScheduledJob, job_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "schedule not found")
    _require_admin(user, db, row.workspace_id)

    if payload.schedule is not None:
        _validate_schedule(payload.schedule)
        row.schedule = payload.schedule
    if payload.spec is not None:
        row.spec = payload.spec
    if payload.enabled is not None:
        row.enabled = payload.enabled

    db.commit()
    db.refresh(row)
    schedule_job(row)
    return _serialise(row)


@router.delete(
    "/schedules/{job_id}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_schedule(
    job_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    row = db.get(ScheduledJob, job_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "schedule not found")
    _require_admin(user, db, row.workspace_id)
    unschedule_job(row.id)
    db.delete(row)
    db.commit()


@router.post("/schedules/{job_id}/run-now")
def run_schedule_now(
    job_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ScheduleRead:
    """Fire the handler synchronously (skipping the trigger). Admin-only."""
    from pycaret_server.scheduler import JOB_HANDLERS

    row = db.get(ScheduledJob, job_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "schedule not found")
    _require_admin(user, db, row.workspace_id)
    handler = JOB_HANDLERS.get(row.kind)
    if handler is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"no handler registered for kind={row.kind!r}",
        )
    handler(job_id=job_id)
    db.refresh(row)
    return _serialise(row)
