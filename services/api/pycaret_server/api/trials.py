"""Phase 0-v2 trial-centric read API.

The legacy ``/runs/{run_id}/trials/...`` routes are the primary surface
(every Trial belongs to one Run via ``Trial.run_id``). These endpoints
exist as a convenience for the cross-experiment views — "every Trial
in this Experiment" — and for direct deep-linking by trial id.

Endpoints:

- ``GET    /experiments/{experiment_id}/trials``
- ``GET    /trials/{trial_id}``                          (direct fetch)
- ``PATCH  /trials/{trial_id}``                          (name, notes)
- ``DELETE /trials/{trial_id}``                          (refuses if promoted)
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import Experiment, Project, Trial, get_db

router = APIRouter(tags=["trials"])


def _experiment_workspace(experiment_id: str, user, db: Session) -> tuple[Experiment, str]:
    e = db.get(Experiment, experiment_id)
    if e is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "experiment not found")
    p = db.get(Project, e.project_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    _require_access(user, db, p.workspace_id)
    return e, p.workspace_id


def _trial_workspace(trial_id: str, user, db: Session) -> Trial:
    t = db.get(Trial, trial_id)
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "trial not found")
    _require_access(user, db, t.workspace_id)
    return t


def _serialise(t: Trial) -> dict:
    return {
        "id": t.id,
        "run_id": t.run_id,
        "experiment_id": t.experiment_id,
        "workspace_id": t.workspace_id,
        "name": t.name,
        "model_id": t.model_id,
        "kind": t.kind or "compare",
        "status": t.status,
        "rank": t.rank,
        "is_best": bool(t.is_best),
        "metrics": dict(t.metrics or {}),
        "params": dict(t.params or {}),
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "finished_at": t.finished_at.isoformat() if t.finished_at else None,
        "duration_ms": t.duration_ms,
        "error": t.error,
        "stored_path": t.stored_path,
        "sha256": t.sha256,
        "size_bytes": t.size_bytes,
        "has_artifact": bool(t.stored_path),
        "parent_trial_ids": t.parent_trial_ids or [],
        "created_by_action_id": t.created_by_action_id,
        "fitted_pipeline_id": t.fitted_pipeline_id,
        "notes": t.notes,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
    }


@router.get("/experiments/{experiment_id}/trials")
def list_experiment_trials(
    experiment_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    kind: str | None = None,
    run_id: str | None = None,
    limit: int = 200,
) -> dict:
    """Every Trial under an Experiment, optionally narrowed.

    Filters:
      - ``kind`` — ``compare`` | ``tuned`` | ``ensembled`` | ``blended`` | ``stacked`` | ``manual``.
      - ``run_id`` — limit to Trials in a single Run.
      - ``limit`` — cap at 1000 (default 200).
    """
    _experiment_workspace(experiment_id, user, db)
    bounded = max(1, min(int(limit), 1000))
    q = (
        select(Trial)
        .where(Trial.experiment_id == experiment_id)
        .order_by(Trial.created_at.desc())
        .limit(bounded)
    )
    if kind:
        q = q.where(Trial.kind == kind)
    if run_id:
        q = q.where(Trial.run_id == run_id)
    rows = db.scalars(q).all()
    return {
        "experiment_id": experiment_id,
        "items": [_serialise(t) for t in rows],
    }


@router.get("/trials/{trial_id}")
def get_trial_direct(
    trial_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Fetch a Trial by id, no Run umbrella required."""
    return _serialise(_trial_workspace(trial_id, user, db))


@router.patch("/trials/{trial_id}")
def patch_trial_direct(
    trial_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Mutate ``name`` / ``notes``. Other fields are workflow-derived."""
    t = _trial_workspace(trial_id, user, db)
    if "name" in payload:
        name = payload["name"]
        if name is not None and not isinstance(name, str):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "name must be a string or null"
            )
        t.name = name if name else None
    if "notes" in payload:
        notes = payload["notes"]
        if notes is not None and not isinstance(notes, str):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "notes must be a string or null"
            )
        t.notes = notes if notes else None
    db.commit()
    db.refresh(t)
    return _serialise(t)


@router.delete("/trials/{trial_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trial(
    trial_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete a Trial. Refused if a Pipeline still references it via
    ``fitted_pipeline_id`` — un-promote first."""
    t = _trial_workspace(trial_id, user, db)
    if t.fitted_pipeline_id:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "trial is promoted to a Pipeline — un-promote it before deleting",
        )
    db.delete(t)
    db.commit()
    return None


__all__ = ["router"]
