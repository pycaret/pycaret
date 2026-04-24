"""Drift report CRUD — distribution-shift snapshots for a Deployment.

Routes under ``/api/v1/``:

  POST   /deployments/{deployment_id}/drift-reports   record a snapshot
  GET    /deployments/{deployment_id}/drift-reports   list snapshots
  GET    /drift-reports/{id}                          single snapshot

v1 semantics:

- There is **no scheduled drift-detection job** yet. That requires a real
  prediction log + Job queue runner, both of which land post-4.0.0. For
  now, drift reports are created on explicit POST — from a CI job hitting
  the API with an ``X-PyCaret-Key`` header, from a notebook, or from the
  UI "Record snapshot" button.
- The caller is responsible for computing feature drift + prediction
  drift upstream. We accept what they submit verbatim + bucket the
  ``drift_status`` from ``drift_score`` server-side so the label is
  consistent.
- The ``drift_analysis`` LLM consultation reads these rows + suggests
  RETRAIN NOW / INVESTIGATE / MONITOR / NO ACTION.

SPEC references: § 4.12 (DriftReport schema), § 11.2 (drift monitoring),
§ 12.2 (drift analyst copilot).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import Deployment, DriftReport, get_db

router = APIRouter(tags=["drift"])


DriftStatus = Literal["none", "mild", "moderate", "severe"]


def bucket_status(score: float) -> DriftStatus:
    """Map a 0..1 drift score to a bucketed label.

    Thresholds chosen to align with common PSI convention (0.1 / 0.25).
    Below 0.1 → none; 0.1–0.25 → mild; 0.25–0.4 → moderate; above → severe.
    Clamped to [0, 1] defensively.
    """
    s = max(0.0, min(1.0, score))
    if s < 0.10:
        return "none"
    if s < 0.25:
        return "mild"
    if s < 0.40:
        return "moderate"
    return "severe"


class DriftReportCreate(BaseModel):
    window_start: datetime
    window_end: datetime
    drift_score: float = Field(ge=0.0, le=1.0)
    feature_drift_json: dict = Field(
        default_factory=dict,
        description=(
            "Per-feature drift values. Shape: {feature_name: "
            "{score: float, kind: 'psi'|'ks'|'chi2'|'missing_rate'}}."
        ),
    )
    prediction_drift_json: dict | None = Field(
        default=None,
        description=(
            "Prediction distribution shift. Shape: {kind: 'js'|'ks', "
            "score: float, baseline_mean?: float, current_mean?: float}."
        ),
    )
    sample_size: int | None = Field(default=None, ge=0)
    baseline_artifact_id: str | None = None


class DriftReportRead(BaseModel):
    id: str
    deployment_id: str
    baseline_artifact_id: str | None
    window_start: datetime
    window_end: datetime
    drift_score: float
    drift_status: DriftStatus
    feature_drift_json: dict
    prediction_drift_json: dict | None
    sample_size: int | None
    created_at: datetime
    created_by: str


def _serialise(r: DriftReport) -> DriftReportRead:
    return DriftReportRead(
        id=r.id,
        deployment_id=r.deployment_id,
        baseline_artifact_id=r.baseline_artifact_id,
        window_start=r.window_start,
        window_end=r.window_end,
        drift_score=r.drift_score,
        drift_status=r.drift_status,  # type: ignore[arg-type]
        feature_drift_json=dict(r.feature_drift_json or {}),
        prediction_drift_json=dict(r.prediction_drift_json) if r.prediction_drift_json else None,
        sample_size=r.sample_size,
        created_at=r.created_at,
        created_by=r.created_by,
    )


# ─────────────────────────────────────────────────────────────── routes


@router.post(
    "/deployments/{deployment_id}/drift-reports",
    response_model=DriftReportRead,
    status_code=status.HTTP_201_CREATED,
)
def create_drift_report(
    deployment_id: str,
    payload: DriftReportCreate,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> DriftReportRead:
    """Record a drift snapshot for a Deployment."""
    dep = db.get(Deployment, deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    _require_access(user, db, dep.workspace_id)

    if payload.window_end < payload.window_start:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "window_end must be >= window_start",
        )

    row = DriftReport(
        deployment_id=deployment_id,
        baseline_artifact_id=payload.baseline_artifact_id,
        window_start=payload.window_start,
        window_end=payload.window_end,
        drift_score=payload.drift_score,
        drift_status=bucket_status(payload.drift_score),
        feature_drift_json=payload.feature_drift_json or {},
        prediction_drift_json=payload.prediction_drift_json,
        sample_size=payload.sample_size,
        created_by=user.id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _serialise(row)


@router.get(
    "/deployments/{deployment_id}/drift-reports",
    response_model=list[DriftReportRead],
)
def list_drift_reports(
    deployment_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    limit: int = 50,
) -> list[DriftReportRead]:
    """List drift reports for a deployment (newest first)."""
    dep = db.get(Deployment, deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    _require_access(user, db, dep.workspace_id)

    rows = db.scalars(
        select(DriftReport)
        .where(DriftReport.deployment_id == deployment_id)
        .order_by(DriftReport.created_at.desc())
        .limit(max(1, min(limit, 500)))
    ).all()
    return [_serialise(r) for r in rows]


@router.get("/drift-reports/{report_id}", response_model=DriftReportRead)
def get_drift_report(
    report_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> DriftReportRead:
    """Single drift report — includes the full feature / prediction JSON."""
    row = db.get(DriftReport, report_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "drift report not found")
    dep = db.get(Deployment, row.deployment_id)
    if dep is None:  # deployment removed but report still here; fall back to deny
        raise HTTPException(status.HTTP_404_NOT_FOUND, "owning deployment missing")
    _require_access(user, db, dep.workspace_id)
    return _serialise(row)
