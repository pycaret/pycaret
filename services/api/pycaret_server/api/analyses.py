"""Phase 11: Analysis (statistical procedures) routes.

Endpoints:

- ``GET    /analysis-kinds``                              registered procedures
- ``GET    /projects/{project_id}/analyses``              list
- ``POST   /projects/{project_id}/analyses``              create (without running)
- ``GET    /analyses/{id}``
- ``PATCH  /analyses/{id}``
- ``DELETE /analyses/{id}``
- ``POST   /analyses/{id}/run``                           execute → result
- ``POST   /analyses/run-once``                           transient: run + return without persist
- ``GET    /analyses/{id}/results``                       past Run rows for this analysis

An Analysis Run reuses the Phase 0 ``runs`` table (kind-agnostic Run
already supports a generic ``metrics`` JSON + ``params`` JSON). The
result envelope (see :class:`AnalysisResult`) lands in ``Run.metrics``;
``Run.params`` carries the analysis input params. No new tables for
the result; everything reuses the existing Run history surface.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.analyses import AnalysisProcedureError, list_kinds, run_analysis
from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import Analysis, DataSource, Project, Run, get_db

router = APIRouter(tags=["analyses"])


def _serialise(a: Analysis) -> dict:
    return {
        "id": a.id,
        "workspace_id": a.workspace_id,
        "project_id": a.project_id,
        "name": a.name,
        "description": a.description,
        "kind": a.kind,
        "params": dict(a.params or {}),
        "data_source_id": a.data_source_id,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "created_by": a.created_by,
    }


def _project_access(project_id: str, user, db: Session) -> Project:
    p = db.get(Project, project_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    _require_access(user, db, p.workspace_id)
    return p


def _analysis_access(analysis_id: str, user, db: Session) -> Analysis:
    a = db.get(Analysis, analysis_id)
    if a is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "analysis not found")
    _require_access(user, db, a.workspace_id)
    return a


def _resolve_df(
    db: Session, data_source_id: str | None, inline_csv: str | None
) -> pd.DataFrame:
    """Load the DataFrame for an analysis run.

    Prefers ``inline_csv`` (raw CSV text in the request) for quick
    ad-hoc analyses; falls back to the DataSource's driver. Future
    cuts will pull through the Dataset version snapshot URI for
    reproducibility.
    """
    if inline_csv:
        return pd.read_csv(io.StringIO(inline_csv))
    if not data_source_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "either inline_csv or data_source_id is required",
        )
    from pycaret_server.datasources import get_driver

    ds = db.get(DataSource, data_source_id)
    if ds is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "data source not found")
    driver = get_driver(ds.kind)
    try:
        from pycaret_server.api.connections import resolve_datasource_secret

        return driver.read_full(
            config=dict(ds.config or {}),
            secret_value=resolve_datasource_secret(db, ds),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"failed to read data source: {exc}",
        ) from exc


# ─────────────────────────────────────────────── kinds


@router.get("/analysis-kinds")
def analysis_kinds() -> dict:
    """List the registered analysis procedure kinds. The UI's
    New-Analysis wizard renders one card per kind."""
    return {"kinds": list_kinds()}


# ─────────────────────────────────────────────── analysis CRUD


@router.get("/projects/{project_id}/analyses")
def list_analyses(
    project_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _project_access(project_id, user, db)
    rows = db.scalars(
        select(Analysis)
        .where(Analysis.project_id == project_id)
        .order_by(Analysis.created_at.desc())
    ).all()
    return [_serialise(a) for a in rows]


@router.post(
    "/projects/{project_id}/analyses",
    status_code=status.HTTP_201_CREATED,
)
def create_analysis(
    project_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Body: ``{name, kind, params, description?, data_source_id?}``."""
    p = _project_access(project_id, user, db)
    name = payload.get("name")
    kind = payload.get("kind")
    if not name or not kind:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "name + kind required"
        )
    if kind not in list_kinds():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown analysis kind {kind!r}; available: {list_kinds()}",
        )
    a = Analysis(
        workspace_id=p.workspace_id,
        project_id=project_id,
        name=str(name),
        description=payload.get("description"),
        kind=str(kind),
        params=dict(payload.get("params") or {}),
        data_source_id=payload.get("data_source_id"),
        created_by=user.id,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return _serialise(a)


@router.get("/analyses/{analysis_id}")
def get_analysis(
    analysis_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    return _serialise(_analysis_access(analysis_id, user, db))


@router.patch("/analyses/{analysis_id}")
def patch_analysis(
    analysis_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    a = _analysis_access(analysis_id, user, db)
    if "name" in payload:
        a.name = str(payload["name"])
    if "description" in payload:
        a.description = payload["description"]
    if "params" in payload and isinstance(payload["params"], dict):
        a.params = dict(payload["params"])
    db.commit()
    db.refresh(a)
    return _serialise(a)


@router.delete(
    "/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_analysis(
    analysis_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    a = _analysis_access(analysis_id, user, db)
    db.delete(a)
    db.commit()
    return None


# ─────────────────────────────────────────────── run


@router.post("/analyses/{analysis_id}/run")
def run_persisted_analysis(
    analysis_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    payload: dict[str, Any] | None = None,
) -> dict:
    """Execute the analysis and persist the result as a Run row.

    Body: ``{params_override?, inline_csv?, data_source_id?}``.
    Overrides merge over the persisted ``Analysis.params``.

    The Run gets ``experiment_id`` set to ``analysis:<analysis_id>`` so
    the existing Run history endpoints can still page it; the
    ``triggered_by`` is ``analysis``. ``metrics`` carries the full
    :class:`AnalysisResult.to_dict()` envelope.
    """
    body = payload or {}
    a = _analysis_access(analysis_id, user, db)
    params = {**dict(a.params or {}), **dict(body.get("params_override") or {})}
    df = _resolve_df(
        db,
        body.get("data_source_id") or a.data_source_id,
        body.get("inline_csv"),
    )
    started = datetime.now(UTC)
    try:
        result = run_analysis(a.kind, df, params)
    except AnalysisProcedureError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"analysis failed: {type(exc).__name__}: {exc}",
        ) from exc
    finished = datetime.now(UTC)

    # Persist as a Run row keyed by a synthetic experiment id so the
    # existing per-experiment Run listing works for analyses too.
    run = Run(
        experiment_id=f"analysis:{analysis_id}",
        trial_id=None,
        sequence=None,
        status="succeeded",
        started_at=started,
        finished_at=finished,
        duration_ms=(finished - started).total_seconds() * 1000,
        metrics=result.to_dict(),
        params=params,
        triggered_by="analysis",
        triggered_by_id=analysis_id,
        created_by=user.id,
        snapshot={
            "kind": a.kind,
            "data_source_id": body.get("data_source_id") or a.data_source_id,
        },
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return {
        "analysis_id": analysis_id,
        "run_id": run.id,
        "duration_ms": run.duration_ms,
        "result": result.to_dict(),
    }


@router.post("/analyses/run-once")
def run_transient_analysis(
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Execute an analysis without persisting it.

    Body: ``{kind, params, inline_csv? | data_source_id?}``. Useful
    for the UI's "preview" button before saving.
    """
    kind = payload.get("kind")
    if kind not in list_kinds():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown analysis kind {kind!r}; available: {list_kinds()}",
        )
    df = _resolve_df(db, payload.get("data_source_id"), payload.get("inline_csv"))
    try:
        result = run_analysis(kind, df, dict(payload.get("params") or {}))
    except AnalysisProcedureError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return {"kind": kind, "result": result.to_dict()}


@router.get("/analyses/{analysis_id}/results")
def list_results(
    analysis_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    limit: int = 50,
) -> list[dict]:
    """Past Run rows for this analysis (most recent first)."""
    _analysis_access(analysis_id, user, db)
    rows = db.scalars(
        select(Run)
        .where(Run.experiment_id == f"analysis:{analysis_id}")
        .order_by(Run.created_at.desc())
        .limit(max(1, min(int(limit), 500)))
    ).all()
    return [
        {
            "run_id": r.id,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "duration_ms": r.duration_ms,
            "metrics": r.metrics or {},
            "params": dict(r.params or {}),
        }
        for r in rows
    ]
