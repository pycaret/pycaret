"""Pipeline registry + Deployment CRUD + in-house serving.

Three concerns in one router because they're tightly coupled:

1. **Pipelines** — a fitted pipeline that came out of a Run can be *promoted*
   into a workspace-scoped `pipelines` row. Pipelines are reusable across
   projects (via `pipeline_project_links`).

2. **Deployments** — a Deployment is a slug-addressable wrapper around a
   Pipeline. Auth mode, status, basic metrics all live on the row.

3. **Serving** — ``POST /api/v1/deployments/{slug}/predict`` dispatches by
   slug through `DeploymentRegistry`. Auth is workspace-scoped for v1.

Route surface (all under ``/api/v1``):

- ``POST   /runs/{run_id}/promote``            — promote a Run's pipeline artifact to a Pipeline row.
- ``GET    /workspaces/{ws_id}/pipelines``     — list.
- ``GET    /pipelines/{id}``                   — fetch.
- ``DELETE /pipelines/{id}``                   — delete.
- ``POST   /pipelines/{id}/deployments``       — create a Deployment.
- ``GET    /workspaces/{ws_id}/deployments``   — list.
- ``GET    /deployments/{id}``                 — fetch (with latency + counts).
- ``DELETE /deployments/{id}``                 — delete.
- ``POST   /deployments/{slug}/predict``       — SERVE.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import (
    Artifact,
    Deployment,
    Experiment,
    Pipeline,
    Project,
    Run,
    get_db,
)
from pycaret_server.serving import get_registry

router = APIRouter(tags=["deployments"])

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")


# ---------------------------------------------------------------- serialise


def _serialise_pipeline(p: Pipeline) -> dict:
    return {
        "id": p.id,
        "workspace_id": p.workspace_id,
        "name": p.name,
        "description": p.description,
        "tags": list(p.tags or []),
        "model_id": p.model_id,
        "origin_run_id": p.origin_run_id,
        "stored_path": p.stored_path,
        "sha256": p.sha256,
        "params": dict(p.params or {}),
        "created_at": p.created_at,
        "created_by": p.created_by,
    }


def _serialise_deployment(d: Deployment, latencies: tuple[float | None, float | None]) -> dict:
    return {
        "id": d.id,
        "workspace_id": d.workspace_id,
        "pipeline_id": d.pipeline_id,
        "endpoint_slug": d.endpoint_slug,
        "status": d.status,
        "auth_mode": d.auth_mode,
        "inference_count": d.inference_count,
        "last_inference_at": d.last_inference_at,
        "p50_latency_ms": latencies[0],
        "p95_latency_ms": latencies[1],
        "error_count": d.error_count,
        "created_at": d.created_at,
        "created_by": d.created_by,
    }


# ----------------------------------------------------- promote Run -> Pipeline


@router.post(
    "/runs/{run_id}/promote",
    status_code=status.HTTP_201_CREATED,
)
def promote_run(
    run_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Promote the ``pipeline_pickle`` artifact of a succeeded Run to a
    workspace-scoped Pipeline row.

    Payload::

        {"name": "iris-v1", "description": "...", "tags": ["baseline"]}
    """
    r = db.get(Run, run_id)
    if r is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    e = db.get(Experiment, r.experiment_id)
    p = db.get(Project, e.project_id)
    _require_access(user, db, p.workspace_id)

    if r.status != "succeeded":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"run status is {r.status!r}; only succeeded runs can be promoted",
        )

    art = db.scalar(
        select(Artifact).where(Artifact.run_id == run_id, Artifact.kind == "pipeline_pickle")
    )
    if art is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "run has no pipeline_pickle artifact")

    name = payload.get("name")
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name is required")

    pipe = Pipeline(
        workspace_id=p.workspace_id,
        name=str(name),
        description=payload.get("description"),
        tags=list(payload.get("tags") or []),
        model_id=(r.snapshot or {}).get("model_id"),
        origin_run_id=run_id,
        stored_path=art.path,
        sha256=art.sha256,
        params=None,
        created_by=user.id,
    )
    db.add(pipe)
    db.commit()
    db.refresh(pipe)
    return _serialise_pipeline(pipe)


# ---------------------------------------------------------------- pipelines


@router.get("/workspaces/{workspace_id}/pipelines")
def list_pipelines(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(Pipeline)
        .where(Pipeline.workspace_id == workspace_id)
        .order_by(Pipeline.created_at.desc())
    ).all()
    return [_serialise_pipeline(p) for p in rows]


@router.get("/pipelines/{pipeline_id}")
def get_pipeline(
    pipeline_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    p = db.get(Pipeline, pipeline_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "pipeline not found")
    _require_access(user, db, p.workspace_id)
    return _serialise_pipeline(p)


@router.delete("/pipelines/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_pipeline(
    pipeline_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    p = db.get(Pipeline, pipeline_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "pipeline not found")
    _require_access(user, db, p.workspace_id)
    # RESTRICT on deployments.pipeline_id means the DB will refuse this delete
    # if any deployment still references it; surface that as a 409.
    if db.scalar(select(Deployment).where(Deployment.pipeline_id == pipeline_id)):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "cannot delete pipeline while deployments still reference it",
        )
    db.delete(p)
    db.commit()


# ---------------------------------------------------------------- deployments


@router.post(
    "/pipelines/{pipeline_id}/deployments",
    status_code=status.HTTP_201_CREATED,
)
def create_deployment(
    pipeline_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Create a Deployment. Payload::

    {"endpoint_slug": "iris-v1", "auth_mode": "workspace"}
    """
    p = db.get(Pipeline, pipeline_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "pipeline not found")
    _require_access(user, db, p.workspace_id)

    slug = payload.get("endpoint_slug")
    if not slug or not _SLUG_RE.match(slug):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "endpoint_slug must match [a-z0-9][a-z0-9-]{1,62}[a-z0-9]",
        )
    auth_mode = payload.get("auth_mode", "workspace")
    if auth_mode not in ("workspace", "api-key", "public"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "auth_mode must be one of workspace | api-key | public",
        )

    if db.scalar(select(Deployment).where(Deployment.endpoint_slug == slug)):
        raise HTTPException(status.HTTP_409_CONFLICT, f"endpoint_slug {slug!r} already in use")

    d = Deployment(
        workspace_id=p.workspace_id,
        pipeline_id=p.id,
        endpoint_slug=slug,
        status="active",
        auth_mode=auth_mode,
        created_by=user.id,
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return _serialise_deployment(d, get_registry().latency_percentiles(slug))


@router.get("/workspaces/{workspace_id}/deployments")
def list_deployments(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(Deployment)
        .where(Deployment.workspace_id == workspace_id)
        .order_by(Deployment.created_at.desc())
    ).all()
    reg = get_registry()
    return [_serialise_deployment(d, reg.latency_percentiles(d.endpoint_slug)) for d in rows]


@router.get("/deployments/{deployment_id}")
def get_deployment(
    deployment_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    d = db.get(Deployment, deployment_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    _require_access(user, db, d.workspace_id)
    return _serialise_deployment(d, get_registry().latency_percentiles(d.endpoint_slug))


@router.delete("/deployments/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_deployment(
    deployment_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    d = db.get(Deployment, deployment_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    _require_access(user, db, d.workspace_id)
    get_registry().evict(d.endpoint_slug)
    db.delete(d)
    db.commit()


# --------------------------------------------------------- SERVE (by slug)


@router.post("/deployments/{endpoint_slug}/predict")
def predict(
    endpoint_slug: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Run inference against a deployed pipeline.

    Payload::

        {"rows": [{"sepal length (cm)": 5.1, ...}, ...]}

    Returns::

        {"predictions": [{"index": 0, "prediction": 0}, ...],
         "latency_ms": 3.14,
         "deployment_id": "<uuid>"}
    """
    d = db.scalar(select(Deployment).where(Deployment.endpoint_slug == endpoint_slug))
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    if d.status != "active":
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"deployment status is {d.status!r}; cannot serve"
        )
    # Access — workspace mode for v1. (api-key + public bypass in future.)
    if d.auth_mode == "workspace":
        _require_access(user, db, d.workspace_id)

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "payload must include a non-empty 'rows' list"
        )

    pipe_row = db.get(Pipeline, d.pipeline_id)
    if pipe_row is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "underlying pipeline row missing — deployment is orphaned",
        )

    try:
        preds, latency = get_registry().predict(endpoint_slug, pipe_row.stored_path, rows)
    except Exception as exc:  # noqa: BLE001
        d.error_count += 1
        db.commit()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"prediction failed: {exc}") from exc

    # Tick counters.
    d.inference_count += len(rows)
    d.last_inference_at = datetime.now(UTC)
    p50, p95 = get_registry().latency_percentiles(endpoint_slug)
    d.p50_latency_ms = p50
    d.p95_latency_ms = p95
    db.commit()

    return {
        "deployment_id": d.id,
        "endpoint_slug": endpoint_slug,
        "predictions": preds,
        "latency_ms": latency,
        "request_id": str(uuid.uuid4()),
    }
