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
    PredictionLog,
    Project,
    RegisteredModel,
    RegisteredModelVersion,
    Run,
    Trial,
    get_db,
)
from pycaret_server.serving import get_registry

# Cap on input/output rows persisted per PredictionLog row. Total prediction
# count remains exact via PredictionLog.n_rows; this just bounds storage of
# the actual feature/label payloads.
_MAX_LOG_ROWS = 50

router = APIRouter(tags=["deployments"])

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")


# ---------------------------------------------------------------- serialise


def _serialise_pipeline(p: Pipeline, db: Session | None = None) -> dict:
    # Session-56 UI merge: include the registered_model_{id,version_id}
    # for this Pipeline so the frontend can deep-link to /models/{id}
    # without a second lookup. Pre-session-56 Pipelines (no matching
    # registry row) get nulls — which is the correct historical fact.
    rmv_id: str | None = None
    rm_id: str | None = None
    if db is not None:
        src_trial = db.scalar(
            select(Trial).where(Trial.fitted_pipeline_id == p.id)
        )
        if src_trial is not None:
            from pycaret_server.db import RegisteredModelVersion as _RMV

            rmv = db.scalar(
                select(_RMV).where(_RMV.trial_id == src_trial.id)
            )
            if rmv is not None:
                rmv_id = rmv.id
                rm_id = rmv.registered_model_id
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
        "family_id": p.family_id,
        "version": p.version,
        "created_at": p.created_at,
        "created_by": p.created_by,
        "registered_model_id": rm_id,
        "registered_model_version_id": rmv_id,
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


def _record_predict_metrics(
    db: Session,
    *,
    deployment: Deployment,
    latency_ms: float | None,
    n_rows: int,
    p50: float | None,
    p95: float | None,
    is_error: bool,
) -> None:
    """Phase 10: stamp per-request metric points on the time-series.

    Bucket to the minute so multiple predictions in the same 60s slot
    aggregate cleanly. The Monitoring chart polls the
    ``GET /deployments/{id}/metrics`` endpoint and renders this stream.

    Metrics emitted:
      * ``latency_ms`` — per-request latency on the success path.
      * ``p50_latency_ms`` / ``p95_latency_ms`` — sliding-window
        percentiles maintained by the serving registry.
      * ``requests`` — running count of predict calls.
      * ``error_rate`` — 0 or 1 (averaged in the chart UI's bucket).
    """
    from pycaret_server.db import MetricPoint

    now = datetime.now(UTC).replace(microsecond=0, second=0)
    points: list[tuple[str, float]] = []
    if latency_ms is not None:
        points.append(("latency_ms", float(latency_ms)))
    if p50 is not None:
        points.append(("p50_latency_ms", float(p50)))
    if p95 is not None:
        points.append(("p95_latency_ms", float(p95)))
    points.append(("requests", float(n_rows)))
    points.append(("error_rate", 1.0 if is_error else 0.0))

    for metric, value in points:
        db.add(
            MetricPoint(
                workspace_id=deployment.workspace_id,
                deployment_id=deployment.id,
                metric=metric,
                ts_bucket=now,
                value=value,
                count=1,
            )
        )


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

    # Phase 0-v2: Run-level promote routes through the best Trial.
    # Prefer a pre-Phase-0 ``pipeline_pickle`` Artifact row if one
    # exists (legacy promote path); fall back to the Run's best Trial
    # (``is_best=True``) and use its directly-owned artifact.
    art = db.scalar(
        select(Artifact).where(Artifact.run_id == run_id, Artifact.kind == "pipeline_pickle")
    )
    best_trial = db.scalar(
        select(Trial)
        .where(Trial.run_id == run_id, Trial.is_best.is_(True))
        .limit(1)
    )
    if art is None and (best_trial is None or not best_trial.stored_path):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "run has no promotable artifact (no Trial with is_best=True, and no legacy pipeline_pickle Artifact)",
        )

    name = payload.get("name")
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name is required")

    # Versioning: pipelines that share (workspace_id, name) are revisions of
    # the same logical model. Compute family_id + next version by looking
    # up the latest existing row with that name.
    prior = db.scalar(
        select(Pipeline)
        .where(Pipeline.workspace_id == p.workspace_id, Pipeline.name == str(name))
        .order_by(Pipeline.version.desc())
        .limit(1)
    )
    if prior is None:
        family_id = str(uuid.uuid4())
        version = 1
    else:
        family_id = prior.family_id or prior.id
        version = (prior.version or 1) + 1

    if art is not None:
        stored_path = art.path
        sha256 = art.sha256
        model_id = (r.snapshot or {}).get("model_id")
        params = None
    else:
        stored_path = best_trial.stored_path
        sha256 = best_trial.sha256
        model_id = best_trial.model_id
        params = dict(best_trial.params or {})

    pipe = Pipeline(
        workspace_id=p.workspace_id,
        name=str(name),
        description=payload.get("description"),
        tags=list(payload.get("tags") or []),
        model_id=model_id,
        origin_run_id=run_id,
        stored_path=stored_path,
        sha256=sha256,
        params=params,
        family_id=family_id,
        version=version,
        created_by=user.id,
    )
    db.add(pipe)
    db.commit()
    db.refresh(pipe)
    return _serialise_pipeline(pipe, db)


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
    return [_serialise_pipeline(p, db) for p in rows]


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
    return _serialise_pipeline(p, db)


@router.get("/pipelines/{pipeline_id}/versions")
def list_pipeline_versions(
    pipeline_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Return every revision in the same ``family`` as ``pipeline_id``.

    Versioning model: Pipelines that share ``(workspace_id, name)`` are
    revisions of the same logical model and share a ``family_id``. This
    endpoint lists them in version-descending order so the UI can build a
    "rollback to..." picker.
    """
    p = db.get(Pipeline, pipeline_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "pipeline not found")
    _require_access(user, db, p.workspace_id)

    if p.family_id:
        stmt = (
            select(Pipeline)
            .where(
                Pipeline.workspace_id == p.workspace_id,
                Pipeline.family_id == p.family_id,
            )
            .order_by(Pipeline.version.desc())
        )
    else:
        # Older rows without a family_id — fall back to name match.
        stmt = (
            select(Pipeline)
            .where(
                Pipeline.workspace_id == p.workspace_id,
                Pipeline.name == p.name,
            )
            .order_by(Pipeline.created_at.desc())
        )
    rows = db.scalars(stmt).all()
    return {"family_id": p.family_id, "items": [_serialise_pipeline(r, db) for r in rows]}


@router.get("/pipelines/{pipeline_id}/input-schema")
def pipeline_input_schema(
    pipeline_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Return ``{columns, sample_row, target}`` derived from the origin run.

    Used by the deployment Test panel so the JSON-rows textarea can
    pre-fill with a real holdout row from the dataset this pipeline was
    trained on — instead of a hardcoded iris example.
    """
    p = db.get(Pipeline, pipeline_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "pipeline not found")
    _require_access(user, db, p.workspace_id)

    origin = p.origin_run_id
    if not origin:
        return {"columns": [], "sample_row": {}, "target": None}
    run = db.get(Run, origin)
    if run is None:
        return {"columns": [], "sample_row": {}, "target": None}

    # Reuse the helper from runs.py — single source of truth so trial
    # detail + deployment test panel can't drift.
    from pycaret_server.api.runs import _input_schema_for_run

    schema = _input_schema_for_run(run, db)
    return schema or {"columns": [], "sample_row": {}, "target": None}


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

    # Back-fill registry IDs so deployments created via the legacy pipeline-
    # based endpoint still show up on the Model registry page. Session-56
    # unified promote stamps both Pipeline + RegisteredModelVersion off the
    # same Trial; here we walk Pipeline → Trial → RegisteredModelVersion to
    # recover the pair. Pipelines from pre-unified-promote era won't have
    # a matching version — those Deployments stay registry-less, which is
    # correct (the old Pipeline never had a governance row).
    rmv_id: str | None = None
    rm_id: str | None = None
    src_trial = db.scalar(
        select(Trial).where(Trial.fitted_pipeline_id == p.id)
    )
    if src_trial is not None:
        rmv = db.scalar(
            select(RegisteredModelVersion).where(
                RegisteredModelVersion.trial_id == src_trial.id
            )
        )
        if rmv is not None:
            rmv_id = rmv.id
            rm_id = rmv.registered_model_id

    d = Deployment(
        workspace_id=p.workspace_id,
        pipeline_id=p.id,
        registered_model_id=rm_id,
        registered_model_version_id=rmv_id,
        trial_id=src_trial.id if src_trial is not None else None,
        run_id=p.origin_run_id,
        endpoint_slug=slug,
        status="active",
        auth_mode=auth_mode,
        created_by=user.id,
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return _serialise_deployment(d, get_registry().latency_percentiles(slug))


@router.post(
    "/registered-models/{model_id}/versions/{version_id}/deployments",
    status_code=status.HTTP_201_CREATED,
)
def create_deployment_from_version(
    model_id: str,
    version_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Phase 7: create a Deployment that references a
    ``RegisteredModelVersion`` directly (no legacy pipeline detour).

    Body: ``{endpoint_slug, auth_mode?}``. The new Deployment row has
    ``pipeline_id`` left NULL; ``registered_model_id`` +
    ``registered_model_version_id`` are stamped so a future rollback
    can swap version without re-creating the Deployment.
    """
    m = db.get(RegisteredModel, model_id)
    if m is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "registered model not found"
        )
    _require_access(user, db, m.workspace_id)
    v = db.get(RegisteredModelVersion, version_id)
    if v is None or v.registered_model_id != model_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "version not found")
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
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"endpoint_slug {slug!r} already in use"
        )

    # Session-56 unification: every promote also creates a Pipeline that the
    # predict path keys off (deployments.py:get_predict reads .pkl from
    # Pipeline.id). Walk version → trial → Pipeline to recover it, so a
    # deploy from the registry is fully equivalent to a deploy from the old
    # Pipelines page. Without this back-fill the test-predict panel + the
    # input-schema fetch both fall back to the iris fixture and the actual
    # /predict route 500s with "no saved pipeline".
    pipe_id: str | None = None
    if v.trial_id:
        src_trial = db.get(Trial, v.trial_id)
        if src_trial is not None and src_trial.fitted_pipeline_id:
            pipe_id = src_trial.fitted_pipeline_id

    d = Deployment(
        workspace_id=m.workspace_id,
        pipeline_id=pipe_id,
        registered_model_id=model_id,
        registered_model_version_id=version_id,
        trial_id=v.trial_id,
        run_id=v.run_id,
        endpoint_slug=slug,
        status="active",
        auth_mode=auth_mode,
        created_by=user.id,
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    # Lineage: registered_model_version → deployment.
    try:
        from pycaret_server.api.connections import record_lineage

        record_lineage(
            db,
            workspace_id=m.workspace_id,
            source_kind="registered_model_version",
            source_id=version_id,
            target_kind="deployment",
            target_id=d.id,
            relation="deployed_from",
            metadata={"slug": slug},
        )
    except Exception:  # noqa: BLE001
        pass
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


@router.post("/deployments/{deployment_id}/rollback")
def rollback_deployment(
    deployment_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Repoint a deployment at a different Pipeline in the same family.

    Payload::

        {"pipeline_id": "<uuid>"}

    The target Pipeline must belong to the same workspace AND share the
    deployment's current family_id (or the same name if family_id is unset).
    The in-memory registry is evicted so the next ``/predict`` reloads the
    new artifact.
    """
    d = db.get(Deployment, deployment_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    _require_access(user, db, d.workspace_id)

    target_id = (payload or {}).get("pipeline_id")
    if not target_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "pipeline_id is required")

    target = db.get(Pipeline, target_id)
    if target is None or target.workspace_id != d.workspace_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "target pipeline not in this workspace",
        )

    current = db.get(Pipeline, d.pipeline_id)
    if current is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "underlying pipeline row missing",
        )
    same_family = (
        current.family_id is not None
        and current.family_id == target.family_id
    )
    same_name_fallback = (
        current.family_id is None and current.name == target.name
    )
    if not (same_family or same_name_fallback):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "target pipeline is not in the same family/name as the current pipeline",
        )

    d.pipeline_id = target.id
    db.commit()
    db.refresh(d)
    get_registry().evict(d.endpoint_slug)

    p50, p95 = get_registry().latency_percentiles(d.endpoint_slug)
    return _serialise_deployment(d, (p50, p95))


# --------------------------------------------------------- SERVE (by slug)


def _execute_prediction(
    *,
    d: Deployment,
    payload: dict,
    user: User,
    db: Session,
) -> dict:
    """Run inference against a specific Deployment row.

    Extracted in session-56 so both the slug-based ``/deployments/{slug}/predict``
    route and the new alias-based ``/inference/{ws}/{model}/predict`` route
    share a single implementation. The alias route picks the right Deployment
    (production version) and forwards here.

    All side effects (PredictionLog, metric ticks, latency stats) are
    bucketed against ``d.endpoint_slug`` so monitoring stays accurate
    regardless of which URL the client hit.
    """
    if d.status != "active":
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"deployment status is {d.status!r}; cannot serve"
        )
    if d.auth_mode == "workspace":
        _require_access(user, db, d.workspace_id)

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "payload must include a non-empty 'rows' list",
        )

    if not d.pipeline_id:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "deployment has no pipeline_id — orphaned row",
        )
    pipe_row = db.get(Pipeline, d.pipeline_id)
    if pipe_row is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "underlying pipeline row missing — deployment is orphaned",
        )

    request_id = str(uuid.uuid4())
    try:
        preds, latency = get_registry().predict(d.endpoint_slug, pipe_row.stored_path, rows)
    except Exception as exc:  # noqa: BLE001
        d.error_count += 1
        db.add(
            PredictionLog(
                deployment_id=d.id,
                workspace_id=d.workspace_id,
                request_id=request_id,
                n_rows=len(rows),
                latency_ms=None,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                request_sample=rows[:_MAX_LOG_ROWS],
                response_sample=None,
                user_id=user.id,
            )
        )
        _record_predict_metrics(
            db,
            deployment=d,
            latency_ms=None,
            n_rows=len(rows),
            p50=None,
            p95=None,
            is_error=True,
        )
        db.commit()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"prediction failed: {exc}") from exc

    d.inference_count += len(rows)
    d.last_inference_at = datetime.now(UTC)
    p50, p95 = get_registry().latency_percentiles(d.endpoint_slug)
    d.p50_latency_ms = p50
    d.p95_latency_ms = p95

    db.add(
        PredictionLog(
            deployment_id=d.id,
            workspace_id=d.workspace_id,
            request_id=request_id,
            n_rows=len(rows),
            latency_ms=latency,
            status="ok",
            error=None,
            request_sample=rows[:_MAX_LOG_ROWS],
            response_sample=preds[:_MAX_LOG_ROWS],
            user_id=user.id,
        )
    )
    _record_predict_metrics(
        db,
        deployment=d,
        latency_ms=latency,
        n_rows=len(rows),
        p50=p50,
        p95=p95,
        is_error=False,
    )
    db.commit()

    return {
        "deployment_id": d.id,
        "endpoint_slug": d.endpoint_slug,
        "predictions": preds,
        "latency_ms": latency,
        "request_id": request_id,
    }


@router.post("/deployments/{endpoint_slug}/predict")
def predict(
    endpoint_slug: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Run inference against a specific deployment by its endpoint slug.

    Version-pinned: a slug always serves one specific Pipeline. For an
    "always-production" URL that survives version rollovers, see
    ``POST /inference/{workspace_id}/{model_name}/predict`` below.
    """
    d = db.scalar(select(Deployment).where(Deployment.endpoint_slug == endpoint_slug))
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    return _execute_prediction(d=d, payload=payload, user=user, db=db)


# ─── ALIAS-BASED INFERENCE (MLflow-style "Champion" routing) ────────────


@router.post("/inference/{workspace_id}/{model_name}/predict")
def predict_via_alias(
    workspace_id: str,
    model_name: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Run inference against the current **production** version of a model.

    URL shape mirrors how MLflow / SageMaker model aliases work::

        POST https://<host>/api/v1/inference/<workspace_id>/<model_name>/predict
        body: {"rows": [...]}

    Resolution chain:

    1. ``RegisteredModel`` keyed by ``(workspace_id, model_name)``.
    2. ``current_version_id`` → the production-status version (we keep this
       in lock-step with the status workflow: ``setStatus(...,
       set_current=True)`` when transitioning to production).
    3. The Deployment whose ``registered_model_version_id`` matches. If
       multiple deployments serve the same version (canary / multi-region),
       the most-recently-created active one wins.

    Why this matters: clients code once against a stable URL. When you
    promote v2 → v3 in the registry, no client config has to change — the
    next request hits v3 automatically. This is the foundation for the
    A/B testing, canary, blue/green features that unlock on top of it
    (each one becomes a routing policy on the alias).

    Errors:
      - 404 "model not found" — no RegisteredModel with that name.
      - 404 "no production version" — model has no version in production.
      - 404 "no deployment for production version" — version is in
        production but no Deployment endpoint serves it yet (the user
        needs to click Deploy on the Model Registry page).
    """
    # Note: do NOT use _require_access here on the model itself — the
    # eventual auth check is per-deployment (workspace mode reuses the
    # workspace membership check; api-key / public are V2).
    m = db.scalar(
        select(RegisteredModel).where(
            RegisteredModel.workspace_id == workspace_id,
            RegisteredModel.name == model_name,
        )
    )
    if m is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"model {model_name!r} not found in workspace {workspace_id}",
        )
    if not m.current_version_id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"model {model_name!r} has no production version — promote one first",
        )
    # Resolve the most-recent active deployment for the production version.
    # If multiple deployments serve the same version (canary / multi-region),
    # this picks the latest-created — sufficient for V1. V2 can add an
    # explicit "production-route" tag on the Deployment row.
    d = db.scalar(
        select(Deployment)
        .where(
            Deployment.registered_model_version_id == m.current_version_id,
            Deployment.status == "active",
        )
        .order_by(Deployment.created_at.desc())
        .limit(1)
    )
    if d is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"model {model_name!r} is in production but no active deployment serves "
            "version {m.current_version_id}. Deploy the production version on the "
            "Model Registry page to enable alias routing.",
        )
    return _execute_prediction(d=d, payload=payload, user=user, db=db)


# --------------------------------------------------------- prediction logs


@router.get("/deployments/{deployment_id}/prediction-logs")
def list_prediction_logs(
    deployment_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    limit: int = 100,
    offset: int = 0,
    status_filter: str | None = None,
) -> dict:
    """Paginated read of prediction logs for a deployment.

    Sorted newest-first. ``limit`` clamped to [1, 500]. Use ``status_filter='ok'``
    or ``status_filter='error'`` to scope to a single class.
    """
    d = db.get(Deployment, deployment_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    _require_access(user, db, d.workspace_id)

    limit = max(1, min(500, limit))
    offset = max(0, offset)

    stmt = select(PredictionLog).where(PredictionLog.deployment_id == deployment_id)
    if status_filter in ("ok", "error"):
        stmt = stmt.where(PredictionLog.status == status_filter)
    stmt = stmt.order_by(PredictionLog.created_at.desc()).limit(limit).offset(offset)

    rows = db.scalars(stmt).all()
    return {
        "deployment_id": deployment_id,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "id": r.id,
                "request_id": r.request_id,
                "created_at": r.created_at.isoformat(),
                "n_rows": r.n_rows,
                "latency_ms": r.latency_ms,
                "status": r.status,
                "error": r.error,
                "request_sample": r.request_sample,
                "response_sample": r.response_sample,
                "user_id": r.user_id,
            }
            for r in rows
        ],
    }
