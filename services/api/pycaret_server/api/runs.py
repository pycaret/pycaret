"""Run lifecycle routes.

Two prefixes mounted from the same router:

- ``/api/v1/experiments/{experiment_id}/runs`` — submit + list runs for an experiment.
- ``/api/v1/runs/{run_id}``                    — fetch, cancel, events, WebSocket.

A ``POST`` to the experiments-nested path enqueues a `RunOrchestrator` job and
returns 202 with a Run row already persisted in ``status="queued"``. The
caller polls ``GET /api/v1/runs/{run_id}`` or opens the WebSocket for live
event fan-out.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse
from pycaret.core.tasks import TaskType
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import EventResponse, RunCreate, RunResponse
from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.auth.tokens import decode_token
from datetime import UTC, datetime

from pycaret_server.db import (
    DataSource,
    Event,
    Experiment,
    Pipeline,
    Project,
    RegisteredModel,
    RegisteredModelVersion,
    Run,
    Trial,
    User,
    get_db,
)
from pycaret_server.runs.broker import event_broker
from pycaret_server.runs.dispatch import dispatch_run
from pycaret_server.runs.orchestrator import RunSpec, get_orchestrator

router = APIRouter(tags=["runs"])


# ------------------------------------------------------------------- helpers


def _serialise_run(r: Run, db: Session | None = None) -> RunResponse:
    """Serialise a Run row.

    When ``db`` is provided we walk Experiment → Project to resolve
    ``project_id`` + ``workspace_id`` so deep-linked clients (the React
    sidebar's `/runs/:id` route) recover their context without a chain of
    extra requests.
    """
    project_id: str | None = None
    workspace_id: str | None = None
    if db is not None:
        exp = db.get(Experiment, r.experiment_id)
        if exp is not None:
            project_id = exp.project_id
            proj = db.get(Project, exp.project_id)
            if proj is not None:
                workspace_id = proj.workspace_id
    return RunResponse(
        id=r.id,
        experiment_id=r.experiment_id,
        project_id=project_id,
        workspace_id=workspace_id,
        status=r.status,
        started_at=r.started_at,
        finished_at=r.finished_at,
        duration_ms=r.duration_ms,
        error=r.error,
        leaderboard=r.leaderboard,
        metrics_summary=r.metrics_summary,
        snapshot=r.snapshot,
        created_at=r.created_at,
        created_by=r.created_by,
    )


def _serialise_event(e: Event) -> EventResponse:
    return EventResponse(
        id=e.id,
        run_id=e.run_id,
        kind=e.kind,
        message=e.message,
        payload=e.payload,
        duration_ms=e.duration_ms,
        emitted_at=e.emitted_at,
    )


def _experiment_access(experiment_id: str, user, db: Session) -> Experiment:
    """Resolve experiment + verify the user can read its workspace."""
    e = db.get(Experiment, experiment_id)
    if e is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "experiment not found")
    p = db.get(Project, e.project_id)
    _require_access(user, db, p.workspace_id)
    return e


def _run_access(run_id: str, user, db: Session) -> Run:
    r = db.get(Run, run_id)
    if r is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    e = db.get(Experiment, r.experiment_id)
    p = db.get(Project, e.project_id)
    _require_access(user, db, p.workspace_id)
    return r


# ------------------------------------------------------- submit + list


@router.post(
    "/experiments/{experiment_id}/runs",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_run(
    experiment_id: str,
    payload: RunCreate,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> RunResponse:
    """Enqueue a run. Returns 202 with a Run row in status='queued'."""
    e = _experiment_access(experiment_id, user, db)
    return _serialise_run(dispatch_run(db, e, payload, user_id=user.id), db)


@router.post("/runs/{run_id}/cancel", response_model=RunResponse)
def cancel_run(
    run_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> RunResponse:
    """Cooperatively cancel a queued or running run.

    Signals the orchestrator's `threading.Event`; the worker picks it up at the
    next stage boundary. If the run is already in a terminal state, the current
    row is returned unchanged.
    """
    r = _run_access(run_id, user, db)
    if r.status in ("succeeded", "failed", "cancelled"):
        return _serialise_run(r, db)
    get_orchestrator().cancel(run_id)
    # Don't refresh — the worker may still be running. The status flip happens
    # when the checkpoint trips; the client can poll or watch the WS for it.
    return _serialise_run(r, db)


@router.get(
    "/experiments/{experiment_id}/runs",
    response_model=list[RunResponse],
)
def list_runs(
    experiment_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[RunResponse]:
    _experiment_access(experiment_id, user, db)
    rows = db.scalars(
        select(Run).where(Run.experiment_id == experiment_id).order_by(Run.created_at.desc())
    ).all()
    return [_serialise_run(r, db) for r in rows]


# ---------------------------------------------------------------- fetch + events


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(
    run_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> RunResponse:
    return _serialise_run(_run_access(run_id, user, db), db)


def _trial_response_dict(trial: Trial, rank_override: int | None = None) -> dict:
    """Serialise a Phase 0-v2 Trial row directly — no compat shim.

    Run owns Trials; each Trial carries its own metrics + artifact +
    status. The response matches the existing wire shape the UI uses
    (so the type-side change is minimal): adds ``status``,
    ``started_at``, ``finished_at``, ``duration_ms``, ``error``
    for the new per-Trial lifecycle.
    """
    return {
        "id": trial.id,
        "run_id": trial.run_id,
        "model_id": trial.model_id,
        "name": trial.name,
        "rank": rank_override if rank_override is not None else trial.rank,
        "metrics": dict(trial.metrics or {}),
        "is_best": bool(trial.is_best),
        "status": trial.status,
        "started_at": trial.started_at.isoformat() if trial.started_at else None,
        "finished_at": trial.finished_at.isoformat() if trial.finished_at else None,
        "duration_ms": trial.duration_ms,
        "error": trial.error,
        "fitted_pipeline_id": trial.fitted_pipeline_id,
        "has_artifact": bool(trial.stored_path),
        "size_bytes": trial.size_bytes,
        "kind": trial.kind or "compare",
        "parent_trial_ids": trial.parent_trial_ids or [],
        "created_at": trial.created_at.isoformat() if trial.created_at else None,
    }


@router.get("/runs/{run_id}/trials")
def list_trials(
    run_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Return the trials this Run contains.

    Phase 0-v2: trials are owned by the Run directly via
    ``Trial.run_id``. Sorted by ``rank`` when set (after Run finishes)
    and by ``created_at`` otherwise.
    """
    _run_access(run_id, user, db)
    trials = db.scalars(
        select(Trial)
        .where(Trial.run_id == run_id)
        .order_by(
            Trial.rank.asc().nulls_last(),
            Trial.created_at.asc(),
        )
    ).all()
    return {
        "run_id": run_id,
        "items": [
            _trial_response_dict(t, rank_override=idx + 1 if t.rank is None else None)
            for idx, t in enumerate(trials)
        ],
    }


def _trial_access(run_id: str, trial_id: str, user, db: Session) -> tuple[Run, Trial]:
    """Resolve (run, trial). The Trial must belong to the Run via ``run_id``."""
    r = _run_access(run_id, user, db)
    t = db.get(Trial, trial_id)
    if t is None or t.run_id != run_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "trial not found")
    return r, t


@router.get("/runs/{run_id}/trials/{trial_id}")
def get_trial(
    run_id: str,
    trial_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Full detail for a single trial.

    Phase 0-v2: the Trial owns its artifact + metrics + params + status
    directly. No more Run-1 indirection.
    """
    r, t = _trial_access(run_id, trial_id, user, db)

    stored_path = t.stored_path
    metrics = dict(t.metrics or {})
    params = dict(t.params or {})
    sha256 = t.sha256
    size_bytes = t.size_bytes

    pipeline_steps: list[dict] = []
    pipeline_tree: dict | None = None
    if stored_path:
        try:
            pipeline_steps = _extract_pipeline_steps(stored_path)
        except Exception:  # noqa: BLE001 — never let a bad pickle 500 the page
            pipeline_steps = []
        try:
            pipeline_tree = _extract_pipeline_tree(stored_path)
        except Exception:  # noqa: BLE001
            pipeline_tree = None

    snap = r.snapshot or {}
    task = (snap.get("task") or snap.get("task_type") or "").lower() or None
    available_plots: list[str] = []
    if task:
        try:
            from pycaret_server.api.plots import _PLOT_REGISTRY

            available_plots = list(_PLOT_REGISTRY.get(task, {}).keys())
        except Exception:  # noqa: BLE001
            available_plots = []

    # Input schema + a real sample row from the holdout dataset so the
    # Predict tab can prefill the textarea with something the pipeline
    # actually accepts. Cheap-ish — one CSV read per detail call.
    input_schema = _input_schema_for_run(r, db)

    return {
        "id": t.id,
        "run_id": t.run_id,
        "workspace_id": t.workspace_id,
        "model_id": t.model_id,
        "name": t.name,
        "rank": t.rank,
        "status": t.status,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "finished_at": t.finished_at.isoformat() if t.finished_at else None,
        "duration_ms": t.duration_ms,
        "error": t.error,
        "metrics": metrics,
        "is_best": bool(t.is_best),
        "fitted_pipeline_id": t.fitted_pipeline_id,
        "params": dict(params),
        "pipeline_steps": pipeline_steps,
        "pipeline_tree": pipeline_tree,
        "available_plots": available_plots,
        "task": task,
        "run_snapshot": snap or {},
        "notes": t.notes,
        "input_schema": input_schema,
        "stored_path": stored_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "has_artifact": bool(stored_path),
        "kind": t.kind or "compare",
        "parent_trial_ids": t.parent_trial_ids or [],
        "created_at": t.created_at.isoformat() if t.created_at else None,
    }


def _input_schema_for_run(run: Run, db: Session) -> dict | None:
    """Return ``{columns: [{name, dtype}], sample_row: {...}, target}``.

    Reads the run's holdout dataset, drops the target, and returns one
    real row plus column types. The Predict tab prefills its textarea
    from this so the user can run inference without knowing the schema.
    """
    from pycaret_server.api.plots import _load_dataset_frame

    snap = run.snapshot or {}
    df = _load_dataset_frame(snap, db)
    if df is None:
        return None
    target = snap.get("target")
    if not target and snap.get("sklearn_dataset"):
        try:
            from pycaret_server.runs.plans import load_sklearn_dataset

            _, target = load_sklearn_dataset(snap["sklearn_dataset"])
        except Exception:  # noqa: BLE001
            pass
    feature_df = df.drop(columns=[target]) if target and target in df.columns else df

    columns: list[dict] = []
    sample_row: dict = {}
    for col in feature_df.columns:
        s = feature_df[col]
        columns.append({"name": str(col), "dtype": str(s.dtype)})
        if len(s) > 0:
            v = s.dropna().iloc[0] if s.dropna().shape[0] else s.iloc[0]
            sample_row[str(col)] = _jsonable(v)
    return {
        "columns": columns,
        "sample_row": sample_row,
        "target": target if target else None,
    }


def _extract_pipeline_tree(stored_path: str) -> dict | None:
    """Recursively serialize the fitted pipeline as a tree the UI can render.

    The structure is intentionally typed so the React diagram knows what
    layout primitive to use:

    - ``type: "pipeline"`` — a sklearn ``Pipeline``: ``children`` holds its
      named steps in execution order.
    - ``type: "column_transformer"`` — sklearn ``ColumnTransformer``:
      ``branches`` holds parallel transformers, each tagged with the
      column list it operates on.
    - ``type: "feature_union"`` — sklearn ``FeatureUnion``: ``branches`` holds
      transformers concatenated horizontally.
    - ``type: "leaf"`` — any other estimator/transformer (terminal node).
    - ``type: "passthrough"`` — the literal strings ``"passthrough"`` /
      ``"drop"`` returned by ColumnTransformer's "remainder" knob.

    Always JSON-safe.
    """
    try:
        import joblib

        obj = joblib.load(_uri_to_local_path(stored_path))
    except Exception:  # noqa: BLE001
        return None
    return _serialise_estimator(name="root", est=obj, is_root=True)


def _safe_params(est) -> dict:
    """Return ``estimator.get_params(deep=False)`` filtered to JSON-safe values."""
    import json

    try:
        raw = est.get_params(deep=False)
    except Exception:  # noqa: BLE001
        return {}
    out: dict = {}
    for k, v in raw.items():
        try:
            json.dumps(v)
            out[k] = v
        except Exception:  # noqa: BLE001
            out[k] = repr(v)
    return out


def _serialise_estimator(name: str, est, *, is_root: bool = False) -> dict:
    """Recursive worker for :func:`_extract_pipeline_tree`."""
    # Passthrough / drop — ColumnTransformer's remainder slot can hold these
    # as bare strings, not estimator instances.
    if isinstance(est, str) and est in ("passthrough", "drop"):
        return {
            "type": "passthrough",
            "name": name,
            "class": est,
            "module": "",
            "params": {},
        }

    cls = type(est)
    base = {
        "name": name,
        "class": cls.__name__,
        "module": cls.__module__,
        "params": _safe_params(est),
    }

    # sklearn.pipeline.Pipeline (and FeatureUnion uses transformer_list).
    if hasattr(est, "steps") and isinstance(getattr(est, "steps"), list):
        children = [_serialise_estimator(n, sub) for n, sub in est.steps]
        if children:
            children[-1]["is_estimator"] = True
        return {
            **base,
            "type": "pipeline",
            "is_root": is_root,
            "children": children,
        }

    # sklearn.compose.ColumnTransformer — fitted has ``transformers_``;
    # unfitted falls back to ``transformers``.
    transformers = (
        getattr(est, "transformers_", None) or getattr(est, "transformers", None)
    )
    if transformers and isinstance(transformers, list):
        branches: list[dict] = []
        for entry in transformers:
            try:
                sub_name, sub_est, columns = entry
            except (TypeError, ValueError):
                continue
            if hasattr(columns, "tolist"):
                col_list = list(columns.tolist())
            elif isinstance(columns, slice):
                col_list = [str(columns)]
            else:
                try:
                    col_list = list(columns)
                except TypeError:
                    col_list = [str(columns)]
            # Filter to JSON-safe primitives.
            col_list = [c if isinstance(c, (str, int, float, bool)) else str(c) for c in col_list]
            branches.append(
                {
                    "name": sub_name,
                    "columns": col_list,
                    "transformer": _serialise_estimator(sub_name, sub_est),
                }
            )
        return {
            **base,
            "type": "column_transformer",
            "is_root": is_root,
            "branches": branches,
        }

    # sklearn.pipeline.FeatureUnion — has ``transformer_list``.
    tl = getattr(est, "transformer_list", None)
    if tl and isinstance(tl, list):
        branches = [
            {
                "name": sub_name,
                "transformer": _serialise_estimator(sub_name, sub_est),
            }
            for sub_name, sub_est in tl
            if sub_est is not None
        ]
        return {
            **base,
            "type": "feature_union",
            "is_root": is_root,
            "branches": branches,
        }

    return {**base, "type": "leaf", "is_root": is_root}


def _extract_pipeline_steps(stored_path: str) -> list[dict]:
    """Walk the fitted Pipeline and serialize each step.

    Returns a list ordered the way sklearn applied them. Final entry is
    flagged ``is_estimator=True``. Param values run through the same
    JSON-or-repr filter used at training time so the response stays
    serializable.
    """
    import json

    try:
        import joblib

        obj = joblib.load(_uri_to_local_path(stored_path))
    except Exception:  # noqa: BLE001
        return []

    raw_steps: list[tuple[str, object]] = []
    if hasattr(obj, "steps") and obj.steps:
        raw_steps = list(obj.steps)
    else:
        # Bare estimator (no Pipeline wrapper) — present it as a single step.
        raw_steps = [("model", obj)]

    out: list[dict] = []
    for idx, (name, step) in enumerate(raw_steps):
        cls = type(step)
        try:
            raw_params = step.get_params(deep=False)
        except Exception:  # noqa: BLE001
            raw_params = {}
        params: dict = {}
        for k, v in raw_params.items():
            try:
                json.dumps(v)
                params[k] = v
            except Exception:  # noqa: BLE001
                params[k] = repr(v)
        out.append(
            {
                "index": idx,
                "name": name,
                "class": cls.__name__,
                "module": cls.__module__,
                "params": params,
                "is_estimator": idx == len(raw_steps) - 1,
            }
        )
    return out


@router.get("/runs/{run_id}/trials/{trial_id}/download")
def download_trial(
    run_id: str,
    trial_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> FileResponse:
    """Stream the trial's pickled fitted Pipeline as ``<model_id>.pkl``.

    Phase 2: stored_path is a URI (``file://`` or ``s3://``). For local-fs
    we serve via FileResponse to keep auth + range-request behavior; for
    cloud back-ends we'd 302 to a presigned URL (left for a future cut).
    """
    from pycaret_server.storage import get_object_store

    _, t = _trial_access(run_id, trial_id, user, db)
    if not t.stored_path:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "trial has no stored pipeline (older runs predate trial-level artifacts)",
        )
    store = get_object_store()
    stored_uri = t.stored_path
    if store.scheme != "file":
        presigned = store.presigned_url(stored_uri)
        if presigned:
            from fastapi.responses import RedirectResponse

            return RedirectResponse(presigned)  # type: ignore[return-value]
    # Local-fs path — resolve URI to filesystem path and stream.
    if stored_uri.startswith("file://"):
        from urllib.parse import urlparse as _up

        raw = _up(stored_uri).path
        if raw.startswith("/") and len(raw) > 3 and raw[2] == ":":
            raw = raw[1:]
        p = Path(raw)
    else:
        p = Path(stored_uri)
    if not p.is_file():
        raise HTTPException(
            status.HTTP_410_GONE, "trial pipeline file no longer exists on disk"
        )
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in t.model_id) or "trial"
    return FileResponse(
        path=str(p),
        media_type="application/octet-stream",
        filename=f"{safe}.pkl",
    )


@router.post(
    "/runs/{run_id}/trials/{trial_id}/promote",
    status_code=status.HTTP_201_CREATED,
)
def promote_trial(
    run_id: str,
    trial_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Promote a trial — unified Pipeline + RegisteredModel write.

    Pre-session-56 there were two promote paths (Pipeline only, and a
    separately-mounted "register as new version" form). The Phase-7 split
    was leaky: the registry form silently AttributeError'd on real data
    and the two parallel surfaces produced disjoint Pipelines / Registry
    tables. We now do both in one atomic transaction:

    1. Write a Pipeline row (legacy contract for serving + rollback).
    2. Find-or-create a RegisteredModel keyed by ``(workspace_id, name)``;
       write a new RegisteredModelVersion off the trial's bytes; bump the
       RegisteredModel.current_version_id.
    3. Drop a lineage edge run → registered_model_version.

    Pipeline.version and RegisteredModelVersion.version are kept in lock-
    step so the two tables read the same number for the same artifact —
    the user doesn't need to mentally map them.
    """
    r, t = _trial_access(run_id, trial_id, user, db)
    if t.status != "succeeded":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"trial status is {t.status!r}; only succeeded trials can be promoted",
        )
    if not t.stored_path or not t.sha256:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "trial has no stored pipeline artifact",
        )

    name = payload.get("name")
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name is required")

    # ── Pipeline (legacy artifact pointer used by predict / rollback)
    prior = db.scalar(
        select(Pipeline)
        .where(
            Pipeline.workspace_id == t.workspace_id,
            Pipeline.name == str(name),
        )
        .order_by(Pipeline.version.desc())
        .limit(1)
    )
    if prior is None:
        family_id = str(uuid.uuid4())
        version = 1
    else:
        family_id = prior.family_id or prior.id
        version = (prior.version or 1) + 1

    pipe = Pipeline(
        workspace_id=t.workspace_id,
        name=str(name),
        description=payload.get("description"),
        tags=list(payload.get("tags") or []),
        model_id=t.model_id,
        origin_run_id=run_id,
        stored_path=t.stored_path,
        sha256=t.sha256,
        params=dict(t.params or {}),
        family_id=family_id,
        version=version,
        created_by=user.id,
    )
    db.add(pipe)
    # Flush so SQLAlchemy generates ``pipe.id`` — the UUID default fires at
    # INSERT time, not at construction, so without this the back-link below
    # would persist NULL and the UI would never show the trial as promoted.
    db.flush()
    t.fitted_pipeline_id = pipe.id

    # ── RegisteredModel + RegisteredModelVersion (governance layer)
    rm = db.scalar(
        select(RegisteredModel).where(
            RegisteredModel.workspace_id == t.workspace_id,
            RegisteredModel.name == str(name),
        )
    )
    if rm is None:
        rm = RegisteredModel(
            workspace_id=t.workspace_id,
            name=str(name),
            description=payload.get("description"),
            tags=list(payload.get("tags") or []),
            created_by=user.id,
        )
        db.add(rm)
        db.flush()

    rmv = RegisteredModelVersion(
        registered_model_id=rm.id,
        version=version,  # kept in lock-step with Pipeline.version
        run_id=run_id,
        trial_id=trial_id,
        stored_path=t.stored_path,
        sha256=t.sha256,
        size_bytes=t.size_bytes,
        params=dict(t.params or {}),
        metrics=dict(t.metrics or {}),
        status="staging",
        promoted_by=user.id,
        promoted_at=datetime.now(UTC),
        notes=payload.get("notes"),
    )
    db.add(rmv)
    db.flush()
    rm.current_version_id = rmv.id

    db.commit()
    db.refresh(pipe)
    db.refresh(rmv)

    # ── Lineage (best-effort; logs WARNING on failure, never raises)
    try:
        from pycaret_server.api.connections import record_lineage

        record_lineage(
            db,
            workspace_id=t.workspace_id,
            source_kind="run",
            source_id=run_id,
            target_kind="registered_model_version",
            target_id=rmv.id,
            relation="produced",
            metadata={"trial_id": trial_id, "version": version, "pipeline_id": pipe.id},
        )
    except Exception:  # noqa: BLE001 — defensive: lineage is decorative here
        pass

    return {
        "id": pipe.id,
        "workspace_id": pipe.workspace_id,
        "name": pipe.name,
        "model_id": pipe.model_id,
        "origin_run_id": pipe.origin_run_id,
        "version": pipe.version,
        "family_id": pipe.family_id,
        "stored_path": pipe.stored_path,
        "sha256": pipe.sha256,
        "params": dict(pipe.params or {}),
        "created_at": pipe.created_at,
        # New: registry IDs are surfaced so the UI doesn't need a follow-up call
        # to know where to send the user next ("View on Model registry →").
        "registered_model_id": rm.id,
        "registered_model_version_id": rmv.id,
    }


# ─────────────────────────────────────────────── un-promote


@router.delete(
    "/runs/{run_id}/trials/{trial_id}/promote",
    status_code=status.HTTP_204_NO_CONTENT,
)
def unpromote_trial(
    run_id: str,
    trial_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Reverse a previous promote.

    Symmetric with the unified promote: clears the Trial back-link, deletes
    the Pipeline row, AND deletes the matching RegisteredModelVersion. The
    parent RegisteredModel is preserved (other versions may still exist; if
    not, the user can delete it explicitly).

    If a Deployment still references either the Pipeline or the version,
    returns 409 — the user must tear those down first.
    """
    from pycaret_server.db import Deployment

    _r, t = _trial_access(run_id, trial_id, user, db)
    pipe_id = t.fitted_pipeline_id
    if not pipe_id:
        # Already not promoted — idempotent 204.
        return None

    pipe = db.get(Pipeline, pipe_id)
    if pipe is not None:
        in_use = db.scalar(
            select(Deployment).where(Deployment.pipeline_id == pipe_id)
        )
        if in_use is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "pipeline has active deployments — delete those first, "
                "then withdraw the promotion",
            )

    # Matching RegisteredModelVersion was created off this same Trial.
    rmv = db.scalar(
        select(RegisteredModelVersion).where(
            RegisteredModelVersion.trial_id == trial_id
        )
    )
    if rmv is not None:
        in_use_rmv = db.scalar(
            select(Deployment).where(
                Deployment.registered_model_version_id == rmv.id
            )
        )
        if in_use_rmv is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "registered model version has active deployments — delete "
                "those first, then withdraw the promotion",
            )

    t.fitted_pipeline_id = None
    if pipe is not None:
        db.delete(pipe)
    if rmv is not None:
        # If this was the parent's current_version_id, clear it first so the
        # FK doesn't dangle. (current_version_id uses ON DELETE SET NULL, but
        # being explicit avoids the post-delete reload weirdness.)
        rm = db.get(RegisteredModel, rmv.registered_model_id)
        if rm is not None and rm.current_version_id == rmv.id:
            rm.current_version_id = None
        db.delete(rmv)
    db.commit()
    return None


# ─────────────────────────────────────────────── follow-on actions
#
# Tune / Ensemble / Blend / Stack run against existing trial(s) and land
# their result as a *new trial* in the same Run with ``kind`` set and
# ``parent_trial_ids`` back-linking the source(s). No new Run row is
# created — events stream into the source run's existing event channel so
# whatever drawer the user has open keeps updating.


def _spec_data_fields_from_run(run: Run, db: Session) -> dict:
    """Pull the data-resolution slice out of a Run's snapshot.

    The snapshot stores ``data_source_id`` (a FK to ``data_sources``) —
    not the absolute CSV path the orchestrator's ``_load_data`` needs.
    Resolve here so the follow-on worker can re-fit the experiment on the
    same dataset the source compare run used.
    """
    snap = run.snapshot or {}
    out: dict = {}
    if snap.get("sklearn_dataset"):
        out["sklearn_dataset"] = snap["sklearn_dataset"]
    if snap.get("data_inline"):
        out["data_inline"] = snap["data_inline"]
    # Direct path (rare — set explicitly by some tests).
    if snap.get("data_source_path"):
        out["data_source_path"] = snap["data_source_path"]
    # The common path: resolve the DataSource row's ``config.path``.
    elif snap.get("data_source_id"):
        ds = db.get(DataSource, snap["data_source_id"])
        if ds is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "data source referenced by this run no longer exists — cannot "
                "tune / blend / stack without the original dataset",
            )
        cfg = ds.config or {}
        path = cfg.get("path") or cfg.get("stored_path")
        if not path:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"data source {ds.id!r} has no on-disk path",
            )
        out["data_source_path"] = str(path)
    return out


def _experiment_setup(run: Run, db: Session) -> dict:
    """Lookup the source experiment's ``setup_params`` so the worker
    reconstructs the same preprocessing pipeline."""
    exp = db.get(Experiment, run.experiment_id)
    if exp is None:
        return {}
    return dict(exp.setup_params or {})


def _workspace_id_for_run(run: Run, db: Session) -> str:
    exp = db.get(Experiment, run.experiment_id)
    if exp is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "experiment row missing for run"
        )
    proj = db.get(Project, exp.project_id)
    if proj is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "project row missing for experiment"
        )
    return proj.workspace_id


def _trial_stored_path(trial: Trial) -> str | None:
    """Phase 0-v2: Trial owns its artifact directly."""
    return trial.stored_path


def _uri_to_local_path(uri: str) -> str:
    """Phase 2 helper: turn a stored_path URI into a filesystem path joblib
    can read. Accepts ``file://`` URIs *and* bare absolute paths (for
    backwards-compat with pre-Phase-2 DB rows). For non-local URIs the
    caller is expected to use the ObjectStore directly.
    """
    if uri.startswith("file://"):
        from urllib.parse import urlparse

        raw = urlparse(uri).path
        if raw.startswith("/") and len(raw) > 3 and raw[2] == ":":
            # Windows: ``file:///C:/foo`` → urlparse path is ``/C:/foo``
            raw = raw[1:]
        return raw
    return uri


def _build_action_spec_from_trial(
    run: Run,
    trial: Trial,
    *,
    action: str,
    action_params: dict,
    db: Session,
) -> "TrialActionSpec":  # noqa: F821  — string forward ref for late import
    from pycaret_server.runs.orchestrator import TrialActionSpec

    stored = _trial_stored_path(trial)
    if not stored:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "trial has no stored pipeline — cannot use as source",
        )
    snap = run.snapshot or {}
    return TrialActionSpec(
        run_id=run.id,
        experiment_id=run.experiment_id,
        workspace_id=_workspace_id_for_run(run, db),
        task=TaskType(snap.get("task") or snap.get("task_type")),
        target=snap.get("target"),
        setup_params=_experiment_setup(run, db),
        action=action,
        action_params=action_params,
        source_stored_paths=[stored],
        source_trial_ids=[trial.id],
        source_model_id=trial.model_id,
        **_spec_data_fields_from_run(run, db),
    )


def _build_action_spec_from_run(
    run: Run,
    source_trials: list[Trial],
    *,
    action: str,
    action_params: dict,
    db: Session,
) -> "TrialActionSpec":  # noqa: F821
    from pycaret_server.runs.orchestrator import TrialActionSpec

    paths: list[str] = []
    missing: list[str] = []
    for t in source_trials:
        sp = _trial_stored_path(t)
        if not sp:
            missing.append(t.id)
        else:
            paths.append(sp)
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"source trials missing stored pipelines: {missing}",
        )
    snap = run.snapshot or {}
    return TrialActionSpec(
        run_id=run.id,
        experiment_id=run.experiment_id,
        workspace_id=_workspace_id_for_run(run, db),
        task=TaskType(snap.get("task") or snap.get("task_type")),
        target=snap.get("target"),
        setup_params=_experiment_setup(run, db),
        action=action,
        action_params=action_params,
        source_stored_paths=paths,
        source_trial_ids=[t.id for t in source_trials],
        source_model_id=source_trials[0].model_id,
        **_spec_data_fields_from_run(run, db),
    )


@router.post(
    "/runs/{run_id}/trials/{trial_id}/tune",
    status_code=status.HTTP_202_ACCEPTED,
)
def tune_trial(
    run_id: str,
    trial_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Spawn a hyperparameter-tuning action on a single trial.

    Body knobs:
      * ``n_iter``: search iterations (default 10)
      * ``optimize``: metric to optimise (e.g. ``"Accuracy"`` / ``"R2"``)
      * ``fold``: CV fold override

    Returns 202 immediately; the new trial lands when the worker finishes.
    """
    r, t = _trial_access(run_id, trial_id, user, db)
    if r.status != "succeeded":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"run status is {r.status!r}; only succeeded runs can be tuned against",
        )
    action_params = {
        k: payload[k]
        for k in ("n_iter", "optimize", "fold", "round")
        if k in payload
    }
    spec = _build_action_spec_from_trial(
        r, t, action="tune", action_params=action_params, db=db
    )
    get_orchestrator().submit_trial_action(spec)
    return {"action": "tune", "run_id": run_id, "source_trial_id": trial_id, "status": "queued"}


@router.post(
    "/runs/{run_id}/trials/{trial_id}/ensemble",
    status_code=status.HTTP_202_ACCEPTED,
)
def ensemble_trial(
    run_id: str,
    trial_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Bag or Boost a single trial. Body: ``{method, n_estimators}``."""
    r, t = _trial_access(run_id, trial_id, user, db)
    if r.status != "succeeded":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"run status is {r.status!r}",
        )
    method = payload.get("method", "Bagging")
    if method not in ("Bagging", "Boosting"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "method must be 'Bagging' or 'Boosting'",
        )
    action_params = {"method": method}
    if "n_estimators" in payload:
        action_params["n_estimators"] = int(payload["n_estimators"])
    if "fold" in payload:
        action_params["fold"] = payload["fold"]
    spec = _build_action_spec_from_trial(
        r, t, action="ensemble", action_params=action_params, db=db
    )
    get_orchestrator().submit_trial_action(spec)
    return {
        "action": "ensemble",
        "run_id": run_id,
        "source_trial_id": trial_id,
        "status": "queued",
    }


def _resolve_source_trials(
    run_id: str, source_trial_ids: list[str], db: Session
) -> list[Trial]:
    if not isinstance(source_trial_ids, list) or len(source_trial_ids) < 2:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "blend/stack require at least 2 source_trial_ids",
        )
    rows: list[Trial] = []
    for tid in source_trial_ids:
        t = db.get(Trial, tid)
        # Phase 0: trials are grouped under the umbrella Run via
        # created_by_action_id, not via the old ``run_id`` FK.
        if t is None or t.created_by_action_id != run_id:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"trial {tid!r} not found in run"
            )
        rows.append(t)
    return rows


@router.post(
    "/runs/{run_id}/blend",
    status_code=status.HTTP_202_ACCEPTED,
)
def blend_run_trials(
    run_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Voting ensemble across N trials. Body: ``{source_trial_ids, method, weights?}``."""
    r = _run_access(run_id, user, db)
    if r.status != "succeeded":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"run status is {r.status!r}",
        )
    sources = _resolve_source_trials(run_id, payload.get("source_trial_ids") or [], db)
    action_params: dict = {}
    method = payload.get("method")
    if method in ("auto", "hard", "soft"):
        action_params["method"] = method
    if isinstance(payload.get("weights"), list):
        action_params["weights"] = [float(w) for w in payload["weights"]]
    if "fold" in payload:
        action_params["fold"] = payload["fold"]
    spec = _build_action_spec_from_run(
        r, sources, action="blend", action_params=action_params, db=db
    )
    get_orchestrator().submit_trial_action(spec)
    return {
        "action": "blend",
        "run_id": run_id,
        "source_trial_ids": [t.id for t in sources],
        "status": "queued",
    }


@router.post(
    "/runs/{run_id}/stack",
    status_code=status.HTTP_202_ACCEPTED,
)
def stack_run_trials(
    run_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Stacking ensemble with optional meta-learner. Body:
    ``{source_trial_ids, meta_model?, restack?}``."""
    r = _run_access(run_id, user, db)
    if r.status != "succeeded":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"run status is {r.status!r}",
        )
    sources = _resolve_source_trials(run_id, payload.get("source_trial_ids") or [], db)
    action_params: dict = {}
    if "meta_model" in payload and payload["meta_model"] not in (None, ""):
        action_params["meta_model"] = payload["meta_model"]
    if "fold" in payload:
        action_params["fold"] = payload["fold"]
    spec = _build_action_spec_from_run(
        r, sources, action="stack", action_params=action_params, db=db
    )
    get_orchestrator().submit_trial_action(spec)
    return {
        "action": "stack",
        "run_id": run_id,
        "source_trial_ids": [t.id for t in sources],
        "status": "queued",
    }


# ─────────────────────────────────────────────── trial annotations


@router.patch("/runs/{run_id}/trials/{trial_id}")
def patch_trial(
    run_id: str,
    trial_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Mutate user-editable fields on a trial — currently just ``notes``.

    Empty string clears the notes; ``None`` (omitted key) leaves them
    unchanged. Returns the updated value so optimistic UIs can confirm.
    """
    _r, t = _trial_access(run_id, trial_id, user, db)
    if "notes" in payload:
        notes = payload["notes"]
        if notes is not None and not isinstance(notes, str):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "notes must be a string or null"
            )
        t.notes = notes if notes else None
    db.commit()
    db.refresh(t)
    return {"id": t.id, "notes": t.notes}


# ─────────────────────────────────────────────── predict against the trial


@router.post("/runs/{run_id}/trials/{trial_id}/predict")
def predict_trial(
    run_id: str,
    trial_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Run inference using *this* trial's pickle — no promotion required.

    Body: ``{"rows": [{col: val, ...}]}``. We load the trial pipeline,
    coerce rows to a DataFrame, and call ``.predict`` (and ``.predict_proba``
    when the estimator supports it). Lets users sanity-check a candidate
    without going through the deploy → predict path.
    """
    _r, t = _trial_access(run_id, trial_id, user, db)
    stored = _trial_stored_path(t)
    if not stored:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "trial has no stored pipeline artifact",
        )

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "rows must be a non-empty list of dicts"
        )

    try:
        import joblib
        import pandas as pd

        pipe = joblib.load(_uri_to_local_path(stored))
        frame = pd.DataFrame.from_records(rows)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"could not coerce rows to DataFrame: {exc}",
        ) from exc

    try:
        preds = pipe.predict(frame)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"pipeline.predict failed: {exc}",
        ) from exc

    out: dict = {"predictions": [_jsonable(v) for v in list(preds)]}
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(frame)
            out["probabilities"] = [
                [float(p) for p in row] for row in list(proba)
            ]
            classes = getattr(pipe, "classes_", None)
            if classes is not None:
                out["classes"] = [_jsonable(c) for c in list(classes)]
        except Exception:  # noqa: BLE001
            pass
    if hasattr(pipe, "decision_function") and "probabilities" not in out:
        try:
            scores = pipe.decision_function(frame)
            out["scores"] = [_jsonable(v) for v in list(scores)]
        except Exception:  # noqa: BLE001
            pass
    return out


def _jsonable(v):
    """Coerce a single value to a JSON-safe primitive."""
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


# ─────────────────────────────────────────────── per-fold metrics


@router.get("/runs/{run_id}/trials/{trial_id}/cv")
def trial_cv(
    run_id: str,
    trial_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    folds: int = 5,
) -> dict:
    """Re-compute cross-validation scores for this trial on demand.

    Uses the fitted Pipeline + the run's holdout dataset + the snapshot's
    seed so results are repeatable. ``folds`` defaults to 5; capped at 10
    to keep the request bounded. Returns per-fold scores for the metrics
    sklearn knows for the task.
    """
    from pycaret_server.api.plots import _load_dataset_frame, _resolve_run_task

    folds = max(2, min(folds, 10))

    r, t = _trial_access(run_id, trial_id, user, db)
    stored = _trial_stored_path(t)
    if not stored:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "trial has no stored pipeline artifact",
        )
    task = _resolve_run_task(r)
    snap = r.snapshot or {}
    df = _load_dataset_frame(snap, db)
    if df is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "could not load the holdout dataset to score",
        )
    target = snap.get("target")
    if not target and snap.get("sklearn_dataset"):
        try:
            from pycaret_server.runs.plans import load_sklearn_dataset

            _, target = load_sklearn_dataset(snap["sklearn_dataset"])
        except Exception:  # noqa: BLE001
            pass
    if not target or target not in df.columns:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "run snapshot is missing the target column"
        )

    try:
        import joblib
        import numpy as np
        from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

        pipe = joblib.load(_uri_to_local_path(stored))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"could not reload pipeline: {exc}",
        ) from exc

    y = df[target]
    X = df.drop(columns=[target])
    seed = int(snap.get("session_id") or 0)
    if task == "classification":
        scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    elif task == "regression":
        scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"per-fold CV not supported for task {task!r}",
        )

    try:
        result = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"cross_validate failed: {exc}"
        ) from exc

    folds_out = []
    for i in range(folds):
        row: dict = {"fold": i + 1}
        for metric in scoring:
            arr = result.get(f"test_{metric}")
            if arr is not None and i < len(arr):
                # neg_* metrics → flip sign so callers see positive values
                v = float(arr[i])
                if metric.startswith("neg_"):
                    v = -v
                row[metric.replace("neg_", "")] = v
        folds_out.append(row)

    summary: dict = {}
    for metric in scoring:
        arr = result.get(f"test_{metric}")
        if arr is None or len(arr) == 0:
            continue
        v = np.array(arr, dtype=float)
        if metric.startswith("neg_"):
            v = -v
        summary[metric.replace("neg_", "")] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    return {
        "task": task,
        "folds": folds,
        "scoring": [m.replace("neg_", "") for m in scoring],
        "rows": folds_out,
        "summary": summary,
    }


# ─────────────────────────────────────────────── cohort / slice metrics


@router.get("/runs/{run_id}/trials/{trial_id}/cohorts")
def trial_cohorts(
    run_id: str,
    trial_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    column: str | None = None,
    max_groups: int = 12,
) -> dict:
    """Compute per-cohort metrics by slicing the holdout on ``column``.

    Surfaces fairness / segment quality: e.g., "accuracy on Store=A vs B".
    Numeric columns are bucketed into quartiles. Requires the run's
    holdout to be re-loadable (CSV or sklearn dataset).

    Calling without a ``column`` returns just the list of available
    columns so the UI can populate a dropdown without a separate route.
    """
    from pycaret_server.api.plots import _load_dataset_frame, _resolve_run_task

    r, t = _trial_access(run_id, trial_id, user, db)
    stored = _trial_stored_path(t)
    if not stored:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "trial has no stored pipeline artifact",
        )
    task = _resolve_run_task(r)
    snap = r.snapshot or {}
    df = _load_dataset_frame(snap, db)
    if df is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "could not load the holdout dataset",
        )
    target = snap.get("target")
    if not target and snap.get("sklearn_dataset"):
        try:
            from pycaret_server.runs.plans import load_sklearn_dataset

            _, target = load_sklearn_dataset(snap["sklearn_dataset"])
        except Exception:  # noqa: BLE001
            pass
    if not target or target not in df.columns:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "run snapshot is missing the target column"
        )

    available = [c for c in df.columns if c != target]
    if not column:
        # Discovery probe — the UI uses this to populate its dropdown.
        return {
            "task": task,
            "column": None,
            "rows": [],
            "available_columns": available,
        }
    if column == target:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "cohort column cannot be the target"
        )
    if column not in df.columns:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"unknown column {column!r}"
        )

    try:
        import joblib
        import numpy as np
        import pandas as pd

        pipe = joblib.load(_uri_to_local_path(stored))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"could not reload pipeline: {exc}",
        ) from exc

    # Re-derive the holdout using the snapshot's session_id so we score the
    # same rows the run did.
    from sklearn.model_selection import train_test_split

    seed = int(snap.get("session_id") or 0)
    train_size = float(snap.get("train_size") or 0.7)
    y = df[target]
    X = df.drop(columns=[target])
    try:
        if task == "classification":
            X_train, X_test, _y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=seed, stratify=y
            )
        else:
            X_train, X_test, _y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=seed
            )
    except Exception:  # noqa: BLE001 — fall back to no-stratify
        X_train, X_test, _y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=seed
        )

    # Bucketise numeric columns into quartiles for grouping.
    series = X_test[column]
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > max_groups:
        try:
            buckets = pd.qcut(series, q=4, duplicates="drop")
            grouping = buckets.astype(str)
        except Exception:  # noqa: BLE001
            grouping = series.astype(str)
    else:
        grouping = series.astype(str).fillna("(missing)")

    preds = pipe.predict(X_test)
    cohorts: list[dict] = []
    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        for value, idx in grouping.groupby(grouping).groups.items():
            sub_y = y_test.loc[idx]
            sub_pred = pd.Series(preds, index=y_test.index).loc[idx]
            if len(sub_y) == 0:
                continue
            cohorts.append(
                {
                    "value": str(value),
                    "n": int(len(sub_y)),
                    "metrics": {
                        "accuracy": float(accuracy_score(sub_y, sub_pred)),
                        "f1": float(f1_score(sub_y, sub_pred, average="weighted", zero_division=0)),
                        "precision": float(
                            precision_score(sub_y, sub_pred, average="weighted", zero_division=0)
                        ),
                        "recall": float(
                            recall_score(sub_y, sub_pred, average="weighted", zero_division=0)
                        ),
                    },
                }
            )
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        for value, idx in grouping.groupby(grouping).groups.items():
            sub_y = y_test.loc[idx]
            sub_pred = pd.Series(preds, index=y_test.index).loc[idx]
            if len(sub_y) == 0:
                continue
            cohorts.append(
                {
                    "value": str(value),
                    "n": int(len(sub_y)),
                    "metrics": {
                        "r2": float(r2_score(sub_y, sub_pred)),
                        "mae": float(mean_absolute_error(sub_y, sub_pred)),
                        "rmse": float(np.sqrt(mean_squared_error(sub_y, sub_pred))),
                    },
                }
            )

    cohorts.sort(key=lambda c: c["n"], reverse=True)
    return {
        "task": task,
        "column": column,
        "rows": cohorts[:max_groups],
        "available_columns": [c for c in X.columns if c != target],
    }


@router.get("/runs/{run_id}/events", response_model=list[EventResponse])
def list_events(
    run_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    limit: int = 500,
    after_id: str | None = None,
    tail: bool = False,
) -> list[EventResponse]:
    """List events for a run in emission order.

    ``after_id`` lets a polling client fetch only events it hasn't seen yet.
    ``tail=true`` returns the *latest* ``limit`` events instead of the first
    ``limit`` — used by the live progress card so it always sees the most
    recent activity even if the run has emitted hundreds of events.
    Result is always returned in chronological (ascending) order.
    """
    _run_access(run_id, user, db)
    bounded = max(1, min(limit, 5000))
    if tail:
        # Pull the latest N by emitted_at desc, then reverse so the response
        # stays in chronological order (consumers expect [..., latest]).
        q = (
            select(Event)
            .where(Event.run_id == run_id)
            .order_by(Event.emitted_at.desc())
            .limit(bounded)
        )
        rows = list(db.scalars(q).all())
        rows.reverse()
        return [_serialise_event(e) for e in rows]

    q = select(Event).where(Event.run_id == run_id).order_by(Event.emitted_at.asc())
    if after_id:
        anchor = db.get(Event, after_id)
        if anchor is not None:
            q = q.where(Event.emitted_at > anchor.emitted_at)
    q = q.limit(bounded)
    return [_serialise_event(e) for e in db.scalars(q).all()]


@router.post("/runs/{run_id}/wait", response_model=RunResponse)
def wait_for_run(
    run_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    timeout_s: float = 30.0,
) -> RunResponse:
    """Block until the run finishes (or timeout). Useful in notebooks + tests;
    the UI uses the WebSocket to stay async."""
    r = _run_access(run_id, user, db)
    if r.status in ("succeeded", "failed", "cancelled"):
        return _serialise_run(r, db)
    try:
        get_orchestrator().wait_for(run_id, timeout=timeout_s)
    except Exception:
        # Propagate the run row as-is; status will reflect what actually happened.
        pass
    db.refresh(r)
    return _serialise_run(r, db)


# ---------------------------------------------------------------- WebSocket


@router.websocket("/runs/{run_id}/events/ws")
async def ws_events(websocket: WebSocket, run_id: str) -> None:
    """Live event stream for a run.

    Auth: we accept the token as a query param ``?token=<access_token>`` since
    most browser WebSocket APIs can't set headers. The client fetches the token
    via the normal login/refresh flow and appends it to the URL.

    Protocol: each engine Event is sent as a JSON object. The server sends a
    final ``{"kind": "run.closed"}`` before closing once the run reaches a
    terminal state.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401)  # unauthorised
        return
    try:
        payload = decode_token(token)
    except Exception:
        await websocket.close(code=4401)
        return

    # Authorise against the DB. Use a scoped session since WebSocket handlers
    # don't run inside the normal Depends(get_db) lifecycle.
    from pycaret_server.db import get_session

    session = get_session()
    try:
        user = session.get(User, payload.sub)
        if user is None or not user.is_active:
            await websocket.close(code=4401)
            return
        try:
            _run_access(run_id, user, session)
        except HTTPException:
            await websocket.close(code=4403)  # forbidden
            return
    finally:
        session.close()

    await websocket.accept()

    # If the run already reached a terminal state before the client subscribed,
    # replay the stored events then close. Otherwise subscribe for live fan-out.
    session = get_session()
    try:
        run = session.get(Run, run_id)
        terminal = run is not None and run.status in ("succeeded", "failed", "cancelled")
        # Always replay stored events first so the client has full history.
        stored = (
            session.scalars(
                select(Event).where(Event.run_id == run_id).order_by(Event.emitted_at.asc())
            ).all()
            if run is not None
            else []
        )
    finally:
        session.close()

    for e in stored:
        await websocket.send_json(
            {
                "kind": e.kind,
                "message": e.message or "",
                "payload": e.payload or {},
                "duration_ms": e.duration_ms,
                "timestamp": e.emitted_at.timestamp(),
                "experiment_id": None,
            }
        )

    if terminal:
        await websocket.send_json({"kind": "run.closed"})
        await websocket.close()
        return

    queue = event_broker.subscribe(run_id)
    try:
        while True:
            item = await queue.get()
            if item is event_broker.END:
                await websocket.send_json({"kind": "run.closed"})
                break
            await websocket.send_json(item)
    except WebSocketDisconnect:
        pass
    finally:
        event_broker.unsubscribe(run_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
