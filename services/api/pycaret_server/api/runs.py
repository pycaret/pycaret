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

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pycaret.core.tasks import TaskType
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import EventResponse, RunCreate, RunResponse
from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.auth.tokens import decode_token
from pycaret_server.db import DataSource, Event, Experiment, Project, Run, User, get_db
from pycaret_server.runs.broker import event_broker
from pycaret_server.runs.orchestrator import RunSpec, get_orchestrator

router = APIRouter(tags=["runs"])


# ------------------------------------------------------------------- helpers


def _serialise_run(r: Run) -> RunResponse:
    return RunResponse(
        id=r.id,
        experiment_id=r.experiment_id,
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

    if payload.plan not in ("setup", "create", "compare"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"invalid plan {payload.plan!r}; must be one of setup|create|compare",
        )
    if payload.plan == "create" and not payload.model_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "plan='create' requires model_id",
        )
    if not (payload.sklearn_dataset or payload.data_inline or payload.data_source_id):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "must supply sklearn_dataset, data_inline, or data_source_id",
        )

    # Resolve data_source_id -> CSV path + guard workspace access.
    data_source_path: str | None = None
    if payload.data_source_id:
        ds = db.get(DataSource, payload.data_source_id)
        if ds is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "data source not found")
        project_ws = db.get(Project, e.project_id).workspace_id
        if ds.workspace_id != project_ws:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "data source belongs to a different workspace than the experiment",
            )
        if ds.kind != "csv_upload":
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"data source kind {ds.kind!r} not yet supported for runs; "
                "only 'csv_upload' is implemented",
            )
        data_source_path = (ds.config or {}).get("path")
        if not data_source_path:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "data source missing 'path' in config",
            )

    # Resolve target: prefer explicit payload.target, fall back to experiment.target.
    effective_target = payload.target or e.target

    # Snapshot the experiment config into the run for reproducibility.
    snapshot = {
        "task": e.task,
        "target": effective_target,
        "setup_params": dict(e.setup_params or {}),
        "plan": payload.plan,
        "model_id": payload.model_id,
        "plan_params": dict(payload.plan_params or {}),
        "sklearn_dataset": payload.sklearn_dataset,
        "data_inline_rows": len(payload.data_inline) if payload.data_inline else 0,
        "data_source_id": payload.data_source_id,
    }

    r = Run(
        experiment_id=e.id,
        status="queued",
        created_by=user.id,
        snapshot=snapshot,
    )
    db.add(r)
    db.commit()
    db.refresh(r)

    spec = RunSpec(
        run_id=r.id,
        experiment_id=e.id,
        task=TaskType(e.task),
        target=effective_target,
        setup_params=dict(e.setup_params or {}),
        plan=payload.plan,  # type: ignore[arg-type]
        model_id=payload.model_id,
        plan_params=dict(payload.plan_params or {}),
        sklearn_dataset=payload.sklearn_dataset,
        data_inline=list(payload.data_inline) if payload.data_inline else None,
        data_source_path=data_source_path,
        target_override=payload.target,
    )
    get_orchestrator().submit(spec)

    return _serialise_run(r)


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
        return _serialise_run(r)
    get_orchestrator().cancel(run_id)
    # Don't refresh — the worker may still be running. The status flip happens
    # when the checkpoint trips; the client can poll or watch the WS for it.
    return _serialise_run(r)


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
    return [_serialise_run(r) for r in rows]


# ---------------------------------------------------------------- fetch + events


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(
    run_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> RunResponse:
    return _serialise_run(_run_access(run_id, user, db))


@router.get("/runs/{run_id}/events", response_model=list[EventResponse])
def list_events(
    run_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    limit: int = 500,
    after_id: str | None = None,
) -> list[EventResponse]:
    """List events for a run in emission order.

    `after_id` lets a polling client fetch only events it hasn't seen yet
    (the UI uses the WebSocket instead in practice, but polling is a
    reasonable fallback)."""
    _run_access(run_id, user, db)
    q = select(Event).where(Event.run_id == run_id).order_by(Event.emitted_at.asc())
    if after_id:
        anchor = db.get(Event, after_id)
        if anchor is not None:
            q = q.where(Event.emitted_at > anchor.emitted_at)
    q = q.limit(max(1, min(limit, 5000)))
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
        return _serialise_run(r)
    try:
        get_orchestrator().wait_for(run_id, timeout=timeout_s)
    except Exception:
        # Propagate the run row as-is; status will reflect what actually happened.
        pass
    db.refresh(r)
    return _serialise_run(r)


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
