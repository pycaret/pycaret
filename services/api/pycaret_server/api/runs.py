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
from pycaret_server.db import DataSource, Event, Experiment, Project, Run, Trial, User, get_db
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


@router.get("/runs/{run_id}/trials")
def list_trials(
    run_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Return the trials persisted for this run.

    A trial is one candidate model from a ``compare_models`` / ``automl`` plan,
    promoted from the leaderboard JSON into a queryable row. Sorted by ``rank``
    ascending (best first).
    """
    _run_access(run_id, user, db)
    rows = db.scalars(
        select(Trial).where(Trial.run_id == run_id).order_by(Trial.rank.asc())
    ).all()
    return {
        "run_id": run_id,
        "items": [
            {
                "id": t.id,
                "model_id": t.model_id,
                "rank": t.rank,
                "metrics": t.metrics,
                "is_best": t.is_best,
                "fitted_pipeline_id": t.fitted_pipeline_id,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in rows
        ],
    }


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
