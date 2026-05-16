"""Phase 8: notebook CRUD + session lifecycle.

Endpoints:

- ``GET    /projects/{project_id}/notebooks``
- ``POST   /projects/{project_id}/notebooks``
- ``GET    /notebooks/{id}``
- ``PATCH  /notebooks/{id}``                       rename, description, tags
- ``DELETE /notebooks/{id}``
- ``POST   /notebooks/{id}/sessions``              start a session
- ``GET    /notebooks/{id}/sessions``              list this user's sessions
- ``POST   /sessions/{id}/heartbeat``              keep-alive
- ``DELETE /sessions/{id}``                        stop

A "session" is a live JupyterLab container. The platform proxies
``/notebooks/sessions/{id}/iframe`` to the per-session URL exposed by
the manager — that's the URL the frontend embeds.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import Notebook, NotebookSession, Project, get_db
from pycaret_server.notebooks import get_notebook_manager

router = APIRouter(tags=["notebooks"])


def _serialise_notebook(n: Notebook) -> dict:
    return {
        "id": n.id,
        "workspace_id": n.workspace_id,
        "project_id": n.project_id,
        "name": n.name,
        "path": n.path,
        "kernel": n.kernel,
        "object_uri": n.object_uri,
        "description": n.description,
        "last_executed_at": (
            n.last_executed_at.isoformat() if n.last_executed_at else None
        ),
        "last_modified_at": (
            n.last_modified_at.isoformat() if n.last_modified_at else None
        ),
        "tags": list(n.tags or []),
        "created_at": n.created_at.isoformat() if n.created_at else None,
        "created_by": n.created_by,
    }


def _serialise_session(s: NotebookSession) -> dict:
    return {
        "id": s.id,
        "workspace_id": s.workspace_id,
        "notebook_id": s.notebook_id,
        "user_id": s.user_id,
        "status": s.status,
        "container_id": s.container_id,
        "port": s.port,
        # NEVER include the token in a list response — only on the
        # POST /sessions response so the iframe URL builder can use it.
        "started_at": s.started_at.isoformat() if s.started_at else None,
        "last_active_at": s.last_active_at.isoformat() if s.last_active_at else None,
        "stopped_at": s.stopped_at.isoformat() if s.stopped_at else None,
        "idle_timeout_seconds": s.idle_timeout_seconds,
        "cpu_limit": s.cpu_limit,
        "memory_mb_limit": s.memory_mb_limit,
        "error": s.error,
    }


def _project_access(project_id: str, user, db: Session) -> Project:
    p = db.get(Project, project_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    _require_access(user, db, p.workspace_id)
    return p


# ─────────────────────────────────────────────── notebook CRUD


@router.get("/projects/{project_id}/notebooks")
def list_notebooks(
    project_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _project_access(project_id, user, db)
    rows = db.scalars(
        select(Notebook)
        .where(Notebook.project_id == project_id)
        .order_by(Notebook.created_at.desc())
    ).all()
    return [_serialise_notebook(n) for n in rows]


@router.post(
    "/projects/{project_id}/notebooks",
    status_code=status.HTTP_201_CREATED,
)
def create_notebook(
    project_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Body: ``{name, kernel?, path?, description?, tags?}``."""
    p = _project_access(project_id, user, db)
    name = payload.get("name")
    if not name or not isinstance(name, str):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name required")
    n = Notebook(
        workspace_id=p.workspace_id,
        project_id=project_id,
        name=name,
        kernel=str(payload.get("kernel") or "python3"),
        path=payload.get("path") or f"notebooks/{name}.ipynb",
        description=payload.get("description"),
        tags=list(payload.get("tags") or []),
        last_modified_at=datetime.now(UTC),
        created_by=user.id,
    )
    db.add(n)
    db.commit()
    db.refresh(n)
    return _serialise_notebook(n)


@router.get("/notebooks/{notebook_id}")
def get_notebook(
    notebook_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    n = db.get(Notebook, notebook_id)
    if n is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "notebook not found")
    _require_access(user, db, n.workspace_id)
    return _serialise_notebook(n)


@router.patch("/notebooks/{notebook_id}")
def patch_notebook(
    notebook_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    n = db.get(Notebook, notebook_id)
    if n is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "notebook not found")
    _require_access(user, db, n.workspace_id)
    for key in ("name", "description", "kernel", "path"):
        if key in payload and (
            payload[key] is None or isinstance(payload[key], str)
        ):
            setattr(n, key, payload[key])
    if "tags" in payload and isinstance(payload["tags"], list):
        n.tags = [str(t) for t in payload["tags"]]
    db.commit()
    db.refresh(n)
    return _serialise_notebook(n)


@router.delete(
    "/notebooks/{notebook_id}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_notebook(
    notebook_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    n = db.get(Notebook, notebook_id)
    if n is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "notebook not found")
    _require_access(user, db, n.workspace_id)
    # Stop any live session attached to it first.
    sessions = db.scalars(
        select(NotebookSession).where(
            NotebookSession.notebook_id == notebook_id,
            NotebookSession.status.in_(("starting", "running")),
        )
    ).all()
    manager = get_notebook_manager()
    for s in sessions:
        if s.container_id:
            try:
                manager.stop(s.container_id)
            except Exception:  # noqa: BLE001
                pass
        s.status = "stopped"
        s.stopped_at = datetime.now(UTC)
    db.delete(n)
    db.commit()
    return None


# ─────────────────────────────────────────────── session lifecycle


@router.post(
    "/notebooks/{notebook_id}/sessions",
    status_code=status.HTTP_201_CREATED,
)
def start_session(
    notebook_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    payload: dict[str, Any] | None = None,
) -> dict:
    """Start (or reuse) a JupyterLab session for this notebook.

    Body knobs (all optional):
      * ``cpu_limit`` — float, e.g. 2.0
      * ``memory_mb_limit`` — int, e.g. 4096
      * ``idle_timeout_seconds`` — int, default 1800
      * ``env`` — extra env vars passed into the container

    Response includes the iframe URL (with token) — this is the ONE
    response that surfaces the JupyterLab token; subsequent reads omit
    it for security.
    """
    n = db.get(Notebook, notebook_id)
    if n is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "notebook not found")
    _require_access(user, db, n.workspace_id)
    body = payload or {}

    # Reuse a live session if one exists for this (user, notebook).
    live = db.scalar(
        select(NotebookSession).where(
            NotebookSession.notebook_id == notebook_id,
            NotebookSession.user_id == user.id,
            NotebookSession.status.in_(("starting", "running")),
        )
    )
    manager = get_notebook_manager()
    if live is not None and live.container_id and manager.is_alive(live.container_id):
        live.last_active_at = datetime.now(UTC)
        live.status = "running"
        db.commit()
        url = manager.proxy_url(
            type(  # build a fake descriptor for the URL helper
                "D",
                (),
                {
                    "container_id": live.container_id,
                    "port": live.port or 0,
                    "token": live.token or "",
                    "image": None,
                },
            )()
        )
        out = _serialise_session(live)
        out["url"] = url
        out["token"] = live.token
        return out

    s = NotebookSession(
        workspace_id=n.workspace_id,
        notebook_id=notebook_id,
        user_id=user.id,
        status="starting",
        idle_timeout_seconds=int(body.get("idle_timeout_seconds") or 1800),
        cpu_limit=(
            float(body["cpu_limit"]) if isinstance(body.get("cpu_limit"), (int, float)) else None
        ),
        memory_mb_limit=(
            int(body["memory_mb_limit"])
            if isinstance(body.get("memory_mb_limit"), int)
            else None
        ),
    )
    db.add(s)
    db.commit()
    db.refresh(s)

    try:
        descriptor = manager.start(
            session_id=s.id,
            notebook_id=notebook_id,
            workspace_id=n.workspace_id,
            user_id=user.id,
            kernel=n.kernel,
            cpu_limit=s.cpu_limit,
            memory_mb_limit=s.memory_mb_limit,
            env=dict(body.get("env") or {}),
        )
    except Exception as exc:  # noqa: BLE001
        s.status = "failed"
        s.error = f"{type(exc).__name__}: {exc}"
        db.commit()
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"failed to start session: {s.error}",
        ) from exc

    now = datetime.now(UTC)
    s.container_id = descriptor.container_id
    s.port = descriptor.port
    s.token = descriptor.token
    s.status = "running"
    s.started_at = now
    s.last_active_at = now
    db.commit()
    db.refresh(s)

    out = _serialise_session(s)
    out["url"] = manager.proxy_url(descriptor)
    # First-response only — token comes back so the iframe URL has it.
    out["token"] = descriptor.token
    return out


@router.get("/notebooks/{notebook_id}/sessions")
def list_sessions(
    notebook_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    n = db.get(Notebook, notebook_id)
    if n is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "notebook not found")
    _require_access(user, db, n.workspace_id)
    rows = db.scalars(
        select(NotebookSession)
        .where(NotebookSession.notebook_id == notebook_id)
        .order_by(NotebookSession.started_at.desc())
    ).all()
    return [_serialise_session(s) for s in rows]


@router.post("/sessions/{session_id}/heartbeat")
def heartbeat(
    session_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    s = db.get(NotebookSession, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    s.last_active_at = datetime.now(UTC)
    db.commit()
    return {"ok": True, "last_active_at": s.last_active_at.isoformat()}


@router.delete(
    "/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT
)
def stop_session(
    session_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    s = db.get(NotebookSession, session_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    _require_access(user, db, s.workspace_id)
    if s.container_id:
        try:
            get_notebook_manager().stop(s.container_id)
        except Exception:  # noqa: BLE001
            pass
    s.status = "stopped"
    s.stopped_at = datetime.now(UTC)
    db.commit()
    return None


# ─────────────────────────────────────────────── content + health


_DEFAULT_NB_BYTES = (
    b'{\n "cells": [\n  {\n   "cell_type": "code",\n   "execution_count": null,\n'
    b'   "metadata": {},\n   "outputs": [],\n   "source": [\n'
    b'    "# pycaret-server scratch\\n",\n'
    b'    "from pycaret_client import client\\n",\n'
    b'    "client.ping()\\n"\n'
    b'   ]\n  }\n ],\n'
    b' "metadata": {\n  "kernelspec": {\n   "name": "python3",\n'
    b'   "display_name": "Python 3"\n  }\n },\n'
    b' "nbformat": 4,\n "nbformat_minor": 5\n}\n'
)


@router.get("/notebooks/{notebook_id}/content")
def get_notebook_content(
    notebook_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Return the persisted ``.ipynb`` JSON (or a starter notebook).

    Phase 8.1 hardening: lets clients (JupyterLab plugin, custom
    runners, the future in-browser editor) round-trip notebook contents
    through the platform without depending on a per-session live
    container. The bytes live in object storage at the URI on
    ``notebooks.object_uri``; if the row has never been saved, we
    return a canonical empty-notebook so the editor can boot cleanly.
    """
    import json

    n = db.get(Notebook, notebook_id)
    if n is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "notebook not found")
    _require_access(user, db, n.workspace_id)
    if not n.object_uri:
        return {
            "notebook_id": notebook_id,
            "object_uri": None,
            "content": json.loads(_DEFAULT_NB_BYTES.decode("utf-8")),
            "last_modified_at": (
                n.last_modified_at.isoformat() if n.last_modified_at else None
            ),
        }
    from pycaret_server.storage import get_object_store

    store = get_object_store()
    try:
        blob = store.get_bytes(n.object_uri)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"could not fetch notebook bytes: {exc}",
        ) from exc
    try:
        content = json.loads(blob.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"notebook bytes are not valid JSON: {exc}",
        ) from exc
    return {
        "notebook_id": notebook_id,
        "object_uri": n.object_uri,
        "content": content,
        "last_modified_at": (
            n.last_modified_at.isoformat() if n.last_modified_at else None
        ),
    }


@router.put("/notebooks/{notebook_id}/content")
def save_notebook_content(
    notebook_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Persist ``.ipynb`` content to object storage.

    Body: ``{content: <ipynb-json>}`` — must be a JSON object with at
    least ``nbformat`` + ``cells``. Returns the new ``object_uri`` plus
    a ``size_bytes``/``sha256`` digest so a stale-tab detector can
    refuse a clobbering save.

    The bytes land at ``notebooks/<notebook_id>.ipynb`` on whichever
    object store ``get_object_store()`` resolves to — same backend as
    Run artifacts.
    """
    import hashlib
    import json

    n = db.get(Notebook, notebook_id)
    if n is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "notebook not found")
    _require_access(user, db, n.workspace_id)
    content = payload.get("content")
    if not isinstance(content, dict):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "content must be a JSON object"
        )
    # Light-weight ipynb shape check: nbformat + cells.
    if "nbformat" not in content or "cells" not in content:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "content must be an .ipynb (nbformat + cells required)",
        )
    try:
        blob = json.dumps(content).encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"content not JSON-serialisable: {exc}"
        ) from exc

    from pycaret_server.storage import get_object_store

    store = get_object_store()
    try:
        uri = store.put_bytes(f"notebooks/{notebook_id}.ipynb", blob)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"could not write notebook: {exc}"
        ) from exc
    n.object_uri = uri
    n.last_modified_at = datetime.now(UTC)
    db.commit()
    return {
        "notebook_id": notebook_id,
        "object_uri": uri,
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "last_modified_at": n.last_modified_at.isoformat(),
    }


@router.get("/sessions/{session_id}/health")
def session_health(
    session_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Cheap reachability probe for a live notebook session.

    Returns ``{alive, status, last_active_at, idle_seconds, reap_in_seconds}``.
    Frontend uses it to flash "session expiring in 3m" before the
    reaper sweeps. ``alive`` is the container's actual liveness as the
    manager reports it, not just the DB status — a container that
    crashed out-of-band shows ``alive=False`` even if the row is still
    ``running``.
    """
    s = db.get(NotebookSession, session_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    _require_access(user, db, s.workspace_id)
    manager = get_notebook_manager()
    alive = False
    if s.container_id and s.status == "running":
        try:
            alive = bool(manager.is_alive(s.container_id))
        except Exception:  # noqa: BLE001
            alive = False
    last = s.last_active_at or s.started_at
    now = datetime.now(UTC)
    # SQLite drops tzinfo on round-trip — coerce to UTC before
    # subtracting so we never trip on naive/aware mismatch.
    if last is not None and last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    idle = (now - last).total_seconds() if last else None
    reap_in = (
        max(0.0, float(s.idle_timeout_seconds) - idle)
        if (idle is not None and s.idle_timeout_seconds is not None)
        else None
    )
    return {
        "session_id": session_id,
        "status": s.status,
        "alive": alive,
        "last_active_at": last.isoformat() if last else None,
        "idle_seconds": idle,
        "reap_in_seconds": reap_in,
        "idle_timeout_seconds": s.idle_timeout_seconds,
    }


# ─────────────────────────────────────────────── reaper helper


def reap_idle_sessions(db: Session) -> int:
    """Stop sessions whose last activity is older than their timeout.

    Called from the scheduler / a worker job. Returns the count of
    sessions actually stopped.
    """
    now = datetime.now(UTC)
    rows = db.scalars(
        select(NotebookSession).where(
            NotebookSession.status == "running",
        )
    ).all()
    manager = get_notebook_manager()
    stopped = 0
    for s in rows:
        last = s.last_active_at or s.started_at
        if last is None:
            continue
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        age = (now - last).total_seconds()
        if age <= (s.idle_timeout_seconds or 1800):
            continue
        if s.container_id:
            try:
                manager.stop(s.container_id)
            except Exception:  # noqa: BLE001
                pass
        s.status = "stopped"
        s.stopped_at = now
        stopped += 1
    if stopped:
        db.commit()
    return stopped
