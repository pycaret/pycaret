"""Phase 4: Connection + Secret + Dataset + Lineage routes.

These are the new "data catalog" endpoints. Existing
``/workspaces/{ws}/data-sources`` keeps working — DataSource is now
optionally backed by a Connection for richer types, but CSV uploads
still operate without one.

Auth: every route is workspace-scoped and runs through
``_require_access``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

_log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.crypto import decrypt, encrypt
from pycaret_server.datasources import get_driver, list_kinds
from pycaret_server.db import (
    Connection,
    DataSource,
    Dataset,
    Lineage,
    Secret,
    get_db,
)

router = APIRouter(tags=["data-catalog"])


# ──────────────────────────────────────────────── secrets


def _serialise_secret(s: Secret) -> dict:
    return {
        "id": s.id,
        "workspace_id": s.workspace_id,
        "name": s.name,
        "kind": s.kind,
        "last4": s.last4,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "created_by": s.created_by,
    }


@router.get("/workspaces/{workspace_id}/secrets")
def list_secrets(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(Secret)
        .where(Secret.workspace_id == workspace_id)
        .order_by(Secret.created_at.desc())
    ).all()
    return [_serialise_secret(s) for s in rows]


@router.post(
    "/workspaces/{workspace_id}/secrets",
    status_code=status.HTTP_201_CREATED,
)
def create_secret(
    workspace_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Create a workspace secret. Body: ``{name, kind?, value}``.

    The plaintext value is encrypted at rest via Fernet (same key as
    LLM provider settings); the response never echoes the plaintext.
    """
    _require_access(user, db, workspace_id)
    name = payload.get("name")
    value = payload.get("value")
    if not name or not isinstance(name, str):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name is required")
    if not value or not isinstance(value, str):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "value is required")
    s = Secret(
        workspace_id=workspace_id,
        name=name,
        kind=str(payload.get("kind") or "opaque"),
        value_encrypted=encrypt(value),
        last4=value[-4:] if len(value) > 4 else None,
        created_by=user.id,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return _serialise_secret(s)


@router.delete(
    "/workspaces/{workspace_id}/secrets/{secret_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_secret(
    workspace_id: str,
    secret_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    _require_access(user, db, workspace_id)
    s = db.get(Secret, secret_id)
    if s is None or s.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "secret not found")
    db.delete(s)
    db.commit()
    return None


# ──────────────────────────────────────────────── connections


def _serialise_connection(c: Connection) -> dict:
    return {
        "id": c.id,
        "workspace_id": c.workspace_id,
        "name": c.name,
        "kind": c.kind,
        "config": dict(c.config or {}),
        "secret_id": c.secret_id,
        "last_tested_at": c.last_tested_at.isoformat() if c.last_tested_at else None,
        "last_test_status": c.last_test_status,
        "last_test_error": c.last_test_error,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "created_by": c.created_by,
    }


@router.get("/workspaces/{workspace_id}/connections")
def list_connections(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(Connection)
        .where(Connection.workspace_id == workspace_id)
        .order_by(Connection.created_at.desc())
    ).all()
    return [_serialise_connection(c) for c in rows]


@router.post(
    "/workspaces/{workspace_id}/connections",
    status_code=status.HTTP_201_CREATED,
)
def create_connection(
    workspace_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Body: ``{name, kind, config, secret_id?}``."""
    _require_access(user, db, workspace_id)
    name = payload.get("name")
    kind = payload.get("kind")
    if not name or not kind:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name + kind required")
    if kind not in list_kinds():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown kind {kind!r}; available: {list_kinds()}",
        )
    c = Connection(
        workspace_id=workspace_id,
        name=str(name),
        kind=str(kind),
        config=dict(payload.get("config") or {}),
        secret_id=payload.get("secret_id"),
        created_by=user.id,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return _serialise_connection(c)


@router.get("/workspaces/{workspace_id}/connections/{connection_id}/tables")
def list_connection_tables(
    workspace_id: str,
    connection_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """List the tables / objects exposed by a Connection's backend.

    Used by the "Register as DataSource" flow: the UI shows this list,
    user picks a table, we create a DataSource pointing at it. Failure
    surfaces as 400 so the UI can show the driver's actual error
    message (e.g. "FATAL: password authentication failed").
    """
    _require_access(user, db, workspace_id)
    c = db.get(Connection, connection_id)
    if c is None or c.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "connection not found")
    secret_value: str | None = None
    if c.secret_id:
        s = db.get(Secret, c.secret_id)
        if s is not None:
            try:
                secret_value = decrypt(s.value_encrypted)
            except Exception:  # noqa: BLE001
                secret_value = None
    driver = get_driver(c.kind)
    try:
        tables = driver.list_tables(
            config=dict(c.config or {}), secret_value=secret_value
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"could not list tables: {type(exc).__name__}: {exc}",
        ) from exc
    return {"connection_id": connection_id, "kind": c.kind, "tables": tables}


@router.post(
    "/workspaces/{workspace_id}/data-sources/from-connection",
    status_code=status.HTTP_201_CREATED,
)
def register_datasource_from_connection(
    workspace_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Register a Connection table/object as a DataSource.

    Body: ``{connection_id, table, name?, description?}`` — ``name``
    defaults to ``"<connection_name>:<table>"`` so each row reads
    self-documentingly in the data-sources list.

    The DataSource ``kind`` is taken from the Connection (e.g.
    ``postgres``); its ``config`` carries the Connection's host/port/
    database plus the table/query the user picked. Credentials stay
    on the Connection's linked Secret — never copied here.
    """
    _require_access(user, db, workspace_id)
    connection_id = payload.get("connection_id")
    table = payload.get("table")
    if not connection_id or not table:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "connection_id + table are required",
        )
    c = db.get(Connection, str(connection_id))
    if c is None or c.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "connection not found")
    name = payload.get("name") or f"{c.name}:{table}"

    # Build the DataSource config from the Connection's params + the
    # picked table. For postgres, split the ``schema.table`` form
    # back into separate fields so the driver doesn't need to parse.
    config = dict(c.config or {})
    if "." in str(table):
        schema, tbl = str(table).split(".", 1)
        config["schema"] = schema
        config["table"] = tbl
    else:
        config["table"] = str(table)
    config["connection_id"] = c.id  # back-link so we can rotate creds

    from pycaret_server.db import DataSource

    ds = DataSource(
        workspace_id=workspace_id,
        name=str(name),
        kind=c.kind,
        description=payload.get("description"),
        config=config,
        created_by=user.id,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {
        "id": ds.id,
        "workspace_id": ds.workspace_id,
        "name": ds.name,
        "kind": ds.kind,
        "config": dict(ds.config or {}),
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
        "created_by": ds.created_by,
    }


@router.post("/workspaces/{workspace_id}/connections/{connection_id}/test")
def test_connection(
    workspace_id: str,
    connection_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Run the driver's ``test_connection`` and stamp the result on the row."""
    _require_access(user, db, workspace_id)
    c = db.get(Connection, connection_id)
    if c is None or c.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "connection not found")
    secret_value: str | None = None
    if c.secret_id:
        s = db.get(Secret, c.secret_id)
        if s is not None:
            try:
                secret_value = decrypt(s.value_encrypted)
            except Exception:  # noqa: BLE001
                secret_value = None
    driver = get_driver(c.kind)
    ok, err = driver.test_connection(config=c.config or {}, secret_value=secret_value)
    c.last_tested_at = datetime.now(UTC)
    c.last_test_status = "ok" if ok else "error"
    c.last_test_error = err
    db.commit()
    return {"ok": ok, "error": err}


@router.delete(
    "/workspaces/{workspace_id}/connections/{connection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_connection(
    workspace_id: str,
    connection_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    _require_access(user, db, workspace_id)
    c = db.get(Connection, connection_id)
    if c is None or c.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "connection not found")
    db.delete(c)
    db.commit()
    return None


@router.get("/datasource-kinds")
def datasource_kinds() -> dict:
    """Return the registered driver kinds; the UI's New-Connection wizard
    populates its kind picker from this."""
    return {"kinds": list_kinds()}


# ──────────────────────────────────────────────── datasets (versioned snapshots)


def _serialise_dataset(d: Dataset) -> dict:
    return {
        "id": d.id,
        "workspace_id": d.workspace_id,
        "data_source_id": d.data_source_id,
        "version": d.version,
        "name": d.name,
        "schema_json": d.schema_json,
        "row_count": d.row_count,
        "byte_count": d.byte_count,
        "snapshot_uri": d.snapshot_uri,
        "sample_uri": d.sample_uri,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }


@router.get("/data-sources/{data_source_id}/datasets")
def list_datasets(
    data_source_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    """Version history for a DataSource."""
    ds = db.get(DataSource, data_source_id)
    if ds is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "data source not found")
    _require_access(user, db, ds.workspace_id)
    rows = db.scalars(
        select(Dataset)
        .where(Dataset.data_source_id == data_source_id)
        .order_by(Dataset.version.desc())
    ).all()
    return [_serialise_dataset(d) for d in rows]


@router.post(
    "/data-sources/{data_source_id}/refresh",
    status_code=status.HTTP_201_CREATED,
)
def refresh_dataset(
    data_source_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Introspect the DataSource via its driver and write a new Dataset row.

    Captures schema, sample rows, row count, byte count. Doesn't yet
    materialise a snapshot URI — that ships when the worker handler for
    ``dataset_refresh`` lands.
    """
    ds = db.get(DataSource, data_source_id)
    if ds is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "data source not found")
    _require_access(user, db, ds.workspace_id)
    driver = get_driver(ds.kind)
    try:
        schema = driver.introspect(
            config=dict(ds.config or {}),
            secret_value=resolve_datasource_secret(db, ds),
            sample_rows=20,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"introspect failed: {type(exc).__name__}: {exc}",
        ) from exc
    # Bump version.
    prior = db.scalar(
        select(Dataset)
        .where(Dataset.data_source_id == data_source_id)
        .order_by(Dataset.version.desc())
        .limit(1)
    )
    version = (prior.version + 1) if prior else 1
    d = Dataset(
        workspace_id=ds.workspace_id,
        data_source_id=data_source_id,
        version=version,
        name=ds.name,
        schema_json={
            "columns": schema.columns,
            "sample_rows": schema.sample_rows,
        },
        row_count=schema.row_count,
        byte_count=schema.byte_count,
        created_by=user.id,
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return _serialise_dataset(d)


# ──────────────────────────────────────────────── lineage graph


@router.get("/workspaces/{workspace_id}/lineage")
def list_lineage(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    node_kind: str | None = None,
    node_id: str | None = None,
    depth: int = 2,
) -> dict:
    """Return lineage edges, optionally rooted at a node.

    Without filters: every edge in the workspace.
    With ``node_kind`` + ``node_id``: BFS up to ``depth`` hops (in both
    directions) so the viewer can render a focused subgraph.
    """
    _require_access(user, db, workspace_id)

    if not (node_kind and node_id):
        rows = db.scalars(
            select(Lineage)
            .where(Lineage.workspace_id == workspace_id)
            .order_by(Lineage.created_at.desc())
            .limit(1000)
        ).all()
        return {"edges": [_serialise_edge(e) for e in rows]}

    bounded = max(0, min(int(depth), 6))
    seen_nodes: set[tuple[str, str]] = {(node_kind, node_id)}
    frontier: set[tuple[str, str]] = {(node_kind, node_id)}
    edges: list[Lineage] = []
    for _ in range(bounded):
        if not frontier:
            break
        next_frontier: set[tuple[str, str]] = set()
        for kind, ident in frontier:
            out_edges = db.scalars(
                select(Lineage).where(
                    Lineage.workspace_id == workspace_id,
                    Lineage.source_kind == kind,
                    Lineage.source_id == ident,
                )
            ).all()
            in_edges = db.scalars(
                select(Lineage).where(
                    Lineage.workspace_id == workspace_id,
                    Lineage.target_kind == kind,
                    Lineage.target_id == ident,
                )
            ).all()
            for e in (*out_edges, *in_edges):
                edges.append(e)
                t = (e.target_kind, e.target_id)
                s = (e.source_kind, e.source_id)
                for n in (t, s):
                    if n not in seen_nodes:
                        seen_nodes.add(n)
                        next_frontier.add(n)
        frontier = next_frontier

    # Dedupe edges by id.
    by_id = {e.id: e for e in edges}
    return {
        "edges": [_serialise_edge(e) for e in by_id.values()],
        "nodes": [
            {"kind": kind, "id": ident} for kind, ident in sorted(seen_nodes)
        ],
    }


def _serialise_edge(e: Lineage) -> dict:
    return {
        "id": e.id,
        "source": {"kind": e.source_kind, "id": e.source_id},
        "target": {"kind": e.target_kind, "id": e.target_id},
        "relation": e.relation,
        "metadata": e.metadata_json,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


def resolve_datasource_secret(db: Session, ds: DataSource) -> str | None:
    """Pull the plaintext secret a DataSource needs to authenticate.

    For ``csv_upload`` sources this is always None (uploaded files don't
    carry credentials). For connection-backed sources (the
    ``Register as DataSource`` flow stamps ``config.connection_id`` on
    the row) we look up the Connection, decrypt its Secret, and hand
    that to the driver.

    Best-effort: returns None if the Connection / Secret rows have
    been deleted out from under the DataSource. The driver will then
    surface a connection error which is the right user-visible state.
    """
    config = dict(ds.config or {})
    connection_id = config.get("connection_id")
    if not connection_id:
        return None
    c = db.get(Connection, str(connection_id))
    if c is None or not c.secret_id:
        return None
    s = db.get(Secret, c.secret_id)
    if s is None:
        return None
    try:
        return decrypt(s.value_encrypted)
    except Exception:  # noqa: BLE001
        return None


def record_lineage(
    db: Session,
    *,
    workspace_id: str | None,
    source_kind: str,
    source_id: str,
    target_kind: str,
    target_id: str,
    relation: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Helper for backend code to drop a lineage edge.

    Best-effort: failures are logged at WARNING but do not propagate, so a
    downstream write never breaks the user-facing action. The graph going
    silently stale was a hidden-rot risk flagged in the session-55 audit —
    `_log.warning` makes regressions visible in `services/api/logs.log`
    instead.
    """
    try:
        db.add(
            Lineage(
                id=str(uuid.uuid4()),
                workspace_id=workspace_id,
                source_kind=source_kind,
                source_id=source_id,
                target_kind=target_kind,
                target_id=target_id,
                relation=relation,
                metadata_json=metadata,
            )
        )
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        _log.warning(
            "lineage write failed (swallowed): "
            "workspace_id=%s edge=%s:%s -[%s]-> %s:%s err=%s: %s",
            workspace_id,
            source_kind,
            source_id,
            relation,
            target_kind,
            target_id,
            type(exc).__name__,
            exc,
        )
