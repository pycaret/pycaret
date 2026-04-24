"""Data-source CRUD + CSV upload route.

Mounted under ``/api/v1/workspaces/{workspace_id}/data-sources`` for listing
and creation; ``/api/v1/data-sources/{id}`` for fetch + delete. CSV uploads
land in ``${PYCARET_ARTIFACT_DIR}/data-sources/<uuid>.csv`` and the resulting
``DataSource`` row carries ``kind="csv_upload"`` with ``config={"path": …,
"sha256": …, "size_bytes": …, "rows": …, "columns": [...]}``.

S3 + Postgres connectors register a config dict without uploading anything
— their data is pulled lazily by the orchestrator at run time.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.config import get_settings
from pycaret_server.db import DataSource, Workspace, get_db

router = APIRouter(tags=["data-sources"])

# Hard guardrails. Tests deliberately push small files; production ops can bump
# these via config later. We keep the value in Python so schema changes don't
# force a migration — the cap is policy, not durable state.
_MAX_CSV_BYTES = 64 * 1024 * 1024  # 64 MB
_ALLOWED_KINDS = {"csv_upload", "s3", "postgres"}


# ---------------------------------------------------------------- serialise


def _serialise(ds: DataSource) -> dict:
    return {
        "id": ds.id,
        "workspace_id": ds.workspace_id,
        "name": ds.name,
        "kind": ds.kind,
        "description": ds.description,
        "config": dict(ds.config or {}),
        "created_at": ds.created_at,
        "created_by": ds.created_by,
    }


# ------------------------------------------------------------------- list


@router.get("/workspaces/{workspace_id}/data-sources")
def list_data_sources(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, ws.id)
    rows = db.scalars(
        select(DataSource)
        .where(DataSource.workspace_id == workspace_id)
        .order_by(DataSource.created_at.desc())
    ).all()
    return [_serialise(r) for r in rows]


# ------------------------------------------------------------------- upload CSV


@router.post(
    "/workspaces/{workspace_id}/data-sources/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload_csv(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile, File()],
    name: Annotated[str, Form()],
    description: Annotated[str | None, Form()] = None,
) -> dict:
    """Upload a CSV and register it as a ``csv_upload`` DataSource.

    The file is read in chunks; we reject anything over `_MAX_CSV_BYTES` before
    it hits disk. A SHA-256 checksum and a quick `pd.read_csv` sample are
    recorded in ``config`` so the UI can preview columns without re-uploading.
    """
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, ws.id)

    if (
        db.scalar(
            select(DataSource).where(
                DataSource.workspace_id == workspace_id, DataSource.name == name
            )
        )
        is not None
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, f"data source {name!r} already exists")

    settings = get_settings()
    uploads_dir: Path = settings.artifact_dir / "data-sources"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Stream to disk with size enforcement + SHA running in parallel.
    file_id = str(uuid.uuid4())
    target = uploads_dir / f"{file_id}.csv"
    hasher = hashlib.sha256()
    total = 0
    try:
        with target.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_CSV_BYTES:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"upload exceeds {_MAX_CSV_BYTES // (1024 * 1024)} MB cap",
                    )
                hasher.update(chunk)
                f.write(chunk)
    except Exception:
        # Best-effort cleanup; propagate the original error.
        target.unlink(missing_ok=True)
        raise

    # Sample the file with pandas so the UI has column metadata immediately.
    try:
        sample = pd.read_csv(target, nrows=5)
        columns = [str(c) for c in sample.columns]
        rows_est = sum(1 for _ in open(target, encoding="utf-8")) - 1
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"could not parse CSV: {exc}") from exc

    ds = DataSource(
        workspace_id=workspace_id,
        name=name,
        kind="csv_upload",
        description=description,
        config={
            "path": str(target),
            "sha256": hasher.hexdigest(),
            "size_bytes": total,
            "rows": max(0, rows_est),
            "columns": columns,
            "uploaded_at": datetime.now(UTC).isoformat(),
        },
        created_by=user.id,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return _serialise(ds)


# --------------------------------------------------------- register connectors


@router.post(
    "/workspaces/{workspace_id}/data-sources",
    status_code=status.HTTP_201_CREATED,
)
def register_connector(
    workspace_id: str,
    payload: dict,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Register a non-upload DataSource (``s3`` or ``postgres``).

    Payload shape::

        {"name": "...", "kind": "s3"|"postgres", "config": {...}, "description": "..."}

    No connectivity check is performed — that happens at run dispatch time.
    """
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, ws.id)

    name = payload.get("name")
    kind = payload.get("kind")
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name is required")
    if kind not in _ALLOWED_KINDS or kind == "csv_upload":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"kind must be one of {sorted(_ALLOWED_KINDS - {'csv_upload'})}; "
            "use /upload for csv_upload",
        )
    if (
        db.scalar(
            select(DataSource).where(
                DataSource.workspace_id == workspace_id, DataSource.name == name
            )
        )
        is not None
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, f"data source {name!r} already exists")

    ds = DataSource(
        workspace_id=workspace_id,
        name=str(name),
        kind=str(kind),
        description=payload.get("description"),
        config=dict(payload.get("config") or {}),
        created_by=user.id,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return _serialise(ds)


# --------------------------------------------------------- get + delete


@router.get("/data-sources/{data_source_id}")
def get_data_source(
    data_source_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    ds = db.get(DataSource, data_source_id)
    if ds is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "data source not found")
    _require_access(user, db, ds.workspace_id)
    return _serialise(ds)


@router.delete("/data-sources/{data_source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_data_source(
    data_source_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    ds = db.get(DataSource, data_source_id)
    if ds is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "data source not found")
    _require_access(user, db, ds.workspace_id)

    # Clean up the uploaded file if any.
    if ds.kind == "csv_upload":
        p = (ds.config or {}).get("path")
        if p:
            Path(p).unlink(missing_ok=True)

    db.delete(ds)
    db.commit()
