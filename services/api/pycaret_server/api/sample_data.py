"""Sample / bundled CSV datasets.

The PyCaret repo ships a catalog of small public datasets under
``<repo>/datasets/`` with an ``index.csv`` describing each one. They're
the same files the 3.x ``get_data(name)`` helper has always served.

This module exposes:

- ``GET /sample-datasets`` — list of bundled samples with metadata
  (task type, target column, instance + attribute count, missing flag).
- ``POST /workspaces/{ws}/data-sources/from-sample`` — registers a
  bundled sample as a workspace ``DataSource`` (``kind="csv_upload"``,
  ``config.path`` pointing at the bundled CSV path; ``config.source =
  "sample"`` flag so the UI can render a badge).

The on-disk CSVs are read-only — DataSource rows referencing them are
just pointers; profile / refresh / EDA all reuse the existing
csv_upload driver.
"""

from __future__ import annotations

import csv
import hashlib
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import DataSource, Workspace, get_db

router = APIRouter(tags=["sample-datasets"])


# ──────────────────────────────────────────────── filesystem resolution


def _datasets_dir() -> Path:
    """Find ``<repo>/datasets/`` containing ``index.csv``.

    Resolution order:
      1. ``PYCARET_SAMPLE_DATASETS_DIR`` env var (absolute path).
      2. Walk parents of this file looking for ``datasets/index.csv``.
    """
    env = os.environ.get("PYCARET_SAMPLE_DATASETS_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "index.csv").is_file():
            return p
        raise RuntimeError(
            f"PYCARET_SAMPLE_DATASETS_DIR={env!r} does not contain index.csv"
        )
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "datasets" / "index.csv"
        if cand.is_file():
            return parent / "datasets"
    raise RuntimeError(
        "could not locate datasets/index.csv anywhere up the tree; "
        "set PYCARET_SAMPLE_DATASETS_DIR to the absolute path"
    )


# ──────────────────────────────────────────────── catalog


class SampleDataset(BaseModel):
    name: str
    data_types: str
    default_task: str
    target_1: str | None
    target_2: str | None
    instances: int
    attributes: int
    has_missing: bool
    size_bytes: int | None
    available: bool  # True iff <name>.csv exists on disk


@lru_cache(maxsize=1)
def _read_catalog() -> list[SampleDataset]:
    """Parse ``index.csv`` once per process. Cached — the file is read-only."""
    root = _datasets_dir()
    index = root / "index.csv"
    out: list[SampleDataset] = []
    with index.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Dataset") or "").strip()
            if not name:
                continue
            csv_path = root / f"{name}.csv"
            available = csv_path.is_file()
            try:
                size = csv_path.stat().st_size if available else None
            except OSError:
                size = None
            t1 = (row.get("Target Variable 1") or "").strip()
            t2 = (row.get("Target Variable 2") or "").strip()
            out.append(
                SampleDataset(
                    name=name,
                    data_types=(row.get("Data Types") or "").strip(),
                    default_task=(row.get("Default Task") or "").strip(),
                    target_1=None if t1 in {"", "None"} else t1,
                    target_2=None if t2 in {"", "None"} else t2,
                    instances=int(row.get("# Instances") or 0),
                    attributes=int(row.get("# Attributes") or 0),
                    has_missing=(row.get("Missing Values") or "").strip().upper() == "Y",
                    size_bytes=size,
                    available=available,
                )
            )
    return out


@router.get("/sample-datasets")
def list_sample_datasets(user: CurrentUser) -> list[SampleDataset]:  # noqa: ARG001 — auth-gate
    """Return the bundled sample-dataset catalog."""
    return _read_catalog()


# ──────────────────────────────────────────────── register-as-DataSource


class FromSampleRequest(BaseModel):
    sample_name: str
    name: str | None = None  # override workspace-display name; default = sample_name


@router.post(
    "/workspaces/{workspace_id}/data-sources/from-sample",
    status_code=status.HTTP_201_CREATED,
)
def register_sample_as_data_source(
    workspace_id: str,
    payload: FromSampleRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Register a bundled sample as a workspace DataSource.

    Mirrors the upload-CSV flow but skips the disk-write: the file
    already exists in the bundled tree, so ``config.path`` points
    directly at it. ``config.source = "sample"`` and ``config.sample_name``
    let the UI render a "sample" badge + offer revert / re-add UX later.
    """
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, workspace_id)

    catalog = {s.name: s for s in _read_catalog()}
    sample = catalog.get(payload.sample_name)
    if sample is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"sample dataset {payload.sample_name!r} not found in catalog",
        )
    if not sample.available:
        raise HTTPException(
            status.HTTP_410_GONE,
            f"sample dataset {payload.sample_name!r} is in the index but the CSV is missing on disk",
        )

    root = _datasets_dir()
    csv_path = (root / f"{sample.name}.csv").resolve()

    display_name = (payload.name or sample.name).strip()
    if not display_name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name must be non-empty")

    # Reject collisions explicitly so the UI can render a clear message
    # instead of a generic DB-uniqueness 500.
    existing = db.scalar(
        select(DataSource).where(
            DataSource.workspace_id == workspace_id, DataSource.name == display_name
        )
    )
    if existing is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"data source {display_name!r} already exists in this workspace",
        )

    # Quick metadata for the row card without re-parsing the CSV.
    try:
        size_bytes = csv_path.stat().st_size
    except OSError:
        size_bytes = None
    sha256: str | None = None
    try:
        h = hashlib.sha256()
        with csv_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        sha256 = h.hexdigest()
    except OSError:
        sha256 = None

    ds = DataSource(
        workspace_id=workspace_id,
        name=display_name,
        kind="csv_upload",
        description=f"Bundled sample · {sample.default_task}"
        + (f" · target={sample.target_1}" if sample.target_1 else ""),
        config={
            "path": str(csv_path),
            "source": "sample",
            "sample_name": sample.name,
            "sample_task": sample.default_task,
            "sample_target": sample.target_1,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "rows": sample.instances,
            "columns": None,  # filled by the refresh / profile path on first read
            "uploaded_at": datetime.now(UTC).isoformat(),
        },
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
        "description": ds.description,
        "config": dict(ds.config or {}),
        "created_at": ds.created_at,
        "created_by": ds.created_by,
    }
