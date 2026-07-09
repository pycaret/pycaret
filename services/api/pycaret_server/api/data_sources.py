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
import threading
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

# Profile cache: ``{data_source_id: (mtime_ns, size, sample_rows, payload)}``.
# Process-local; that's fine for now since the dev server runs in-process and
# the prod deploy is single-worker. When file mtime/size or requested
# ``sample_rows`` changes we recompute. Keeps the EDA dashboard instant on
# repeat loads of the same dataset.
# Tuple layout supports both CSV (mtime_ns, size, sample_rows, payload)
# and remote drivers ("remote" sentinel, sample_rows, None, payload).
_PROFILE_CACHE: dict[str, tuple] = {}
_PROFILE_CACHE_LOCK = threading.Lock()


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
        with open(target, encoding="utf-8") as f:
            rows_est = sum(1 for _ in f) - 1
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"could not parse CSV: {exc}") from exc

    ds = DataSource(
        workspace_id=workspace_id,
        name=name,
        kind="csv_upload",
        description=description,
        config={
            "path": str(target.resolve()),
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


@router.get("/data-sources/{data_source_id}/profile")
def dataset_profile(
    data_source_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    sample_rows: int = 10,
) -> dict:
    """Return a rich one-shot profile of the dataset.

    Drives the EDA dashboard. Single endpoint so the dashboard renders
    without a chain of round-trips:

    - shape, memory, duplicate count, total + per-column missing
    - per-column profile: dtype + kind (numeric / categorical / datetime /
      boolean / text), missing %, unique, cardinality, stats (numeric:
      min/max/mean/median/std/skew/kurtosis/q1/q3/zeros%/IQR-outliers),
      30-bin histogram (numeric), top-10 value counts (categorical)
    - Pearson correlation matrix between numeric columns
    - heuristic warnings (constant, ID-like, high cardinality, lots of
      missing)
    - small head sample for the Sample tab
    """
    ds = db.get(DataSource, data_source_id)
    if ds is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "data source not found")
    _require_access(user, db, ds.workspace_id)
    sample_rows = max(1, min(sample_rows, 50))
    cfg = ds.config or {}

    # CSV upload path keeps its fast file-stat cache; remote drivers
    # fall back to a (data_source_id, sample_rows) cache that the user
    # can bust with the "Refresh" button on the Data screen (it calls
    # POST /data-sources/{id}/refresh which clears this entry).
    if ds.kind == "csv_upload":
        csv_path = cfg.get("path") or cfg.get("stored_path")
        if not csv_path or not Path(str(csv_path)).is_file():
            raise HTTPException(
                status.HTTP_410_GONE,
                "uploaded file no longer exists on disk",
            )
        p = Path(str(csv_path))
        try:
            st = p.stat()
            mtime_ns, size = int(st.st_mtime_ns), int(st.st_size)
        except OSError:
            mtime_ns, size = 0, 0
        with _PROFILE_CACHE_LOCK:
            cached = _PROFILE_CACHE.get(data_source_id)
        if (
            cached is not None
            and cached[0] == mtime_ns
            and cached[1] == size
            and cached[2] == sample_rows
        ):
            return cached[3]

        try:
            import pandas as pd

            df = pd.read_csv(str(csv_path))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"could not read CSV: {exc}"
            ) from exc

        payload = _compute_dataset_profile(
            df, sample_rows=sample_rows, name=ds.name
        )
        with _PROFILE_CACHE_LOCK:
            _PROFILE_CACHE[data_source_id] = (mtime_ns, size, sample_rows, payload)
        return payload

    # Driver-backed source (postgres, s3, http, …) — go through the
    # registered DatasourceDriver.read_full so the EDA dashboard
    # works for every connection type without per-kind branching.
    from pycaret_server.api.connections import resolve_datasource_secret
    from pycaret_server.datasources import get_driver
    from pycaret_server.datasources.base import DatasourceDriverError

    with _PROFILE_CACHE_LOCK:
        cached = _PROFILE_CACHE.get(data_source_id)
    if (
        cached is not None
        and cached[0] == "remote"
        and cached[1] == sample_rows
    ):
        return cached[3]

    try:
        driver = get_driver(ds.kind)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"no driver registered for data source kind {ds.kind!r}: {exc}",
        ) from exc
    try:
        df = driver.read_full(
            config=dict(cfg),
            secret_value=resolve_datasource_secret(db, ds),
        )
    except DatasourceDriverError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"could not read {ds.kind} source: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"unexpected error reading data source: {exc}",
        ) from exc
    if df is None or getattr(df, "empty", False):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "data source returned no rows — nothing to profile",
        )

    payload = _compute_dataset_profile(
        df, sample_rows=sample_rows, name=ds.name
    )
    with _PROFILE_CACHE_LOCK:
        # ("remote", sample_rows) sentinel keeps remote entries from
        # being invalidated by a CSV upload's mtime check.
        _PROFILE_CACHE[data_source_id] = ("remote", sample_rows, None, payload)
    return payload


def _compute_dataset_profile(
    df, *, sample_rows: int = 10, name: str | None = None
) -> dict:
    """Pure pandas profile — no engine deps so unit tests can hit it cheap.

    Kept intentionally JSON-safe; numpy scalars are coerced to floats and
    NaNs are replaced with ``None`` so the response serialises cleanly
    without ``allow_nan`` games.
    """
    import math

    import numpy as np
    import pandas as pd

    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    memory_bytes = int(df.memory_usage(deep=True).sum())

    try:
        duplicates = int(df.duplicated().sum())
    except Exception:  # noqa: BLE001
        duplicates = 0
    cells = max(1, n_rows * n_cols)
    total_missing = int(df.isna().sum().sum())

    def _safe(x):
        if x is None:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        if isinstance(x, np.generic):
            v = x.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        if isinstance(x, (np.datetime64, pd.Timestamp)):
            return str(x)
        if isinstance(x, (str, int, bool)):
            return x
        if isinstance(x, float):
            return x
        return str(x)

    columns: list[dict] = []
    type_counts = {"numeric": 0, "categorical": 0, "datetime": 0, "boolean": 0, "text": 0}
    warnings: list[dict] = []

    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        non_null = max(0, n_rows - missing)
        unique = int(s.nunique(dropna=True))

        if pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        else:
            # Distinguish "categorical" (low cardinality) from free-form "text"
            ratio = (unique / non_null) if non_null else 0
            kind = "text" if (non_null > 0 and ratio > 0.5 and unique > 50) else "categorical"
        type_counts[kind] = type_counts.get(kind, 0) + 1

        col_info: dict = {
            "name": str(col),
            "dtype": str(s.dtype),
            "kind": kind,
            "missing": missing,
            "missing_pct": missing / max(1, n_rows),
            "unique": unique,
            "cardinality_pct": (unique / non_null) if non_null else 0.0,
            "is_constant": unique <= 1,
            "is_id_like": unique == n_rows and n_rows > 0,
            "stats": None,
            "histogram": [],
            "top_values": [],
            "sample_values": [],
        }

        clean = s.dropna()
        if kind == "numeric" and not clean.empty:
            try:
                q1 = float(clean.quantile(0.25))
                q3 = float(clean.quantile(0.75))
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = int(((clean < lo) | (clean > hi)).sum())
                col_info["stats"] = {
                    "min": _safe(clean.min()),
                    "max": _safe(clean.max()),
                    "mean": _safe(clean.mean()),
                    "median": _safe(clean.median()),
                    "std": _safe(clean.std()),
                    "skew": _safe(clean.skew()),
                    "kurtosis": _safe(clean.kurt()),
                    "q1": _safe(q1),
                    "q3": _safe(q3),
                    "zeros_pct": float((clean == 0).mean()),
                    "outliers_iqr": outliers,
                }
                # Use 30 bins; degenerate flat columns get 1 bin and that's fine.
                bins = max(1, min(30, max(1, int(np.sqrt(len(clean)))) + 1))
                if clean.min() == clean.max():
                    col_info["histogram"] = [
                        {
                            "bin_start": _safe(clean.min()),
                            "bin_end": _safe(clean.max()),
                            "count": int(len(clean)),
                        }
                    ]
                else:
                    counts, edges = np.histogram(clean.values, bins=bins)
                    col_info["histogram"] = [
                        {
                            "bin_start": float(edges[i]),
                            "bin_end": float(edges[i + 1]),
                            "count": int(counts[i]),
                        }
                        for i in range(len(counts))
                    ]
            except Exception:  # noqa: BLE001
                pass
        elif kind in ("categorical", "boolean", "text") and not clean.empty:
            try:
                vc = clean.value_counts(dropna=True).head(10)
                col_info["top_values"] = [
                    {
                        "value": str(idx),
                        "count": int(cnt),
                        "pct": float(cnt) / max(1, n_rows),
                    }
                    for idx, cnt in vc.items()
                ]
            except Exception:  # noqa: BLE001
                pass
        elif kind == "datetime" and not clean.empty:
            try:
                col_info["stats"] = {
                    "min": str(clean.min()),
                    "max": str(clean.max()),
                }
            except Exception:  # noqa: BLE001
                pass

        # First five distinct non-null values for the sidebar preview.
        try:
            col_info["sample_values"] = [
                _safe(v) for v in clean.drop_duplicates().head(5).tolist()
            ]
        except Exception:  # noqa: BLE001
            pass

        columns.append(col_info)

        # Heuristic warnings ─────────────────────────────────────────
        if col_info["is_constant"]:
            warnings.append(
                {
                    "kind": "constant",
                    "severity": "warn",
                    "column": col_info["name"],
                    "message": "Column is constant — drop before training.",
                }
            )
        if col_info["is_id_like"]:
            warnings.append(
                {
                    "kind": "id_like",
                    "severity": "warn",
                    "column": col_info["name"],
                    "message": "Every row has a unique value — looks like an ID column.",
                }
            )
        if (
            kind in ("categorical", "text")
            and col_info["cardinality_pct"] > 0.5
            and unique > 50
        ):
            warnings.append(
                {
                    "kind": "high_cardinality",
                    "severity": "info",
                    "column": col_info["name"],
                    "message": f"High cardinality ({unique} unique values).",
                }
            )
        if col_info["missing_pct"] > 0.5:
            warnings.append(
                {
                    "kind": "missing",
                    "severity": "warn",
                    "column": col_info["name"],
                    "message": f"{col_info['missing_pct'] * 100:.1f}% missing.",
                }
            )

    # Pearson correlation between numeric columns.
    numeric_names = [c["name"] for c in columns if c["kind"] == "numeric"]
    correlations: dict | None = None
    if len(numeric_names) >= 2:
        try:
            corr = df[numeric_names].corr().fillna(0).round(4)
            correlations = {
                "columns": numeric_names,
                "matrix": [
                    [float(v) for v in row] for row in corr.values.tolist()
                ],
            }
        except Exception:  # noqa: BLE001
            correlations = None

    if duplicates > 0:
        warnings.append(
            {
                "kind": "duplicates",
                "severity": "info",
                "column": None,
                "message": f"{duplicates} duplicate row{'s' if duplicates != 1 else ''}.",
            }
        )

    # Head sample for the Sample tab.
    try:
        sample = df.head(sample_rows).copy()
        for c in sample.columns:
            sample[c] = sample[c].map(_safe)
        sample_records = sample.to_dict(orient="records")
    except Exception:  # noqa: BLE001
        sample_records = []

    return {
        "name": name,
        "shape": {"rows": n_rows, "cols": n_cols},
        "memory_bytes": memory_bytes,
        "duplicates": duplicates,
        "missing_total": total_missing,
        "missing_pct": total_missing / cells,
        "type_counts": type_counts,
        "columns": columns,
        "correlations": correlations,
        "warnings": warnings,
        "sample": sample_records,
    }


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
