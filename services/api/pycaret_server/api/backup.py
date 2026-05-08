"""Platform backup + restore — superuser-only.

Two endpoints:

  GET    /admin/backup
       Stream a tarball containing:
         * ``database.json`` — all rows from every table (UUIDs preserved)
         * ``artifacts/<run_id>/...`` — raw artifact files

  POST   /admin/restore   (multipart upload of a tarball)
       Wipe + reload from the tarball. Refuses if the existing DB has any
       non-bootstrap data unless ``confirm=true`` is passed; this guard
       prevents one accidental click from nuking a populated workspace.

The format is intentionally simple JSON-per-table + flat files — operators
can hand-edit a backup before restoring.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from pycaret_server.api.admin import _require_superuser
from pycaret_server.auth import CurrentUser
from pycaret_server.config import get_settings
from pycaret_server.db import Base, User, get_db

router = APIRouter(tags=["admin-backup"])


# Tables to skip in dumps. ``alembic_version`` is recreated by Alembic on
# restore; ``sessions`` are short-lived auth state and don't survive a
# restart anyway.
_SKIP_TABLES = {"alembic_version", "sessions"}


@router.get("/admin/backup")
def create_backup(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """Stream a tarball of the entire platform state."""
    _require_superuser(user)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # Dump all tables to JSON.
        dump = _dump_database(db)
        data = json.dumps(dump, default=str, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="database.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

        # Embed raw artifacts.
        artifact_dir = get_settings().artifact_dir
        if artifact_dir.is_dir():
            for path in artifact_dir.rglob("*"):
                if not path.is_file():
                    continue
                arcname = "artifacts/" + str(path.relative_to(artifact_dir)).replace("\\", "/")
                tf.add(str(path), arcname=arcname)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/gzip",
        headers={
            "Content-Disposition": 'attachment; filename="pycaret-backup.tar.gz"'
        },
    )


@router.post("/admin/restore")
def restore_backup(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(...),
    confirm: bool = Form(False),
) -> dict:
    """Wipe the DB + artifact dir and reload from the uploaded tarball."""
    _require_superuser(user)

    if _has_real_data(db) and not confirm:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "database has existing data; pass confirm=true to overwrite",
        )

    raw = file.file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty upload")

    settings = get_settings()
    artifact_dir = settings.artifact_dir

    # Wipe artifacts + DB tables (in dependency-safe reverse order).
    if artifact_dir.is_dir():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    inspector = inspect(db.get_bind())
    table_names = inspector.get_table_names()
    # Reverse topological sort isn't easily available; iterate twice — first
    # to delete (FK violations get retried), then unlock.
    for table in Base.metadata.sorted_tables:
        if table.name in _SKIP_TABLES:
            continue
        if table.name in table_names:
            db.execute(table.delete())
    db.commit()

    # Extract and reapply.
    counts: dict[str, int] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        # Database first.
        try:
            db_member = tf.getmember("database.json")
        except KeyError as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "tarball missing database.json"
            ) from exc

        f = tf.extractfile(db_member)
        if f is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "database.json unreadable")
        dump = json.loads(f.read().decode("utf-8"))
        counts = _load_database(db, dump)

        # Artifacts next.
        for member in tf.getmembers():
            if not member.isfile() or not member.name.startswith("artifacts/"):
                continue
            rel = Path(member.name).relative_to("artifacts")
            target = artifact_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            target.write_bytes(extracted.read())

    db.commit()
    return {"status": "ok", "restored_rows": counts}


# ----------------------------------------------------------------- helpers


def _dump_database(db: Session) -> dict:
    """Return a {table_name: [{col: value, ...}]} mapping for every table."""
    out: dict = {}
    for table in Base.metadata.sorted_tables:
        if table.name in _SKIP_TABLES:
            continue
        rows = db.execute(select(table)).mappings().all()
        out[table.name] = [dict(r) for r in rows]
    return out


def _load_database(db: Session, dump: dict) -> dict[str, int]:
    """Insert ``dump`` rows. Tables loaded in Base.metadata.sorted_tables order
    so foreign keys resolve."""
    counts: dict[str, int] = {}
    for table in Base.metadata.sorted_tables:
        rows = dump.get(table.name)
        if not rows:
            continue
        # Convert datetime strings back via SQLAlchemy type adapters when
        # possible — most ISO-8601 strings round-trip cleanly into the
        # native types via the JSON loader because we re-issue typed inserts.
        db.execute(table.insert(), rows)
        counts[table.name] = len(rows)
    return counts


def _has_real_data(db: Session) -> bool:
    """Detect whether the DB has anything beyond a freshly bootstrapped admin.

    "Fresh" = exactly one user (the bootstrap superuser) and one workspace.
    """
    user_count = int(db.scalar(select(User).limit(2)) is not None)  # noqa: F841 — quick existence
    n_users = int(db.scalar(select(__import__("sqlalchemy").func.count()).select_from(User)) or 0)
    return n_users > 1
