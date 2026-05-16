"""Local-filesystem ObjectStore driver.

Default dev backend. Stores blobs under ``settings.artifact_dir`` and
returns ``file://`` URIs. The directory tree mirrors the key — so
``key="runs/abc/pipeline.pkl"`` lands at
``<artifact_dir>/runs/abc/pipeline.pkl`` — keeping the on-disk layout
human-debuggable.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

from pycaret_server.storage.base import ObjectStore, ObjectStoreError


class LocalFsObjectStore(ObjectStore):
    """File-system backed ObjectStore. Thread-safe (no shared state)."""

    scheme = "file"

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────── path helpers

    def _path_for(self, key: str) -> Path:
        # Strip any leading slashes so the key doesn't escape ``root``.
        cleaned = key.lstrip("/")
        p = (self._root / cleaned).resolve()
        if self._root not in p.parents and p != self._root:
            # Defence-in-depth: refuse keys that traversal-walk out.
            raise ObjectStoreError(f"key {key!r} escapes the artifact root")
        return p

    def _path_for_uri(self, uri: str) -> Path:
        if uri.startswith("file://"):
            parsed = urlparse(uri)
            # urlparse on Windows ``file:///C:/...`` keeps the colon —
            # handle both POSIX and Windows shapes.
            raw = parsed.path
            if raw.startswith("/") and len(raw) > 3 and raw[2] == ":":
                raw = raw[1:]
            return Path(raw)
        if uri.startswith("/") or (len(uri) > 2 and uri[1] == ":"):
            # Bare absolute path — accept for backwards-compat with the
            # pre-Phase-2 absolute-path strings already in the DB.
            return Path(uri)
        # Treat as a key relative to the root.
        return self._path_for(uri)

    def _uri_for(self, p: Path) -> str:
        return p.resolve().as_uri()

    # ─────────────────────────────────────────────────── protocol impl

    def put_bytes(self, key: str, blob: bytes) -> str:
        p = self._path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
        return self._uri_for(p)

    def put_file(self, key: str, fp: BinaryIO) -> str:
        p = self._path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as out:
            while True:
                chunk = fp.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        return self._uri_for(p)

    def get_bytes(self, uri: str) -> bytes:
        p = self._path_for_uri(uri)
        if not p.is_file():
            raise ObjectStoreError(f"file not found: {uri}")
        return p.read_bytes()

    def open(self, uri: str) -> BinaryIO:
        p = self._path_for_uri(uri)
        if not p.is_file():
            raise ObjectStoreError(f"file not found: {uri}")
        return p.open("rb")

    def exists(self, uri: str) -> bool:
        try:
            return self._path_for_uri(uri).is_file()
        except ObjectStoreError:
            return False

    def delete(self, uri: str) -> None:
        p = self._path_for_uri(uri)
        if p.is_file():
            p.unlink()

    def size(self, uri: str) -> int | None:
        try:
            return self._path_for_uri(uri).stat().st_size
        except (OSError, ObjectStoreError):
            return None

    def presigned_url(self, uri: str, *, expires_in: int = 3600) -> str | None:  # noqa: ARG002
        # Local-fs has no concept of a presigned URL — callers stream
        # through the API instead.
        return None
