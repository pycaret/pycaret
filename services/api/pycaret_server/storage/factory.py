"""ObjectStore factory — single entry point the rest of the app uses.

The driver is selected once at process start based on
``PYCARET_STORAGE_BACKEND`` and cached. Tests can override via
``reset_for_tests`` so each fixture spins up its own local-fs root.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from pycaret_server.config import get_settings
from pycaret_server.storage.local_fs import LocalFsObjectStore

if TYPE_CHECKING:
    from pycaret_server.storage.base import ObjectStore


_lock = threading.Lock()
_singleton: "ObjectStore | None" = None


def get_object_store() -> "ObjectStore":
    """Return the process-wide ObjectStore singleton.

    Resolved from ``PYCARET_STORAGE_BACKEND`` on first call. Subsequent
    calls reuse the same instance; ``reset_for_tests`` clears it.
    """
    global _singleton
    with _lock:
        if _singleton is not None:
            return _singleton
        settings = get_settings()
        backend = (settings.storage_backend or "local").lower()
        if backend == "local":
            _singleton = LocalFsObjectStore(root=settings.artifact_dir)
        elif backend in ("s3", "minio"):
            from pycaret_server.storage.s3 import S3ObjectStore  # late import

            _singleton = S3ObjectStore(
                bucket=settings.storage_bucket or "",
                region=settings.storage_region,
                endpoint_url=settings.storage_endpoint_url,
                access_key=settings.storage_access_key,
                secret_key=settings.storage_secret_key,
            )
        else:
            raise RuntimeError(
                f"unknown PYCARET_STORAGE_BACKEND={backend!r}; "
                "expected 'local' | 's3' | 'minio'"
            )
        return _singleton


def reset_for_tests() -> None:
    """Drop the cached singleton so the next call re-resolves from settings."""
    global _singleton
    with _lock:
        _singleton = None
