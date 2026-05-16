"""Phase 2 object-storage abstraction.

Goal: stop hard-coding local-fs paths into the DB. Every artifact (Run
pickle, dataset snapshot, notebook output, registered-model bytes)
lands on an ``ObjectStore`` driver — same interface whether the bytes
live on disk, in S3, in MinIO, or in GCS/Azure Blob.

DB rows store **URIs**: ``file:///abs/path/run.pkl`` (local) or
``s3://bucket/key`` (cloud). The driver knows how to read/write each
scheme; the application never assembles a path by hand again.

Selection is driven by ``PYCARET_STORAGE_BACKEND``:

- ``local`` (default) — pickles land under ``settings.artifact_dir``;
  URIs are ``file://``.
- ``s3`` — boto3 client, ``PYCARET_STORAGE_BUCKET`` is the destination;
  URIs are ``s3://bucket/key``.
- ``minio`` — same as s3 but with ``PYCARET_STORAGE_ENDPOINT_URL`` set
  to the MinIO endpoint.

Phase 2 ships the LocalFs driver fully wired into the orchestrator;
S3 / MinIO drivers are in place behind the env toggle, ready to flip on.
"""

from pycaret_server.storage.base import ObjectStore, ObjectStoreError
from pycaret_server.storage.factory import get_object_store, reset_for_tests

__all__ = [
    "ObjectStore",
    "ObjectStoreError",
    "get_object_store",
    "reset_for_tests",
]
