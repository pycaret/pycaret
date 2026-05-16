"""ObjectStore protocol — every driver implements this.

The interface is intentionally narrow. Anything that's not implementable
across all back-ends (versioning, multipart upload, bucket creation)
lives on the driver itself, not the base. Application code targets only
``ObjectStore``.
"""

from __future__ import annotations

from typing import BinaryIO, Protocol


class ObjectStoreError(RuntimeError):
    """Raised by any driver on a non-retryable failure."""


class ObjectStore(Protocol):
    """The subset of an object-storage API our application actually uses."""

    scheme: str
    """The URI scheme this driver produces (``file``, ``s3``, ``gs``…)."""

    def put_bytes(self, key: str, blob: bytes) -> str:
        """Upload raw bytes and return a fully-qualified URI."""

    def put_file(self, key: str, fp: BinaryIO) -> str:
        """Streamed upload from an open file handle. Returns the URI."""

    def get_bytes(self, uri: str) -> bytes:
        """Fetch a previously-stored blob by URI. Raises on miss."""

    def open(self, uri: str) -> BinaryIO:
        """Open the URI for streaming reads. Caller closes the returned handle."""

    def exists(self, uri: str) -> bool:
        """Return True if the URI resolves to an existing object."""

    def delete(self, uri: str) -> None:
        """Remove the object. No-op on miss."""

    def size(self, uri: str) -> int | None:
        """Return the object size in bytes if knowable, else None."""

    def presigned_url(self, uri: str, *, expires_in: int = 3600) -> str | None:
        """Return a time-limited downloadable URL, or None when not applicable
        (``file://`` URIs return None — caller streams via the API)."""
