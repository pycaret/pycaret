"""S3-compatible ObjectStore driver.

Used for both AWS S3 and MinIO (the latter via ``endpoint_url``).
``boto3`` is an optional dependency — kept out of the base install so a
SQLite + local-fs dev path doesn't pay the cost.

The constructor refuses to start without an explicit ``bucket`` —
losing the bucket name silently was the most common foot-gun in v1.
"""

from __future__ import annotations

from typing import BinaryIO
from urllib.parse import urlparse

from pycaret_server.storage.base import ObjectStore, ObjectStoreError


class S3ObjectStore(ObjectStore):
    """Boto3-backed ObjectStore. Stores blobs at ``s3://<bucket>/<key>``."""

    scheme = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        if not bucket:
            raise ObjectStoreError(
                "S3ObjectStore requires a bucket name "
                "(set PYCARET_STORAGE_BUCKET)"
            )
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:  # noqa: BLE001
            raise ObjectStoreError(
                "boto3 is not installed; pip install pycaret-server[s3]"
            ) from exc
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    # ─────────────────────────────────────────────────── URI helpers

    def _uri_for(self, key: str) -> str:
        return f"s3://{self._bucket}/{key.lstrip('/')}"

    def _key_for(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ObjectStoreError(f"not an s3 URI: {uri}")
        if parsed.netloc != self._bucket:
            raise ObjectStoreError(
                f"URI bucket {parsed.netloc!r} != driver bucket {self._bucket!r}"
            )
        return parsed.path.lstrip("/")

    # ─────────────────────────────────────────────────── protocol impl

    def put_bytes(self, key: str, blob: bytes) -> str:
        self._client.put_object(Bucket=self._bucket, Key=key.lstrip("/"), Body=blob)
        return self._uri_for(key)

    def put_file(self, key: str, fp: BinaryIO) -> str:
        self._client.upload_fileobj(fp, self._bucket, key.lstrip("/"))
        return self._uri_for(key)

    def get_bytes(self, uri: str) -> bytes:
        key = self._key_for(uri)
        resp = self._client.get_object(Bucket=self._bucket, Key=key)
        return resp["Body"].read()

    def open(self, uri: str) -> BinaryIO:
        # boto3's StreamingBody quacks like a file enough for our reads.
        key = self._key_for(uri)
        resp = self._client.get_object(Bucket=self._bucket, Key=key)
        return resp["Body"]  # type: ignore[no-any-return]

    def exists(self, uri: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._key_for(uri))
            return True
        except Exception:  # noqa: BLE001 — botocore.exceptions.ClientError 404 etc
            return False

    def delete(self, uri: str) -> None:
        try:
            self._client.delete_object(
                Bucket=self._bucket, Key=self._key_for(uri)
            )
        except Exception:  # noqa: BLE001
            pass  # idempotent

    def size(self, uri: str) -> int | None:
        try:
            resp = self._client.head_object(
                Bucket=self._bucket, Key=self._key_for(uri)
            )
            return int(resp["ContentLength"])
        except Exception:  # noqa: BLE001
            return None

    def presigned_url(self, uri: str, *, expires_in: int = 3600) -> str | None:
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": self._key_for(uri)},
                ExpiresIn=expires_in,
            )
        except Exception:  # noqa: BLE001
            return None
