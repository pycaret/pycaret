"""DatasourceDriver protocol — every backend implements this.

The protocol is intentionally small. Anything backend-specific (Snowflake's
warehouse selection, BigQuery's location, S3's read-using-DuckDB
optimization) hangs off the driver itself, not the base interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class DatasourceDriverError(RuntimeError):
    """Raised by any driver on a non-retryable failure."""


@dataclass
class SchemaDescription:
    """Self-describing schema returned by :meth:`DatasourceDriver.introspect`."""

    columns: list[dict[str, Any]]
    """[{"name": str, "dtype": str, "nullable": bool, "sample": Any}, ...]"""

    row_count: int | None
    """Total rows; None when the backend doesn't cheaply know."""

    byte_count: int | None
    """On-disk byte count; None when not applicable (DB tables)."""

    sample_rows: list[dict[str, Any]] | None
    """First N rows for the dataset profile view."""


class DatasourceDriver(Protocol):
    """The subset of a data backend we actually use."""

    kind: str
    """``csv_upload`` | ``postgres`` | ``snowflake`` | ``bigquery`` | …"""

    def test_connection(
        self, *, config: dict[str, Any], secret_value: str | None
    ) -> tuple[bool, str | None]:
        """Return (ok, error_message)."""

    def list_tables(
        self, *, config: dict[str, Any], secret_value: str | None
    ) -> list[str]:
        """List addressable objects (tables, views, files) for the connection.

        CSV drivers return a single-item list (the file). DB drivers return
        ``schema.table`` strings. Returning an empty list is valid (no
        permissions, empty bucket).
        """

    def introspect(
        self,
        *,
        config: dict[str, Any],
        secret_value: str | None,
        sample_rows: int = 20,
    ) -> SchemaDescription:
        """Read schema + a sample. Doesn't materialise the full dataset."""

    def read_full(
        self,
        *,
        config: dict[str, Any],
        secret_value: str | None,
    ) -> Any:
        """Return a pandas DataFrame of the entire object/table.

        Callers materialise the result; the driver doesn't stream. For
        anything that won't fit in memory, the future ``read_stream``
        method (post-v1) is the right answer.
        """
