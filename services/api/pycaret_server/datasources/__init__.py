"""Phase 4 datasource driver layer.

Each backend (CSV upload, Postgres table, S3 file, …) implements the
``DatasourceDriver`` protocol. The factory resolves the right driver by
``DataSource.kind`` and the API/orchestrator never touches a backend
SDK directly.

Phase 4 ships two drivers end-to-end:

- ``csv_upload`` — reads a CSV that was already uploaded via
  ``data_sources`` and lives on the object store (Phase 2).
- ``postgres`` — pulls a table or query result from a Postgres
  Connection. Credentials live in ``secrets``.

Adding the next driver (Snowflake, BigQuery, S3 parquet) is a single
new module under this package + one ``DRIVERS`` entry.
"""

from pycaret_server.datasources.base import (
    DatasourceDriver,
    DatasourceDriverError,
    SchemaDescription,
)
from pycaret_server.datasources.factory import get_driver, list_kinds

__all__ = [
    "DatasourceDriver",
    "DatasourceDriverError",
    "SchemaDescription",
    "get_driver",
    "list_kinds",
]
