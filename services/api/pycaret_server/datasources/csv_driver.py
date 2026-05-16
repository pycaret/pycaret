"""CSV-upload datasource driver.

Reads a CSV that the user uploaded via ``POST /data-sources``. The
``config`` carries the object-storage URI (``file://`` or ``s3://``);
the driver pulls bytes through ``ObjectStore``.

The ``secret_value`` arg is unused (CSV uploads have no credentials),
kept in the signature for protocol uniformity.
"""

from __future__ import annotations

import io
from typing import Any

from pycaret_server.datasources.base import (
    DatasourceDriver,
    DatasourceDriverError,
    SchemaDescription,
)


class CsvUploadDriver(DatasourceDriver):
    kind = "csv_upload"

    def _read_bytes(self, config: dict[str, Any]) -> bytes:
        from pycaret_server.storage import get_object_store

        uri = config.get("path") or config.get("stored_path") or config.get("uri")
        if not uri:
            raise DatasourceDriverError(
                "csv_upload config missing 'path' / 'stored_path' / 'uri'"
            )
        store = get_object_store()
        return store.get_bytes(uri)

    def test_connection(
        self, *, config: dict[str, Any], secret_value: str | None  # noqa: ARG002
    ) -> tuple[bool, str | None]:
        try:
            self._read_bytes(config)
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    def list_tables(
        self, *, config: dict[str, Any], secret_value: str | None  # noqa: ARG002
    ) -> list[str]:
        # A CSV upload is one logical "table" — the file itself.
        return [str(config.get("name") or config.get("path") or "csv")]

    def introspect(
        self,
        *,
        config: dict[str, Any],
        secret_value: str | None,  # noqa: ARG002
        sample_rows: int = 20,
    ) -> SchemaDescription:
        import pandas as pd

        blob = self._read_bytes(config)
        df = pd.read_csv(io.BytesIO(blob))
        columns = []
        for col in df.columns:
            s = df[col]
            sample = s.dropna().iloc[0] if s.dropna().shape[0] else None
            columns.append(
                {
                    "name": str(col),
                    "dtype": str(s.dtype),
                    "nullable": bool(s.isna().any()),
                    "sample": _jsonable(sample),
                }
            )
        sample_df = df.head(sample_rows).to_dict(orient="records")
        return SchemaDescription(
            columns=columns,
            row_count=int(len(df)),
            byte_count=len(blob),
            sample_rows=[
                {k: _jsonable(v) for k, v in row.items()} for row in sample_df
            ],
        )

    def read_full(
        self,
        *,
        config: dict[str, Any],
        secret_value: str | None,  # noqa: ARG002
    ) -> Any:
        import pandas as pd

        return pd.read_csv(io.BytesIO(self._read_bytes(config)))


def _jsonable(v: Any) -> Any:
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:  # noqa: BLE001
        pass
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)
