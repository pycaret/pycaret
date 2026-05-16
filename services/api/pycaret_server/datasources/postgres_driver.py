"""Postgres datasource driver.

Connects to a Postgres database described in ``config`` (host, port,
database, optional schema) with the password in ``secret_value``. Reads
either a named table (``config['table']``) or a custom query
(``config['query']``).

``psycopg`` / ``sqlalchemy`` are already deps for the server itself,
so this driver pulls in no new install-time cost. The connection is
opened per call and closed in a ``finally`` — for tight loops a
connection-pooled variant lands when we have a real workload to
benchmark against.
"""

from __future__ import annotations

from typing import Any

from pycaret_server.datasources.base import (
    DatasourceDriver,
    DatasourceDriverError,
    SchemaDescription,
)


def _build_url(config: dict[str, Any], secret_value: str | None) -> str:
    host = config.get("host", "localhost")
    port = config.get("port", 5432)
    db = config.get("database")
    user = config.get("user", "postgres")
    if not db:
        raise DatasourceDriverError("postgres config missing 'database'")
    pw_part = f":{secret_value}" if secret_value else ""
    return f"postgresql+psycopg://{user}{pw_part}@{host}:{port}/{db}"


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier defensively. Refuses anything with
    embedded double-quotes since our use-case never legitimately needs
    one."""
    if '"' in name or "\x00" in name:
        raise DatasourceDriverError(f"invalid identifier {name!r}")
    return '"' + name + '"'


def _resolve_query(config: dict[str, Any]) -> str:
    """Return the SELECT used by introspect / read_full.

    Prefers ``config['query']`` when provided; otherwise builds a
    ``SELECT * FROM "schema"."table"`` from the table fields. Names are
    quoted to dodge keyword collisions; embedded quotes are refused.
    """
    q = config.get("query")
    if q:
        return q
    table = config.get("table")
    if not table:
        raise DatasourceDriverError("postgres config needs 'table' or 'query'")
    schema = config.get("schema")
    if schema:
        return f"SELECT * FROM {_quote_ident(schema)}.{_quote_ident(table)}"
    return f"SELECT * FROM {_quote_ident(table)}"


class PostgresDriver(DatasourceDriver):
    kind = "postgres"

    def test_connection(
        self, *, config: dict[str, Any], secret_value: str | None
    ) -> tuple[bool, str | None]:
        from sqlalchemy import create_engine, text

        try:
            eng = create_engine(_build_url(config, secret_value))
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    def list_tables(
        self, *, config: dict[str, Any], secret_value: str | None
    ) -> list[str]:
        from sqlalchemy import create_engine, text

        eng = create_engine(_build_url(config, secret_value))
        schema = config.get("schema") or "public"
        with eng.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT table_schema, table_name "
                    "FROM information_schema.tables "
                    "WHERE table_schema = :schema "
                    "ORDER BY table_name"
                ),
                {"schema": schema},
            ).all()
        return [f"{s}.{t}" for s, t in rows]

    def introspect(
        self,
        *,
        config: dict[str, Any],
        secret_value: str | None,
        sample_rows: int = 20,
    ) -> SchemaDescription:
        import pandas as pd
        from sqlalchemy import create_engine

        eng = create_engine(_build_url(config, secret_value))
        query = _resolve_query(config)
        # LIMIT only the sample; row_count is a separate cheap query.
        with eng.connect() as conn:
            df = pd.read_sql(f"{query} LIMIT {int(sample_rows)}", conn)
            # ``SELECT COUNT(*)`` against the same source — wrap in a
            # subquery so it works for both raw tables and user queries.
            count = pd.read_sql(
                f"SELECT COUNT(*) AS c FROM ({query}) AS src", conn
            )
        columns = []
        for col in df.columns:
            s = df[col]
            sample = s.dropna().iloc[0] if s.dropna().shape[0] else None
            columns.append(
                {
                    "name": str(col),
                    "dtype": str(s.dtype),
                    "nullable": True,
                    "sample": _jsonable(sample),
                }
            )
        return SchemaDescription(
            columns=columns,
            row_count=int(count.iloc[0, 0]) if len(count) else None,
            byte_count=None,
            sample_rows=[
                {k: _jsonable(v) for k, v in row.items()}
                for row in df.to_dict(orient="records")
            ],
        )

    def read_full(
        self,
        *,
        config: dict[str, Any],
        secret_value: str | None,
    ) -> Any:
        import pandas as pd
        from sqlalchemy import create_engine

        eng = create_engine(_build_url(config, secret_value))
        with eng.connect() as conn:
            return pd.read_sql(_resolve_query(config), conn)


def _jsonable(v: Any) -> Any:
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
    except Exception:  # noqa: BLE001
        pass
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)
