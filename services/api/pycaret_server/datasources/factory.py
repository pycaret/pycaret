"""DatasourceDriver factory.

Resolves a driver by ``DataSource.kind``. Drivers register themselves
here so adding a new backend is a single new import + ``register()``
call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pycaret_server.datasources.base import DatasourceDriver


_REGISTRY: dict[str, "DatasourceDriver"] = {}


def register(kind: str, driver: "DatasourceDriver") -> None:
    """Register a driver under its kind discriminator."""
    _REGISTRY[kind] = driver


def get_driver(kind: str) -> "DatasourceDriver":
    """Return the driver for a kind. Raises if no driver is registered."""
    if kind not in _REGISTRY:
        raise KeyError(
            f"no datasource driver registered for kind={kind!r}; "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[kind]


def list_kinds() -> list[str]:
    """Return the registered driver kinds (for ``GET /datasources/kinds``)."""
    return sorted(_REGISTRY)


# Built-in drivers register on import.
def _bootstrap() -> None:
    from pycaret_server.datasources.csv_driver import CsvUploadDriver
    from pycaret_server.datasources.postgres_driver import PostgresDriver

    register("csv_upload", CsvUploadDriver())
    register("postgres", PostgresDriver())


_bootstrap()
