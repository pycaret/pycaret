"""API route modules.

Each module registers a FastAPI ``APIRouter`` exported as ``router``. The app
factory mounts them at ``/api/v1``.
"""

from pycaret_server.api import (
    api_keys,
    auth,
    data_sources,
    deployments,
    describe,
    experiments,
    llm,
    projects,
    runs,
    setup,
    workspaces,
)

__all__ = [
    "api_keys",
    "auth",
    "data_sources",
    "deployments",
    "describe",
    "experiments",
    "llm",
    "projects",
    "runs",
    "setup",
    "workspaces",
]
