"""API route modules.

Each module registers a FastAPI ``APIRouter`` exported as ``router``. The app
factory mounts them at ``/api/v1``.
"""

from pycaret_server.api import (
    api_keys,
    audit,
    auth,
    data_sources,
    deployments,
    describe,
    drift,
    experiments,
    llm,
    members,
    projects,
    runs,
    setup,
    workspaces,
)

__all__ = [
    "api_keys",
    "audit",
    "auth",
    "data_sources",
    "deployments",
    "describe",
    "drift",
    "experiments",
    "llm",
    "members",
    "projects",
    "runs",
    "setup",
    "workspaces",
]
