"""API route modules.

Each module registers a FastAPI ``APIRouter`` exported as ``router``. The app
factory mounts them at ``/api/v1``.
"""

from pycaret_server.api import (
    analyses,
    api_keys,
    audit,
    auth,
    connections,
    data_sources,
    deployments,
    describe,
    drift,
    experiments,
    git_repos,
    governance,
    llm,
    members,
    monitoring,
    notebooks,
    plots,
    projects,
    queue_admin,
    registry,
    runs,
    setup,
    trials,
    workspaces,
)

__all__ = [
    "analyses",
    "api_keys",
    "audit",
    "auth",
    "connections",
    "data_sources",
    "deployments",
    "describe",
    "drift",
    "experiments",
    "git_repos",
    "governance",
    "llm",
    "members",
    "monitoring",
    "notebooks",
    "plots",
    "projects",
    "queue_admin",
    "registry",
    "runs",
    "setup",
    "trials",
    "workspaces",
]
