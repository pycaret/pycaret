"""API route modules.

Each module registers a FastAPI ``APIRouter`` exported as ``router``. The app
factory mounts them at ``/api/v1``.
"""

from pycaret_server.api import (
    auth,
    describe,
    experiments,
    projects,
    runs,
    setup,
    workspaces,
)

__all__ = [
    "auth",
    "describe",
    "experiments",
    "projects",
    "runs",
    "setup",
    "workspaces",
]
