"""Phase 8 notebook runtime package.

A ``NotebookManager`` spawns isolated JupyterLab containers per session.
Two drivers ship:

- ``LocalManager`` — for dev. Doesn't actually spawn a container; just
  marks the session ``running`` with a fake port + token so the rest of
  the system can be exercised end-to-end without Docker.
- ``DockerManager`` — talks to the host Docker daemon via ``docker``
  CLI (kept dep-free; ``docker-py`` adds nothing we need). Spawns a
  ``jupyter/scipy-notebook`` (or any image set via env) container with
  resource caps, mounts a workspace-scoped volume as ``/data``, and
  returns a port the platform proxy can forward to.

Selection is via ``PYCARET_NOTEBOOK_BACKEND=local|docker``. Kubernetes
backend ships in a future cut; the protocol is the same.
"""

from pycaret_server.notebooks.base import (
    NotebookManager,
    NotebookManagerError,
    SessionDescriptor,
)
from pycaret_server.notebooks.factory import get_notebook_manager, reset_for_tests

__all__ = [
    "NotebookManager",
    "NotebookManagerError",
    "SessionDescriptor",
    "get_notebook_manager",
    "reset_for_tests",
]
