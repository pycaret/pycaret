"""NotebookManager factory."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from pycaret_server.config import get_settings

if TYPE_CHECKING:
    from pycaret_server.notebooks.base import NotebookManager


_lock = threading.Lock()
_singleton: "NotebookManager | None" = None


def get_notebook_manager() -> "NotebookManager":
    """Return the process-wide NotebookManager singleton."""
    global _singleton
    with _lock:
        if _singleton is not None:
            return _singleton
        settings = get_settings()
        backend = (settings.notebook_backend or "local").lower()
        if backend == "docker":
            from pycaret_server.notebooks.docker import DockerManager

            _singleton = DockerManager(
                image=settings.notebook_image
                or "quay.io/jupyter/scipy-notebook:latest",
                data_dir=settings.notebook_data_dir,
                network=settings.notebook_network,
            )
        else:
            from pycaret_server.notebooks.local import LocalManager

            _singleton = LocalManager()
        return _singleton


def reset_for_tests() -> None:
    global _singleton
    with _lock:
        _singleton = None
