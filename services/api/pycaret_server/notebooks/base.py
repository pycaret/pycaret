"""NotebookManager protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class NotebookManagerError(RuntimeError):
    """Raised by any manager when a session lifecycle op fails."""


@dataclass
class SessionDescriptor:
    """What a Manager returns after spawning (or finding) a session.

    Stamped onto the ``NotebookSession`` row by the API layer.
    """

    container_id: str
    port: int
    token: str
    image: str | None = None


class NotebookManager(Protocol):
    """The container-spawning surface our application uses."""

    backend: str
    """Free-form id: ``local`` | ``docker`` | ``k8s``."""

    def start(
        self,
        *,
        session_id: str,
        notebook_id: str,
        workspace_id: str,
        user_id: str,
        kernel: str = "python3",
        cpu_limit: float | None = None,
        memory_mb_limit: int | None = None,
        env: dict[str, str] | None = None,
    ) -> SessionDescriptor:
        """Spawn / claim a Jupyter container for this session.

        Idempotent: calling twice with the same ``session_id`` returns
        the existing descriptor.
        """

    def stop(self, container_id: str) -> None:
        """Terminate the container. No-op on miss."""

    def is_alive(self, container_id: str) -> bool:
        """Return True when the container is still running."""

    def proxy_url(self, descriptor: SessionDescriptor) -> str:
        """Return the URL the platform's iframe loads.

        For ``local`` this is a marker URL the frontend renders as
        "container manager unavailable, run pycaret-server in
        docker-compose to enable notebooks". For ``docker`` it's the
        per-port localhost URL with the token query string.
        """
