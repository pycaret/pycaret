"""LocalManager — dev-only stub that records sessions but spawns nothing.

Lets the rest of the system be exercised end-to-end (create notebook,
start session, view "running" status, stop) on a machine that doesn't
have Docker. The frontend renders a placeholder iframe with a hint
about enabling the docker backend.
"""

from __future__ import annotations

import secrets
import threading
from typing import Any

from pycaret_server.notebooks.base import NotebookManager, SessionDescriptor


_state_lock = threading.Lock()
_state: dict[str, dict[str, Any]] = {}


class LocalManager(NotebookManager):
    backend = "local"

    def start(
        self,
        *,
        session_id: str,
        notebook_id: str,  # noqa: ARG002
        workspace_id: str,  # noqa: ARG002
        user_id: str,  # noqa: ARG002
        kernel: str = "python3",  # noqa: ARG002
        cpu_limit: float | None = None,  # noqa: ARG002
        memory_mb_limit: int | None = None,  # noqa: ARG002
        env: dict[str, str] | None = None,  # noqa: ARG002
    ) -> SessionDescriptor:
        with _state_lock:
            existing = _state.get(session_id)
            if existing is not None:
                return SessionDescriptor(
                    container_id=existing["container_id"],
                    port=existing["port"],
                    token=existing["token"],
                    image="local-stub",
                )
            descriptor = SessionDescriptor(
                container_id=f"local-{session_id}",
                # Pretend port — never bound to anything.
                port=18888,
                token=secrets.token_urlsafe(24),
                image="local-stub",
            )
            _state[session_id] = {
                "container_id": descriptor.container_id,
                "port": descriptor.port,
                "token": descriptor.token,
                "alive": True,
            }
            return descriptor

    def stop(self, container_id: str) -> None:
        with _state_lock:
            for sid, info in list(_state.items()):
                if info.get("container_id") == container_id:
                    info["alive"] = False
                    _state.pop(sid, None)

    def is_alive(self, container_id: str) -> bool:
        with _state_lock:
            for info in _state.values():
                if info.get("container_id") == container_id and info.get("alive"):
                    return True
        return False

    def proxy_url(self, descriptor: SessionDescriptor) -> str:
        # Frontend-only marker so the UI can render the placeholder
        # "notebook backend unavailable" tile instead of a broken iframe.
        return f"pycaret://local-notebook?cid={descriptor.container_id}"
