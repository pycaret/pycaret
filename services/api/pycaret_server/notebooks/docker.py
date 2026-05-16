"""DockerManager — spawns one JupyterLab container per session.

Talks to the host Docker daemon via the ``docker`` CLI. Picks an
ephemeral port; passes ``--memory`` / ``--cpus`` for resource caps;
mounts the workspace's artifact dir into ``/data`` so notebooks read
the same files as the API process.

The image defaults to ``quay.io/jupyter/scipy-notebook:latest`` but is
overridable per-environment via ``PYCARET_NOTEBOOK_IMAGE``. A custom
image bakes a ``pycaret_client`` SDK so notebooks can call back into
the platform without a manual install.
"""

from __future__ import annotations

import logging
import secrets
import socket
import subprocess
from typing import Any

from pycaret_server.notebooks.base import (
    NotebookManager,
    NotebookManagerError,
    SessionDescriptor,
)

_log = logging.getLogger(__name__)


def _pick_free_port() -> int:
    """OS-assigned free TCP port. Race-y but cheap; the worst case is
    one retry on a session start."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run(args: list[str]) -> tuple[int, str, str]:
    res = subprocess.run(args, capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


class DockerManager(NotebookManager):
    backend = "docker"

    def __init__(
        self,
        *,
        image: str = "quay.io/jupyter/scipy-notebook:latest",
        data_dir: str | None = None,
        network: str | None = None,
    ) -> None:
        self._image = image
        self._data_dir = data_dir
        self._network = network

    def start(
        self,
        *,
        session_id: str,
        notebook_id: str,  # noqa: ARG002 — kept for future per-notebook isolation
        workspace_id: str,
        user_id: str,  # noqa: ARG002
        kernel: str = "python3",  # noqa: ARG002
        cpu_limit: float | None = None,
        memory_mb_limit: int | None = None,
        env: dict[str, str] | None = None,
    ) -> SessionDescriptor:
        name = f"pyc-nb-{session_id[:12]}"
        # Idempotency: check if a container with this name already exists.
        rc, out, _err = _run(["docker", "ps", "-a", "-q", "-f", f"name={name}"])
        if rc == 0 and out:
            # Pull its port from inspect.
            cid = out.split("\n", 1)[0]
            port = self._inspect_port(cid) or _pick_free_port()
            return SessionDescriptor(container_id=cid, port=port, token="", image=self._image)

        port = _pick_free_port()
        token = secrets.token_urlsafe(24)
        args: list[str] = [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--publish",
            f"{port}:8888",
            "--label",
            f"pycaret.workspace={workspace_id}",
            "--label",
            f"pycaret.session={session_id}",
        ]
        if cpu_limit:
            args += ["--cpus", str(cpu_limit)]
        if memory_mb_limit:
            args += ["--memory", f"{memory_mb_limit}m"]
        if self._network:
            args += ["--network", self._network]
        if self._data_dir:
            args += ["-v", f"{self._data_dir}:/data:rw"]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        # Jupyter command + token. ``LabApp.token`` is the auth gate;
        # baseurl=/ matches our iframe routing.
        args += [
            self._image,
            "start-notebook.sh",
            "--NotebookApp.token=" + token,
            "--NotebookApp.password=",
            "--NotebookApp.allow_origin=*",
            "--NotebookApp.disable_check_xsrf=True",
        ]
        rc, out, err = _run(args)
        if rc != 0:
            raise NotebookManagerError(f"docker run failed: {err or out}")
        container_id = out.split("\n", 1)[0]
        return SessionDescriptor(
            container_id=container_id,
            port=port,
            token=token,
            image=self._image,
        )

    def _inspect_port(self, container_id: str) -> int | None:
        rc, out, _err = _run(
            [
                "docker",
                "inspect",
                "--format",
                "{{ (index (index .NetworkSettings.Ports \"8888/tcp\") 0).HostPort }}",
                container_id,
            ]
        )
        if rc != 0 or not out:
            return None
        try:
            return int(out)
        except ValueError:
            return None

    def stop(self, container_id: str) -> None:
        # ``docker rm -f`` covers both still-running and exited cases.
        _run(["docker", "rm", "-f", container_id])

    def is_alive(self, container_id: str) -> bool:
        rc, out, _err = _run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_id]
        )
        return rc == 0 and out.strip().lower() == "true"

    def proxy_url(self, descriptor: SessionDescriptor) -> str:
        return f"http://localhost:{descriptor.port}/lab?token={descriptor.token}"
