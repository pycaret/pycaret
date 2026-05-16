"""Session 33 — Notebook runtime hardening.

Smoke tests for:

- ``LocalManager``: start / is_alive / stop on the dev-stub backend.
  (Docker backend tests require a daemon and are skipped in CI; see
  ``test_docker_manager_*`` below.)
- ``GET /notebooks/{id}/content`` returns the default starter when no
  bytes are persisted yet.
- ``PUT /notebooks/{id}/content`` writes the ipynb to object storage
  + stamps ``object_uri``, and a subsequent ``GET`` round-trips the
  same JSON.
- ``GET /sessions/{id}/health`` reports liveness for an active local
  session (and ``alive=False`` once stopped).
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient]:
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("PYCARET_DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("PYCARET_JWT_SECRET", "test-secret-32-bytes-long-string!!")
    monkeypatch.setenv("PYCARET_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PYCARET_SECRETS_KEY", Fernet.generate_key().decode())
    # Force local notebook backend so the tests don't shell out to docker.
    monkeypatch.setenv("PYCARET_NOTEBOOK_BACKEND", "local")

    from pycaret_server.config import get_settings

    get_settings.cache_clear()

    from pycaret_server.crypto import reset_for_tests

    reset_for_tests()

    from pycaret_server.notebooks import reset_for_tests as reset_nb_for_tests

    reset_nb_for_tests()

    from pycaret_server.db import session as sess_mod

    sess_mod.engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    sess_mod.session_factory = sessionmaker(
        bind=sess_mod.engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    from pycaret_server.app import create_app
    from pycaret_server.db import Base

    Base.metadata.create_all(sess_mod.engine)

    with TestClient(create_app()) as c:
        yield c

    reset_nb_for_tests()
    reset_for_tests()


def _bootstrap(client: TestClient) -> dict:
    r = client.post(
        "/api/v1/setup/bootstrap",
        json={
            "email": "admin@example.com",
            "password": "supersecret",
            "display_name": "Admin",
            "workspace_name": "Default",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def _headers(tok: dict) -> dict:
    return {"Authorization": f"Bearer {tok['access_token']}"}


def _make_project(client: TestClient, headers: dict) -> tuple[str, str]:
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    p_id = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "NB"},
        headers=headers,
    ).json()["id"]
    return ws_id, p_id


def _make_notebook(client: TestClient, headers: dict, p_id: str) -> str:
    r = client.post(
        f"/api/v1/projects/{p_id}/notebooks",
        json={"name": "scratch"},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ============================================ local manager smoke


def test_local_manager_lifecycle() -> None:
    """Stub-backend manager spawns a descriptor + reports liveness."""
    from pycaret_server.notebooks.local import LocalManager

    mgr = LocalManager()
    d = mgr.start(
        session_id="sess-abc",
        notebook_id="nb",
        workspace_id="ws",
        user_id="user",
    )
    assert d.container_id == "local-sess-abc"
    assert d.port == 18888
    assert d.token  # randomly-generated
    assert mgr.is_alive(d.container_id) is True

    # Idempotent — same descriptor on re-start.
    d2 = mgr.start(
        session_id="sess-abc",
        notebook_id="nb",
        workspace_id="ws",
        user_id="user",
    )
    assert d2.container_id == d.container_id

    mgr.stop(d.container_id)
    assert mgr.is_alive(d.container_id) is False


# ============================================ content round-trip


def test_notebook_content_round_trip(client: TestClient) -> None:
    """PUT then GET returns the exact same ipynb content."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, p_id = _make_project(client, headers)
    nb_id = _make_notebook(client, headers, p_id)

    # GET before any save returns the canonical starter.
    r0 = client.get(f"/api/v1/notebooks/{nb_id}/content", headers=headers)
    assert r0.status_code == 200, r0.text
    body0 = r0.json()
    assert body0["object_uri"] is None
    assert body0["content"]["nbformat"] == 4
    assert isinstance(body0["content"]["cells"], list)

    payload = {
        "content": {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": ["print('hello pycaret')\n"],
                }
            ],
            "metadata": {
                "kernelspec": {"name": "python3", "display_name": "Python 3"}
            },
        }
    }
    r1 = client.put(
        f"/api/v1/notebooks/{nb_id}/content", json=payload, headers=headers
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["object_uri"] is not None
    assert body1["size_bytes"] > 0
    assert len(body1["sha256"]) == 64

    r2 = client.get(f"/api/v1/notebooks/{nb_id}/content", headers=headers)
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["object_uri"] == body1["object_uri"]
    assert body2["content"] == payload["content"]


def test_notebook_content_rejects_non_ipynb(client: TestClient) -> None:
    """Save with a payload missing ``nbformat`` is a 400."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, p_id = _make_project(client, headers)
    nb_id = _make_notebook(client, headers, p_id)

    r = client.put(
        f"/api/v1/notebooks/{nb_id}/content",
        json={"content": {"hello": "world"}},
        headers=headers,
    )
    assert r.status_code == 400


# ============================================ session health


def test_session_health_reports_liveness(client: TestClient) -> None:
    """Start a session on the local backend, hit /health, then stop it."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, p_id = _make_project(client, headers)
    nb_id = _make_notebook(client, headers, p_id)

    r = client.post(f"/api/v1/notebooks/{nb_id}/sessions", headers=headers)
    assert r.status_code == 201, r.text
    sess = r.json()
    session_id = sess["id"]

    h = client.get(f"/api/v1/sessions/{session_id}/health", headers=headers)
    assert h.status_code == 200, h.text
    health = h.json()
    assert health["alive"] is True
    assert health["status"] == "running"
    assert health["idle_seconds"] is not None
    assert health["idle_seconds"] < 60  # just spawned
    # idle_timeout default is 1800 → reap_in should be ~ 1800.
    assert health["reap_in_seconds"] >= 1700

    stop = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
    assert stop.status_code == 204, stop.text

    h2 = client.get(f"/api/v1/sessions/{session_id}/health", headers=headers)
    assert h2.status_code == 200
    assert h2.json()["alive"] is False
    assert h2.json()["status"] == "stopped"


# ============================================ docker backend smoke (opt-in)


@pytest.mark.skipif(
    os.environ.get("PYCARET_NB_DOCKER_SMOKE") != "1",
    reason="set PYCARET_NB_DOCKER_SMOKE=1 to run the docker-backed test",
)
def test_docker_manager_inspect_smoke() -> None:
    """Sanity-check the DockerManager wiring without spawning a container.

    Verifies the ``docker ps`` idempotency probe works end-to-end on
    machines that *do* have a daemon. Real container spawn is gated
    behind the env-var so CI doesn't need docker-in-docker.
    """
    from pycaret_server.notebooks.docker import DockerManager

    mgr = DockerManager(image="python:3.11-slim")
    # is_alive on a bogus container is False, not raising.
    assert mgr.is_alive("does-not-exist") is False
