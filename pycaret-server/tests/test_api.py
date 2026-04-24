"""Integration tests for the PyCaret server API.

Uses FastAPI's TestClient + an in-memory SQLite to exercise every route.
Fresh DB per test — no state leakage.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient]:
    """Fresh DB + app per test."""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("PYCARET_DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("PYCARET_JWT_SECRET", "test-secret-only")
    monkeypatch.setenv("PYCARET_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    # settings is cached; re-read by clearing the lru_cache
    from pycaret_server.config import get_settings

    get_settings.cache_clear()

    # Rebind the engine to the test DB
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

    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- meta


def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["app"]
    assert r.json()["version"]


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_openapi_schema(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"]
    # All the expected route prefixes are present
    paths = schema["paths"]
    assert any(p.startswith("/api/v1/setup/") for p in paths)
    assert any(p.startswith("/api/v1/auth/") for p in paths)
    assert any(p.startswith("/api/v1/describe/") for p in paths)
    assert any(p.startswith("/api/v1/workspaces") for p in paths)


# ---------------------------------------------------------------- setup


def test_setup_status_before_bootstrap(client: TestClient) -> None:
    r = client.get("/api/v1/setup/status")
    assert r.status_code == 200
    body = r.json()
    assert body["is_bootstrapped"] is False
    assert body["user_count"] == 0


def test_bootstrap_creates_admin_and_workspace(client: TestClient) -> None:
    r = client.post(
        "/api/v1/setup/bootstrap",
        json={
            "email": "admin@example.com",
            "password": "supersecret",
            "display_name": "Admin",
            "workspace_name": "Test Workspace",
        },
    )
    assert r.status_code == 201, r.text
    tokens = r.json()
    assert tokens["access_token"]
    assert tokens["refresh_token"]
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] > 0

    # Second attempt is rejected
    r2 = client.post(
        "/api/v1/setup/bootstrap",
        json={
            "email": "other@example.com",
            "password": "anotherone",
            "workspace_name": "Second Try",
        },
    )
    assert r2.status_code == 409

    # Status flipped
    r3 = client.get("/api/v1/setup/status")
    assert r3.json()["is_bootstrapped"] is True
    assert r3.json()["user_count"] == 1


# ---------------------------------------------------------------- auth + workspaces


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
    assert r.status_code == 201
    return r.json()


def _auth(tok: dict) -> dict:
    return {"Authorization": f"Bearer {tok['access_token']}"}


def test_auth_me(client: TestClient) -> None:
    tokens = _bootstrap(client)
    r = client.get("/api/v1/auth/me", headers=_auth(tokens))
    assert r.status_code == 200
    me = r.json()
    assert me["email"] == "admin@example.com"
    assert me["is_superuser"] is True


def test_login_and_refresh(client: TestClient) -> None:
    _bootstrap(client)
    r = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "supersecret"},
    )
    assert r.status_code == 200
    tokens = r.json()

    # Refresh returns a new pair
    r2 = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert r2.status_code == 200
    new_tokens = r2.json()
    assert new_tokens["refresh_token"] != tokens["refresh_token"]

    # Old refresh token is now revoked
    r3 = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert r3.status_code == 401


def test_workspace_crud(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _auth(tokens)

    r = client.get("/api/v1/workspaces", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) == 1  # the bootstrap workspace
    first = r.json()[0]
    assert first["name"] == "Default"

    # Create a second
    r = client.post(
        "/api/v1/workspaces",
        json={"name": "Another", "description": "second"},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    second_id = r.json()["id"]

    # Dup fails
    r = client.post(
        "/api/v1/workspaces",
        json={"name": "Another"},
        headers=headers,
    )
    assert r.status_code == 409

    # Delete
    r = client.delete(f"/api/v1/workspaces/{second_id}", headers=headers)
    assert r.status_code == 204
    r = client.get("/api/v1/workspaces", headers=headers)
    assert len(r.json()) == 1


def test_project_crud(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _auth(tokens)
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]

    # Empty list first
    r = client.get(f"/api/v1/workspaces/{ws_id}/projects", headers=headers)
    assert r.json() == []

    # Create
    r = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "Churn", "description": "predict churn", "tags": ["v1"]},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    p_id = r.json()["id"]

    # Fetch
    r = client.get(f"/api/v1/workspaces/{ws_id}/projects/{p_id}", headers=headers)
    assert r.status_code == 200
    assert r.json()["name"] == "Churn"
    assert r.json()["tags"] == ["v1"]


def test_experiment_crud(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _auth(tokens)
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    p_id = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "Churn"},
        headers=headers,
    ).json()["id"]

    r = client.post(
        f"/api/v1/projects/{p_id}/experiments",
        json={
            "name": "baseline",
            "task": "classification",
            "target": "churn",
            "setup_params": {"session_id": 42, "fold": 5},
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    exp = r.json()
    assert exp["task"] == "classification"
    assert exp["setup_params"]["fold"] == 5

    # Bad task
    r_bad = client.post(
        f"/api/v1/projects/{p_id}/experiments",
        json={"name": "bad", "task": "elephant"},
        headers=headers,
    )
    assert r_bad.status_code == 400


# ---------------------------------------------------------------- describe (engine proxy)


def test_describe_models(client: TestClient) -> None:
    r = client.get("/api/v1/describe/models?task=classification")
    assert r.status_code == 200
    cards = r.json()
    assert len(cards) >= 15
    assert any(c["id"] == "lr" for c in cards)


def test_describe_model_not_found(client: TestClient) -> None:
    r = client.get("/api/v1/describe/models/nope?task=classification")
    assert r.status_code == 404


def test_describe_setup_params_roundtrip(client: TestClient) -> None:
    r = client.get("/api/v1/describe/setup-params?task=classification")
    assert r.status_code == 200
    schema = r.json()
    assert schema["task"] == "classification"
    assert len(schema["parameters"]) >= 10
    assert "Preprocessing" in schema["groups"]


def test_unauthenticated_routes_reject(client: TestClient) -> None:
    r = client.get("/api/v1/workspaces")
    assert r.status_code == 401
    r = client.get("/api/v1/auth/me")
    assert r.status_code == 401
