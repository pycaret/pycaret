"""Session 23 — Model Library + admin/users routes.

- ``/workspaces/{id}/model-library`` lazily seeds from the engine's
  ``list_models`` on first read; subsequent reads serve DB rows.
- ``/admin/users`` is gated on ``user.is_superuser``; non-superusers get 403.
- Patching ``is_superuser`` / ``is_active`` works, with last-superuser and
  self-deactivation guards.
"""

from __future__ import annotations

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
    monkeypatch.setenv("PYCARET_JWT_SECRET", "test-secret-only-please")
    monkeypatch.setenv("PYCARET_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PYCARET_SECRETS_KEY", Fernet.generate_key().decode())

    from pycaret_server.config import get_settings

    get_settings.cache_clear()

    from pycaret_server.crypto import reset_for_tests

    reset_for_tests()

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
    from pycaret_server.runs.broker import event_broker
    from pycaret_server.runs.orchestrator import reset_orchestrator
    from pycaret_server.serving import reset_registry

    Base.metadata.create_all(sess_mod.engine)
    reset_orchestrator()
    reset_registry()
    event_broker.clear()

    with TestClient(create_app()) as c:
        yield c

    reset_orchestrator()
    reset_registry()
    event_broker.clear()
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


def _register_second_user(client: TestClient) -> dict:
    """Insert a second non-superuser directly into the DB.

    The platform doesn't expose a public /auth/register endpoint; non-bootstrap
    users normally arrive via workspace-member invites. Tests skip that ceremony
    and write the row directly.
    """
    import bcrypt

    from pycaret_server.db import User, get_session

    with get_session() as s:
        u = User(
            email="alice@example.com",
            display_name="Alice",
            password_hash=bcrypt.hashpw(b"alicepassword", bcrypt.gensalt()).decode(),
            is_superuser=False,
            is_active=True,
        )
        s.add(u)
        s.commit()
        s.refresh(u)
        return {"id": u.id, "email": u.email}


# ============================================================ model library


def test_model_library_lazy_seed_on_first_read(client: TestClient) -> None:
    """First GET seeds from the engine; subsequent reads serve DB rows."""
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]

    r = client.get(
        f"/api/v1/workspaces/{ws_id}/model-library?task=classification",
        headers=headers,
    )
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert len(items) > 0
    # All rows are for classification, all enabled by default.
    assert {i["task_type"] for i in items} == {"classification"}
    assert all(i["enabled"] for i in items)
    # Must include common classifiers.
    model_ids = {i["model_id"] for i in items}
    assert "lr" in model_ids
    assert "rf" in model_ids


def test_model_library_filter_by_task_rejects_unknown(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    r = client.get(
        f"/api/v1/workspaces/{ws_id}/model-library?task=quantum_computing",
        headers=headers,
    )
    assert r.status_code == 400


def test_model_library_patch_toggles_enabled(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    items = client.get(
        f"/api/v1/workspaces/{ws_id}/model-library?task=classification",
        headers=headers,
    ).json()["items"]
    target = next(i for i in items if i["model_id"] == "rf")

    r = client.patch(
        f"/api/v1/workspaces/{ws_id}/model-library/{target['id']}",
        headers=headers,
        json={"enabled": False, "custom_params": {"n_estimators": 200}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enabled"] is False
    assert body["custom_params"] == {"n_estimators": 200}


def test_model_library_sync_idempotent(client: TestClient) -> None:
    """Second sync is a no-op (no duplicate rows)."""
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    # Lazy-seed.
    initial = client.get(
        f"/api/v1/workspaces/{ws_id}/model-library?task=classification",
        headers=headers,
    ).json()["items"]

    # Resync — should not duplicate.
    r = client.post(
        f"/api/v1/workspaces/{ws_id}/model-library/sync?task=classification",
        headers=headers,
    )
    assert r.status_code == 200

    after = client.get(
        f"/api/v1/workspaces/{ws_id}/model-library?task=classification",
        headers=headers,
    ).json()["items"]
    assert len(after) == len(initial)


# =================================================================== admin


def test_admin_users_requires_superuser(client: TestClient) -> None:
    """Bootstrap user IS superuser; a freshly-registered second user is not."""
    bootstrap_tokens = _bootstrap(client)

    # Bootstrap user can list.
    r1 = client.get("/api/v1/admin/users", headers=_headers(bootstrap_tokens))
    assert r1.status_code == 200
    assert len(r1.json()["items"]) >= 1

    # Second user (non-superuser) is forbidden.
    _register_second_user(client)
    alice_tokens = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "alicepassword"},
    ).json()
    r2 = client.get("/api/v1/admin/users", headers=_headers(alice_tokens))
    assert r2.status_code == 403


def test_admin_patch_user_promote_then_demote(client: TestClient) -> None:
    """Bootstrap admin promotes Alice, then demotes her."""
    headers = _headers(_bootstrap(client))
    _register_second_user(client)

    users = client.get("/api/v1/admin/users", headers=headers).json()["items"]
    alice = next(u for u in users if u["email"] == "alice@example.com")
    assert alice["is_superuser"] is False

    # Promote.
    r = client.patch(
        f"/api/v1/admin/users/{alice['id']}",
        json={"is_superuser": True},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["is_superuser"] is True

    # Demote (the bootstrap admin still exists, so this is allowed).
    r = client.patch(
        f"/api/v1/admin/users/{alice['id']}",
        json={"is_superuser": False},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["is_superuser"] is False


def test_admin_cannot_demote_last_superuser(client: TestClient) -> None:
    """Demoting the only superuser raises 409."""
    bootstrap_tokens = _bootstrap(client)
    headers = _headers(bootstrap_tokens)
    me = client.get("/api/v1/auth/me", headers=headers).json()

    r = client.patch(
        f"/api/v1/admin/users/{me['id']}",
        json={"is_superuser": False},
        headers=headers,
    )
    assert r.status_code == 409


def test_admin_cannot_deactivate_self(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    me = client.get("/api/v1/auth/me", headers=headers).json()
    r = client.patch(
        f"/api/v1/admin/users/{me['id']}",
        json={"is_active": False},
        headers=headers,
    )
    assert r.status_code == 409
