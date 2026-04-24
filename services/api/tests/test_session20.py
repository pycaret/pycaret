"""Session 20: workspace members + X-PyCaret-Key middleware.

Tests cover:
  - Member CRUD + the last-admin guard on both demote + remove
  - API keys actually work as a bearer alternative end-to-end against
    a normal protected route (workspaces list)
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient]:
    db_file = tmp_path / "s20.db"
    monkeypatch.setenv("PYCARET_DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("PYCARET_JWT_SECRET", "test-secret-only-please")
    monkeypatch.setenv("PYCARET_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    from pycaret_server.config import get_settings

    get_settings.cache_clear()

    from pycaret_server.db import session as sess_mod

    sess_mod.engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    sess_mod.session_factory = sessionmaker(
        bind=sess_mod.engine, autocommit=False, autoflush=False, expire_on_commit=False
    )

    from pycaret_server.app import create_app
    from pycaret_server.db import Base

    Base.metadata.create_all(sess_mod.engine)

    with TestClient(create_app()) as c:
        yield c


# ---------------------------------------------------------------- helpers


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


def _auth(tok: dict) -> dict:
    return {"Authorization": f"Bearer {tok['access_token']}"}


def _create_second_user(client: TestClient) -> tuple[str, dict]:
    """Bootstrap is idempotent in terms of 'the first user is superuser'; to
    create a second user for membership tests we hit the engine directly
    (no user-invite endpoint in v1) by hashing the password and inserting.
    """
    # Easier path: hit the auth pieces directly through the same code the
    # bootstrap uses. We don't have a self-service signup in v1, so use
    # the password hasher + insert directly.
    from pycaret_server.auth.passwords import hash_password
    from pycaret_server.db import User, get_session

    session = get_session()
    try:
        u = User(
            email="alice@example.com",
            display_name="Alice",
            password_hash=hash_password("alicepassword"),
            is_active=True,
        )
        session.add(u)
        session.commit()
        session.refresh(u)
        user_id = u.id
    finally:
        session.close()

    # Log Alice in so we can exercise role-based access.
    r = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "alicepassword"},
    )
    assert r.status_code == 200, r.text
    return user_id, r.json()


# ============================================================ members CRUD


def test_list_members_shows_bootstrap_admin(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]

    r = client.get(f"/api/v1/workspaces/{ws}/members", headers=_auth(tok))
    assert r.status_code == 200
    members = r.json()
    assert len(members) == 1
    assert members[0]["email"] == "admin@example.com"
    assert members[0]["role"] == "admin"


def test_invite_existing_user_adds_member(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    _alice_id, _ = _create_second_user(client)

    r = client.post(
        f"/api/v1/workspaces/{ws}/members",
        headers=_auth(tok),
        json={"email": "alice@example.com", "role": "member"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["email"] == "alice@example.com"
    assert body["role"] == "member"

    # Duplicate invite is 409.
    r = client.post(
        f"/api/v1/workspaces/{ws}/members",
        headers=_auth(tok),
        json={"email": "alice@example.com", "role": "member"},
    )
    assert r.status_code == 409


def test_invite_unknown_email_returns_404(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    r = client.post(
        f"/api/v1/workspaces/{ws}/members",
        headers=_auth(tok),
        json={"email": "nobody@example.com", "role": "member"},
    )
    assert r.status_code == 404


def test_member_cannot_invite_other_members(client: TestClient) -> None:
    """Non-admins should be blocked from mutating membership."""
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    alice_id, alice_tok = _create_second_user(client)

    client.post(
        f"/api/v1/workspaces/{ws}/members",
        headers=_auth(tok),
        json={"email": "alice@example.com", "role": "member"},
    )

    # Alice (non-admin) tries to invite yet another user — denied.
    r = client.post(
        f"/api/v1/workspaces/{ws}/members",
        headers=_auth(alice_tok),
        json={"email": "admin@example.com", "role": "member"},
    )
    assert r.status_code == 403
    _ = alice_id


def test_change_role_promotes_and_demotes(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    alice_id, _ = _create_second_user(client)
    client.post(
        f"/api/v1/workspaces/{ws}/members",
        headers=_auth(tok),
        json={"email": "alice@example.com", "role": "member"},
    )

    # Promote alice to admin.
    r = client.patch(
        f"/api/v1/workspaces/{ws}/members/{alice_id}",
        headers=_auth(tok),
        json={"role": "admin"},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "admin"

    # Admin count is now 2 — safely demote alice back.
    r = client.patch(
        f"/api/v1/workspaces/{ws}/members/{alice_id}",
        headers=_auth(tok),
        json={"role": "member"},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "member"


def test_cannot_demote_last_admin(client: TestClient) -> None:
    """Demoting the only admin is refused with 400."""
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]

    # Find the admin's user id.
    members = client.get(f"/api/v1/workspaces/{ws}/members", headers=_auth(tok)).json()
    admin_id = members[0]["user_id"]

    r = client.patch(
        f"/api/v1/workspaces/{ws}/members/{admin_id}",
        headers=_auth(tok),
        json={"role": "member"},
    )
    assert r.status_code == 400
    assert "last admin" in r.json()["detail"]


def test_cannot_remove_last_admin(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    admin_id = client.get(f"/api/v1/workspaces/{ws}/members", headers=_auth(tok)).json()[0][
        "user_id"
    ]

    r = client.delete(f"/api/v1/workspaces/{ws}/members/{admin_id}", headers=_auth(tok))
    assert r.status_code == 400


def test_remove_member_succeeds_when_not_last_admin(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    alice_id, _ = _create_second_user(client)
    client.post(
        f"/api/v1/workspaces/{ws}/members",
        headers=_auth(tok),
        json={"email": "alice@example.com", "role": "member"},
    )
    r = client.delete(f"/api/v1/workspaces/{ws}/members/{alice_id}", headers=_auth(tok))
    assert r.status_code == 204

    members = client.get(f"/api/v1/workspaces/{ws}/members", headers=_auth(tok)).json()
    assert len(members) == 1
    assert members[0]["email"] == "admin@example.com"


# ============================================================ API key auth


def test_api_key_authenticates_a_protected_route(client: TestClient) -> None:
    """Mint an API key + use it on `/workspaces` instead of the JWT."""
    tok = _bootstrap(client)
    created = client.post("/api/v1/auth/api-keys", headers=_auth(tok), json={"name": "ci"}).json()
    plaintext = created["token"]

    r = client.get(
        "/api/v1/workspaces",
        headers={"X-PyCaret-Key": plaintext},
    )
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_revoked_api_key_is_rejected(client: TestClient) -> None:
    tok = _bootstrap(client)
    created = client.post(
        "/api/v1/auth/api-keys", headers=_auth(tok), json={"name": "doomed"}
    ).json()
    plaintext = created["token"]

    client.delete(f"/api/v1/auth/api-keys/{created['id']}", headers=_auth(tok))

    r = client.get("/api/v1/workspaces", headers={"X-PyCaret-Key": plaintext})
    assert r.status_code == 401
    assert "revoked" in r.json()["detail"].lower()


def test_bogus_api_key_is_rejected(client: TestClient) -> None:
    r = client.get("/api/v1/workspaces", headers={"X-PyCaret-Key": "pck_not_a_real_key"})
    assert r.status_code == 401


def test_expired_api_key_is_rejected(client: TestClient, monkeypatch) -> None:
    """Forge an expired key by flipping its expires_at backwards."""
    tok = _bootstrap(client)
    created = client.post(
        "/api/v1/auth/api-keys",
        headers=_auth(tok),
        json={"name": "short", "expires_in_days": 1},
    ).json()
    plaintext = created["token"]

    # Directly tweak expires_at to 1 day ago.
    from datetime import UTC, datetime, timedelta

    from pycaret_server.db import ApiKey, get_session

    session = get_session()
    try:
        key = session.get(ApiKey, created["id"])
        assert key is not None
        key.expires_at = datetime.now(UTC) - timedelta(days=1)
        session.commit()
    finally:
        session.close()

    r = client.get("/api/v1/workspaces", headers={"X-PyCaret-Key": plaintext})
    assert r.status_code == 401
    assert "expired" in r.json()["detail"].lower()


def test_jwt_takes_precedence_over_api_key_header(client: TestClient) -> None:
    """When both creds are present, the JWT wins — matches the common
    developer pattern of a long-lived key in env plus a short-lived session."""
    tok = _bootstrap(client)
    # Create + revoke an API key so that if it were consulted, we'd 401.
    created = client.post(
        "/api/v1/auth/api-keys", headers=_auth(tok), json={"name": "red-herring"}
    ).json()
    client.delete(f"/api/v1/auth/api-keys/{created['id']}", headers=_auth(tok))

    # Hit the route with both a valid JWT and a revoked key header — must succeed.
    r = client.get(
        "/api/v1/workspaces",
        headers={
            "Authorization": f"Bearer {tok['access_token']}",
            "X-PyCaret-Key": created["token"],
        },
    )
    assert r.status_code == 200


def test_missing_both_credentials_is_rejected(client: TestClient) -> None:
    r = client.get("/api/v1/workspaces")
    assert r.status_code == 401
    assert "missing" in r.json()["detail"].lower()
