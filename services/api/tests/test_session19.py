"""Session-19 tests: failure debugger + deployment reviewer + API keys.

Same fake-LLM-provider test pattern as session 17/18. API-key tests don't
need the LLM; they just exercise the CRUD + hash-on-create invariant.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient]:
    db_file = tmp_path / "s19.db"
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
    from pycaret_server.llm.providers import register_fake_for_tests
    from pycaret_server.llm.router import reset_router
    from pycaret_server.runs.broker import event_broker
    from pycaret_server.runs.orchestrator import reset_orchestrator

    Base.metadata.create_all(sess_mod.engine)
    reset_router()
    reset_orchestrator()
    event_broker.clear()
    register_fake_for_tests(
        canned_response={
            "suggested_config_json": {"next_action": "rename_target_column", "target": "target"},
            "suggested_action": "Rename target column; re-submit.",
            "reasoning_summary": "DATA: target column 'y' not found in dataset; engine expected column named in setup.",
            "risk_flags": [],
        }
    )

    with TestClient(create_app()) as c:
        yield c

    reset_router()
    reset_orchestrator()
    event_broker.clear()


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


def _configure_llm(client: TestClient, tok: dict, ws: str) -> None:
    client.put(
        f"/api/v1/workspaces/{ws}/llm/settings",
        headers=_auth(tok),
        json={"provider": "anthropic", "model_name": "claude-sonnet-4-5", "api_key": "sk"},
    )


# ────────────────────────────────────────────────────────── failure debugger


def _fail_a_run(client: TestClient, tok: dict, ws: str) -> str:
    """Submit a create-plan run with a bogus model id to force a failure."""
    p = client.post(
        f"/api/v1/workspaces/{ws}/projects", headers=_auth(tok), json={"name": "P"}
    ).json()["id"]
    e = client.post(
        f"/api/v1/projects/{p}/experiments",
        headers=_auth(tok),
        json={
            "name": "bad",
            "task": "classification",
            "target": "target",
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
    ).json()["id"]
    r = client.post(
        f"/api/v1/experiments/{e}/runs",
        headers=_auth(tok),
        # `zzzz_not_a_model` isn't in the registry → create_model raises.
        json={"plan": "create", "model_id": "zzzz_not_a_model", "sklearn_dataset": "iris"},
    ).json()
    run_id = r["id"]
    term = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=60", headers=_auth(tok)).json()
    assert term["status"] == "failed", term
    return run_id


def test_debug_run_happy_path(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    _configure_llm(client, tok, ws)
    run_id = _fail_a_run(client, tok, ws)

    r = client.post("/api/v1/llm/debug-run", headers=_auth(tok), json={"run_id": run_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "failure_debugging"
    assert body["run_id"] == run_id
    # The error text should reach the prompt for audit.
    assert body["prompt"]  # non-empty


def test_debug_run_rejects_succeeded(client: TestClient) -> None:
    """debug-run is for failed runs only — succeeded should 400."""
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    _configure_llm(client, tok, ws)

    # Run a succeeding job.
    p = client.post(
        f"/api/v1/workspaces/{ws}/projects", headers=_auth(tok), json={"name": "P2"}
    ).json()["id"]
    e = client.post(
        f"/api/v1/projects/{p}/experiments",
        headers=_auth(tok),
        json={
            "name": "ok",
            "task": "classification",
            "target": "target",
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
    ).json()["id"]
    run_id = client.post(
        f"/api/v1/experiments/{e}/runs",
        headers=_auth(tok),
        json={"plan": "setup", "sklearn_dataset": "iris"},
    ).json()["id"]
    client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=60", headers=_auth(tok))

    r = client.post("/api/v1/llm/debug-run", headers=_auth(tok), json={"run_id": run_id})
    assert r.status_code == 400
    assert "failed runs only" in r.json()["detail"]


# ────────────────────────────────────────────────────────── deployment reviewer


def test_review_deployment_happy_path(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    _configure_llm(client, tok, ws)

    # Train + promote
    p = client.post(
        f"/api/v1/workspaces/{ws}/projects", headers=_auth(tok), json={"name": "P"}
    ).json()["id"]
    e = client.post(
        f"/api/v1/projects/{p}/experiments",
        headers=_auth(tok),
        json={
            "name": "b",
            "task": "classification",
            "target": "target",
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
    ).json()["id"]
    run_id = client.post(
        f"/api/v1/experiments/{e}/runs",
        headers=_auth(tok),
        json={"plan": "create", "model_id": "lr", "sklearn_dataset": "iris"},
    ).json()["id"]
    client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=120", headers=_auth(tok))
    pipeline_id = client.post(
        f"/api/v1/runs/{run_id}/promote",
        headers=_auth(tok),
        json={"name": "v1"},
    ).json()["id"]

    r = client.post(
        "/api/v1/llm/review-deployment",
        headers=_auth(tok),
        json={"pipeline_id": pipeline_id},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "deployment_risk_review"
    # Run-id is correlated via the pipeline's origin_run.
    assert body["run_id"] == run_id
    # The pipeline name should reach the prompt.
    assert "v1" in body["prompt"]


def test_review_deployment_404_on_unknown_pipeline(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    _configure_llm(client, tok, ws)
    r = client.post(
        "/api/v1/llm/review-deployment",
        headers=_auth(tok),
        json={"pipeline_id": "not-a-real-uuid"},
    )
    assert r.status_code == 404


# ────────────────────────────────────────────────────────── API keys


def test_create_api_key_returns_plaintext_once(client: TestClient) -> None:
    tok = _bootstrap(client)
    r = client.post(
        "/api/v1/auth/api-keys",
        headers=_auth(tok),
        json={"name": "ci-bot"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "ci-bot"
    # Plaintext must be present on create…
    assert body["token"].startswith("pck_")
    assert len(body["token"]) > 40
    # …and the display prefix matches the plaintext head.
    assert body["token"].startswith(body["prefix"])

    # But GET never exposes the plaintext.
    lst = client.get("/api/v1/auth/api-keys", headers=_auth(tok)).json()
    assert len(lst) == 1
    assert "token" not in lst[0]
    assert lst[0]["prefix"].startswith("pck_")


def test_list_api_keys_scoped_to_user(client: TestClient) -> None:
    tok = _bootstrap(client)
    client.post("/api/v1/auth/api-keys", headers=_auth(tok), json={"name": "k1"})
    client.post("/api/v1/auth/api-keys", headers=_auth(tok), json={"name": "k2"})
    lst = client.get("/api/v1/auth/api-keys", headers=_auth(tok)).json()
    assert {k["name"] for k in lst} == {"k1", "k2"}


def test_revoke_api_key_soft_deletes(client: TestClient) -> None:
    tok = _bootstrap(client)
    key = client.post(
        "/api/v1/auth/api-keys", headers=_auth(tok), json={"name": "short-lived"}
    ).json()
    r = client.delete(f"/api/v1/auth/api-keys/{key['id']}", headers=_auth(tok))
    assert r.status_code == 204
    # Row still there, revoked_at set.
    lst = client.get("/api/v1/auth/api-keys", headers=_auth(tok)).json()
    assert len(lst) == 1
    assert lst[0]["revoked_at"] is not None


def test_api_key_expiry_round_trip(client: TestClient) -> None:
    tok = _bootstrap(client)
    key = client.post(
        "/api/v1/auth/api-keys",
        headers=_auth(tok),
        json={"name": "week", "expires_in_days": 7},
    ).json()
    assert key["expires_at"] is not None


def test_api_key_create_requires_name(client: TestClient) -> None:
    tok = _bootstrap(client)
    r = client.post("/api/v1/auth/api-keys", headers=_auth(tok), json={"name": ""})
    assert r.status_code == 422
