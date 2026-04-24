"""Integration tests for the LLM advisory surface.

Uses `FakeLLMProvider` registered under every provider name so the router's
full dispatch path is exercised without hitting real APIs.
"""

from __future__ import annotations

import io
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient]:
    db_file = tmp_path / "llm.db"
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
        bind=sess_mod.engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    from pycaret_server.app import create_app
    from pycaret_server.db import Base
    from pycaret_server.llm.providers import register_fake_for_tests
    from pycaret_server.llm.router import reset_router

    Base.metadata.create_all(sess_mod.engine)
    reset_router()
    # Install the fake under every provider name so real SDKs are never hit.
    register_fake_for_tests(
        canned_response={
            "suggested_config_json": {
                "task_type": "classification",
                "target": "target",
                "primary_metric": "auc",
            },
            "suggested_action": "Run a classification compare on this dataset.",
            "reasoning_summary": "Target column has 3 classes; dataset is balanced and small.",
            "risk_flags": ["small_sample"],
        }
    )

    app = create_app()
    with TestClient(app) as c:
        yield c

    reset_router()


# ------------------------------------------------------------ helpers


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


def _auth(tokens: dict) -> dict:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


# ------------------------------------------------------------ settings CRUD


def test_settings_empty_initially(client: TestClient) -> None:
    tokens = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tokens)).json()[0]["id"]
    r = client.get(f"/api/v1/workspaces/{ws}/llm/settings", headers=_auth(tokens))
    assert r.status_code == 200
    assert r.json() is None


def test_upsert_settings_admin_gated_and_hides_key(client: TestClient) -> None:
    tokens = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tokens)).json()[0]["id"]

    payload = {
        "provider": "anthropic",
        "api_key": "sk-test-abc-123",
        "model_name": "claude-sonnet-4-5",
        "enabled": True,
    }
    r = client.put(
        f"/api/v1/workspaces/{ws}/llm/settings",
        json=payload,
        headers=_auth(tokens),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "anthropic"
    assert body["model_name"] == "claude-sonnet-4-5"
    assert body["enabled"] is True
    assert body["has_api_key"] is True
    assert "api_key" not in body  # never leaked
    assert "api_key_encrypted" not in body


def test_upsert_settings_rejects_unknown_provider(client: TestClient) -> None:
    tokens = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tokens)).json()[0]["id"]
    r = client.put(
        f"/api/v1/workspaces/{ws}/llm/settings",
        json={"provider": "elephant", "model_name": "x", "api_key": "k"},
        headers=_auth(tokens),
    )
    assert r.status_code == 400


def test_switching_provider_disables_previous(client: TestClient) -> None:
    """Uniqueness is on (workspace_id, provider); switching providers flips
    the prior `enabled` row to False but keeps the row for audit."""
    tokens = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tokens)).json()[0]["id"]
    client.put(
        f"/api/v1/workspaces/{ws}/llm/settings",
        json={"provider": "anthropic", "model_name": "c", "api_key": "a"},
        headers=_auth(tokens),
    )
    client.put(
        f"/api/v1/workspaces/{ws}/llm/settings",
        json={"provider": "openai", "model_name": "gpt-x", "api_key": "b"},
        headers=_auth(tokens),
    )
    active = client.get(f"/api/v1/workspaces/{ws}/llm/settings", headers=_auth(tokens)).json()
    assert active["provider"] == "openai"
    assert active["model_name"] == "gpt-x"


def test_test_connection_against_fake_provider(client: TestClient) -> None:
    tokens = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tokens)).json()[0]["id"]
    client.put(
        f"/api/v1/workspaces/{ws}/llm/settings",
        json={"provider": "anthropic", "model_name": "c", "api_key": "a"},
        headers=_auth(tokens),
    )
    r = client.post(
        f"/api/v1/workspaces/{ws}/llm/test-connection",
        headers=_auth(tokens),
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    assert r.json()["provider"] == "anthropic"
    assert r.json()["latency_ms"] is not None


def test_test_connection_400_when_no_provider_configured(client: TestClient) -> None:
    tokens = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tokens)).json()[0]["id"]
    r = client.post(f"/api/v1/workspaces/{ws}/llm/test-connection", headers=_auth(tokens))
    assert r.status_code == 400


# ------------------------------------------------------------ analyze-dataset


def _upload_iris_csv(client: TestClient, tokens: dict, ws_id: str) -> str:
    import sklearn.datasets as sk

    bundle = sk.load_iris(as_frame=True)
    df = bundle.frame.copy()
    if bundle.target.name != "target":
        df = df.rename(columns={bundle.target.name: "target"})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    r = client.post(
        f"/api/v1/workspaces/{ws_id}/data-sources/upload",
        headers=_auth(tokens),
        data={"name": "iris"},
        files={"file": ("iris.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_analyze_dataset_happy_path(client: TestClient) -> None:
    """End-to-end: upload CSV → configure LLM → analyze → read back via history."""
    tokens = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tokens)).json()[0]["id"]

    # Must configure LLM before consulting.
    client.put(
        f"/api/v1/workspaces/{ws}/llm/settings",
        json={"provider": "anthropic", "model_name": "claude-sonnet-4-5", "api_key": "k"},
        headers=_auth(tokens),
    )

    ds_id = _upload_iris_csv(client, tokens, ws)

    r = client.post(
        "/api/v1/llm/analyze-dataset",
        headers=_auth(tokens),
        json={"data_source_id": ds_id, "workspace_id": ws, "task_type_hint": "classification"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "dataset_analysis"
    assert body["provider"] == "anthropic"
    assert body["model_name"] == "claude-sonnet-4-5"
    assert body["generated_config_json"]["task_type"] == "classification"
    assert body["response_json"]["suggested_action"]
    assert body["latency_ms"] is not None
    # Prompt is captured for audit.
    assert "iris" in body["prompt"].lower() or "target" in body["prompt"].lower()

    # History lists it.
    hist = client.get(f"/api/v1/workspaces/{ws}/llm/consultations", headers=_auth(tokens)).json()
    assert len(hist) == 1
    assert hist[0]["id"] == body["id"]

    # Single fetch.
    one = client.get(f"/api/v1/llm/consultations/{body['id']}", headers=_auth(tokens)).json()
    assert one["id"] == body["id"]
    assert one["response_json"] == body["response_json"]


def test_analyze_dataset_requires_configured_llm(client: TestClient) -> None:
    tokens = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tokens)).json()[0]["id"]
    ds_id = _upload_iris_csv(client, tokens, ws)
    r = client.post(
        "/api/v1/llm/analyze-dataset",
        headers=_auth(tokens),
        json={"data_source_id": ds_id, "workspace_id": ws},
    )
    assert r.status_code == 400
    assert "No LLM provider configured" in r.json()["detail"]


def test_analyze_dataset_rejects_non_csv_source(client: TestClient) -> None:
    tokens = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tokens)).json()[0]["id"]
    client.put(
        f"/api/v1/workspaces/{ws}/llm/settings",
        json={"provider": "anthropic", "model_name": "c", "api_key": "k"},
        headers=_auth(tokens),
    )
    # Register an s3 connector — LLM analyze is csv-only for v1.
    s3 = client.post(
        f"/api/v1/workspaces/{ws}/data-sources",
        headers=_auth(tokens),
        json={"name": "bucket", "kind": "s3", "config": {"bucket": "b", "key": "k"}},
    ).json()
    r = client.post(
        "/api/v1/llm/analyze-dataset",
        headers=_auth(tokens),
        json={"data_source_id": s3["id"], "workspace_id": ws},
    )
    assert r.status_code == 400
