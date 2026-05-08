"""Session 22 — control plane progress: secrets encryption + prediction logs + trials.

Three concerns wired in this slice:

1. ``pycaret_server.crypto`` — Fernet round-trip + legacy plaintext fallback
   + encryption is applied when LLM provider settings are written.
2. ``PredictionLog`` — every ``/deployments/{slug}/predict`` call writes one
   row (ok or error path); a ``GET /deployments/{id}/prediction-logs``
   endpoint paginates them.
3. ``Trial`` — every successful ``compare_models`` plan persists one row per
   leaderboard entry; ``GET /runs/{id}/trials`` lists them ordered by rank.
"""

from __future__ import annotations

import io
from collections.abc import Generator

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ---------------------------------------------------------------- fixtures


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


# ------------------------------------------------------------------- helpers


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
        json={"name": "Demo"},
        headers=headers,
    ).json()["id"]
    return ws_id, p_id


def _make_classification_experiment(client: TestClient, headers: dict) -> tuple[str, str]:
    ws_id, p_id = _make_project(client, headers)
    r = client.post(
        f"/api/v1/projects/{p_id}/experiments",
        json={
            "name": "exp",
            "task": "classification",
            "target": "target",
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return ws_id, r.json()["id"]


def _wait_for_run(client: TestClient, headers: dict, run_id: str, timeout: int = 120) -> dict:
    r = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s={timeout}", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


# ============================================================= crypto


def test_crypto_round_trip(monkeypatch):
    """``encrypt(decrypt(x)) == x`` for any utf-8 string."""
    monkeypatch.setenv("PYCARET_SECRETS_KEY", Fernet.generate_key().decode())
    from pycaret_server.config import get_settings
    from pycaret_server.crypto import decrypt, encrypt, is_encrypted, reset_for_tests

    get_settings.cache_clear()
    reset_for_tests()

    plaintext = "sk-ant-deadbeefcafef00d-1234567890"
    cipher = encrypt(plaintext)
    assert cipher != plaintext
    assert is_encrypted(cipher)
    assert decrypt(cipher) == plaintext


def test_crypto_legacy_plaintext_passes_through(monkeypatch):
    """Stored values without the ENC: prefix are returned as-is for back-compat."""
    monkeypatch.setenv("PYCARET_SECRETS_KEY", Fernet.generate_key().decode())
    from pycaret_server.config import get_settings
    from pycaret_server.crypto import decrypt, is_encrypted, reset_for_tests

    get_settings.cache_clear()
    reset_for_tests()

    legacy = "this-was-stored-before-encryption"
    assert not is_encrypted(legacy)
    assert decrypt(legacy) == legacy  # legacy plaintext untouched


def test_crypto_decrypt_with_wrong_key_raises(monkeypatch):
    """Rotating ``PYCARET_SECRETS_KEY`` makes old ciphertext unreadable."""
    monkeypatch.setenv("PYCARET_SECRETS_KEY", Fernet.generate_key().decode())
    from pycaret_server.config import get_settings
    from pycaret_server.crypto import decrypt, encrypt, reset_for_tests

    get_settings.cache_clear()
    reset_for_tests()
    cipher = encrypt("super-secret")

    # Rotate the key and re-resolve the singleton.
    monkeypatch.setenv("PYCARET_SECRETS_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    reset_for_tests()

    with pytest.raises(RuntimeError, match="Could not decrypt"):
        decrypt(cipher)


def test_llm_settings_api_key_stored_encrypted(client: TestClient) -> None:
    """The PUT /llm/settings endpoint encrypts api_key before persisting."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]

    api_key_plaintext = "sk-ant-test-zzz-9876"
    r = client.put(
        f"/api/v1/workspaces/{ws_id}/llm/settings",
        json={
            "provider": "anthropic",
            "api_key": api_key_plaintext,
            "model_name": "claude-3-5-sonnet-latest",
            "enabled": True,
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text

    # Inspect the DB row directly to confirm it isn't stored plaintext.
    from pycaret_server.db import LLMProviderSetting, get_session

    with get_session() as s:
        row = s.query(LLMProviderSetting).filter_by(workspace_id=ws_id).one()
        assert row.api_key_encrypted is not None
        assert row.api_key_encrypted != api_key_plaintext
        assert row.api_key_encrypted.startswith("ENC:v1:")


# ====================================================== prediction logs


def _full_compare_promote_deploy(
    client: TestClient, headers: dict
) -> tuple[str, str, str, str]:
    """Train on iris → promote → deploy. Returns (ws_id, deployment_id, slug, run_id)."""
    ws_id, exp_id = _make_classification_experiment(client, headers)

    run = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "compare", "sklearn_dataset": "iris"},
        headers=headers,
    )
    assert run.status_code == 202, run.text
    run_id = run.json()["id"]
    out = _wait_for_run(client, headers, run_id)
    assert out["status"] == "succeeded", out.get("error")

    promo = client.post(
        f"/api/v1/runs/{run_id}/promote",
        json={"name": "iris-pipeline"},
        headers=headers,
    )
    assert promo.status_code == 201, promo.text
    pipe_id = promo.json()["id"]

    dep = client.post(
        f"/api/v1/pipelines/{pipe_id}/deployments",
        json={"endpoint_slug": "iris-test", "auth_mode": "workspace"},
        headers=headers,
    )
    assert dep.status_code == 201, dep.text
    return ws_id, dep.json()["id"], "iris-test", run_id


def test_prediction_log_written_on_success(client: TestClient) -> None:
    """A successful ``/predict`` writes one PredictionLog row."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    ws_id, dep_id, slug, _ = _full_compare_promote_deploy(client, headers)

    rows = [
        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    ]
    r = client.post(
        f"/api/v1/deployments/{slug}/predict",
        json={"rows": rows},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "request_id" in body

    # Logs endpoint shows our row.
    logs = client.get(f"/api/v1/deployments/{dep_id}/prediction-logs", headers=headers)
    assert logs.status_code == 200, logs.text
    items = logs.json()["items"]
    assert len(items) == 1
    log = items[0]
    assert log["status"] == "ok"
    assert log["n_rows"] == 1
    assert log["request_id"] == body["request_id"]
    assert log["latency_ms"] is not None and log["latency_ms"] >= 0
    assert log["request_sample"] == rows
    assert isinstance(log["response_sample"], list) and len(log["response_sample"]) == 1


def test_prediction_log_written_on_error(client: TestClient) -> None:
    """A bad payload still produces an error PredictionLog row."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, dep_id, slug, _ = _full_compare_promote_deploy(client, headers)

    # Send rows missing all required features → pipeline.predict will raise.
    bad = client.post(
        f"/api/v1/deployments/{slug}/predict",
        json={"rows": [{"unrelated_column": 1}]},
        headers=headers,
    )
    assert bad.status_code == 400, bad.text

    logs = client.get(
        f"/api/v1/deployments/{dep_id}/prediction-logs?status_filter=error",
        headers=headers,
    )
    items = logs.json()["items"]
    assert len(items) == 1
    assert items[0]["status"] == "error"
    assert items[0]["error"]
    assert items[0]["latency_ms"] is None


def test_prediction_logs_pagination(client: TestClient) -> None:
    """``limit`` + ``offset`` both work; results are newest-first."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, dep_id, slug, _ = _full_compare_promote_deploy(client, headers)

    rows = [
        {"sepal length (cm)": 5.0, "sepal width (cm)": 3.5, "petal length (cm)": 1.3, "petal width (cm)": 0.2}
    ]
    for _ in range(3):
        r = client.post(
            f"/api/v1/deployments/{slug}/predict",
            json={"rows": rows},
            headers=headers,
        )
        assert r.status_code == 200

    page1 = client.get(
        f"/api/v1/deployments/{dep_id}/prediction-logs?limit=2&offset=0",
        headers=headers,
    ).json()["items"]
    page2 = client.get(
        f"/api/v1/deployments/{dep_id}/prediction-logs?limit=2&offset=2",
        headers=headers,
    ).json()["items"]
    assert len(page1) == 2 and len(page2) == 1
    # No overlap.
    ids = {x["id"] for x in page1} | {x["id"] for x in page2}
    assert len(ids) == 3


# =================================================================== trials


def test_trials_persisted_for_compare_run(client: TestClient) -> None:
    """A compare plan run produces queryable Trial rows."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)

    run = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "compare", "sklearn_dataset": "iris"},
        headers=headers,
    )
    assert run.status_code == 202, run.text
    run_id = run.json()["id"]
    out = _wait_for_run(client, headers, run_id)
    assert out["status"] == "succeeded", out.get("error")

    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers)
    assert trials.status_code == 200, trials.text
    items = trials.json()["items"]
    assert len(items) >= 2, items
    # Sorted by rank ascending.
    ranks = [t["rank"] for t in items]
    assert ranks == sorted(ranks)
    # Exactly one is_best.
    bests = [t for t in items if t["is_best"]]
    assert len(bests) == 1
    assert bests[0]["rank"] == 1
    # Each has a model_id and metrics dict.
    for t in items:
        assert t["model_id"]
        assert isinstance(t["metrics"], dict)


def test_trials_route_404_for_unknown_run(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    r = client.get(
        "/api/v1/runs/00000000-0000-0000-0000-000000000000/trials",
        headers=headers,
    )
    assert r.status_code == 404
