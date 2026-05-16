"""Run lifecycle integration tests.

Exercises:
- POST /api/v1/experiments/{id}/runs       (queue a run)
- GET  /api/v1/experiments/{id}/runs       (list)
- GET  /api/v1/runs/{id}                   (status)
- POST /api/v1/runs/{id}/wait              (block)
- GET  /api/v1/runs/{id}/events            (replay)
- WS   /api/v1/runs/{id}/events/ws         (live fan-out)

Uses the same per-test SQLite fixture as ``test_api.py`` plus a reset of the
orchestrator singleton so every test gets a clean thread-pool.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient]:
    """Fresh DB + app + orchestrator per test."""
    db_file = tmp_path / "test.db"
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
    from pycaret_server.runs.broker import event_broker
    from pycaret_server.runs.orchestrator import reset_orchestrator

    Base.metadata.create_all(sess_mod.engine)
    reset_orchestrator()
    event_broker.clear()

    app = create_app()
    with TestClient(app) as c:
        yield c

    reset_orchestrator()
    event_broker.clear()


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


def _headers(tok: dict) -> dict:
    return {"Authorization": f"Bearer {tok['access_token']}"}


def _make_classification_experiment(client: TestClient, headers: dict) -> str:
    """Create workspace + project + classification experiment. Returns experiment id."""
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    p_id = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "Iris"},
        headers=headers,
    ).json()["id"]
    r = client.post(
        f"/api/v1/projects/{p_id}/experiments",
        json={
            "name": "baseline",
            "task": "classification",
            "target": "target",
            # Keep setup deterministic + fast for the test suite.
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------- tests


def test_submit_run_validation(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    exp_id = _make_classification_experiment(client, headers)

    # No data source
    r = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "setup"},
        headers=headers,
    )
    assert r.status_code == 400
    assert "sklearn_dataset" in r.json()["detail"]

    # create without model_id
    r = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "create", "sklearn_dataset": "iris"},
        headers=headers,
    )
    assert r.status_code == 400

    # invalid plan
    r = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "explode", "sklearn_dataset": "iris"},
        headers=headers,
    )
    assert r.status_code == 400


def test_setup_run_lifecycle(client: TestClient) -> None:
    """A setup-only run reaches 'succeeded' and captures start/end events."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    exp_id = _make_classification_experiment(client, headers)

    r = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "setup", "sklearn_dataset": "iris"},
        headers=headers,
    )
    assert r.status_code == 202, r.text
    run = r.json()
    run_id = run["id"]
    assert run["status"] == "queued"
    assert run["snapshot"]["sklearn_dataset"] == "iris"

    # Block until done — 60s is generous for setup on tiny iris.
    r = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=60", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "succeeded", body
    assert body["duration_ms"] and body["duration_ms"] > 0
    assert body["error"] is None

    # Event stream recorded something.
    r = client.get(f"/api/v1/runs/{run_id}/events", headers=headers)
    assert r.status_code == 200
    events = r.json()
    kinds = [e["kind"] for e in events]
    assert "experiment.started" in kinds
    assert "experiment.fitted" in kinds


def test_create_run_produces_artifact(client: TestClient) -> None:
    """A 'create' run trains one model and persists a pipeline artifact."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    exp_id = _make_classification_experiment(client, headers)

    r = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "create",
            "model_id": "lr",
            "sklearn_dataset": "iris",
            "plan_params": {"verbose": False},
        },
        headers=headers,
    )
    assert r.status_code == 202, r.text
    run_id = r.json()["id"]

    r = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=120", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "succeeded", body.get("error")

    # Phase 0-v2: artifact lives on the per-Trial pickle, not on a
    # run-level ``pipeline.pkl`` — Trials are the unit of work.
    from pathlib import Path

    from pycaret_server.config import get_settings

    trials_dir = get_settings().artifact_dir / "runs" / run_id / "trials"
    assert trials_dir.is_dir(), trials_dir
    pickles = list(trials_dir.glob("*.pkl"))
    assert len(pickles) == 1, f"expected exactly one trial pickle, found {pickles}"
    assert pickles[0].stat().st_size > 0
    _ = Path


def test_list_runs_for_experiment(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    exp_id = _make_classification_experiment(client, headers)

    r = client.get(f"/api/v1/experiments/{exp_id}/runs", headers=headers)
    assert r.json() == []

    client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "setup", "sklearn_dataset": "iris"},
        headers=headers,
    )
    client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "setup", "sklearn_dataset": "wine"},
        headers=headers,
    )

    r = client.get(f"/api/v1/experiments/{exp_id}/runs", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_websocket_replay_after_run_finishes(client: TestClient) -> None:
    """Connect after the run terminates; WS replays stored events + sends run.closed."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    exp_id = _make_classification_experiment(client, headers)

    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "setup", "sklearn_dataset": "iris"},
        headers=headers,
    )
    assert submit.status_code == 202
    run_id = submit.json()["id"]

    # Wait for terminal state via the blocking endpoint.
    r = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=60", headers=headers)
    assert r.json()["status"] == "succeeded"

    # Now connect the WebSocket — it should replay and close cleanly.
    url = f"/api/v1/runs/{run_id}/events/ws?token={tokens['access_token']}"
    kinds: list[str] = []
    with client.websocket_connect(url) as ws:
        while True:
            msg = ws.receive_json()
            kinds.append(msg.get("kind", ""))
            if msg.get("kind") == "run.closed":
                break

    assert "run.closed" in kinds
    # At minimum we saw the experiment start + finish.
    assert "experiment.started" in kinds
    assert "experiment.fitted" in kinds


def test_ws_rejects_unauth(client: TestClient) -> None:
    """WebSocket without a token is closed with 4401."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    exp_id = _make_classification_experiment(client, headers)
    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "setup", "sklearn_dataset": "iris"},
        headers=headers,
    )
    run_id = submit.json()["id"]

    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(f"/api/v1/runs/{run_id}/events/ws"):
            pass
    assert excinfo.value.code == 4401
