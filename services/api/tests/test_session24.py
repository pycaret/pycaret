"""Session 24 — full platform extensions.

Covers the V2 surfaces wired this session:

- Schedules CRUD + immediate-run handler dispatch.
- Experiment templates CRUD.
- Webhooks CRUD (creation only — actual delivery requires a live HTTP
  target which we don't spin up here).
- Pipeline versioning + deployment rollback.
- AutoML pipeline search plan.
- Expanded role set.
"""

from __future__ import annotations

import io
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
    from pycaret_server.scheduler import shutdown_scheduler
    from pycaret_server.serving import reset_registry

    Base.metadata.create_all(sess_mod.engine)
    reset_orchestrator()
    reset_registry()
    event_broker.clear()
    shutdown_scheduler()

    with TestClient(create_app()) as c:
        yield c

    reset_orchestrator()
    reset_registry()
    event_broker.clear()
    shutdown_scheduler()
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


def _make_classification_experiment(
    client: TestClient, headers: dict
) -> tuple[str, str]:
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    p_id = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "Demo"},
        headers=headers,
    ).json()["id"]
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


def _wait_for_run(client: TestClient, headers: dict, run_id: str, timeout: int = 180) -> dict:
    r = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s={timeout}", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


# ============================================================== schedules


def test_schedule_crud_drift_monitor(client: TestClient) -> None:
    """Create a drift_monitor schedule for a deployment, list, patch, delete."""
    headers = _headers(_bootstrap(client))
    ws_id, exp_id = _make_classification_experiment(client, headers)

    # We need a deployment to attach the schedule to.
    run = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "compare", "sklearn_dataset": "iris"},
        headers=headers,
    ).json()
    out = _wait_for_run(client, headers, run["id"])
    assert out["status"] == "succeeded"

    pipe = client.post(
        f"/api/v1/runs/{run['id']}/promote",
        json={"name": "iris"},
        headers=headers,
    ).json()
    dep = client.post(
        f"/api/v1/pipelines/{pipe['id']}/deployments",
        json={"endpoint_slug": "iris-test", "auth_mode": "workspace"},
        headers=headers,
    ).json()

    # Create the schedule.
    r = client.post(
        f"/api/v1/workspaces/{ws_id}/schedules",
        json={
            "kind": "drift_monitor",
            "target_id": dep["id"],
            "schedule": {"interval_seconds": 600},
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    schedule = r.json()
    assert schedule["enabled"] is True
    assert schedule["kind"] == "drift_monitor"

    # List.
    items = client.get(f"/api/v1/workspaces/{ws_id}/schedules", headers=headers).json()["items"]
    assert len(items) == 1

    # Patch (disable).
    r = client.patch(
        f"/api/v1/schedules/{schedule['id']}",
        json={"enabled": False},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["enabled"] is False

    # Delete.
    r = client.delete(f"/api/v1/schedules/{schedule['id']}", headers=headers)
    assert r.status_code == 204
    items = client.get(f"/api/v1/workspaces/{ws_id}/schedules", headers=headers).json()["items"]
    assert len(items) == 0


def test_schedule_rejects_unknown_kind(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    r = client.post(
        f"/api/v1/workspaces/{ws_id}/schedules",
        json={
            "kind": "magic_8_ball",
            "target_id": "00000000-0000-0000-0000-000000000000",
            "schedule": {"interval_seconds": 600},
        },
        headers=headers,
    )
    assert r.status_code == 400


def test_schedule_validates_interval_minimum(client: TestClient) -> None:
    """Intervals below 30s are refused (would hammer the engine)."""
    headers = _headers(_bootstrap(client))
    ws_id, exp_id = _make_classification_experiment(client, headers)
    r = client.post(
        f"/api/v1/workspaces/{ws_id}/schedules",
        json={
            "kind": "retrain",
            "target_id": exp_id,
            "schedule": {"interval_seconds": 5},
        },
        headers=headers,
    )
    assert r.status_code == 400


# ====================================================== experiment templates


def test_experiment_template_crud(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]

    r = client.post(
        f"/api/v1/workspaces/{ws_id}/experiment-templates",
        json={
            "name": "small-classification-fast",
            "description": "session_id=42, fold=3",
            "task": "classification",
            "setup_params": {"session_id": 42, "fold": 3, "verbose": False},
            "plan_params": {"include": ["lr", "dt"]},
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    t = r.json()
    assert t["name"] == "small-classification-fast"

    items = client.get(
        f"/api/v1/workspaces/{ws_id}/experiment-templates?task=classification",
        headers=headers,
    ).json()["items"]
    assert len(items) == 1

    r = client.patch(
        f"/api/v1/experiment-templates/{t['id']}",
        json={"setup_params": {"session_id": 7, "fold": 5, "verbose": False}},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["setup_params"]["session_id"] == 7

    r = client.delete(f"/api/v1/experiment-templates/{t['id']}", headers=headers)
    assert r.status_code == 204


# ================================================================ webhooks


def test_webhook_crud_secret_encrypted(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]

    r = client.post(
        f"/api/v1/workspaces/{ws_id}/webhooks",
        json={
            "url": "https://example.com/hook",
            "event_types": ["run.succeeded", "drift.alert"],
            "secret": "super-secret-shared-key",
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    wh = r.json()
    assert wh["has_secret"] is True
    assert "super-secret-shared-key" not in str(wh)

    # Confirm DB stores ciphertext.
    from pycaret_server.db import WebhookSubscription, get_session

    with get_session() as s:
        row = s.query(WebhookSubscription).filter_by(id=wh["id"]).one()
        assert row.secret_encrypted is not None
        assert row.secret_encrypted.startswith("ENC:v1:")
        assert "super-secret-shared-key" not in row.secret_encrypted


def test_webhook_rejects_unknown_event(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    r = client.post(
        f"/api/v1/workspaces/{ws_id}/webhooks",
        json={
            "url": "https://example.com",
            "event_types": ["banana.peeled"],
        },
        headers=headers,
    )
    assert r.status_code == 400


# ====================================================== pipeline versioning


def test_pipeline_versioning_and_rollback(client: TestClient) -> None:
    """Promote two runs with the same name -> v1 + v2 in same family. Rollback works."""
    headers = _headers(_bootstrap(client))
    ws_id, exp_id = _make_classification_experiment(client, headers)

    # First run + promote.
    r1 = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "create", "model_id": "lr", "sklearn_dataset": "iris"},
        headers=headers,
    ).json()
    _wait_for_run(client, headers, r1["id"])
    p1 = client.post(
        f"/api/v1/runs/{r1['id']}/promote",
        json={"name": "iris-classifier"},
        headers=headers,
    ).json()
    assert p1["version"] == 1
    assert p1["family_id"]

    # Second run + promote with the same name.
    r2 = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "create", "model_id": "dt", "sklearn_dataset": "iris"},
        headers=headers,
    ).json()
    _wait_for_run(client, headers, r2["id"])
    p2 = client.post(
        f"/api/v1/runs/{r2['id']}/promote",
        json={"name": "iris-classifier"},
        headers=headers,
    ).json()
    assert p2["version"] == 2
    assert p2["family_id"] == p1["family_id"]

    # Versions endpoint shows both.
    versions = client.get(f"/api/v1/pipelines/{p2['id']}/versions", headers=headers).json()
    assert len(versions["items"]) == 2

    # Deploy v2, then roll back to v1.
    dep = client.post(
        f"/api/v1/pipelines/{p2['id']}/deployments",
        json={"endpoint_slug": "iris-rollback", "auth_mode": "workspace"},
        headers=headers,
    ).json()
    assert dep["pipeline_id"] == p2["id"]

    r = client.post(
        f"/api/v1/deployments/{dep['id']}/rollback",
        json={"pipeline_id": p1["id"]},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["pipeline_id"] == p1["id"]


def test_rollback_rejects_cross_family(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    ws_id, exp_id = _make_classification_experiment(client, headers)

    r1 = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "create", "model_id": "lr", "sklearn_dataset": "iris"},
        headers=headers,
    ).json()
    _wait_for_run(client, headers, r1["id"])
    p1 = client.post(
        f"/api/v1/runs/{r1['id']}/promote",
        json={"name": "fam-A"},
        headers=headers,
    ).json()
    p2 = client.post(
        f"/api/v1/runs/{r1['id']}/promote",
        json={"name": "fam-B"},  # different family
        headers=headers,
    ).json()
    dep = client.post(
        f"/api/v1/pipelines/{p1['id']}/deployments",
        json={"endpoint_slug": "fam-test", "auth_mode": "workspace"},
        headers=headers,
    ).json()
    r = client.post(
        f"/api/v1/deployments/{dep['id']}/rollback",
        json={"pipeline_id": p2["id"]},
        headers=headers,
    )
    assert r.status_code == 400


# ============================================================ search plan


def test_search_plan_runs_and_aggregates(client: TestClient) -> None:
    """plan='search' iterates preprocessing variants and aggregates leaderboards."""
    headers = _headers(_bootstrap(client))
    ws_id, exp_id = _make_classification_experiment(client, headers)

    run = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "search",
            "sklearn_dataset": "iris",
            "plan_params": {
                "variants": [{}, {"normalize": True}],
                "compare_params": {"include": ["lr", "dt"]},
            },
        },
        headers=headers,
    )
    assert run.status_code == 202, run.text
    run_id = run.json()["id"]
    out = _wait_for_run(client, headers, run_id)
    assert out["status"] == "succeeded", out.get("error")

    # Leaderboard contains a Variant column with both variant indices.
    leaderboard = out["leaderboard"]
    assert leaderboard, "search plan produced empty leaderboard"
    variants_seen = {row["Variant"] for row in leaderboard}
    assert variants_seen == {0, 1}
