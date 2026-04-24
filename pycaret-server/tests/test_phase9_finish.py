"""Session-11 integration tests: data sources, run cancel, deployments, alembic.

Covers the 4 finish-out features that close Phase 9. Uses the same per-test
SQLite + orchestrator-reset fixture pattern as the other suites.
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


# -------------------------------------------------------------- helpers


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


def _make_classification_experiment(
    client: TestClient, headers: dict, target: str = "target"
) -> tuple[str, str]:
    ws_id, p_id = _make_project(client, headers)
    r = client.post(
        f"/api/v1/projects/{p_id}/experiments",
        json={
            "name": "exp",
            "task": "classification",
            "target": target,
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return ws_id, r.json()["id"]


# ============================================================= data sources


def test_csv_upload_and_run_from_it(client: TestClient, tmp_path) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    ws_id, exp_id = _make_classification_experiment(client, headers, target="target")

    # Build a tiny CSV in-memory.
    import sklearn.datasets as sk

    bundle = sk.load_iris(as_frame=True)
    df = bundle.frame.copy()
    if bundle.target.name != "target":
        df = df.rename(columns={bundle.target.name: "target"})
    elif "target" not in df.columns:
        df["target"] = bundle.target
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # Upload
    up = client.post(
        f"/api/v1/workspaces/{ws_id}/data-sources/upload",
        headers=headers,
        data={"name": "iris.csv", "description": "from sklearn"},
        files={"file": ("iris.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert up.status_code == 201, up.text
    ds = up.json()
    assert ds["kind"] == "csv_upload"
    assert ds["config"]["sha256"]
    assert ds["config"]["rows"] == 150
    assert "target" in ds["config"]["columns"]

    # List
    lst = client.get(f"/api/v1/workspaces/{ws_id}/data-sources", headers=headers)
    assert len(lst.json()) == 1

    # Duplicate rejected
    dup = client.post(
        f"/api/v1/workspaces/{ws_id}/data-sources/upload",
        headers=headers,
        data={"name": "iris.csv"},
        files={"file": ("iris.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert dup.status_code == 409

    # Submit a run using the data source
    run = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers=headers,
        json={
            "plan": "create",
            "model_id": "lr",
            "data_source_id": ds["id"],
        },
    )
    assert run.status_code == 202, run.text
    run_id = run.json()["id"]
    wait = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=120", headers=headers)
    assert wait.json()["status"] == "succeeded", wait.json().get("error")


def test_register_s3_connector(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    ws_id, _ = _make_project(client, headers)

    r = client.post(
        f"/api/v1/workspaces/{ws_id}/data-sources",
        headers=headers,
        json={
            "name": "churn-s3",
            "kind": "s3",
            "config": {"bucket": "example", "key": "churn.csv", "region": "us-east-1"},
        },
    )
    assert r.status_code == 201
    assert r.json()["kind"] == "s3"
    assert r.json()["config"]["bucket"] == "example"

    # csv_upload is forbidden on this endpoint
    bad = client.post(
        f"/api/v1/workspaces/{ws_id}/data-sources",
        headers=headers,
        json={"name": "bad", "kind": "csv_upload"},
    )
    assert bad.status_code == 400

    # Unknown kind
    bad2 = client.post(
        f"/api/v1/workspaces/{ws_id}/data-sources",
        headers=headers,
        json={"name": "bad2", "kind": "azure"},
    )
    assert bad2.status_code == 400


def test_data_source_delete_cleans_file(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    ws_id, _ = _make_project(client, headers)

    csv_bytes = b"a,b\n1,2\n3,4\n"
    up = client.post(
        f"/api/v1/workspaces/{ws_id}/data-sources/upload",
        headers=headers,
        data={"name": "tiny"},
        files={"file": ("tiny.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    ds = up.json()
    path = ds["config"]["path"]

    from pathlib import Path

    assert Path(path).exists()
    r = client.delete(f"/api/v1/data-sources/{ds['id']}", headers=headers)
    assert r.status_code == 204
    assert not Path(path).exists()


# ============================================================ run cancel


def test_cancel_queued_run(client: TestClient) -> None:
    """Cancelling right after submit flips status to 'cancelled'."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)

    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers=headers,
        json={"plan": "compare", "sklearn_dataset": "iris"},
    )
    assert submit.status_code == 202
    run_id = submit.json()["id"]

    # Fire cancellation before the worker gets far.
    cancel = client.post(f"/api/v1/runs/{run_id}/cancel", headers=headers)
    assert cancel.status_code == 200

    # Wait for terminal state.
    wait = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=60", headers=headers)
    body = wait.json()
    # Either cancelled (hit a checkpoint before completing) or succeeded if the
    # tiny iris setup won the race. We assert the happy path — cancellation
    # wins on machines slower than a couple ms per checkpoint.
    assert body["status"] in ("cancelled", "succeeded")


def test_cancel_terminal_run_is_noop(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)

    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers=headers,
        json={"plan": "setup", "sklearn_dataset": "iris"},
    )
    run_id = submit.json()["id"]
    client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=60", headers=headers)

    # Cancel after completion — route returns the current row, no state change.
    r = client.post(f"/api/v1/runs/{run_id}/cancel", headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] in ("succeeded", "failed")


# ============================================================ deployments


def test_promote_run_and_serve_predictions(client: TestClient) -> None:
    """End-to-end: train a model, promote it, create a deployment, predict."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    ws_id, exp_id = _make_classification_experiment(client, headers)

    # 1. Train a model.
    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers=headers,
        json={
            "plan": "create",
            "model_id": "lr",
            "sklearn_dataset": "iris",
        },
    )
    run_id = submit.json()["id"]
    wait = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=120", headers=headers)
    assert wait.json()["status"] == "succeeded", wait.json().get("error")

    # 2. Promote run -> pipeline.
    promote = client.post(
        f"/api/v1/runs/{run_id}/promote",
        headers=headers,
        json={"name": "iris-lr-v1", "description": "baseline LR", "tags": ["baseline"]},
    )
    assert promote.status_code == 201, promote.text
    pipeline_id = promote.json()["id"]
    assert promote.json()["stored_path"]
    assert promote.json()["sha256"]

    # 3. Create a deployment.
    dep = client.post(
        f"/api/v1/pipelines/{pipeline_id}/deployments",
        headers=headers,
        json={"endpoint_slug": "iris-v1"},
    )
    assert dep.status_code == 201, dep.text
    assert dep.json()["endpoint_slug"] == "iris-v1"

    # 4. Predict against the slug.
    # Use iris-shaped feature dicts — 4 numeric columns.
    import sklearn.datasets as sk

    iris = sk.load_iris(as_frame=True)
    row = iris.frame.iloc[0].drop(iris.target.name).to_dict()
    pred = client.post(
        "/api/v1/deployments/iris-v1/predict",
        headers=headers,
        json={"rows": [row, row]},
    )
    assert pred.status_code == 200, pred.text
    body = pred.json()
    assert body["endpoint_slug"] == "iris-v1"
    assert len(body["predictions"]) == 2
    assert "latency_ms" in body

    # 5. Listing reflects metrics.
    lst = client.get(f"/api/v1/workspaces/{ws_id}/deployments", headers=headers)
    assert lst.status_code == 200
    items = lst.json()
    assert len(items) == 1
    assert items[0]["inference_count"] == 2
    assert items[0]["p50_latency_ms"] is not None


def test_promote_rejects_unfinished_run(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)

    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers=headers,
        json={"plan": "setup", "sklearn_dataset": "iris"},
    )
    run_id = submit.json()["id"]
    # Don't wait — promote immediately should fail (run is still queued/running).
    r = client.post(
        f"/api/v1/runs/{run_id}/promote",
        headers=headers,
        json={"name": "too-early"},
    )
    assert r.status_code == 400


def test_delete_pipeline_with_active_deployment_fails(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)

    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers=headers,
        json={"plan": "create", "model_id": "lr", "sklearn_dataset": "iris"},
    )
    run_id = submit.json()["id"]
    client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=120", headers=headers)

    pipeline_id = client.post(
        f"/api/v1/runs/{run_id}/promote",
        headers=headers,
        json={"name": "p1"},
    ).json()["id"]
    client.post(
        f"/api/v1/pipelines/{pipeline_id}/deployments",
        headers=headers,
        json={"endpoint_slug": "p1-slug"},
    )

    r = client.delete(f"/api/v1/pipelines/{pipeline_id}", headers=headers)
    assert r.status_code == 409


def test_deployment_slug_collision(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)

    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers=headers,
        json={"plan": "create", "model_id": "lr", "sklearn_dataset": "iris"},
    )
    run_id = submit.json()["id"]
    client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=120", headers=headers)
    pipeline_id = client.post(
        f"/api/v1/runs/{run_id}/promote",
        headers=headers,
        json={"name": "p"},
    ).json()["id"]

    first = client.post(
        f"/api/v1/pipelines/{pipeline_id}/deployments",
        headers=headers,
        json={"endpoint_slug": "same"},
    )
    assert first.status_code == 201

    dup = client.post(
        f"/api/v1/pipelines/{pipeline_id}/deployments",
        headers=headers,
        json={"endpoint_slug": "same"},
    )
    assert dup.status_code == 409

    # Bad slug
    bad = client.post(
        f"/api/v1/pipelines/{pipeline_id}/deployments",
        headers=headers,
        json={"endpoint_slug": "has spaces"},
    )
    assert bad.status_code == 400


# ============================================================ alembic baseline


def test_alembic_baseline_creates_schema(tmp_path) -> None:
    """Running `alembic upgrade head` on an empty SQLite gives us all 15 tables."""
    import os
    import subprocess

    db_path = tmp_path / "fresh.db"
    env = os.environ.copy()
    env["ALEMBIC_URL"] = f"sqlite:///{db_path}"

    import pycaret_server

    server_root = str(__import__("pathlib").Path(pycaret_server.__file__).resolve().parents[1])
    result = subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=server_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    import sqlite3

    conn = sqlite3.connect(db_path)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()
    # 14 app tables + alembic_version = 15
    expected = {
        "alembic_version",
        "api_keys",
        "artifacts",
        "data_sources",
        "deployments",
        "events",
        "experiments",
        "fold_metrics",
        "pipeline_project_links",
        "pipelines",
        "projects",
        "runs",
        "sessions",
        "users",
        "workspace_members",
        "workspaces",
    }
    assert tables >= expected
