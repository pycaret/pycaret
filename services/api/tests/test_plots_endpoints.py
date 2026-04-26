"""Plot endpoint integration tests (session 53).

Exercises:

- ``GET /api/v1/plots/registry`` — discoverability of plot kinds.
- Authenticated 404 paths for run-bound plots when there's no promoted
  pipeline / unknown task.
- ``GET /api/v1/datasets/{ds}/plots/eda/{kind}`` end-to-end on a real
  CSV upload (covers correlation heatmap + missingness map +
  profile_summary).

Heavier model-card render tests live in the engine plot-module tests
(s47-52). Here we just verify the HTTP wiring: routing, auth,
serialization shape.
"""

from __future__ import annotations

from collections.abc import Generator
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient]:
    """Fresh DB + app per test."""
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


def _bootstrap_and_workspace(client: TestClient) -> tuple[dict, str]:
    """First-run bootstrap → returns (auth headers, workspace id)."""
    r = client.post(
        "/api/v1/setup/bootstrap",
        json={
            "email": "owner@example.com",
            "password": "supersecret",
            "display_name": "Owner",
            "workspace_name": "Plotsville",
        },
    )
    assert r.status_code == 201, r.text
    tok = r.json()
    headers = {"Authorization": f"Bearer {tok['access_token']}"}
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    return headers, ws_id


def _upload_csv(client: TestClient, headers: dict, ws_id: str, csv_bytes: bytes, name: str) -> str:
    resp = client.post(
        f"/api/v1/workspaces/{ws_id}/data-sources/upload",
        files={"file": (name, BytesIO(csv_bytes), "text/csv")},
        data={"name": name},
        headers=headers,
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Registry — public, no auth needed.
# ---------------------------------------------------------------------------


def test_plot_registry_lists_all_tasks(client):
    resp = client.get("/api/v1/plots/registry")
    assert resp.status_code == 200
    body = resp.json()
    # All five 4.0 tasks plus EDA category.
    for task in ("classification", "regression", "clustering", "anomaly", "time_series", "eda"):
        assert task in body["tasks"]
    # Per-kind detail dict.
    assert "details" in body
    assert "confusion_matrix" in body["tasks"]["classification"]


def test_plot_registry_describes_kind_requirements(client):
    resp = client.get("/api/v1/plots/registry")
    body = resp.json()
    cm = body["details"]["classification"]["confusion_matrix"]
    assert cm["requires"] == ["pipeline", "X_test", "y_test"]
    cal = body["details"]["classification"]["calibration_curve"]
    assert cal["binary_only"] is True


# ---------------------------------------------------------------------------
# Run-bound endpoints — error paths (we don't have a fitted run in the
# fixture, so success path is exercised by the engine plot tests).
# ---------------------------------------------------------------------------


def test_run_plot_unknown_run_returns_404(client):
    headers, ws_id = _bootstrap_and_workspace(client)
    resp = client.get("/api/v1/runs/no-such-run/plots/confusion_matrix", headers=headers)
    assert resp.status_code in (403, 404)  # access check or not-found


def test_run_plot_requires_auth(client):
    resp = client.get("/api/v1/runs/anything/plots/confusion_matrix")
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Dataset-only EDA — full happy path.
# ---------------------------------------------------------------------------


_TINY_CSV = b"a,b,c\n1,2.0,x\n3,4.5,y\n5,,x\n7,8.1,y\n9,10.2,x\n"


def test_eda_correlation_heatmap_end_to_end(client):
    headers, ws_id = _bootstrap_and_workspace(client)

    ds_id = _upload_csv(client, headers, ws_id, _TINY_CSV, "tiny.csv")
    resp = client.get(f"/api/v1/datasets/{ds_id}/plots/eda/correlation_heatmap", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "correlation_heatmap"
    assert body["task"] == "eda"
    # Plotly figure dict has data + layout.
    fig = body["figure"]
    assert "data" in fig and "layout" in fig
    assert fig["data"][0]["type"] == "heatmap"


def test_eda_missingness_map_end_to_end(client):
    headers, ws_id = _bootstrap_and_workspace(client)

    ds_id = _upload_csv(client, headers, ws_id, _TINY_CSV, "tiny.csv")
    resp = client.get(f"/api/v1/datasets/{ds_id}/plots/eda/missingness_map", headers=headers)
    assert resp.status_code == 200, resp.text
    fig = resp.json()["figure"]
    # Bar chart with one row per column.
    assert fig["data"][0]["type"] == "bar"


def test_eda_column_distribution_requires_column_param(client):
    headers, ws_id = _bootstrap_and_workspace(client)

    ds_id = _upload_csv(client, headers, ws_id, _TINY_CSV, "tiny.csv")
    resp = client.get(f"/api/v1/datasets/{ds_id}/plots/eda/column_distribution", headers=headers)
    assert resp.status_code == 400


def test_eda_column_distribution_with_column(client):
    headers, ws_id = _bootstrap_and_workspace(client)

    ds_id = _upload_csv(client, headers, ws_id, _TINY_CSV, "tiny.csv")
    resp = client.get(
        f"/api/v1/datasets/{ds_id}/plots/eda/column_distribution",
        params={"column": "a"},
        headers=headers,
    )
    assert resp.status_code == 200
    fig = resp.json()["figure"]
    # Numeric column → histogram.
    assert fig["data"][0]["type"] == "histogram"


def test_eda_profile_summary(client):
    headers, ws_id = _bootstrap_and_workspace(client)

    ds_id = _upload_csv(client, headers, ws_id, _TINY_CSV, "tiny.csv")
    resp = client.get(f"/api/v1/datasets/{ds_id}/plots/eda/profile_summary", headers=headers)
    assert resp.status_code == 200
    fig = resp.json()["figure"]
    assert fig["data"][0]["type"] == "table"


def test_eda_unknown_kind_returns_400(client):
    headers, ws_id = _bootstrap_and_workspace(client)

    ds_id = _upload_csv(client, headers, ws_id, _TINY_CSV, "tiny.csv")
    resp = client.get(f"/api/v1/datasets/{ds_id}/plots/eda/not_a_real_plot", headers=headers)
    assert resp.status_code == 400
