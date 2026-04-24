"""Session-21 tests: drift reports + drift analyst + audit logs.

Three concerns:

1. Drift reports — create + list + get + bucketing of drift_status.
2. ``POST /llm/analyze-drift`` — reads the drift report + hits the fake LLM.
3. Audit-log middleware — every mutating /api/v1/* call writes one row,
   with sensitive fields scrubbed, and list routes enforce admin access.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient]:
    db_file = tmp_path / "s21.db"
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
            "suggested_config_json": {
                "retrain_window_days": 30,
                "refresh_baseline": True,
            },
            "suggested_action": (
                "INVESTIGATE: feature 'amount' drove drift_score=0.31 with a "
                "missing-rate spike; check upstream ETL before retraining."
            ),
            "reasoning_summary": (
                "Drift is concentrated in one feature ('amount') with a "
                "missing-rate kind, which is more consistent with a pipeline "
                "breakage than genuine concept drift. Prediction distribution "
                "barely moved (js=0.02). Sample size (400) is adequate."
            ),
            "risk_flags": [
                "concentrated_drift",
                "missing_rate_spike",
                "possible_data_source_change",
            ],
        }
    )

    with TestClient(create_app()) as c:
        yield c

    reset_router()
    reset_orchestrator()
    event_broker.clear()


# ───────────────────────────────────────────────────────────── helpers


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


def _configure_llm(client: TestClient, tok: dict, ws: str) -> None:
    client.put(
        f"/api/v1/workspaces/{ws}/llm/settings",
        headers=_auth(tok),
        json={"provider": "anthropic", "model_name": "claude-sonnet-4-5", "api_key": "sk"},
    )


def _create_deployment(client: TestClient, tok: dict, ws: str) -> str:
    """Create a Pipeline + Deployment directly in the DB so we have
    something to attach drift reports to. The real promote/deploy loop
    is exercised in the session-16 tests; here we just need a row.
    """
    from pycaret_server.db import Deployment, Pipeline, get_session

    s = get_session()
    try:
        admin = s.execute(
            __import__("sqlalchemy").text("SELECT id FROM users WHERE email = 'admin@example.com'")
        ).scalar_one()
        pipeline = Pipeline(
            workspace_id=ws,
            name="fraud-detector-v1",
            description="test pipeline",
            model_id="lr",
            sha256="a" * 64,
            stored_path="/tmp/x.pkl",
            tags=[],
            created_by=admin,
        )
        s.add(pipeline)
        s.flush()
        dep = Deployment(
            workspace_id=ws,
            pipeline_id=pipeline.id,
            endpoint_slug="fraud-v1",
            status="active",
            auth_mode="workspace",
            created_by=admin,
        )
        s.add(dep)
        s.commit()
        return dep.id
    finally:
        s.close()


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ════════════════════════════════════════════════════════════ drift reports


def test_create_drift_report_buckets_status(client: TestClient) -> None:
    """POST creates a report + server buckets drift_status from drift_score."""
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    dep = _create_deployment(client, tok, ws)

    end = datetime.now(UTC)
    start = end - timedelta(days=7)
    body = {
        "window_start": _iso(start),
        "window_end": _iso(end),
        "drift_score": 0.31,  # moderate
        "feature_drift_json": {
            "amount": {"score": 0.42, "kind": "missing_rate"},
            "age": {"score": 0.05, "kind": "psi"},
        },
        "prediction_drift_json": {"kind": "js", "score": 0.02},
        "sample_size": 400,
    }
    r = client.post(
        f"/api/v1/deployments/{dep}/drift-reports",
        headers=_auth(tok),
        json=body,
    )
    assert r.status_code == 201, r.text
    got = r.json()
    assert got["drift_score"] == 0.31
    assert got["drift_status"] == "moderate"
    assert got["sample_size"] == 400
    assert got["feature_drift_json"]["amount"]["kind"] == "missing_rate"


def test_drift_status_none_mild_severe(client: TestClient) -> None:
    """Bucket boundaries: 0.05 → none, 0.15 → mild, 0.6 → severe."""
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    dep = _create_deployment(client, tok, ws)

    end = datetime.now(UTC)
    start = end - timedelta(hours=1)
    for score, want in [(0.05, "none"), (0.15, "mild"), (0.60, "severe")]:
        r = client.post(
            f"/api/v1/deployments/{dep}/drift-reports",
            headers=_auth(tok),
            json={
                "window_start": _iso(start),
                "window_end": _iso(end),
                "drift_score": score,
                "feature_drift_json": {},
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["drift_status"] == want, (score, want, r.json())


def test_list_and_get_drift_reports(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    dep = _create_deployment(client, tok, ws)
    end = datetime.now(UTC)
    start = end - timedelta(hours=1)
    created = client.post(
        f"/api/v1/deployments/{dep}/drift-reports",
        headers=_auth(tok),
        json={
            "window_start": _iso(start),
            "window_end": _iso(end),
            "drift_score": 0.2,
            "feature_drift_json": {"a": {"score": 0.2, "kind": "psi"}},
        },
    ).json()

    listed = client.get(f"/api/v1/deployments/{dep}/drift-reports", headers=_auth(tok)).json()
    assert len(listed) == 1
    assert listed[0]["id"] == created["id"]

    single = client.get(f"/api/v1/drift-reports/{created['id']}", headers=_auth(tok)).json()
    assert single["id"] == created["id"]
    assert single["feature_drift_json"] == {"a": {"score": 0.2, "kind": "psi"}}


def test_drift_report_window_end_must_be_after_start(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    dep = _create_deployment(client, tok, ws)
    end = datetime.now(UTC)
    start = end + timedelta(hours=1)  # backwards
    r = client.post(
        f"/api/v1/deployments/{dep}/drift-reports",
        headers=_auth(tok),
        json={
            "window_start": _iso(start),
            "window_end": _iso(end),
            "drift_score": 0.1,
            "feature_drift_json": {},
        },
    )
    assert r.status_code == 400, r.text
    assert "window_end" in r.text


# ═══════════════════════════════════════════════════════════ drift analyst


def test_analyze_drift_runs_the_llm(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    _configure_llm(client, tok, ws)
    dep = _create_deployment(client, tok, ws)
    end = datetime.now(UTC)
    start = end - timedelta(days=1)
    report = client.post(
        f"/api/v1/deployments/{dep}/drift-reports",
        headers=_auth(tok),
        json={
            "window_start": _iso(start),
            "window_end": _iso(end),
            "drift_score": 0.31,
            "feature_drift_json": {
                "amount": {"score": 0.42, "kind": "missing_rate"},
            },
            "prediction_drift_json": {"kind": "js", "score": 0.02},
            "sample_size": 400,
        },
    ).json()

    r = client.post(
        "/api/v1/llm/analyze-drift",
        headers=_auth(tok),
        json={"drift_report_id": report["id"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "drift_analysis"
    advice = body["response_json"]
    assert advice["suggested_action"].startswith("INVESTIGATE")
    assert "concentrated_drift" in advice["risk_flags"]


def test_analyze_drift_404_on_unknown_report(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    _configure_llm(client, tok, ws)
    r = client.post(
        "/api/v1/llm/analyze-drift",
        headers=_auth(tok),
        json={"drift_report_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert r.status_code == 404, r.text


# ════════════════════════════════════════════════════════════ audit logs


def test_audit_log_records_mutating_requests(client: TestClient) -> None:
    """Creating a workspace lands a row in audit_logs with the scrubbed body."""
    tok = _bootstrap(client)

    # Create a new workspace — that's a POST /api/v1/workspaces.
    r = client.post(
        "/api/v1/workspaces",
        headers=_auth(tok),
        json={"name": "AuditTarget"},
    )
    assert r.status_code == 201, r.text

    # Read back the admin log (bootstrap user is superuser).
    listed = client.get("/api/v1/admin/audit-logs", headers=_auth(tok)).json()
    # Should have at least the bootstrap + the workspace creation.
    assert len(listed) >= 2
    actions = [row["action"] for row in listed]
    assert "workspaces.create" in actions

    # The workspaces.create row should carry the scrubbed payload.
    ws_row = next(r for r in listed if r["action"] == "workspaces.create")
    assert ws_row["method"] == "POST"
    assert ws_row["path"] == "/api/v1/workspaces"
    assert ws_row["status_code"] == 201
    assert ws_row["payload"] == {"name": "AuditTarget"}
    assert ws_row["user_id"] is not None  # we attributed the caller


def test_audit_log_scrubs_password_on_bootstrap(client: TestClient) -> None:
    """Bootstrap POSTs a password; it must never land in audit_logs.payload."""
    _bootstrap(client)
    # Log in to get a token we can use to read back the admin log.
    tok = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "supersecret"},
    ).json()

    listed = client.get("/api/v1/admin/audit-logs", headers=_auth(tok)).json()
    # Find the bootstrap row. Path is /api/v1/setup/bootstrap.
    row = next(r for r in listed if r["path"] == "/api/v1/setup/bootstrap")
    assert row["payload"] is not None
    assert row["payload"]["password"] == "***REDACTED***"
    assert row["payload"]["email"] == "admin@example.com"


def test_audit_log_workspace_scoped_requires_admin(client: TestClient) -> None:
    """A non-member gets 403 on /workspaces/{id}/audit-logs."""
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]

    # Build a second user, log in, hit the workspace audit endpoint.
    from pycaret_server.auth.passwords import hash_password
    from pycaret_server.db import User, get_session

    s = get_session()
    try:
        s.add(
            User(
                email="outsider@example.com",
                display_name="Outsider",
                password_hash=hash_password("outsiderpassword"),
                is_active=True,
            )
        )
        s.commit()
    finally:
        s.close()
    other = client.post(
        "/api/v1/auth/login",
        json={"email": "outsider@example.com", "password": "outsiderpassword"},
    ).json()
    r = client.get(f"/api/v1/workspaces/{ws}/audit-logs", headers=_auth(other))
    assert r.status_code == 403, r.text


def test_audit_log_admin_route_requires_superuser(client: TestClient) -> None:
    """A non-superuser gets 403 on /admin/audit-logs."""
    _bootstrap(client)  # admin is the FIRST user → superuser.

    from pycaret_server.auth.passwords import hash_password
    from pycaret_server.db import User, get_session

    s = get_session()
    try:
        s.add(
            User(
                email="notsuper@example.com",
                display_name="NotSuper",
                password_hash=hash_password("notsuperpassword"),
                is_active=True,
                is_superuser=False,
            )
        )
        s.commit()
    finally:
        s.close()
    other = client.post(
        "/api/v1/auth/login",
        json={"email": "notsuper@example.com", "password": "notsuperpassword"},
    ).json()
    r = client.get("/api/v1/admin/audit-logs", headers=_auth(other))
    assert r.status_code == 403, r.text


def test_audit_log_filters_by_action(client: TestClient) -> None:
    tok = _bootstrap(client)
    ws = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]

    # Create 2 workspaces + 1 project — different actions.
    client.post("/api/v1/workspaces", headers=_auth(tok), json={"name": "W2"})
    client.post(
        f"/api/v1/workspaces/{ws}/projects",
        headers=_auth(tok),
        json={"name": "P"},
    )

    only_ws = client.get(
        "/api/v1/admin/audit-logs",
        headers=_auth(tok),
        params={"action": "workspaces.create"},
    ).json()
    assert len(only_ws) >= 1
    assert all(r["action"] == "workspaces.create" for r in only_ws)

    only_projects = client.get(
        "/api/v1/admin/audit-logs",
        headers=_auth(tok),
        params={"action": "projects.create"},
    ).json()
    assert len(only_projects) >= 1
    assert all(r["action"] == "projects.create" for r in only_projects)


def test_audit_log_workspace_scope_filters_by_workspace(client: TestClient) -> None:
    """Workspace-admin gets only their own workspace's rows."""
    tok = _bootstrap(client)
    ws1 = client.get("/api/v1/workspaces", headers=_auth(tok)).json()[0]["id"]
    ws2 = client.post("/api/v1/workspaces", headers=_auth(tok), json={"name": "Other"}).json()["id"]

    # Create a project in each workspace.
    client.post(f"/api/v1/workspaces/{ws1}/projects", headers=_auth(tok), json={"name": "P1"})
    client.post(f"/api/v1/workspaces/{ws2}/projects", headers=_auth(tok), json={"name": "P2"})

    rows = client.get(f"/api/v1/workspaces/{ws1}/audit-logs", headers=_auth(tok)).json()
    # Every returned row is either workspace_id=ws1 or has a path under it.
    assert all((r["workspace_id"] == ws1) or (f"/workspaces/{ws1}" in r["path"]) for r in rows), (
        rows
    )
