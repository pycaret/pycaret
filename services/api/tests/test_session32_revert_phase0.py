"""Session 32 — Phase 0-v2 (Trial/Run revert + parallel dispatch).

The data model is back to ``Run contains Trials`` (PyCaret 3.x
convention). Each Trial:

- Has its own ``run_id`` parent
- Carries metrics + artifact + status directly
- Goes through its own ``queued → running → succeeded`` lifecycle
- Is dispatched as a separate ``kind="trial"`` Job — workers run
  ``create_model`` in parallel

The Run's status is derived from the aggregate of its Trials'
statuses. The reconciler flips the Run when its last Trial finishes.
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
    monkeypatch.setenv("PYCARET_JWT_SECRET", "test-secret-32-bytes-long-string!!")
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


def _make_classification_experiment(client: TestClient, headers: dict) -> tuple[str, str]:
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


def _wait_for_run(client: TestClient, headers: dict, run_id: str, timeout: int = 240) -> dict:
    r = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s={timeout}", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


# ============================================ schema invariants


def test_trial_belongs_to_run_via_run_id(client: TestClient) -> None:
    """Phase 0-v2 invariant: Run > Trial. ``Trial.run_id`` is the FK."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)

    # Restrict compare to a few cheap algorithms so the test is fast.
    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "compare",
            "sklearn_dataset": "iris",
            "plan_params": {"include_models": ["lr", "dt", "nb"]},
        },
        headers=headers,
    )
    assert submit.status_code == 202, submit.text
    run_id = submit.json()["id"]

    # Poll the trials endpoint — even before the Run is done, Trial
    # rows should already exist (they're created at dispatch).
    trials_before = client.get(
        f"/api/v1/runs/{run_id}/trials", headers=headers
    ).json()["items"]
    assert len(trials_before) == 3
    # Status should be queued or running at this point.
    statuses_before = {t["status"] for t in trials_before}
    assert statuses_before.issubset({"queued", "running", "succeeded"})

    out = _wait_for_run(client, headers, run_id)
    assert out["status"] == "succeeded", out.get("error")

    trials = client.get(
        f"/api/v1/runs/{run_id}/trials", headers=headers
    ).json()["items"]
    assert len(trials) == 3
    for t in trials:
        assert t["run_id"] == run_id
        assert t["status"] == "succeeded"
        assert t["model_id"] in ("lr", "dt", "nb")
        assert isinstance(t["metrics"], dict)
        assert t["has_artifact"] is True

    # Rank stamped, exactly one is_best.
    ranks = sorted(t["rank"] for t in trials if t["rank"])
    assert ranks == [1, 2, 3]
    bests = [t for t in trials if t["is_best"]]
    assert len(bests) == 1
    assert bests[0]["rank"] == 1


def test_compare_decomposes_into_multiple_jobs(client: TestClient) -> None:
    """Each Trial gets its own kind='trial' Job — that's how
    parallelism falls out under Redis mode."""
    from pycaret_server.db import Job, Trial, get_session
    from sqlalchemy import select

    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)

    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "compare",
            "sklearn_dataset": "iris",
            "plan_params": {"include_models": ["lr", "dt", "nb"]},
        },
        headers=headers,
    )
    run_id = submit.json()["id"]
    _wait_for_run(client, headers, run_id)

    # Confirm in the DB that N Trial-Jobs exist for this Run.
    with get_session() as s:
        trial_jobs = s.scalars(
            select(Job).where(Job.run_id == run_id, Job.kind == "trial")
        ).all()
        trials = s.scalars(select(Trial).where(Trial.run_id == run_id)).all()
    assert len(trial_jobs) == 3
    assert len(trials) == 3
    assert {j.correlation_id for j in trial_jobs} == {t.id for t in trials}


def test_run_succeeds_when_all_trials_succeed(client: TestClient) -> None:
    """The reconciler flips the Run after the last Trial terminates."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)

    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "compare",
            "sklearn_dataset": "iris",
            "plan_params": {"include_models": ["lr", "nb"]},
        },
        headers=headers,
    )
    run_id = submit.json()["id"]
    out = _wait_for_run(client, headers, run_id)
    assert out["status"] == "succeeded"
    # Leaderboard cached on the Run for legacy callers.
    assert out["leaderboard"]
    assert len(out["leaderboard"]) == 2


def test_experiment_scoped_trials_endpoint(client: TestClient) -> None:
    """``GET /experiments/{id}/trials`` reads from the new schema."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)
    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "compare",
            "sklearn_dataset": "iris",
            "plan_params": {"include_models": ["lr", "nb"]},
        },
        headers=headers,
    )
    run_id = submit.json()["id"]
    _wait_for_run(client, headers, run_id)

    r = client.get(f"/api/v1/experiments/{exp_id}/trials", headers=headers)
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert len(items) == 2
    for t in items:
        assert t["run_id"] == run_id
        assert t["status"] == "succeeded"
        assert t["experiment_id"] == exp_id


def test_trial_promote_uses_trial_artifact_directly(client: TestClient) -> None:
    """Promotion reads stored_path off the Trial row — no Run-1 detour."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)
    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "compare",
            "sklearn_dataset": "iris",
            "plan_params": {"include_models": ["lr", "nb"]},
        },
        headers=headers,
    )
    run_id = submit.json()["id"]
    _wait_for_run(client, headers, run_id)
    items = client.get(
        f"/api/v1/runs/{run_id}/trials", headers=headers
    ).json()["items"]
    best = next(t for t in items if t["is_best"])
    p = client.post(
        f"/api/v1/runs/{run_id}/trials/{best['id']}/promote",
        headers=headers,
        json={"name": "iris-best", "description": "best from compare"},
    )
    assert p.status_code == 201, p.text
    assert p.json()["model_id"] == best["model_id"]


def test_direct_trial_endpoints(client: TestClient) -> None:
    """``GET /trials/{id}``, ``PATCH``, ``DELETE`` work via the new
    workspace-scoped trial routes."""
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    _, exp_id = _make_classification_experiment(client, headers)
    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "compare",
            "sklearn_dataset": "iris",
            "plan_params": {"include_models": ["lr", "nb"]},
        },
        headers=headers,
    )
    run_id = submit.json()["id"]
    _wait_for_run(client, headers, run_id)
    items = client.get(
        f"/api/v1/runs/{run_id}/trials", headers=headers
    ).json()["items"]
    trial_id = items[0]["id"]

    fetch = client.get(f"/api/v1/trials/{trial_id}", headers=headers)
    assert fetch.status_code == 200, fetch.text
    assert fetch.json()["run_id"] == run_id

    patched = client.patch(
        f"/api/v1/trials/{trial_id}",
        json={"name": "my-renamed", "notes": "looks promising"},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "my-renamed"
    assert patched.json()["notes"] == "looks promising"

    # Delete an unpromoted Trial → 204.
    delete = client.delete(f"/api/v1/trials/{trial_id}", headers=headers)
    assert delete.status_code == 204
