"""Session 33 — GPU routing + distributed-worker validation.

Validates the Phase 14 hand-off points without standing up a real
multi-host worker pool:

- The dispatcher routes Trial-Jobs to the ``gpu`` queue when the
  experiment opted into ``use_gpu`` (and stamps
  ``requested_resources={"gpu": 1}``).
- A CPU-only worker (``detect_gpus`` reports unavailable) releases a
  GPU-tagged Job back to the queue rather than failing it.
- The ``/admin/system`` route surfaces a usable inventory dict.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
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

    from pycaret_server.runtime import reset_for_tests as reset_gpu_for_tests

    reset_gpu_for_tests()

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
    from pycaret_server.runs.orchestrator import reset_orchestrator

    Base.metadata.create_all(sess_mod.engine)
    reset_orchestrator()

    with TestClient(create_app()) as c:
        yield c

    reset_orchestrator()
    reset_gpu_for_tests()
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


# ============================================ GPU detection probe


def test_gpu_detection_via_env_override(monkeypatch) -> None:
    from pycaret_server.runtime import detect_gpus, reset_for_tests as reset

    reset()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    inv = detect_gpus()
    assert inv.available is True
    assert inv.count == 2
    assert inv.source == "env"
    assert inv.devices == ["cuda:0", "cuda:1"]


def test_gpu_detection_env_empty_means_none(monkeypatch) -> None:
    from pycaret_server.runtime import detect_gpus, reset_for_tests as reset

    reset()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    inv = detect_gpus()
    assert inv.available is False
    assert inv.source == "env"


# ============================================ dispatcher GPU routing


@pytest.fixture
def _stub_executor(monkeypatch) -> None:
    """No-op the trial executor so dispatcher tests don't trigger
    PyCaret's GPU import path. We only care about Job-row state at
    dispatch time, not actual training."""
    from pycaret_server.runs import dispatch as _dispatch_mod
    from pycaret_server.runs import orchestrator as _orch_mod

    class _StubOrch:
        def submit_trial_job(self, _job_id: str) -> None:  # noqa: D401
            return None

    monkeypatch.setattr(_orch_mod, "get_orchestrator", lambda: _StubOrch())
    monkeypatch.setattr(_dispatch_mod, "get_orchestrator", lambda: _StubOrch(), raising=False)


def test_dispatcher_routes_use_gpu_to_gpu_queue(
    client: TestClient, _stub_executor: None
) -> None:
    """A compare Run with ``use_gpu=True`` lands every Trial-Job on the
    ``gpu`` queue + stamps ``requested_resources.gpu=1``."""
    from pycaret_server.db import Job, get_session

    tokens = _bootstrap(client)
    headers = _headers(tokens)
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    p_id = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "GPU"},
        headers=headers,
    ).json()["id"]
    e = client.post(
        f"/api/v1/projects/{p_id}/experiments",
        json={
            "name": "exp",
            "task": "classification",
            "target": "target",
            "setup_params": {
                "session_id": 42,
                "fold": 2,
                "verbose": False,
                "use_gpu": True,
            },
        },
        headers=headers,
    )
    assert e.status_code == 201, e.text
    exp_id = e.json()["id"]
    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "compare",
            "sklearn_dataset": "iris",
            "plan_params": {"include_models": ["lr", "dt"]},
        },
        headers=headers,
    )
    assert submit.status_code == 202, submit.text
    run_id = submit.json()["id"]

    # Don't wait_for — the workers might or might not actually run on
    # this CI box. We only care about *dispatch-time* routing.
    with get_session() as s:
        jobs = s.scalars(
            select(Job).where(Job.run_id == run_id, Job.kind == "trial")
        ).all()
    assert len(jobs) == 2
    for j in jobs:
        assert j.queue == "gpu", f"expected gpu queue, got {j.queue!r}"
        assert (j.requested_resources or {}).get("gpu") == 1


def test_dispatcher_defaults_to_default_queue(
    client: TestClient, _stub_executor: None
) -> None:
    """Without ``use_gpu`` Trial-Jobs land on the default queue."""
    from pycaret_server.db import Job, get_session

    tokens = _bootstrap(client)
    headers = _headers(tokens)
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    p_id = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "CPU"},
        headers=headers,
    ).json()["id"]
    e = client.post(
        f"/api/v1/projects/{p_id}/experiments",
        json={
            "name": "exp",
            "task": "classification",
            "target": "target",
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
        headers=headers,
    )
    exp_id = e.json()["id"]
    submit = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={
            "plan": "compare",
            "sklearn_dataset": "iris",
            "plan_params": {"include_models": ["lr", "dt"]},
        },
        headers=headers,
    )
    run_id = submit.json()["id"]
    with get_session() as s:
        jobs = s.scalars(
            select(Job).where(Job.run_id == run_id, Job.kind == "trial")
        ).all()
    for j in jobs:
        assert j.queue == "default"
        # No resource ask on default queue.
        assert (j.requested_resources or {}).get("gpu") is None


# ============================================ worker GPU gate


def test_worker_releases_gpu_job_when_no_gpu(monkeypatch) -> None:
    """``_can_run_job`` refuses a GPU-tagged Job on a CPU-only worker."""
    from pycaret_server.db import Job
    from pycaret_server.runtime import reset_for_tests as reset
    from pycaret_server.worker import _can_run_job

    reset()
    # Force "no gpu" — empty env wins over nvidia-smi.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    j_gpu = Job(
        kind="trial",
        status="queued",
        queue="gpu",
        requested_resources={"gpu": 1},
    )
    ok, reason = _can_run_job(j_gpu)
    assert ok is False
    assert reason is not None and "no GPU" in reason

    # Same worker, default-queue Job → fine.
    j_cpu = Job(kind="trial", status="queued", queue="default")
    ok2, _ = _can_run_job(j_cpu)
    assert ok2 is True


def test_worker_runs_gpu_job_with_gpu_present(monkeypatch) -> None:
    """When ``CUDA_VISIBLE_DEVICES`` is set, GPU jobs are claimable."""
    from pycaret_server.db import Job
    from pycaret_server.runtime import reset_for_tests as reset
    from pycaret_server.worker import _can_run_job

    reset()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    j_gpu = Job(
        kind="trial",
        status="queued",
        queue="gpu",
        requested_resources={"gpu": 1},
    )
    ok, reason = _can_run_job(j_gpu)
    assert ok is True
    assert reason is None


# ============================================ /admin/system


def test_admin_system_inventory_shape(client: TestClient) -> None:
    tokens = _bootstrap(client)
    headers = _headers(tokens)
    r = client.get("/api/v1/admin/system", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "runs_backend" in body
    assert "gpu" in body and "available" in body["gpu"]
    assert "redis" in body and "healthy" in body["redis"]
    assert isinstance(body["worker_queues"], list)
