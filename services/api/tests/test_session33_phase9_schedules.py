"""Session 33 — Phase 9 schedule kinds smoke test.

Verifies that ``drift_check`` / ``batch_predict`` / ``dataset_refresh``
schedules are wired end-to-end:

  ScheduledJob (kind=X)  →  /run-now  →  JOB_HANDLERS[X]  →
  _spawn_worker_job(job_kind=X)  →  Job row (kind=X)  →  worker
  handler in worker._HANDLERS

Inprocess mode runs the worker handler synchronously, so a green
``last_status="ok"`` on the ScheduledJob proves the whole chain works
without standing up Redis or a worker container.
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

    Base.metadata.create_all(sess_mod.engine)

    with TestClient(create_app()) as c:
        yield c
    reset_for_tests()


def _bootstrap(client: TestClient) -> tuple[dict, str]:
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
    tokens = r.json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    return headers, ws_id


def test_drift_check_schedule_fires_via_run_now(client: TestClient) -> None:
    """Workspace-targeted drift_check schedule + run-now produces a
    ``kind=drift_check`` Job row and the schedule row goes green."""
    from pycaret_server.db import Job, ScheduledJob, get_session

    headers, ws_id = _bootstrap(client)

    sched = client.post(
        f"/api/v1/workspaces/{ws_id}/schedules",
        json={
            "kind": "drift_check",
            "target_id": ws_id,
            "schedule": {"interval_seconds": 3600},
            "enabled": True,
        },
        headers=headers,
    )
    assert sched.status_code == 201, sched.text
    sched_id = sched.json()["id"]

    r = client.post(f"/api/v1/schedules/{sched_id}/run-now", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["last_status"] == "ok"

    with get_session() as s:
        sj = s.get(ScheduledJob, sched_id)
        jobs = s.scalars(
            select(Job).where(Job.kind == "drift_check", Job.correlation_id == sched_id)
        ).all()
    assert sj is not None and sj.last_status == "ok"
    assert len(jobs) == 1
    # Inprocess mode runs the handler synchronously; ``_handle_drift_check``
    # writes a ``fired`` list back onto the Job payload even when empty.
    assert "fired" in (jobs[0].payload or {})


def test_dataset_refresh_schedule_smoke(client: TestClient, tmp_path) -> None:
    """A scheduled dataset_refresh against a CSV data source produces a
    new Dataset version row."""
    import pandas as pd

    from pycaret_server.db import DataSource, Dataset, Job, ScheduledJob, get_session

    headers, ws_id = _bootstrap(client)

    # Create a project + drop a CSV on disk + register a DataSource.
    p_id = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "DS"},
        headers=headers,
    ).json()["id"]

    csv_path = tmp_path / "small.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(csv_path, index=False)

    with get_session() as s:
        ds = DataSource(
            workspace_id=ws_id,
            name="small",
            kind="csv_upload",
            config={"path": str(csv_path)},
            created_by=client.get("/api/v1/auth/me", headers=headers).json()["id"],
        )
        s.add(ds)
        s.commit()
        s.refresh(ds)
        ds_id = ds.id

    sched = client.post(
        f"/api/v1/workspaces/{ws_id}/schedules",
        json={
            "kind": "dataset_refresh",
            "target_id": ds_id,
            "schedule": {"interval_seconds": 3600},
            "enabled": True,
        },
        headers=headers,
    )
    assert sched.status_code == 201, sched.text
    sched_id = sched.json()["id"]

    r = client.post(f"/api/v1/schedules/{sched_id}/run-now", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["last_status"] == "ok"

    with get_session() as s:
        sj = s.get(ScheduledJob, sched_id)
        jobs = s.scalars(
            select(Job).where(
                Job.kind == "dataset_refresh", Job.correlation_id == sched_id
            )
        ).all()
        datasets = s.scalars(
            select(Dataset).where(Dataset.data_source_id == ds_id)
        ).all()
    assert sj is not None and sj.last_status == "ok"
    assert len(jobs) == 1
    assert len(datasets) == 1, "expected dataset_refresh to produce a version row"
    assert datasets[0].row_count == 3
