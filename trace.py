import os
import tempfile
import time

db = tempfile.mktemp(suffix=".db").replace(chr(92), "/")
url = f"sqlite:///{db}"
os.environ["PYCARET_DATABASE_URL"] = url
os.environ["PYCARET_JWT_SECRET"] = "test-secret-32-bytes-long-string!!"
os.environ["PYCARET_ARTIFACT_DIR"] = "/tmp/artifacts"
from cryptography.fernet import Fernet

os.environ["PYCARET_SECRETS_KEY"] = Fernet.generate_key().decode()
from pycaret_server.config import get_settings

get_settings.cache_clear()
from pycaret_server.crypto import reset_for_tests

reset_for_tests()
from pycaret_server.db import session as sess_mod
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sess_mod.engine = create_engine(
    url, connect_args={"check_same_thread": False}, future=True
)
sess_mod.session_factory = sessionmaker(
    bind=sess_mod.engine, autoflush=False, expire_on_commit=False
)
from fastapi.testclient import TestClient
from pycaret_server.app import create_app
from pycaret_server.db import Base

Base.metadata.create_all(sess_mod.engine)
from pycaret_server.runs.orchestrator import get_orchestrator, reset_orchestrator

reset_orchestrator()

with TestClient(create_app()) as c:
    tok = c.post(
        "/api/v1/setup/bootstrap",
        json={
            "email": "a@x.com",
            "password": "topsecret123",
            "display_name": "A",
            "workspace_name": "W",
        },
    ).json()
    h = {"Authorization": f"Bearer {tok['access_token']}"}
    ws = c.get("/api/v1/workspaces", headers=h).json()[0]["id"]
    p = c.post(f"/api/v1/workspaces/{ws}/projects", headers=h, json={"name": "P"}).json()["id"]
    e = c.post(
        f"/api/v1/projects/{p}/experiments",
        headers=h,
        json={
            "name": "E",
            "task": "classification",
            "target": "target",
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
    ).json()["id"]
    r = c.post(
        f"/api/v1/experiments/{e}/runs",
        headers=h,
        json={
            "plan": "compare",
            "sklearn_dataset": "iris",
            "plan_params": {"include_models": ["lr"]},
        },
    )
    run_id = r.json()["id"]
    orch = get_orchestrator()
    print(f"AFTER SUBMIT: futures keys: {list(orch._futures.keys())}")
    # Poll trial state for 30s.
    from pycaret_server.db import Trial, Job, get_session
    from sqlalchemy import select

    for i in range(30):
        with get_session() as s:
            ts = s.scalars(select(Trial).where(Trial.run_id == run_id)).all()
            statuses = [(t.model_id, t.status) for t in ts]
        with orch._lock:
            n_futures = len(orch._futures)
            done = [k for k, v in orch._futures.items() if v.done()]
        print(f"t={i:02d}s trials={statuses} futures_total={n_futures} done={len(done)}")
        if statuses and all(s in ("succeeded", "failed") for _, s in statuses):
            break
        time.sleep(1)
    # Now check Run status.
    with get_session() as s:
        from pycaret_server.db import Run

        run = s.get(Run, run_id)
        print(f"FINAL Run.status: {run.status}, error: {run.error}")
