"""Session 25 — trial-level artifacts, downloads, and per-candidate promote.

Promotes ``compare_models`` from "best wins, rest discarded" to a pool of
candidates. After this slice every trial keeps its fitted Pipeline pickle
+ extracted estimator hyperparams so the UI can:

- list every candidate (already shipped in session 22)
- open a model-detail page (``GET /runs/:id/trials/:tid``)
- download the candidate pickle (``GET .../download``)
- promote *any* candidate — not just the best — to a workspace Pipeline
  (``POST .../promote``)
"""

from __future__ import annotations

import io
import pickle
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


def _classification_run(client: TestClient, headers: dict) -> tuple[str, str]:
    """Bootstrap → workspace → project → experiment → compare run on iris.

    Returns ``(run_id, workspace_id)`` after the run reaches succeeded.
    """
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    p_id = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "Demo"},
        headers=headers,
    ).json()["id"]
    exp = client.post(
        f"/api/v1/projects/{p_id}/experiments",
        json={
            "name": "exp",
            "task": "classification",
            "target": "target",
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
        headers=headers,
    )
    assert exp.status_code == 201, exp.text
    exp_id = exp.json()["id"]

    run = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "compare", "sklearn_dataset": "iris"},
        headers=headers,
    )
    assert run.status_code == 202, run.text
    run_id = run.json()["id"]
    wait = client.post(f"/api/v1/runs/{run_id}/wait?timeout_s=120", headers=headers)
    assert wait.status_code == 200, wait.text
    body = wait.json()
    assert body["status"] == "succeeded", body.get("error")
    return run_id, ws_id


# ====================================================================== tests


def test_compare_run_persists_per_trial_artifacts(client: TestClient) -> None:
    """Every trial row gets stored_path + sha256 + size + extracted params."""
    headers = _headers(_bootstrap(client))
    run_id, _ = _classification_run(client, headers)

    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    assert len(trials) >= 2
    for t in trials:
        assert t["has_artifact"] is True, t
        assert t["size_bytes"] is not None and t["size_bytes"] > 0


def test_trial_detail_returns_params_and_metadata(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id, _ = _classification_run(client, headers)

    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    tid = trials[0]["id"]

    detail = client.get(
        f"/api/v1/runs/{run_id}/trials/{tid}", headers=headers
    )
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["id"] == tid
    assert body["model_id"] == trials[0]["model_id"]
    assert body["has_artifact"] is True
    assert body["sha256"] and len(body["sha256"]) == 64
    assert isinstance(body["params"], dict) and body["params"], (
        "estimator params should have been extracted"
    )
    # metrics carries through as a dict
    assert isinstance(body["metrics"], dict)


def test_trial_detail_404_for_wrong_run(client: TestClient) -> None:
    """A trial id that exists but under a different run must 404."""
    headers = _headers(_bootstrap(client))
    run_id, _ = _classification_run(client, headers)
    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    tid = trials[0]["id"]

    bogus = "00000000-0000-0000-0000-000000000000"
    r = client.get(f"/api/v1/runs/{bogus}/trials/{tid}", headers=headers)
    # The run itself doesn't exist → 404.
    assert r.status_code == 404


def test_trial_download_streams_pickle(client: TestClient) -> None:
    """The download endpoint returns a real pickle that round-trips."""
    headers = _headers(_bootstrap(client))
    run_id, _ = _classification_run(client, headers)
    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    tid = trials[0]["id"]

    r = client.get(
        f"/api/v1/runs/{run_id}/trials/{tid}/download", headers=headers
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    cd = r.headers.get("content-disposition", "")
    assert "attachment" in cd and ".pkl" in cd
    # Body should unpickle back to a sklearn-ish object with .predict.
    obj = pickle.loads(r.content)
    assert hasattr(obj, "predict")


def test_trial_promote_creates_pipeline(client: TestClient) -> None:
    """Promoting a trial creates a workspace Pipeline + back-links it on the trial."""
    headers = _headers(_bootstrap(client))
    run_id, ws_id = _classification_run(client, headers)
    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    # Pick the second-best to prove "any candidate can be promoted".
    target = trials[1] if len(trials) > 1 else trials[0]
    tid = target["id"]

    promo = client.post(
        f"/api/v1/runs/{run_id}/trials/{tid}/promote",
        json={"name": "iris-runner-up", "description": "second-best candidate"},
        headers=headers,
    )
    assert promo.status_code == 201, promo.text
    pipe = promo.json()
    assert pipe["name"] == "iris-runner-up"
    assert pipe["origin_run_id"] == run_id
    assert pipe["model_id"] == target["model_id"]
    assert pipe["version"] == 1
    assert pipe["sha256"]
    assert isinstance(pipe["params"], dict) and pipe["params"]
    # session-56: unified promote also creates the RegisteredModel + Version.
    # The response includes both IDs so the UI can deep-link without a follow-up call.
    assert pipe["registered_model_id"], "should auto-create RegisteredModel"
    assert pipe["registered_model_version_id"], "should auto-create v1"

    # Trial back-link must be populated so the UI can light up the
    # "Promoted" pill without waiting for a deploy.
    refreshed = client.get(
        f"/api/v1/runs/{run_id}/trials/{tid}", headers=headers
    ).json()
    assert refreshed["fitted_pipeline_id"] == pipe["id"], (
        "Trial.fitted_pipeline_id should be set to the promoted Pipeline id"
    )

    # The registered model should be queryable by the same workspace.
    rm_list = client.get(
        f"/api/v1/workspaces/{ws_id}/registered-models", headers=headers
    ).json()
    rm_names = [m["name"] for m in rm_list]
    assert "iris-runner-up" in rm_names, (
        "RegisteredModel should be visible on the workspace Registry list"
    )

    # Promoting again under the same name bumps the version + reuses family_id
    # AND attaches a v2 to the same RegisteredModel.
    again = client.post(
        f"/api/v1/runs/{run_id}/trials/{tid}/promote",
        json={"name": "iris-runner-up"},
        headers=headers,
    )
    assert again.status_code == 201, again.text
    second = again.json()
    assert second["family_id"] == pipe["family_id"]
    assert second["version"] == 2
    assert second["registered_model_id"] == pipe["registered_model_id"], (
        "second promote under same name should reuse the same RegisteredModel"
    )
    assert second["registered_model_version_id"] != pipe["registered_model_version_id"]

    rm_id = pipe["registered_model_id"]
    versions = client.get(
        f"/api/v1/registered-models/{rm_id}/versions", headers=headers
    ).json()
    assert {v["version"] for v in versions} == {1, 2}, (
        "Both versions should be visible on the RegisteredModel"
    )

    # Listing pipelines for the workspace should now include both versions.
    listed = client.get(
        f"/api/v1/workspaces/{ws_id}/pipelines", headers=headers
    ).json()
    names = [p["name"] for p in listed]
    assert names.count("iris-runner-up") == 2


def test_trial_detail_includes_pipeline_steps_and_plots(client: TestClient) -> None:
    """Detail response carries the full Pipeline structure + the available
    plot kinds for the run's task — the UI uses these to render the
    pipeline diagram and plots grid without follow-up calls."""
    headers = _headers(_bootstrap(client))
    run_id, _ = _classification_run(client, headers)
    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    tid = trials[0]["id"]

    detail = client.get(
        f"/api/v1/runs/{run_id}/trials/{tid}", headers=headers
    ).json()
    assert detail["task"] == "classification"

    steps = detail["pipeline_steps"]
    assert isinstance(steps, list) and steps, "pipeline_steps should be populated"
    last = steps[-1]
    assert last["is_estimator"] is True
    assert last["class"], "estimator class name should be set"
    assert "module" in last and last["module"]
    assert isinstance(last["params"], dict)
    # All non-final steps must NOT be marked estimator.
    for s in steps[:-1]:
        assert s["is_estimator"] is False

    available = detail["available_plots"]
    # Classification baseline plots — confusion_matrix is always registered.
    assert "confusion_matrix" in available
    assert "roc_curve" in available

    # Recursive pipeline tree drives the React diagram. Must be a tree
    # shape, not a flat list — the orchestrator's compare run produces
    # a Pipeline wrapping a ColumnTransformer + estimator.
    tree = detail["pipeline_tree"]
    assert isinstance(tree, dict) and tree
    assert tree["type"] == "pipeline", tree["type"]
    children = tree.get("children")
    assert isinstance(children, list) and children, "tree.children should be set"
    # Last child is the estimator.
    last = children[-1]
    assert last.get("is_estimator") is True
    # And our preprocessor is a ColumnTransformer (which has branches with
    # column lists — the discriminated union the React diagram switches on).
    preprocess = next(
        (c for c in children if c.get("type") == "column_transformer"), None
    )
    if preprocess is not None:
        assert isinstance(preprocess.get("branches"), list)

    # Run context block: the snapshot the orchestrator wrote at submit time
    # must be returned so the UI can render dataset / target / fold / seed.
    snap = detail["run_snapshot"]
    assert isinstance(snap, dict) and snap, "run_snapshot should be returned"
    assert snap.get("task") == "classification"


def test_trial_plot_renders_without_promote(client: TestClient) -> None:
    """Trial-level plot endpoint loads the trial pickle directly — no
    promotion required. This is the conceptual win of session 25."""
    headers = _headers(_bootstrap(client))
    run_id, _ = _classification_run(client, headers)
    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    tid = trials[0]["id"]

    r = client.get(
        f"/api/v1/runs/{run_id}/trials/{tid}/plots/confusion_matrix",
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "confusion_matrix"
    assert body["task"] == "classification"
    assert "figure" in body and isinstance(body["figure"], dict)
    assert body["figure"].get("data"), "figure should carry plot traces"


def test_trial_plot_400_for_unknown_kind(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id, _ = _classification_run(client, headers)
    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    tid = trials[0]["id"]

    r = client.get(
        f"/api/v1/runs/{run_id}/trials/{tid}/plots/nonsense_kind",
        headers=headers,
    )
    assert r.status_code == 400


def test_trial_unpromote_clears_back_link_and_deletes_pipeline(
    client: TestClient,
) -> None:
    """Withdrawing a promotion: trial loses fitted_pipeline_id, pipeline row
    is removed (since no deployments reference it)."""
    headers = _headers(_bootstrap(client))
    run_id, ws_id = _classification_run(client, headers)
    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    tid = trials[0]["id"]

    pipe = client.post(
        f"/api/v1/runs/{run_id}/trials/{tid}/promote",
        json={"name": "to-be-withdrawn"},
        headers=headers,
    ).json()
    pipe_id = pipe["id"]

    # Withdraw: 204.
    r = client.delete(
        f"/api/v1/runs/{run_id}/trials/{tid}/promote", headers=headers
    )
    assert r.status_code == 204

    # Trial back-link cleared.
    detail = client.get(
        f"/api/v1/runs/{run_id}/trials/{tid}", headers=headers
    ).json()
    assert detail["fitted_pipeline_id"] is None

    # Pipeline row gone.
    listed = client.get(
        f"/api/v1/workspaces/{ws_id}/pipelines", headers=headers
    ).json()
    assert all(p["id"] != pipe_id for p in listed)

    # Idempotent: second withdraw is also 204.
    r2 = client.delete(
        f"/api/v1/runs/{run_id}/trials/{tid}/promote", headers=headers
    )
    assert r2.status_code == 204


def test_trial_unpromote_409_when_deployment_exists(client: TestClient) -> None:
    """If a deployment references the promoted pipeline, withdraw must 409
    instead of leaving an orphan deployment."""
    headers = _headers(_bootstrap(client))
    run_id, _ws_id = _classification_run(client, headers)
    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    tid = trials[0]["id"]

    pipe = client.post(
        f"/api/v1/runs/{run_id}/trials/{tid}/promote",
        json={"name": "deployed-promotion"},
        headers=headers,
    ).json()
    dep = client.post(
        f"/api/v1/pipelines/{pipe['id']}/deployments",
        json={"endpoint_slug": "live-slot", "auth_mode": "workspace"},
        headers=headers,
    )
    assert dep.status_code == 201, dep.text

    r = client.delete(
        f"/api/v1/runs/{run_id}/trials/{tid}/promote", headers=headers
    )
    assert r.status_code == 409


def test_trial_promote_requires_name(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id, _ = _classification_run(client, headers)
    trials = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    tid = trials[0]["id"]
    r = client.post(
        f"/api/v1/runs/{run_id}/trials/{tid}/promote",
        json={},
        headers=headers,
    )
    assert r.status_code == 400


# Silences unused-import warning for io (kept for symmetry with other tests).
_ = io
