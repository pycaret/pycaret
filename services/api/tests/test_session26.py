"""Session 26 — trial dashboard expansion.

Backs the redesigned ``/runs/:id/trials/:tid`` surface:

- ``PATCH /runs/:id/trials/:tid``         — free-form notes annotation
- ``POST  /runs/:id/trials/:tid/predict`` — try inference without promoting
- ``GET   /runs/:id/trials/:tid/cv``      — per-fold cross-validation
- ``GET   /runs/:id/trials/:tid/cohorts`` — slice metrics by a column
- detail response carries a recursive ``pipeline_tree`` (replaces the
  old flat step list + sklearn HTML repr)

The fixture mirrors session 25's so individual tests stay isolated.
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


def _classification_run(client: TestClient, headers: dict) -> str:
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    p_id = client.post(
        f"/api/v1/workspaces/{ws_id}/projects",
        json={"name": "Demo"},
        headers=headers,
    ).json()["id"]
    exp_id = client.post(
        f"/api/v1/projects/{p_id}/experiments",
        json={
            "name": "exp",
            "task": "classification",
            "target": "target",
            "setup_params": {"session_id": 42, "fold": 2, "verbose": False},
        },
        headers=headers,
    ).json()["id"]
    run_id = client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        json={"plan": "compare", "sklearn_dataset": "iris"},
        headers=headers,
    ).json()["id"]
    body = client.post(
        f"/api/v1/runs/{run_id}/wait?timeout_s=120", headers=headers
    ).json()
    assert body["status"] == "succeeded", body.get("error")
    return run_id


def _first_trial(client: TestClient, headers: dict, run_id: str) -> dict:
    items = client.get(f"/api/v1/runs/{run_id}/trials", headers=headers).json()[
        "items"
    ]
    return items[0]


# ============================================================ pipeline_tree


def test_detail_returns_recursive_pipeline_tree(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)
    detail = client.get(
        f"/api/v1/runs/{run_id}/trials/{t['id']}", headers=headers
    ).json()
    tree = detail["pipeline_tree"]
    assert isinstance(tree, dict) and tree["type"] == "pipeline"
    assert isinstance(tree["children"], list) and tree["children"]
    last = tree["children"][-1]
    assert last["is_estimator"] is True


# ===================================================================== notes


def test_patch_notes_round_trip(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)

    # Initial: no notes.
    detail = client.get(
        f"/api/v1/runs/{run_id}/trials/{t['id']}", headers=headers
    ).json()
    assert detail["notes"] in (None, "")

    r = client.patch(
        f"/api/v1/runs/{run_id}/trials/{t['id']}",
        json={"notes": "Worth tuning, check on cohort=A"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["notes"] == "Worth tuning, check on cohort=A"

    # Read-back picks up the change.
    detail = client.get(
        f"/api/v1/runs/{run_id}/trials/{t['id']}", headers=headers
    ).json()
    assert detail["notes"] == "Worth tuning, check on cohort=A"

    # Empty string clears it.
    r = client.patch(
        f"/api/v1/runs/{run_id}/trials/{t['id']}",
        json={"notes": ""},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["notes"] is None


def test_patch_notes_validates_type(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)
    r = client.patch(
        f"/api/v1/runs/{run_id}/trials/{t['id']}",
        json={"notes": 42},
        headers=headers,
    )
    assert r.status_code == 400


# ============================================================== predict


def test_predict_runs_inference(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)

    rows = [
        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3},
    ]
    r = client.post(
        f"/api/v1/runs/{run_id}/trials/{t['id']}/predict",
        json={"rows": rows},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "predictions" in body and len(body["predictions"]) == 2
    # Iris classifiers expose predict_proba — assert when present.
    if "probabilities" in body:
        assert len(body["probabilities"]) == 2
        assert len(body["probabilities"][0]) >= 2
        assert "classes" in body


def test_predict_rejects_empty_rows(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)
    r = client.post(
        f"/api/v1/runs/{run_id}/trials/{t['id']}/predict",
        json={"rows": []},
        headers=headers,
    )
    assert r.status_code == 400


# ============================================================ per-fold CV


def test_cv_returns_per_fold_metrics(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)
    r = client.get(
        f"/api/v1/runs/{run_id}/trials/{t['id']}/cv?folds=3",
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "classification"
    assert body["folds"] == 3
    assert "accuracy" in body["scoring"]
    assert len(body["rows"]) == 3
    for row in body["rows"]:
        assert "fold" in row
        assert isinstance(row.get("accuracy"), (int, float))
    assert "accuracy" in body["summary"]
    s = body["summary"]["accuracy"]
    assert all(k in s for k in ("mean", "std", "min", "max"))


# ================================================================ cohorts


def test_cohorts_probe_returns_columns(client: TestClient) -> None:
    """Empty `column` → discovery: just the available columns."""
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)
    r = client.get(
        f"/api/v1/runs/{run_id}/trials/{t['id']}/cohorts",
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["column"] is None
    assert body["rows"] == []
    assert isinstance(body["available_columns"], list) and body["available_columns"]
    # Iris has 4 features.
    assert len(body["available_columns"]) >= 1


def test_cohorts_slices_holdout_metrics(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)
    cols = client.get(
        f"/api/v1/runs/{run_id}/trials/{t['id']}/cohorts", headers=headers
    ).json()["available_columns"]
    column = cols[0]

    r = client.get(
        f"/api/v1/runs/{run_id}/trials/{t['id']}/cohorts?column={column}",
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["column"] == column
    assert isinstance(body["rows"], list) and body["rows"]
    for row in body["rows"]:
        assert isinstance(row["n"], int) and row["n"] > 0
        # Classification cohort metrics include accuracy.
        assert "accuracy" in row["metrics"]


def test_cohorts_rejects_target_column(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)
    r = client.get(
        f"/api/v1/runs/{run_id}/trials/{t['id']}/cohorts?column=target",
        headers=headers,
    )
    assert r.status_code == 400


# =================================================== input_schema on detail


def test_detail_includes_input_schema(client: TestClient) -> None:
    """Detail response carries an input_schema with a real holdout row.

    The Predict tab uses sample_row to prefill its textarea so the user
    can run inference immediately without knowing the column names.
    """
    headers = _headers(_bootstrap(client))
    run_id = _classification_run(client, headers)
    t = _first_trial(client, headers, run_id)
    detail = client.get(
        f"/api/v1/runs/{run_id}/trials/{t['id']}", headers=headers
    ).json()

    schema = detail["input_schema"]
    assert isinstance(schema, dict) and schema
    assert schema["target"] == "target"
    cols = schema["columns"]
    # Iris has 4 features once target is dropped.
    assert isinstance(cols, list) and len(cols) == 4
    for c in cols:
        assert "name" in c and "dtype" in c
        # target must NOT appear in the schema columns.
        assert c["name"] != "target"
    # The sample row must use the same keys as columns.
    assert set(schema["sample_row"].keys()) == {c["name"] for c in cols}


# =================================================== dataset profile


_TINY_PROFILE_CSV = (
    b"id,age,score,group,note\n"
    b"1,23,0.5,A,foo\n"
    b"2,35,0.8,B,bar\n"
    b"3,28,,A,foo\n"
    b"4,42,0.2,B,baz\n"
    b"5,30,0.6,A,foo\n"
).strip() + b"\n"


def _upload_csv(
    client: TestClient,
    headers: dict,
    ws_id: str,
    csv_bytes: bytes,
    name: str,
) -> str:
    from io import BytesIO

    r = client.post(
        f"/api/v1/workspaces/{ws_id}/data-sources/upload",
        files={"file": (name, BytesIO(csv_bytes), "text/csv")},
        data={"name": name},
        headers=headers,
    )
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def test_dataset_profile_returns_rich_json(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    ds_id = _upload_csv(client, headers, ws_id, _TINY_PROFILE_CSV, "tiny.csv")

    r = client.get(f"/api/v1/data-sources/{ds_id}/profile", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["name"] == "tiny.csv"
    assert body["shape"] == {"rows": 5, "cols": 5}
    assert body["memory_bytes"] > 0
    assert body["duplicates"] == 0
    assert body["missing_total"] == 1
    assert body["missing_pct"] > 0

    # Type counts: id (int) + age (int) + score (float) numeric, group / note text/cat.
    assert body["type_counts"]["numeric"] >= 2
    assert (
        body["type_counts"]["categorical"] + body["type_counts"]["text"] >= 2
    )

    # Per-column profile shape.
    by_name = {c["name"]: c for c in body["columns"]}
    assert set(by_name) == {"id", "age", "score", "group", "note"}
    assert by_name["score"]["missing"] == 1
    assert by_name["score"]["kind"] == "numeric"
    assert by_name["score"]["stats"] is not None
    assert by_name["score"]["stats"]["min"] == 0.2
    assert isinstance(by_name["score"]["histogram"], list) and by_name["score"][
        "histogram"
    ]
    assert by_name["group"]["kind"] == "categorical"
    assert by_name["group"]["top_values"], "categorical column should have top_values"

    # `id` is constant-cardinality == n_rows → flagged ID-like.
    assert by_name["id"]["is_id_like"] is True

    # Correlations only between numeric columns.
    corrs = body["correlations"]
    assert corrs is not None
    assert len(corrs["columns"]) == body["type_counts"]["numeric"]

    # Sample includes head rows with column-keyed dicts.
    assert isinstance(body["sample"], list) and body["sample"]
    assert "id" in body["sample"][0]


def test_dataset_profile_404_for_unknown_id(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    r = client.get(
        "/api/v1/data-sources/00000000-0000-0000-0000-000000000000/profile",
        headers=headers,
    )
    assert r.status_code == 404


def test_dataset_profile_warns_on_constant_column(client: TestClient) -> None:
    headers = _headers(_bootstrap(client))
    ws_id = client.get("/api/v1/workspaces", headers=headers).json()[0]["id"]
    csv = b"const,value\nx,1\nx,2\nx,3\nx,4\n"
    ds_id = _upload_csv(client, headers, ws_id, csv, "const.csv")
    body = client.get(
        f"/api/v1/data-sources/{ds_id}/profile", headers=headers
    ).json()
    kinds = {w["kind"] for w in body["warnings"]}
    assert "constant" in kinds
