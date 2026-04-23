"""End-to-end OOP smoke tests for PyCaret 4.0.

Exercises the canonical 4.0 API (``pycaret.tasks.*Experiment`` + the top-level
``save_model`` / ``load_model`` utilities). One test per task — deliberately
small so CI stays fast. Deep coverage per task gets re-authored in later
sessions as each verb is rewritten natively on top of sklearn.

These tests replace the large `test_classification.py` / `test_regression.py` /
etc. that depended on the 3.x module-level functional API (removed in 4.0).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# --------------------------------------------------------------- Classification


@pytest.mark.slow
def test_classification_e2e_oop():
    import pycaret.datasets
    from pycaret import load_model, save_model
    from pycaret.core import CompareResult, CreateResult, PredictResult
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)

    exp = ClassificationExperiment(
        target="Purchase",
        session_id=42,
        n_jobs=1,
        log_experiment=True,
    ).fit(df)
    assert exp.__sklearn_is_fitted__() is True

    created = exp.create_model("lr", verbose=False)
    assert isinstance(created, CreateResult)
    assert created.model_id == "lr"
    assert created.pipeline is not None

    compare = exp.compare_models(include=["lr", "dt"], verbose=False)
    assert isinstance(compare, CompareResult)
    assert len(compare.models) >= 1
    assert compare.best is not None
    assert compare.leaderboard.shape[0] >= 1

    preds = exp.predict_model(compare.best, verbose=False)
    assert isinstance(preds, PredictResult)
    assert "prediction_label" in preds.predictions.columns

    # persistence roundtrip
    with tempfile.TemporaryDirectory() as tmp:
        path = save_model(compare.best, Path(tmp) / "model")
        restored = load_model(path)
        assert type(restored).__name__ == type(compare.best).__name__

    # event stream captured
    events = exp.events
    kinds = {e.kind.value for e in events}
    assert "experiment.started" in kinds
    assert "experiment.fitted" in kinds


# --------------------------------------------------------------- Regression


@pytest.mark.slow
def test_regression_e2e_oop():
    import pycaret.datasets
    from pycaret.core import CompareResult, PredictResult
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)

    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1).fit(df)
    assert exp.__sklearn_is_fitted__() is True

    compare = exp.compare_models(include=["lr", "ridge"], verbose=False)
    assert isinstance(compare, CompareResult)
    assert compare.best is not None

    preds = exp.predict_model(compare.best, verbose=False)
    assert isinstance(preds, PredictResult)
    assert "prediction_label" in preds.predictions.columns


# --------------------------------------------------------------- Clustering


@pytest.mark.slow
def test_clustering_e2e_oop():
    import pycaret.datasets
    from pycaret.core import CreateResult
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)

    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    kmeans = exp.create_model("kmeans", num_clusters=4, verbose=False)
    assert isinstance(kmeans, CreateResult)
    assert kmeans.model_id == "kmeans"

    labelled = exp.assign_model(kmeans.pipeline)
    # legacy returns a DataFrame with a "Cluster" column
    assert labelled is not None


# --------------------------------------------------------------- Anomaly


@pytest.mark.slow
def test_anomaly_e2e_oop():
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment

    df = pycaret.datasets.get_data("anomaly", verbose=False)

    exp = AnomalyExperiment(session_id=42, n_jobs=1).fit(df)
    iforest = exp.create_model("iforest", verbose=False)
    labelled = exp.assign_model(iforest.pipeline)
    assert labelled is not None


# --------------------------------------------------------------- Introspection


def test_api_introspection_surface():
    """Engine-level introspection works without any Experiment construction."""
    import json

    from pycaret.api import describe_model, describe_setup_params, list_metrics, list_models

    # Static listing
    cls_models = list_models("classification")
    reg_models = list_models("regression")
    assert len(cls_models) >= 15
    assert len(reg_models) >= 15
    assert any(m.id == "xgboost" for m in cls_models)
    assert any(m.id == "lightgbm" for m in reg_models)

    # Per-id describe
    card = describe_model("classification", "lr")
    assert card.name == "Logistic Regression"

    # Metric registry
    cls_metrics = list_metrics("classification")
    assert any(m.id == "Accuracy" and m.greater_is_better for m in cls_metrics)

    # Setup-form schema → must round-trip through JSON for React forms
    schema = describe_setup_params("classification")
    payload = json.dumps(schema.to_dict())
    assert "target" in payload
    assert "session_id" in payload
    assert "Preprocessing" in schema.groups


# --------------------------------------------------------------- Logger


def test_memory_logger_fan_out_over_subscribe():
    """A React-UI-like subscriber receives events as they're emitted."""
    from pycaret.logging import EventKind, MemoryLogger

    log = MemoryLogger()
    received = []
    unsub = log.subscribe(received.append)
    try:
        log.log(EventKind.EXPERIMENT_STARTED, message="hi")
        log.log(EventKind.MODEL_CREATED, payload={"model_id": "lr"})
    finally:
        unsub()

    assert len(received) == 2
    assert received[1].payload["model_id"] == "lr"
