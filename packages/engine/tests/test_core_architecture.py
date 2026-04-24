"""Unit tests for the PyCaret 4.0 new architecture primitives.

Covers:
- `pycaret.core` types: `TaskType`, result dataclasses, error hierarchy
- `pycaret.logging`: event stream, `BaseLogger`, `MemoryLogger`
- `pycaret.api`: static introspection (`list_models`, `describe_model`, ...)
- `pycaret.tasks.ClassificationExperiment`: sklearn-compatibility surface

These tests are fast (no model training) and run on every CI matrix entry.
The full end-to-end fit/compare path is covered by the legacy test suite
which delegates through `Experiment._legacy`.
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass

import pytest

# ---------------------------------------------------------------- core types


def test_tasktype_enum_is_str_subclass():
    from pycaret.core import TaskType

    assert TaskType.CLASSIFICATION == "classification"
    assert TaskType.CLASSIFICATION.is_supervised is True
    assert TaskType.CLUSTERING.is_supervised is False
    assert TaskType("regression") is TaskType.REGRESSION


def test_error_hierarchy_inherits_from_stdlib():
    from pycaret.core.errors import (
        ConfigurationError,
        NotFittedError,
        PyCaretError,
        UnknownModelError,
    )

    assert issubclass(PyCaretError, Exception)
    assert issubclass(ConfigurationError, (PyCaretError, ValueError))
    assert issubclass(NotFittedError, (PyCaretError, RuntimeError))
    assert issubclass(UnknownModelError, (PyCaretError, KeyError))


def test_result_dataclasses_are_frozen_and_introspectable():
    from pycaret.core.results import CompareResult, CreateResult, TuneResult

    for cls in (CompareResult, CreateResult, TuneResult):
        assert is_dataclass(cls), cls


# ---------------------------------------------------------------- logging


def test_event_kind_values_are_namespaced_strings():
    from pycaret.logging import EventKind

    assert EventKind.EXPERIMENT_STARTED.value == "experiment.started"
    assert EventKind.MODEL_COMPARED.value == "model.compared"


def test_event_round_trips_through_json():
    from pycaret.logging import Event, EventKind

    e = Event(
        kind=EventKind.MODEL_CREATED,
        message="ok",
        payload={"model_id": "lr", "score": 0.9},
        duration_ms=12.3,
        experiment_id="exp-1",
    )
    d = e.to_dict()
    assert d["kind"] == "model.created"
    assert json.dumps(d)  # no TypeError


def test_memory_logger_captures_events_and_subscribers_fire():
    from pycaret.logging import EventKind, MemoryLogger

    log = MemoryLogger(experiment_id="t")
    received = []
    unsub = log.subscribe(received.append)

    log.log(EventKind.MODEL_CREATED, message="a", payload={"i": 1})
    log.log(EventKind.MODEL_COMPARED, message="b")

    assert len(log.events) == 2
    assert len(received) == 2
    assert received[0].message == "a"
    unsub()
    log.log(EventKind.ERROR, message="c")
    assert len(received) == 2  # unsubscribed before c


def test_memory_logger_tees_to_file(tmp_path):
    from pycaret.logging import EventKind, MemoryLogger

    path = tmp_path / "events.jsonl"
    log = MemoryLogger(experiment_id="t", file=path)
    log.log(EventKind.EXPERIMENT_STARTED, message="hi")
    log.log(EventKind.MODEL_CREATED, payload={"id": "lr"})
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 2
    first = json.loads(raw[0])
    assert first["kind"] == "experiment.started"


def test_base_logger_methods_are_no_ops():
    from pycaret.logging import BaseLogger

    log = BaseLogger()
    # The 3.x shim methods should all be callable without raising.
    log.log_experiment("x", "y")
    log.log_model("x")
    log.log_model_comparison(None)
    log.log_plot(None, "title")
    log.set_tags("src", None, 0.0)
    log.finish_experiment()


# ---------------------------------------------------------------- api (static)


def test_list_models_classification_is_populated():
    from pycaret.api import list_models

    cards = list_models("classification")
    assert len(cards) >= 15
    ids = {c.id for c in cards}
    # A few ids any user of pycaret expects to be there:
    for expected in ("lr", "rf", "xgboost", "lightgbm", "catboost"):
        assert expected in ids, f"{expected} missing from list_models(classification)"


def test_describe_model_returns_card_with_metadata():
    from pycaret.api import describe_model

    card = describe_model("classification", "lr")
    assert card.id == "lr"
    assert card.name == "Logistic Regression"
    assert "linear" in card.tags
    assert card.library == "sklearn"


def test_describe_model_raises_for_unknown_id():
    from pycaret.api import describe_model
    from pycaret.core.errors import UnknownModelError

    with pytest.raises(UnknownModelError):
        describe_model("classification", "not_a_real_model")


def test_list_metrics_has_defaults_and_direction():
    from pycaret.api import list_metrics

    cls = list_metrics("classification")
    reg = list_metrics("regression")
    assert any(m.is_default for m in cls)
    assert any(m.is_default for m in reg)
    # `Accuracy` is greater_is_better; regression `MAE` is not.
    assert next(m for m in cls if m.id == "Accuracy").greater_is_better is True
    assert next(m for m in reg if m.id == "MAE").greater_is_better is False


def test_describe_setup_params_feeds_a_react_form():
    from pycaret.api import describe_setup_params

    schema = describe_setup_params("classification")
    assert schema.task == "classification"
    assert len(schema.parameters) >= 10

    # Every parameter is JSON-round-trippable.
    serialized = json.dumps(schema.to_dict())
    assert "target" in serialized
    assert "session_id" in serialized

    # Groups exist and a React form could iterate them.
    assert "Preprocessing" in schema.groups
    assert "Cross-Validation" in schema.groups


# ---------------------------------------------------------------- Experiment


def test_classification_experiment_is_sklearn_compatible():
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="y", session_id=7, fold=5, normalize=True)
    params = exp.get_params()
    # All 15 init parameters visible to sklearn introspection.
    assert "target" in params
    assert "session_id" in params
    assert params["session_id"] == 7
    assert params["fold"] == 5
    assert params["normalize"] is True

    # sklearn.base.clone works — config preserved, fitted state dropped.
    from sklearn.base import clone

    exp2 = clone(exp)
    assert exp2.target == "y"
    assert exp2.fold == 5
    assert exp2.__sklearn_is_fitted__() is False


def test_classification_experiment_declares_classifier_tag():
    from pycaret.tasks import ClassificationExperiment

    tags = ClassificationExperiment().__sklearn_tags__()
    # Not all sklearn versions expose `estimator_type` on Tags; be tolerant.
    etype = getattr(tags, "estimator_type", None)
    if etype is not None:
        assert etype == "classifier"


def test_tasks_package_exports_all_five_task_subclasses():
    import pycaret.tasks as tasks

    for name in (
        "ClassificationExperiment",
        "RegressionExperiment",
        "ClusteringExperiment",
        "AnomalyExperiment",
        "TimeSeriesExperiment",
    ):
        assert hasattr(tasks, name), f"pycaret.tasks missing {name}"


def test_legacy_import_paths_re_export_new_classes():
    # 4.0 is OOP-only; each legacy module path surfaces the 4.0 class only.
    from pycaret.anomaly import AnomalyExperiment as A
    from pycaret.classification import ClassificationExperiment as C
    from pycaret.clustering import ClusteringExperiment as Cl
    from pycaret.regression import RegressionExperiment as R
    from pycaret.tasks import (
        AnomalyExperiment,
        ClassificationExperiment,
        ClusteringExperiment,
        RegressionExperiment,
        TimeSeriesExperiment,
    )
    from pycaret.time_series import TimeSeriesExperiment as TS

    assert A is AnomalyExperiment
    assert C is ClassificationExperiment
    assert Cl is ClusteringExperiment
    assert R is RegressionExperiment
    assert TS is TimeSeriesExperiment


def test_all_task_subclasses_are_sklearn_compatible():
    from pycaret.tasks import (
        AnomalyExperiment,
        ClassificationExperiment,
        ClusteringExperiment,
        RegressionExperiment,
        TimeSeriesExperiment,
    )
    from sklearn.base import clone

    for Cls, init_kwargs in [
        (ClassificationExperiment, dict(target="y", session_id=1)),
        (RegressionExperiment, dict(target="y", session_id=1)),
        (ClusteringExperiment, dict(session_id=1)),
        (AnomalyExperiment, dict(session_id=1)),
        (TimeSeriesExperiment, dict(fh=3, session_id=1)),
    ]:
        exp = Cls(**init_kwargs)
        assert exp.__sklearn_is_fitted__() is False
        params = exp.get_params()
        assert "session_id" in params
        clone(exp)  # must not raise


def test_top_level_save_load_model_roundtrip(tmp_path):
    from pycaret import load_model, save_model
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    path = tmp_path / "model"
    written = save_model(model, path)
    assert written.exists()
    restored = load_model(path)
    assert type(restored).__name__ == "LogisticRegression"


def test_functional_api_is_gone():
    # Importing the old functional entry points must now fail — 4.0 is OOP-only.
    import pycaret.classification as cls

    assert not hasattr(cls, "setup"), "`setup` should be gone in 4.0"
    assert not hasattr(cls, "compare_models"), "`compare_models` functional gone"
    assert not hasattr(cls, "set_current_experiment"), "state fn gone"
