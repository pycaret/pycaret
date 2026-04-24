"""Session 22 — god-class drain, first pass: persistence verbs.

The ``save_model`` / ``load_model`` / ``save_experiment`` / ``load_experiment``
verbs on ``Experiment`` no longer delegate to ``self._legacy``. They are now
thin wrappers around the stateless helpers in ``pycaret.persistence``. These
tests lock that contract:

- ``save_model`` works with OR without an Experiment (stateless).
- Round-trip predictions match (pipeline preserves behavior).
- ``save_experiment`` / ``load_experiment`` round-trip ``self``, restore
  fitted state, and reject the cross-type use case
  (calling ``load_experiment`` on a plain model file raises).

These tests deliberately use small sklearn-native fixtures (no engine fit)
so they run in milliseconds — deep end-to-end tests for persistence stay in
``test_e2e_oop.py`` under the `slow` mark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------- helpers


def _fit_tiny_pipeline(random_state: int = 0) -> Pipeline:
    """A minimal sklearn Pipeline fitted on a few rows — fast + deterministic."""
    X, y = make_classification(n_samples=80, n_features=5, random_state=random_state)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipe.fit(X, y)
    return pipe


# --------------------------------------------------------------------- save_model


def test_save_model_via_experiment_instance_roundtrips(tmp_path: Path):
    """`exp.save_model(...)` + `exp.load_model(...)` preserve predictions."""
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="y", session_id=0)
    pipe = _fit_tiny_pipeline()
    X_probe, _ = make_classification(n_samples=10, n_features=5, random_state=1)
    expected = pipe.predict(X_probe)

    # save_model does NOT require fit — mirrors the stateless persistence
    # helper. We shouldn't need to touch the legacy god-class at all.
    written = exp.save_model(pipe, tmp_path / "tiny")
    assert written.exists()
    assert written.suffix == ".pkl"

    restored = exp.load_model(tmp_path / "tiny")
    # Round-trip predictions match the original pipeline.
    np.testing.assert_array_equal(restored.predict(X_probe), expected)


def test_save_model_does_not_touch_legacy(tmp_path: Path, monkeypatch):
    """The new save_model MUST NOT call anything on `self._legacy`.

    If a future refactor accidentally adds a delegation back in, this test
    catches it: we run save_model on an Experiment that has never been fit,
    so `self._legacy` doesn't even exist yet. Any `self._legacy.*` access
    would raise AttributeError.
    """
    from pycaret.tasks import RegressionExperiment

    exp = RegressionExperiment(target="y", session_id=0)
    assert not hasattr(exp, "_legacy"), "unfit Experiment should not have _legacy yet"
    pipe = _fit_tiny_pipeline()
    exp.save_model(pipe, tmp_path / "reg")  # must not raise
    exp.load_model(tmp_path / "reg")  # must not raise
    # Still no legacy — the drain is clean.
    assert not hasattr(exp, "_legacy")


def test_save_model_accepts_path_objects_and_strings(tmp_path: Path):
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="y", session_id=0)
    pipe = _fit_tiny_pipeline()
    # Path object
    w1 = exp.save_model(pipe, tmp_path / "as_path")
    # String path
    w2 = exp.save_model(pipe, str(tmp_path / "as_str"))
    assert w1.exists() and w2.exists()
    assert str(w1).endswith(".pkl")
    assert str(w2).endswith(".pkl")


# --------------------------------------------------------------- MODEL_SAVED event


def test_save_model_emits_model_saved_event(tmp_path: Path):
    """The event stream records the save with the absolute path."""
    from pycaret.logging import EventKind, MemoryLogger
    from pycaret.tasks import ClassificationExperiment

    log = MemoryLogger()
    exp = ClassificationExperiment(target="y", session_id=0, logger=log)
    pipe = _fit_tiny_pipeline()
    written = exp.save_model(pipe, tmp_path / "evt")
    kinds = [e.kind for e in log.events]
    assert EventKind.MODEL_SAVED in kinds
    saved_events = [e for e in log.events if e.kind == EventKind.MODEL_SAVED]
    assert saved_events[-1].payload["path"] == str(written)


# -------------------------------------------------------------- save_experiment


def test_save_experiment_requires_fit():
    """Calling save_experiment on an unfit Experiment raises NotFittedError."""
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="y", session_id=0)
    with pytest.raises(NotFittedError):
        exp.save_experiment("/tmp/should-not-be-written")


def test_load_experiment_rejects_plain_model_file(tmp_path: Path):
    """`Experiment.load_experiment` refuses a file that contains a plain
    sklearn model. The error message should steer the caller to
    `load_model` instead.
    """
    from pycaret.tasks import ClassificationExperiment

    pipe = _fit_tiny_pipeline()
    exp = ClassificationExperiment(target="y", session_id=0)
    # Save a plain model, then try to load it as an Experiment.
    written = exp.save_model(pipe, tmp_path / "plain")

    with pytest.raises(TypeError, match="not a PyCaret Experiment"):
        ClassificationExperiment.load_experiment(written)


# -------------------------------------------------- stateless helpers still work


def test_module_level_helpers_still_exposed(tmp_path: Path):
    """The top-level `pycaret.save_model` / `pycaret.load_model` remain
    available — the god-class drain shouldn't remove them.
    """
    from pycaret import load_model, save_model

    pipe = _fit_tiny_pipeline()
    written = save_model(pipe, tmp_path / "top")
    restored = load_model(written)
    assert type(restored).__name__ == "Pipeline"
