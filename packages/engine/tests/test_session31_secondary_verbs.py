"""Session 31 — drain the secondary verbs: pull / models / get_metrics.

After session 30 finished the internal-state drain, only setup() and a
handful of advisory verbs remained on self._legacy. Session 31 drains
the three that have a clean native equivalent:

  - ``pull()`` reads from ``self._fit_state["last_metrics"]``, which
    every native modeling verb updates before returning.
  - ``models()`` builds a DataFrame from the snapshot's
    ``model_registry``.
  - ``get_metrics()`` reads directly from the task's metric registry.

The remaining secondary verbs (``add_metric``, ``remove_metric``,
``get_config``, ``set_config``, ``plot_model``, ``evaluate_model``)
keep delegating — they each need a bigger registry-side refactor that's
out of scope for this session.
"""

from __future__ import annotations

import pytest

# ============================================================ pull


@pytest.mark.slow
def test_pull_returns_create_model_metrics(monkeypatch):
    """`exp.pull()` after create_model returns the same DataFrame as
    `CreateResult.metrics`. Drain-locks: legacy.pull is poisoned.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-31 drain regression: pull() called self._legacy.pull "
            "after a native verb populated _fit_state['last_metrics']."
        )

    monkeypatch.setattr(exp._legacy, "pull", _poison)

    created = exp.create_model("lr", verbose=False)
    pulled = exp.pull()
    assert pulled.equals(created.metrics)


@pytest.mark.slow
def test_pull_tracks_compare_models_leaderboard():
    """After compare_models, pull() returns the leaderboard."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    result = exp.compare_models(include=["lr", "dt"], n_select=2, verbose=False)
    assert exp.pull().equals(result.leaderboard)


@pytest.mark.slow
def test_pull_tracks_tune_model_metrics():
    """After tune_model, pull() returns the tuned metrics DataFrame."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    created = exp.create_model("lr", verbose=False)
    tuned = exp.tune_model(created.pipeline, n_iter=3, verbose=False)
    assert exp.pull().equals(tuned.metrics)


# ============================================================ models


@pytest.mark.slow
def test_models_returns_native_dataframe_from_snapshot(monkeypatch):
    """`exp.models()` reads from _fit_state["model_registry"], not legacy."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-31 drain regression: models() called self._legacy.models "
            "instead of building from _fit_state['model_registry']."
        )

    monkeypatch.setattr(exp._legacy, "models", _poison)

    df_models = exp.models()
    # Standard PyCaret columns: Name, Reference, Turbo. Index is the model ID.
    assert df_models.index.name == "ID"
    assert {"Name", "Reference", "Turbo"}.issubset(set(df_models.columns))
    # Classification has at least 15 entries in the registry.
    assert len(df_models) >= 15
    # Reference is a fully-qualified module path.
    assert df_models.loc["lr", "Reference"].endswith("LogisticRegression")


@pytest.mark.slow
def test_models_internal_true_falls_back_to_legacy():
    """`models(internal=True)` keeps delegating to legacy (preserves the
    richer ModelContainer view that callers expect).
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    df_internal = exp.models(internal=True)
    assert df_internal is not None
    assert len(df_internal) >= 15


# ============================================================ get_metrics


@pytest.mark.slow
def test_get_metrics_reads_metric_registry_directly(monkeypatch):
    """`exp.get_metrics()` reads from the task's metric registry, not legacy."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-31 drain regression: get_metrics() called self._legacy.get_metrics."
        )

    monkeypatch.setattr(exp._legacy, "get_metrics", _poison)

    df_metrics = exp.get_metrics()
    assert df_metrics.index.name == "ID"
    cols = set(df_metrics.columns)
    assert {"Name", "Display Name", "Greater is Better"}.issubset(cols)
    # Standard classification metrics.
    metric_ids = set(df_metrics.index.tolist())
    assert "acc" in metric_ids or "accuracy" in metric_ids


@pytest.mark.slow
def test_get_metrics_for_regression_has_neg_mae():
    """Regression metric registry has MAE and R2 (or equivalent)."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    df_metrics = exp.get_metrics()
    names = [str(n).lower() for n in df_metrics["Name"].tolist()]
    # At least one MAE / R2 entry.
    assert any("mae" in n for n in names)
    assert any("r2" in n for n in names)


# ============================================================ require fit


def test_secondary_verbs_require_fit():
    """All three drained verbs raise NotFittedError on an unfit experiment."""
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="Purchase", session_id=0)
    with pytest.raises(NotFittedError):
        exp.pull()
    with pytest.raises(NotFittedError):
        exp.models()
    with pytest.raises(NotFittedError):
        exp.get_metrics()
