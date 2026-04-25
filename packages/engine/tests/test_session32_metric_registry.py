"""Session 32 — drain add_metric / remove_metric onto a per-Experiment registry.

After session 31, the only secondary verbs still on `self._legacy` were
``add_metric`` / ``remove_metric`` / ``get_config`` / ``set_config`` /
``plot_model`` / ``evaluate_model``. Session 32 drains the first two by
promoting the metric registry to a per-Experiment dict on
``self._fit_state["metric_registry"]``. ``add_metric`` mutates that dict;
``create_model`` / ``tune_model`` / ``compare_models`` /
``predict_model`` all read from it via the new ``_get_metric_registry()``
helper.

Most importantly, **a custom metric registered via ``add_metric``
actually shows up in subsequent CV results** — that wasn't true before
the drain (the custom metric stayed in the legacy holder while the
native CV path read directly from the global container helpers).
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_add_metric_appears_in_subsequent_create_model_cv():
    """The killer test: after add_metric, the new metric column is in CV results."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.metrics import balanced_accuracy_score

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    exp.add_metric(
        id="balanced_acc",
        name="Balanced Accuracy",
        score_func=balanced_accuracy_score,
        target="pred",
        greater_is_better=True,
    )
    created = exp.create_model("lr", verbose=False)
    assert created.metrics is not None
    cols = list(created.metrics.columns)
    assert any("Balanced" in c for c in cols), f"Balanced Accuracy missing from {cols}"


@pytest.mark.slow
def test_add_metric_appears_in_get_metrics_dataframe():
    """`exp.get_metrics()` reflects added metrics + drops removed ones."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.metrics import matthews_corrcoef

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    base_count = len(exp.get_metrics())
    exp.add_metric(
        id="mcc_custom",
        name="Custom MCC",
        score_func=matthews_corrcoef,
        target="pred",
    )
    assert len(exp.get_metrics()) == base_count + 1
    assert "mcc_custom" in exp.get_metrics().index
    # Custom flag is True for added metrics.
    assert bool(exp.get_metrics().loc["mcc_custom", "Custom"]) is True


@pytest.mark.slow
def test_remove_metric_drops_from_registry():
    """remove_metric drops the metric so subsequent CV doesn't compute it."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.metrics import balanced_accuracy_score

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    # Add then remove.
    exp.add_metric(
        id="balanced_acc",
        name="Balanced Accuracy",
        score_func=balanced_accuracy_score,
    )
    exp.remove_metric("balanced_acc")
    created = exp.create_model("lr", verbose=False)
    cols = list(created.metrics.columns)
    assert not any("Balanced" in c for c in cols)


@pytest.mark.slow
def test_remove_metric_accepts_display_name_too():
    """remove_metric matches by ID or display name (legacy semantics)."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    base_count = len(exp.get_metrics())
    # 'acc' is the built-in Accuracy metric (id='acc', name='Accuracy').
    exp.remove_metric("Accuracy")
    assert len(exp.get_metrics()) == base_count - 1
    assert "acc" not in exp.get_metrics().index


@pytest.mark.slow
def test_remove_metric_unknown_raises():
    """ValueError when the metric isn't in the registry."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    with pytest.raises(ValueError, match="No metric matching"):
        exp.remove_metric("zzz_not_real")


@pytest.mark.slow
def test_add_metric_does_not_call_legacy(monkeypatch):
    """Drain-lock: add_metric does NOT call self._legacy.add_metric for
    classification/regression.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.metrics import balanced_accuracy_score

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Session-32 drain regression: add_metric called legacy.")

    monkeypatch.setattr(exp._legacy, "add_metric", _poison)
    exp.add_metric(
        id="balanced_acc",
        name="Balanced Accuracy",
        score_func=balanced_accuracy_score,
    )
    assert "balanced_acc" in exp.get_metrics().index


@pytest.mark.slow
def test_remove_metric_does_not_call_legacy(monkeypatch):
    """Drain-lock for remove_metric."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Session-32 drain regression: remove_metric called legacy.")

    monkeypatch.setattr(exp._legacy, "remove_metric", _poison)
    exp.remove_metric("acc")  # built-in Accuracy
    assert "acc" not in exp.get_metrics().index


@pytest.mark.slow
def test_metric_registry_persists_across_verbs():
    """A metric added once shows up in tune_model + compare_models too."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.metrics import balanced_accuracy_score

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    exp.add_metric(
        id="balanced_acc",
        name="Balanced Accuracy",
        score_func=balanced_accuracy_score,
    )

    created = exp.create_model("lr", verbose=False)
    assert any("Balanced" in c for c in created.metrics.columns)

    tuned = exp.tune_model(created.pipeline, n_iter=3, verbose=False)
    assert any("Balanced" in c for c in tuned.metrics.columns)

    cm = exp.compare_models(include=["lr", "dt"], n_select=2, verbose=False)
    assert any("Balanced" in c for c in cm.leaderboard.columns)


@pytest.mark.slow
def test_regression_add_metric_works():
    """Regression accepts add_metric too."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment
    from sklearn.metrics import mean_absolute_error

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    exp.add_metric(
        id="custom_mae",
        name="Custom MAE",
        score_func=mean_absolute_error,
        greater_is_better=False,
    )
    created = exp.create_model("lr", verbose=False)
    assert any("Custom MAE" == c or "Custom" in c for c in created.metrics.columns)


def test_add_metric_requires_fit():
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="Purchase", session_id=0)
    with pytest.raises(NotFittedError):
        exp.add_metric(id="a", name="a", score_func=lambda y, p: 0.0)
    with pytest.raises(NotFittedError):
        exp.remove_metric("acc")
