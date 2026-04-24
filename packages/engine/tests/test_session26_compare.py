"""Session 26 — god-class drain: ``compare_models`` (supervised path).

Supervised ``SupervisedExperiment.compare_models`` no longer delegates to
``self._legacy.compare_models``. It iterates the engine's model registry,
calls the (already-drained) ``self.create_model`` for each entry, and
assembles the leaderboard from each model's ``Mean`` metrics row.

Drains 7→8 of the 10 supervised verbs.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_compare_models_returns_top_n_pipelines_classification():
    """compare_models(include=['lr', 'dt'], n_select=2) returns 2 fitted Pipelines."""
    import pycaret.datasets
    from pycaret.core import CompareResult
    from pycaret.tasks import ClassificationExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    result: CompareResult = exp.compare_models(include=["lr", "dt"], n_select=2, verbose=False)
    assert isinstance(result, CompareResult)
    assert len(result.models) == 2
    assert isinstance(result.best, SkPipeline)
    assert all(isinstance(m, SkPipeline) for m in result.models)
    # Leaderboard has both models in it, with the standard classification
    # metric columns.
    assert set(result.leaderboard["Model"]) == {"lr", "dt"}
    cols = [c.lower() for c in result.leaderboard.columns]
    assert any("accuracy" in c for c in cols)


@pytest.mark.slow
def test_compare_models_default_sort_is_accuracy_classification():
    """Default `sort` for classification is `Accuracy`; leaderboard
    should be sorted desc by it.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    # Force two distinguishable models.
    result = exp.compare_models(include=["lr", "dt"], n_select=2, verbose=False)
    accs = result.leaderboard["Accuracy"].tolist()
    assert accs == sorted(accs, reverse=True)
    # ranked_ids matches the leaderboard order.
    assert result.ranked_ids == result.leaderboard["Model"].astype(str).tolist()


@pytest.mark.slow
def test_compare_models_regression_default_sort_is_r2():
    """Regression default `sort` is R2 (greater-is-better)."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    result = exp.compare_models(include=["lr", "ridge"], n_select=2, verbose=False)
    # R2 column present + sorted desc.
    cols = result.leaderboard.columns.tolist()
    assert any(c == "R2" or c == "R²" for c in cols)
    r2_col = next(c for c in cols if c == "R2" or c == "R²")
    r2s = result.leaderboard[r2_col].tolist()
    assert r2s == sorted(r2s, reverse=True)


@pytest.mark.slow
def test_compare_models_sort_with_ascending_metric():
    """`sort='MAE'` (regression error) → leaderboard sorted ASCENDING."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    result = exp.compare_models(include=["lr", "ridge"], n_select=2, sort="MAE", verbose=False)
    maes = result.leaderboard["MAE"].tolist()
    assert maes == sorted(maes)


@pytest.mark.slow
def test_compare_models_exclude_drops_models():
    """exclude= drops a model from the registry walk."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    result = exp.compare_models(
        include=["lr", "dt", "knn"],
        exclude=["dt"],
        n_select=3,
        verbose=False,
    )
    assert "dt" not in result.ranked_ids
    assert set(result.ranked_ids).issubset({"lr", "knn"})


@pytest.mark.slow
def test_compare_models_turbo_skips_slow_models():
    """turbo=True skips rbfsvm / gpc / mlp by default."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    # Pass them in include= but rely on turbo=True (default) to skip them.
    result = exp.compare_models(include=["lr", "rbfsvm", "gpc", "mlp"], n_select=4, verbose=False)
    assert "rbfsvm" not in result.ranked_ids
    assert "gpc" not in result.ranked_ids
    assert "mlp" not in result.ranked_ids
    assert "lr" in result.ranked_ids


@pytest.mark.slow
def test_compare_models_does_not_call_legacy_compare_models(monkeypatch):
    """Drain-lock: self._legacy.compare_models must NOT be invoked."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-26 drain regression: SupervisedExperiment.compare_models "
            "called self._legacy.compare_models. The native path must "
            "iterate registry + call self.create_model."
        )

    monkeypatch.setattr(exp._legacy, "compare_models", _poison)

    result = exp.compare_models(include=["lr", "dt"], n_select=1, verbose=False)
    assert result.best is not None


@pytest.mark.slow
def test_compare_models_predict_chain_from_best():
    """compare_models → predict_model on result.best works without
    transitional branches (best is a real Pipeline).
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    result = exp.compare_models(include=["lr", "dt"], n_select=1, verbose=False)
    preds = exp.predict_model(result.best)
    assert "prediction_label" in preds.predictions.columns
    assert preds.metrics is not None


@pytest.mark.slow
def test_compare_models_errors_ignore_skips_failing_model():
    """A model that raises during create_model is skipped, the rest succeed."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    # Inject a fake "bogus" string id alongside a real one. Bogus → ConfigurationError
    # in create_model → swallowed by errors='ignore'.
    result = exp.compare_models(
        include=["lr", "zzzz_bogus_id"],
        n_select=1,
        errors="ignore",
        verbose=False,
    )
    assert "zzzz_bogus_id" not in result.ranked_ids
    assert "lr" in result.ranked_ids


def test_compare_models_requires_fit():
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="Purchase", session_id=0)
    with pytest.raises(NotFittedError):
        exp.compare_models(include=["lr"])
