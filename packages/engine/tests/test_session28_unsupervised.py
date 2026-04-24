"""Session 28 — god-class drain: unsupervised verbs.

``UnsupervisedExperiment.create_model`` and ``assign_model`` no longer
delegate to ``self._legacy.<verb>`` for clustering + anomaly. They run
the registry's sklearn / pyod estimator directly on
``self._legacy.X_transformed`` and return real sklearn Pipelines.

This finishes the OOP drain — the only `_legacy` callsites that remain
are the time-series experiment + a few remaining read-only attribute
properties (X / X_train / pipeline / etc.) that are part of a separate
"property drain" follow-up.
"""

from __future__ import annotations

import pytest

# ============================================================ clustering


@pytest.mark.slow
def test_clustering_create_model_returns_pipeline_with_kmeans():
    """create_model('kmeans') on a clustering exp returns a fitted Pipeline
    whose last step is a KMeans estimator with `.labels_` populated.
    """
    import pycaret.datasets
    from pycaret.core import CreateResult
    from pycaret.tasks import ClusteringExperiment
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    result: CreateResult = exp.create_model("kmeans", num_clusters=4, verbose=False)

    assert result.model_id == "kmeans"
    assert isinstance(result.pipeline, SkPipeline)
    name, step = result.pipeline.steps[-1]
    assert name == "kmeans"
    assert isinstance(step, KMeans)
    assert hasattr(step, "labels_")
    assert step.n_clusters == 4


@pytest.mark.slow
def test_clustering_assign_model_decorates_with_cluster_column():
    """assign_model returns a DataFrame with a 'Cluster' column."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    created = exp.create_model("kmeans", num_clusters=3, verbose=False)
    labelled = exp.assign_model(created.pipeline)
    assert "Cluster" in labelled.columns
    # Labels look like "Cluster 0", "Cluster 1", ...
    sample = labelled["Cluster"].iloc[0]
    assert sample.startswith("Cluster ")
    # Same row count as the original X.
    assert len(labelled) == len(exp.X)


@pytest.mark.slow
def test_clustering_create_model_does_not_call_legacy_create_model(monkeypatch):
    """Drain-lock: ClusteringExperiment.create_model must NOT call legacy."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-28 drain regression: ClusteringExperiment.create_model "
            "called self._legacy.create_model. Native path must "
            "fit the registry estimator directly."
        )

    monkeypatch.setattr(exp._legacy, "create_model", _poison)
    result = exp.create_model("kmeans", num_clusters=4, verbose=False)
    assert result.pipeline is not None


@pytest.mark.slow
def test_clustering_assign_model_does_not_call_legacy_assign_model(monkeypatch):
    """Drain-lock for assign_model on clustering."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    created = exp.create_model("kmeans", num_clusters=3, verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("Session-28 drain regression: assign_model called legacy.")

    monkeypatch.setattr(exp._legacy, "assign_model", _poison)
    labelled = exp.assign_model(created.pipeline)
    assert "Cluster" in labelled.columns


# ============================================================== anomaly


@pytest.mark.slow
def test_anomaly_create_model_returns_pipeline_with_iforest():
    """create_model('iforest') on an anomaly exp returns a fitted Pipeline."""
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("anomaly", verbose=False)
    exp = AnomalyExperiment(session_id=42, n_jobs=1).fit(df)
    result = exp.create_model("iforest", fraction=0.05, verbose=False)

    assert result.model_id == "iforest"
    assert isinstance(result.pipeline, SkPipeline)
    step = result.pipeline.steps[-1][1]
    # pyod's IForest exposes labels_ + decision_scores_ post fit.
    assert hasattr(step, "labels_")
    assert hasattr(step, "decision_scores_")


@pytest.mark.slow
def test_anomaly_assign_model_decorates_with_score():
    """assign_model returns a DataFrame with Anomaly + Anomaly_Score columns."""
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment

    df = pycaret.datasets.get_data("anomaly", verbose=False)
    exp = AnomalyExperiment(session_id=42, n_jobs=1).fit(df)
    created = exp.create_model("iforest", fraction=0.05, verbose=False)
    labelled = exp.assign_model(created.pipeline)
    assert "Anomaly" in labelled.columns
    assert "Anomaly_Score" in labelled.columns
    # Anomaly is 0/1.
    assert set(labelled["Anomaly"].unique()).issubset({0, 1})


@pytest.mark.slow
def test_anomaly_assign_model_score_false_skips_score_column():
    """score=False omits the Anomaly_Score column."""
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment

    df = pycaret.datasets.get_data("anomaly", verbose=False)
    exp = AnomalyExperiment(session_id=42, n_jobs=1).fit(df)
    created = exp.create_model("iforest", fraction=0.05, verbose=False)
    labelled = exp.assign_model(created.pipeline, score=False)
    assert "Anomaly" in labelled.columns
    assert "Anomaly_Score" not in labelled.columns


@pytest.mark.slow
def test_anomaly_create_model_does_not_call_legacy_create_model(monkeypatch):
    """Drain-lock for anomaly create_model."""
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment

    df = pycaret.datasets.get_data("anomaly", verbose=False)
    exp = AnomalyExperiment(session_id=42, n_jobs=1).fit(df)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-28 drain regression: AnomalyExperiment.create_model "
            "called self._legacy.create_model."
        )

    monkeypatch.setattr(exp._legacy, "create_model", _poison)
    result = exp.create_model("iforest", fraction=0.05, verbose=False)
    assert result.pipeline is not None


# ============================================================ misc


@pytest.mark.slow
def test_clustering_create_model_unknown_id_raises():
    """Unknown cluster algorithm ID raises ConfigurationError."""
    import pycaret.datasets
    from pycaret.core.errors import ConfigurationError
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    with pytest.raises(ConfigurationError, match="Unknown model id"):
        exp.create_model("zzzz_not_real", verbose=False)


@pytest.mark.slow
def test_clustering_create_model_predict_chain_for_kmeans():
    """KMeans supports predict; chain works on new data."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    created = exp.create_model("kmeans", num_clusters=3, verbose=False)
    # Pipeline should accept new data via predict_model.
    new_data = df.head(5)
    preds = exp.predict_model(created.pipeline, data=new_data)
    assert "Cluster" in preds.predictions.columns


def test_unsupervised_verbs_require_fit():
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import AnomalyExperiment, ClusteringExperiment

    cexp = ClusteringExperiment(session_id=0)
    aexp = AnomalyExperiment(session_id=0)
    with pytest.raises(NotFittedError):
        cexp.create_model("kmeans")
    with pytest.raises(NotFittedError):
        cexp.assign_model("dummy")
    with pytest.raises(NotFittedError):
        aexp.create_model("iforest")
    with pytest.raises(NotFittedError):
        aexp.assign_model("dummy")
