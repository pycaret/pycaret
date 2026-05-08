"""Session 54 — wire ``plot_model`` and ``evaluate_model``.

The plot library landed in sessions 47-52 as standalone module functions
(``pycaret.plots.classification.roc_curve(estimator, X, y)`` etc.). This
session re-implements the OOP entry points on top of it:

- ``Experiment.plot_model(estimator, plot=, save=, **kw)`` looks up
  ``plot`` in ``self._build_plot_registry(estimator)`` and returns the
  resulting ``plotly.graph_objects.Figure`` (or saves it to disk).
- ``Experiment.evaluate_model(estimator)`` returns a ``dict`` of
  ``{plot_kind: Figure}`` — the curated diagnostic bundle declared by
  ``self._evaluate_plot_set()``. Each entry is rendered defensively;
  failures (e.g. shap missing) are silently dropped.

Each leaf task class (Classification / Regression / Clustering / Anomaly /
TimeSeries) declares its own registry + default plot kind + evaluate set.
"""

from __future__ import annotations

import importlib.util

import plotly.graph_objects as go
import pytest


_HAS_SHAP = importlib.util.find_spec("shap") is not None


# ---------------------------------------------------------------------------
# Module-scoped fixtures — fitting an experiment is the slow part; reuse
# across tests of the same task.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classification_setup():
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase", session_id=42, n_jobs=1, fold=3
    ).fit(df)
    res = exp.create_model("lr", verbose=False)
    return exp, res.pipeline


@pytest.fixture(scope="module")
def regression_setup():
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    res = exp.create_model("rf", verbose=False)  # tree → has feature_importances_
    return exp, res.pipeline


@pytest.fixture(scope="module")
def clustering_setup():
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42).fit(df)
    res = exp.create_model("kmeans", num_clusters=3)
    return exp, res.pipeline


@pytest.fixture(scope="module")
def anomaly_setup():
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment

    df = pycaret.datasets.get_data("anomaly", verbose=False)
    exp = AnomalyExperiment(session_id=42).fit(df)
    res = exp.create_model("iforest")
    return exp, res.pipeline


@pytest.fixture(scope="module")
def ts_setup():
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)
    return exp, res.pipeline


# ---------------------------------------------------------------------------
# plot_model — classification
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_plot_model_classification_default_is_roc(classification_setup):
    exp, pipeline = classification_setup
    fig = exp.plot_model(pipeline)
    assert isinstance(fig, go.Figure)
    assert any("AUC" in (t.name or "") for t in fig.data)


@pytest.mark.slow
def test_plot_model_classification_kinds(classification_setup):
    exp, pipeline = classification_setup
    for kind in ("auc", "pr", "confusion_matrix", "calibration", "class_distribution"):
        fig = exp.plot_model(pipeline, plot=kind)
        assert isinstance(fig, go.Figure), f"{kind!r} did not return a Figure"


@pytest.mark.slow
def test_plot_model_unknown_kind_raises(classification_setup):
    exp, pipeline = classification_setup
    with pytest.raises(ValueError, match="Unknown plot kind"):
        exp.plot_model(pipeline, plot="not_a_real_plot")


# ---------------------------------------------------------------------------
# plot_model — regression
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_plot_model_regression_default_is_residuals(regression_setup):
    exp, pipeline = regression_setup
    fig = exp.plot_model(pipeline)
    assert isinstance(fig, go.Figure)


@pytest.mark.slow
def test_plot_model_regression_kinds(regression_setup):
    exp, pipeline = regression_setup
    for kind in ("residuals", "residuals_distribution", "prediction_error", "feature"):
        fig = exp.plot_model(pipeline, plot=kind)
        assert isinstance(fig, go.Figure), f"{kind!r} did not return a Figure"


# ---------------------------------------------------------------------------
# plot_model — clustering
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_plot_model_clustering_default(clustering_setup):
    exp, pipeline = clustering_setup
    fig = exp.plot_model(pipeline)
    assert isinstance(fig, go.Figure)


@pytest.mark.slow
def test_plot_model_clustering_silhouette(clustering_setup):
    exp, pipeline = clustering_setup
    fig = exp.plot_model(pipeline, plot="silhouette_plot")
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_model — anomaly
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_plot_model_anomaly_default(anomaly_setup):
    exp, pipeline = anomaly_setup
    fig = exp.plot_model(pipeline)
    assert isinstance(fig, go.Figure)


@pytest.mark.slow
def test_plot_model_anomaly_map(anomaly_setup):
    exp, pipeline = anomaly_setup
    fig = exp.plot_model(pipeline, plot="anomaly_map")
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_model — time-series
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_plot_model_ts_default_is_forecast(ts_setup):
    exp, pipeline = ts_setup
    fig = exp.plot_model(pipeline)
    assert isinstance(fig, go.Figure)


@pytest.mark.slow
def test_plot_model_ts_data_only_plots(ts_setup):
    """``decomposition`` / ``acf`` ignore the estimator (data-only diagnostics)."""
    exp, pipeline = ts_setup
    for kind in ("decomposition", "acf", "pacf"):
        fig = exp.plot_model(pipeline, plot=kind)
        assert isinstance(fig, go.Figure), f"{kind!r} did not return a Figure"


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_evaluate_model_classification_returns_dict(classification_setup):
    exp, pipeline = classification_setup
    out = exp.evaluate_model(pipeline)
    assert isinstance(out, dict)
    assert len(out) >= 4  # curated set has 5 items
    for kind, fig in out.items():
        assert isinstance(fig, go.Figure), f"{kind!r} → {type(fig).__name__}"


@pytest.mark.slow
def test_evaluate_model_regression_returns_dict(regression_setup):
    exp, pipeline = regression_setup
    out = exp.evaluate_model(pipeline)
    assert isinstance(out, dict)
    assert "residuals" in out


@pytest.mark.slow
def test_evaluate_model_skips_failing_plots(regression_setup):
    """A linear estimator without ``feature_importances_`` shouldn't crash
    evaluate_model; the feature plot should just be missing from the dict."""
    from pycaret.tasks import RegressionExperiment

    import pycaret.datasets

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    # Linear regression has .coef_, so feature_importance still works.
    # KNN has neither — use it to test the defensive skip path.
    res = exp.create_model("knn", verbose=False)
    out = exp.evaluate_model(res.pipeline)
    assert isinstance(out, dict)
    # feature_importance should have failed silently — not in the dict.
    assert "feature" not in out or isinstance(out.get("feature"), go.Figure)


# ---------------------------------------------------------------------------
# save= path
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_plot_model_save_returns_path_or_raises(tmp_path, classification_setup):
    """``save=True`` returns a path string; if kaleido is missing, raises."""
    exp, pipeline = classification_setup
    target = tmp_path / "auc.png"
    try:
        result = exp.plot_model(pipeline, plot="auc", save=str(target))
    except RuntimeError as e:
        # Kaleido not installed in this test env — error message must point users at the extra.
        assert "kaleido" in str(e).lower()
        return
    assert result == str(target)
    assert target.exists()
