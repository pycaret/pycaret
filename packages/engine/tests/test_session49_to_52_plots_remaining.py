"""Sessions 49-52 — Track A · clustering, anomaly, time-series, EDA plots.

Batched into a single file because each module is shaped identically:
take a fitted estimator (or raw data, for EDA) and return a Plotly
``Figure``. The dashboard composition is the same across tasks — a
tabbed model-card view that swaps in the per-task module.

Coverage:

- **Clustering (s49)**: elbow_curve, silhouette_curve, silhouette_plot,
  cluster_distribution, embedding_2d (PCA / t-SNE / UMAP).
- **Anomaly (s50)**: score_distribution, anomaly_map (PCA / t-SNE /
  UMAP), feature_anomaly_scatter, score_vs_feature.
- **Time-series (s51)**: forecast (with optional CI band), decomposition,
  acf, pacf, residual_diagnostics, cv_split_visualizer.
- **EDA (s52)**: column_distribution (numeric → histogram, categorical →
  count bar), correlation_heatmap, missingness_map, target_vs_feature
  (4-way joint distribution dispatch), profile_summary (table figure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clustering_setup():
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    res = exp.create_model("kmeans", verbose=False, num_clusters=3)
    return exp, res.pipeline, df


@pytest.fixture(scope="module")
def anomaly_setup():
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment

    df = pycaret.datasets.get_data("anomaly", verbose=False)
    exp = AnomalyExperiment(session_id=42, n_jobs=1).fit(df)
    res = exp.create_model("iforest", verbose=False)
    return exp, res.pipeline, df


@pytest.fixture(scope="module")
def ts_setup():
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    return exp


@pytest.fixture(scope="module")
def eda_dataframe():
    import pycaret.datasets

    return pycaret.datasets.get_data("juice", verbose=False)


# ---------------------------------------------------------------------------
# Session 49 — clustering plots.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_elbow_curve(clustering_setup):
    from pycaret.plots import clustering as cplots

    exp, pipeline, _df = clustering_setup
    fig = cplots.elbow_curve(pipeline, exp._fit_state["X_transformed"], k_range=range(2, 6))
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) > 0


@pytest.mark.slow
def test_silhouette_curve(clustering_setup):
    from pycaret.plots import clustering as cplots

    exp, pipeline, _df = clustering_setup
    fig = cplots.silhouette_curve(pipeline, exp._fit_state["X_transformed"], k_range=range(2, 5))
    assert isinstance(fig, go.Figure)


@pytest.mark.slow
def test_silhouette_plot(clustering_setup):
    from pycaret.plots import clustering as cplots

    exp, pipeline, _df = clustering_setup
    fig = cplots.silhouette_plot(pipeline, exp._fit_state["X_transformed"], sample=200)
    assert isinstance(fig, go.Figure)
    # One trace per cluster.
    assert len(fig.data) >= 2


@pytest.mark.slow
def test_cluster_distribution(clustering_setup):
    from pycaret.plots import clustering as cplots

    _exp, pipeline, _df = clustering_setup
    fig = cplots.cluster_distribution(pipeline)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"


@pytest.mark.slow
def test_embedding_2d_pca(clustering_setup):
    from pycaret.plots import clustering as cplots

    exp, pipeline, _df = clustering_setup
    fig = cplots.embedding_2d(pipeline, exp._fit_state["X_transformed"], method="pca", sample=200)
    assert isinstance(fig, go.Figure)
    # One trace per cluster.
    assert len(fig.data) >= 2


@pytest.mark.slow
def test_embedding_2d_unknown_method_raises(clustering_setup):
    from pycaret.plots import clustering as cplots

    exp, pipeline, _df = clustering_setup
    with pytest.raises(ValueError, match="Unknown embedding method"):
        cplots.embedding_2d(pipeline, exp._fit_state["X_transformed"], method="bogus")


# ---------------------------------------------------------------------------
# Session 50 — anomaly plots.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_score_distribution(anomaly_setup):
    from pycaret.plots import anomaly as aplots

    exp, pipeline, _df = anomaly_setup
    fig = aplots.score_distribution(pipeline, exp._fit_state["X_transformed"])
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "histogram"


@pytest.mark.slow
def test_anomaly_map(anomaly_setup):
    from pycaret.plots import anomaly as aplots

    exp, pipeline, _df = anomaly_setup
    fig = aplots.anomaly_map(pipeline, exp._fit_state["X_transformed"], method="pca", sample=300)
    assert isinstance(fig, go.Figure)
    # Inlier + anomaly traces.
    assert len(fig.data) == 2


@pytest.mark.slow
def test_feature_anomaly_scatter(anomaly_setup):
    from pycaret.plots import anomaly as aplots

    exp, pipeline, _df = anomaly_setup
    cols = list(exp._fit_state["X_transformed"].columns)[:2]
    fig = aplots.feature_anomaly_scatter(
        pipeline, exp._fit_state["X_transformed"], cols[0], cols[1]
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


@pytest.mark.slow
def test_score_vs_feature(anomaly_setup):
    from pycaret.plots import anomaly as aplots

    exp, pipeline, _df = anomaly_setup
    feat = list(exp._fit_state["X_transformed"].columns)[0]
    fig = aplots.score_vs_feature(pipeline, exp._fit_state["X_transformed"], feat)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Session 51 — time-series plots.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ts_forecast_basic(ts_setup):
    from pycaret.plots import time_series as tsplots

    exp = ts_setup
    res = exp.create_model("naive", verbose=False)
    preds = exp.predict_model(res.pipeline)
    fig = tsplots.forecast(
        y_true=exp.y_test,
        y_pred=preds.predictions["y_pred"],
        history=exp.y_train,
    )
    assert isinstance(fig, go.Figure)
    # History + forecast + actual = 3 traces.
    assert len(fig.data) == 3


@pytest.mark.slow
def test_ts_forecast_with_intervals(ts_setup):
    from pycaret.plots import time_series as tsplots

    exp = ts_setup
    res = exp.create_model("arima", verbose=False)
    preds = exp.predict_model(res.pipeline, return_pred_int=True)
    df = preds.predictions
    fig = tsplots.forecast(
        y_true=exp.y_test,
        y_pred=df["y_pred"],
        lower=df["lower"],
        upper=df["upper"],
    )
    assert isinstance(fig, go.Figure)
    # Without history: upper, lower (CI), actual, forecast = 4 traces.
    assert len(fig.data) == 4


@pytest.mark.slow
def test_ts_decomposition(ts_setup):
    from pycaret.plots import time_series as tsplots

    exp = ts_setup
    fig = tsplots.decomposition(exp.y_train, period=12)
    assert isinstance(fig, go.Figure)
    # 4 subplots → 4 scatter traces.
    assert len(fig.data) == 4


@pytest.mark.slow
def test_ts_acf_and_pacf(ts_setup):
    from pycaret.plots import time_series as tsplots

    exp = ts_setup
    a = tsplots.acf(exp.y_train, nlags=20)
    p = tsplots.pacf(exp.y_train, nlags=20)
    assert isinstance(a, go.Figure)
    assert isinstance(p, go.Figure)


@pytest.mark.slow
def test_ts_residual_diagnostics(ts_setup):
    from pycaret.plots import time_series as tsplots

    exp = ts_setup
    res = exp.create_model("naive", verbose=False)
    preds = exp.predict_model(res.pipeline)
    fig = tsplots.residual_diagnostics(
        y_true=exp.y_test, y_pred=preds.predictions["y_pred"], nlags=12
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.slow
def test_ts_cv_split_visualizer(ts_setup):
    from pycaret.plots import time_series as tsplots

    exp = ts_setup
    fold_gen = exp._fit_state["fold_generator"]
    fig = tsplots.cv_split_visualizer(fold_gen, exp.y_train)
    assert isinstance(fig, go.Figure)
    # Train + test traces per fold.
    assert len(fig.data) >= 2


# ---------------------------------------------------------------------------
# Session 52 — EDA plots.
# ---------------------------------------------------------------------------


def test_eda_column_distribution_numeric(eda_dataframe):
    from pycaret.plots import eda

    df = eda_dataframe
    numeric_col = df.select_dtypes(include="number").columns[0]
    fig = eda.column_distribution(df, numeric_col)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "histogram"


def test_eda_column_distribution_categorical(eda_dataframe):
    from pycaret.plots import eda

    df = eda_dataframe
    cat_col = "Purchase"
    fig = eda.column_distribution(df, cat_col)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"


def test_eda_correlation_heatmap(eda_dataframe):
    from pycaret.plots import eda

    fig = eda.correlation_heatmap(eda_dataframe, annotate=False)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "heatmap"


def test_eda_missingness_map_no_missing(eda_dataframe):
    from pycaret.plots import eda

    fig = eda.missingness_map(eda_dataframe)
    assert isinstance(fig, go.Figure)
    # Title gets a "no missing values" suffix when none.
    assert "no missing" in fig.layout.title.text


def test_eda_missingness_map_with_missing():
    """Synthetic frame with missing values gives a populated bar chart."""
    from pycaret.plots import eda

    df = pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4, 5],
            "b": [np.nan, np.nan, 3, 4, 5],
            "c": [1, 2, 3, 4, 5],
        }
    )
    fig = eda.missingness_map(df)
    # First trace's text values reflect non-zero missing %.
    text = list(fig.data[0].text)
    # Should see "20.0%" and "40.0%" for cols a + b.
    assert any("20.0" in t for t in text)
    assert any("40.0" in t for t in text)


def test_eda_target_vs_feature_dispatch(eda_dataframe):
    from pycaret.plots import eda

    df = eda_dataframe
    # numeric × categorical.
    fig = eda.target_vs_feature(df, "PriceCH", "Purchase")
    assert isinstance(fig, go.Figure)
    # categorical × numeric.
    fig2 = eda.target_vs_feature(df, "Purchase", "PriceCH")
    assert isinstance(fig2, go.Figure)


def test_eda_profile_summary(eda_dataframe):
    from pycaret.plots import eda

    fig = eda.profile_summary(eda_dataframe)
    assert isinstance(fig, go.Figure)
    # Plotly Table.
    assert fig.data[0].type == "table"


# ---------------------------------------------------------------------------
# Module surface.
# ---------------------------------------------------------------------------


def test_all_modules_exposed():
    from pycaret.plots import (
        anomaly,
        classification,
        clustering,
        eda,
        feature,
        regression,
        time_series,
    )

    for mod in (classification, regression, feature, clustering, anomaly, time_series, eda):
        assert hasattr(mod, "__all__")
        assert len(mod.__all__) > 0
