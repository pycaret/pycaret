"""Plotly-native clustering diagnostics.

Functions:

- ``elbow_curve`` — sum-of-squared-distances vs. k for KMeans-family
  models.
- ``silhouette_curve`` — silhouette score vs. k.
- ``silhouette_plot`` — per-sample silhouette for the chosen model.
- ``cluster_distribution`` — bar of cluster sizes.
- ``embedding_2d`` — t-SNE / UMAP / PCA scatter colored by cluster.

The screen-side composition: the "Clustering" tab in the Run-detail
view shows elbow + silhouette side-by-side for tuning k, then
embedding_2d for the chosen model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import plotly.graph_objects as go

from pycaret.plots._base import PALETTE, apply_layout, categorical_sequence

__all__ = [
    "cluster_distribution",
    "elbow_curve",
    "embedding_2d",
    "silhouette_curve",
    "silhouette_plot",
]


def _unwrap_for_clustering(estimator: Any) -> Any:
    """Pull the bare clusterer from a sktime/sklearn pipeline."""
    if hasattr(estimator, "steps"):
        return estimator.steps[-1][1]
    return estimator


def elbow_curve(
    estimator: Any,
    X: Any,
    *,
    k_range: range | list[int] | None = None,
    title: str | None = "Elbow curve",
) -> go.Figure:
    """Sum-of-squared-distances vs. k. The "elbow" is where adding clusters
    stops meaningfully reducing inertia — pick k there.

    Works on any clusterer that accepts ``n_clusters`` and exposes
    ``inertia_`` after fit (KMeans-family).
    """
    bare = _unwrap_for_clustering(estimator)
    if not hasattr(bare, "set_params"):
        raise ValueError("elbow_curve requires an estimator with set_params (e.g. KMeans).")
    if k_range is None:
        k_range = list(range(2, 11))

    inertias: list[float] = []
    ks: list[int] = []
    for k in k_range:
        try:
            est = deepcopy(bare).set_params(n_clusters=k)
            est.fit(X)
            if hasattr(est, "inertia_"):
                inertias.append(float(est.inertia_))
                ks.append(int(k))
        except Exception:  # noqa: BLE001
            continue
    if not inertias:
        raise ValueError(
            "Could not compute an elbow curve — no inertia_ values produced. "
            "Use silhouette_curve for non-KMeans algorithms."
        )

    fig = go.Figure(
        data=go.Scatter(
            x=ks,
            y=inertias,
            mode="lines+markers",
            line={"color": PALETTE["accent"], "width": 2},
            marker={"size": 9},
            hovertemplate="k = %{x}<br>Inertia: %{y:.2f}<extra></extra>",
        )
    )
    return apply_layout(
        fig,
        title=title,
        xaxis_title="Number of clusters (k)",
        yaxis_title="Inertia",
        showlegend=False,
    )


def silhouette_curve(
    estimator: Any,
    X: Any,
    *,
    k_range: range | list[int] | None = None,
    sample: int | None = 1000,
    random_state: int = 0,
    title: str | None = "Silhouette score vs. k",
) -> go.Figure:
    """Mean silhouette score vs. k. Higher is better; ranges -1..1."""
    from sklearn.metrics import silhouette_score

    bare = _unwrap_for_clustering(estimator)
    if not hasattr(bare, "set_params"):
        raise ValueError("silhouette_curve requires an estimator with set_params.")
    if k_range is None:
        k_range = list(range(2, 11))

    rng = np.random.default_rng(random_state)
    if sample is not None and hasattr(X, "shape") and X.shape[0] > sample:
        idx = rng.choice(X.shape[0], size=sample, replace=False)
        X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_sub = X

    scores: list[float] = []
    ks: list[int] = []
    for k in k_range:
        try:
            est = deepcopy(bare).set_params(n_clusters=k)
            labels = est.fit_predict(X_sub)
            if len(set(labels)) > 1:
                scores.append(float(silhouette_score(X_sub, labels)))
                ks.append(int(k))
        except Exception:  # noqa: BLE001
            continue

    if not scores:
        raise ValueError("silhouette_curve produced no valid scores — check the estimator + data.")

    fig = go.Figure(
        data=go.Scatter(
            x=ks,
            y=scores,
            mode="lines+markers",
            line={"color": PALETTE["accent"], "width": 2},
            marker={"size": 9},
            hovertemplate="k = %{x}<br>Silhouette: %{y:.3f}<extra></extra>",
        )
    )
    return apply_layout(
        fig,
        title=title,
        xaxis_title="Number of clusters (k)",
        yaxis_title="Silhouette score",
        showlegend=False,
    )


def silhouette_plot(
    estimator: Any,
    X: Any,
    *,
    sample: int | None = 1000,
    random_state: int = 0,
    title: str | None = "Per-cluster silhouette",
) -> go.Figure:
    """Per-sample silhouette grouped by cluster label.

    A good clustering shows similar widths across clusters and high
    average silhouette. Negative silhouettes indicate likely
    misclassification.
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    rng = np.random.default_rng(random_state)
    if sample is not None and hasattr(X, "shape") and X.shape[0] > sample:
        idx = rng.choice(X.shape[0], size=sample, replace=False)
        X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_sub = X

    bare = _unwrap_for_clustering(estimator)
    if hasattr(bare, "labels_") and len(getattr(bare, "labels_", [])) == len(X_sub):
        labels = np.asarray(bare.labels_)
    else:
        labels = np.asarray(bare.fit_predict(X_sub))

    if len(set(labels)) < 2:
        raise ValueError("silhouette_plot requires ≥ 2 clusters in the labels.")

    sample_scores = silhouette_samples(X_sub, labels)
    avg = float(silhouette_score(X_sub, labels))
    cluster_ids = sorted(set(labels.tolist()))
    colors = categorical_sequence(len(cluster_ids))

    fig = go.Figure()
    y_lower = 10
    for i, cluster_id in enumerate(cluster_ids):
        cluster_scores = np.sort(sample_scores[labels == cluster_id])
        size = len(cluster_scores)
        y_upper = y_lower + size
        fig.add_trace(
            go.Scatter(
                x=cluster_scores,
                y=list(range(y_lower, y_upper)),
                mode="lines",
                fill="tozerox",
                fillcolor=colors[i],
                line={"width": 0.5, "color": colors[i]},
                name=f"Cluster {cluster_id}",
                hovertemplate=f"Cluster {cluster_id}<br>Silhouette: %{{x:.3f}}<extra></extra>",
            )
        )
        y_lower = y_upper + 10

    fig.add_vline(
        x=avg,
        line={"color": PALETTE["danger"], "width": 1.5, "dash": "dash"},
        annotation_text=f"avg = {avg:.3f}",
        annotation_position="top",
    )

    return apply_layout(
        fig,
        title=title,
        xaxis_title="Silhouette coefficient",
        yaxis_title="Sample (grouped by cluster)",
    )


def cluster_distribution(
    estimator: Any,
    X: Any | None = None,
    *,
    title: str | None = "Cluster sizes",
) -> go.Figure:
    """Bar chart of how many samples fall into each cluster."""
    bare = _unwrap_for_clustering(estimator)
    if hasattr(bare, "labels_"):
        labels = np.asarray(bare.labels_)
    elif X is not None:
        labels = np.asarray(bare.fit_predict(X))
    else:
        raise ValueError("cluster_distribution requires either a fitted estimator or X.")

    cluster_ids, counts = np.unique(labels, return_counts=True)
    colors = categorical_sequence(len(cluster_ids))
    fig = go.Figure(
        data=go.Bar(
            x=[f"Cluster {cid}" for cid in cluster_ids],
            y=counts,
            marker={"color": colors},
            text=counts,
            textposition="outside",
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        )
    )
    return apply_layout(
        fig, title=title, xaxis_title="Cluster", yaxis_title="Count", showlegend=False
    )


def embedding_2d(
    estimator: Any,
    X: Any,
    *,
    method: str = "pca",
    sample: int | None = 1500,
    random_state: int = 0,
    title: str | None = None,
) -> go.Figure:
    """2-D embedding of X colored by the clusterer's labels.

    Methods:
    - ``pca`` — sklearn PCA (fast, deterministic, no extra dep).
    - ``tsne`` — sklearn TSNE (slower, captures non-linear structure).
    - ``umap`` — only if ``umap-learn`` is installed.
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(random_state)
    if sample is not None and hasattr(X, "shape") and X.shape[0] > sample:
        idx = rng.choice(X.shape[0], size=sample, replace=False)
        X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_sub = X

    bare = _unwrap_for_clustering(estimator)
    if hasattr(bare, "labels_") and len(getattr(bare, "labels_", [])) == len(X_sub):
        labels = np.asarray(bare.labels_)
    else:
        labels = np.asarray(bare.fit_predict(X_sub))

    method_lower = method.lower()
    if method_lower == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        emb = reducer.fit_transform(X_sub)
        method_label = "PCA"
    elif method_lower == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=random_state, init="pca", perplexity=30)
        emb = reducer.fit_transform(X_sub)
        method_label = "t-SNE"
    elif method_lower == "umap":
        try:
            import umap  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ImportError(
                "umap-learn is not installed. `pip install umap-learn` or use method='pca' / 'tsne'."
            ) from exc
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        emb = reducer.fit_transform(X_sub)
        method_label = "UMAP"
    else:
        raise ValueError(f"Unknown embedding method {method!r}; choose pca / tsne / umap.")

    cluster_ids = sorted(set(labels.tolist()))
    colors = categorical_sequence(len(cluster_ids))
    fig = go.Figure()
    for cid, color in zip(cluster_ids, colors, strict=True):
        mask = labels == cid
        fig.add_trace(
            go.Scattergl(
                x=emb[mask, 0],
                y=emb[mask, 1],
                mode="markers",
                name=f"Cluster {cid}",
                marker={"size": 6, "color": color, "opacity": 0.7},
                hovertemplate=f"Cluster {cid}<extra></extra>",
            )
        )

    return apply_layout(
        fig,
        title=title or f"{method_label} embedding (colored by cluster)",
        xaxis_title=f"{method_label} 1",
        yaxis_title=f"{method_label} 2",
    )
