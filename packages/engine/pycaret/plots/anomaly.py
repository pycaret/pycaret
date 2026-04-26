"""Plotly-native anomaly-detection diagnostics.

Functions:

- ``score_distribution`` — histogram of anomaly scores with the
  detection threshold marked.
- ``anomaly_map`` — 2-D scatter colored by anomaly flag (PCA / t-SNE /
  UMAP).
- ``feature_anomaly_scatter`` — pick two features, scatter raw data
  colored by anomaly status.
- ``score_vs_feature`` — anomaly score against a feature value (helps
  identify which features drive anomaly).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from pycaret.plots._base import PALETTE, apply_layout

__all__ = [
    "anomaly_map",
    "feature_anomaly_scatter",
    "score_distribution",
    "score_vs_feature",
]


def _unwrap(estimator: Any) -> Any:
    if hasattr(estimator, "steps"):
        return estimator.steps[-1][1]
    return estimator


def _scores_and_labels(estimator: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compute scores + labels for the rows of ``X``.

    Avoids reading ``.decision_scores_`` / ``.labels_`` because those are
    fixed-size to the training data and don't apply to sampled / new
    rows. Calls ``predict`` + ``decision_function`` (pyod) or
    ``predict`` + ``decision_function`` (sklearn) directly on ``X``.
    """
    bare = _unwrap(estimator)

    # pyod convention: bare.predict(X) returns 0/1 and bare.decision_function(X)
    # returns continuous scores (higher = more anomalous).
    if hasattr(bare, "predict") and hasattr(bare, "decision_function"):
        try:
            preds_raw = np.asarray(bare.predict(X))
            scores_raw = np.asarray(bare.decision_function(X), dtype=float)
        except Exception:  # noqa: BLE001
            preds_raw = scores_raw = None  # type: ignore[assignment]

        if preds_raw is not None and scores_raw is not None and len(preds_raw) == len(X):
            # Map sklearn's {-1, 1} → {1, 0}; pyod already uses {0, 1}.
            unique = set(preds_raw.tolist())
            if unique <= {-1, 1}:
                labels = np.where(preds_raw == -1, 1, 0)
                # sklearn decision_function: higher = more inlier; flip.
                scores = -scores_raw
            else:
                labels = preds_raw.astype(int)
                scores = scores_raw
            return scores, labels

    # Last-resort fallback: cached attrs (training-set only — won't match
    # sampled X but better than nothing for the score_distribution case).
    if hasattr(bare, "decision_scores_") and hasattr(bare, "labels_"):
        return np.asarray(bare.decision_scores_, dtype=float), np.asarray(bare.labels_, dtype=int)

    raise ValueError(
        f"{type(bare).__name__} doesn't expose anomaly scores. Expected "
        "either pyod-style (predict + decision_function) or sklearn-style "
        "(predict returning ±1 + decision_function)."
    )


def score_distribution(
    estimator: Any,
    X: Any,
    *,
    threshold: float | None = None,
    nbins: int = 60,
    title: str | None = "Anomaly score distribution",
) -> go.Figure:
    """Histogram of anomaly scores with the detection threshold marked."""
    scores, labels = _scores_and_labels(estimator, X)
    if threshold is None:
        # Best-effort: minimum score among flagged anomalies.
        flagged = scores[labels == 1]
        threshold = float(flagged.min()) if len(flagged) else float(np.percentile(scores, 95))

    fig = go.Figure(
        data=go.Histogram(
            x=scores,
            nbinsx=nbins,
            marker={"color": PALETTE["accent"]},
            opacity=0.85,
            hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold,
        line={"color": PALETTE["danger"], "width": 1.5, "dash": "dash"},
        annotation_text=f"threshold = {threshold:.3f}",
        annotation_position="top right",
    )
    return apply_layout(
        fig, title=title, xaxis_title="Anomaly score", yaxis_title="Count", showlegend=False
    )


def anomaly_map(
    estimator: Any,
    X: Any,
    *,
    method: str = "pca",
    sample: int | None = 2000,
    random_state: int = 0,
    title: str | None = None,
) -> go.Figure:
    """2-D embedding of X colored by anomaly flag (red = anomaly).

    Methods: ``pca`` (default), ``tsne``, ``umap`` (optional dep).
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(random_state)
    if sample is not None and hasattr(X, "shape") and X.shape[0] > sample:
        idx = rng.choice(X.shape[0], size=sample, replace=False)
        X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_sub = X

    scores, labels = _scores_and_labels(estimator, X_sub)

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

    inlier_mask = labels == 0
    anomaly_mask = labels == 1

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=emb[inlier_mask, 0],
            y=emb[inlier_mask, 1],
            mode="markers",
            name="Inlier",
            marker={"size": 5, "color": PALETTE["accent"], "opacity": 0.5},
            hovertemplate="Inlier<br>Score: %{customdata:.3f}<extra></extra>",
            customdata=scores[inlier_mask],
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=emb[anomaly_mask, 0],
            y=emb[anomaly_mask, 1],
            mode="markers",
            name="Anomaly",
            marker={
                "size": 8,
                "color": PALETTE["danger"],
                "opacity": 0.85,
                "line": {"color": "white", "width": 0.5},
            },
            hovertemplate="Anomaly<br>Score: %{customdata:.3f}<extra></extra>",
            customdata=scores[anomaly_mask],
        )
    )
    return apply_layout(
        fig,
        title=title or f"{method_label} map (anomalies highlighted)",
        xaxis_title=f"{method_label} 1",
        yaxis_title=f"{method_label} 2",
    )


def feature_anomaly_scatter(
    estimator: Any,
    X: Any,
    feature_x: str | int,
    feature_y: str | int,
    *,
    feature_names: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Scatter of two raw features colored by anomaly status.

    More interpretable than the embedding map when the user knows which
    features are likely to drive anomaly (e.g. transaction_amount vs.
    time_of_day).
    """
    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)

    def _idx(f: str | int) -> int:
        if isinstance(f, str):
            return (feature_names or list(range(X.shape[1]))).index(f)
        return int(f)

    ix = _idx(feature_x)
    iy = _idx(feature_y)
    name_x = (feature_names or [f"feature_{ix}"])[ix]
    name_y = (feature_names or [f"feature_{iy}"])[iy]

    if hasattr(X, "iloc"):
        x_vals = X.iloc[:, ix].to_numpy()
        y_vals = X.iloc[:, iy].to_numpy()
    else:
        x_vals = np.asarray(X)[:, ix]
        y_vals = np.asarray(X)[:, iy]

    scores, labels = _scores_and_labels(estimator, X)
    inlier_mask = labels == 0
    anomaly_mask = labels == 1

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_vals[inlier_mask],
            y=y_vals[inlier_mask],
            mode="markers",
            name="Inlier",
            marker={"size": 5, "color": PALETTE["accent"], "opacity": 0.5},
            hovertemplate=f"{name_x}: %{{x:.3f}}<br>{name_y}: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=x_vals[anomaly_mask],
            y=y_vals[anomaly_mask],
            mode="markers",
            name="Anomaly",
            marker={
                "size": 8,
                "color": PALETTE["danger"],
                "opacity": 0.85,
                "line": {"color": "white", "width": 0.5},
            },
            customdata=scores[anomaly_mask],
            hovertemplate=(
                f"{name_x}: %{{x:.3f}}<br>{name_y}: %{{y:.3f}}<br>Score: %{{customdata:.3f}}<extra></extra>"
            ),
        )
    )
    return apply_layout(
        fig,
        title=title or f"{name_x} vs. {name_y} (anomalies highlighted)",
        xaxis_title=name_x,
        yaxis_title=name_y,
    )


def score_vs_feature(
    estimator: Any,
    X: Any,
    feature: str | int,
    *,
    feature_names: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Anomaly score against a feature value. Anomaly cluster shape
    reveals which feature ranges drive detection.
    """
    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)
    if isinstance(feature, str):
        idx = (feature_names or list(range(X.shape[1]))).index(feature)
    else:
        idx = int(feature)
    name = (feature_names or [f"feature_{idx}"])[idx]

    if hasattr(X, "iloc"):
        feat_vals = X.iloc[:, idx].to_numpy()
    else:
        feat_vals = np.asarray(X)[:, idx]

    scores, labels = _scores_and_labels(estimator, X)
    inlier_mask = labels == 0
    anomaly_mask = labels == 1

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=feat_vals[inlier_mask],
            y=scores[inlier_mask],
            mode="markers",
            name="Inlier",
            marker={"size": 5, "color": PALETTE["accent"], "opacity": 0.5},
            hovertemplate=f"{name}: %{{x:.3f}}<br>Score: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=feat_vals[anomaly_mask],
            y=scores[anomaly_mask],
            mode="markers",
            name="Anomaly",
            marker={"size": 8, "color": PALETTE["danger"], "opacity": 0.85},
            hovertemplate=f"{name}: %{{x:.3f}}<br>Score: %{{y:.3f}}<extra></extra>",
        )
    )
    return apply_layout(
        fig,
        title=title or f"Anomaly score vs. {name}",
        xaxis_title=name,
        yaxis_title="Anomaly score",
    )
