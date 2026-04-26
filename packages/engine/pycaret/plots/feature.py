"""Plotly-native feature attribution plots — model-agnostic.

Built on sklearn primitives (``permutation_importance``,
``partial_dependence``) so they work with any fitted estimator without
requiring SHAP. SHAP-based plots will be added as an optional layer
(install via ``pip install pycaret[explain]``).

Functions:

- ``permutation_importance`` — drop-in replacement for the legacy
  feature-importance plot, but works on any model (not just trees).
- ``partial_dependence`` — 1-D / 2-D PDP showing how the model output
  changes with one or two features.
- ``ice_curve`` — individual conditional expectation; per-row PDP
  trajectories with the average overlaid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from pycaret.plots._base import PALETTE, apply_layout, categorical_sequence

__all__ = [
    "ice_curve",
    "partial_dependence",
    "permutation_importance",
]


def permutation_importance(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    feature_names: list[str] | None = None,
    n_repeats: int = 10,
    random_state: int | None = 0,
    scoring: str | None = None,
    top_n: int = 20,
    title: str | None = "Permutation feature importance",
) -> go.Figure:
    """Model-agnostic feature importance via sklearn's permutation_importance.

    Each feature is permuted ``n_repeats`` times; the drop in score is
    its importance. Returns a horizontal bar chart with error bars (±σ
    across repeats).

    Parameters
    ----------
    estimator : fitted model or Pipeline
    X, y : holdout data + true labels / values
    feature_names : column names in the order ``X`` columns appear
    n_repeats : permutation repeats per feature (default 10)
    scoring : sklearn scorer string; defaults to the estimator's task
    top_n : how many features to show
    """
    from sklearn.inspection import permutation_importance as _pi

    result = _pi(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1,
    )

    importances = np.asarray(result.importances_mean)
    stds = np.asarray(result.importances_std)
    n_features = len(importances)

    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    order = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in order]
    sorted_imp = importances[order]
    sorted_std = stds[order]

    fig = go.Figure(
        data=go.Bar(
            x=sorted_imp[::-1],
            y=sorted_names[::-1],
            orientation="h",
            marker={"color": PALETTE["accent"]},
            error_x={"type": "data", "array": sorted_std[::-1], "color": PALETTE["muted"]},
            hovertemplate="%{y}<br>Δscore: %{x:.4f} ±σ<extra></extra>",
        )
    )
    return apply_layout(
        fig,
        title=title,
        xaxis_title="Permutation importance (Δ score)",
        yaxis_title="Feature",
        showlegend=False,
        height=max(360, 24 * len(sorted_names) + 60),
    )


def partial_dependence(
    estimator: Any,
    X: Any,
    feature: str | int | tuple,
    *,
    feature_names: list[str] | None = None,
    grid_resolution: int = 50,
    kind: str = "average",
    title: str | None = None,
) -> go.Figure:
    """1-D or 2-D partial dependence plot.

    Parameters
    ----------
    estimator : fitted classifier / regressor / Pipeline
    X : data to compute the dependence over (typically X_train)
    feature : single feature (str / int) for 1-D PDP; tuple of two for 2-D.
    feature_names : column names; auto-derived from a DataFrame
    grid_resolution : how many grid points along each axis
    kind : ``"average"`` (PDP) | ``"individual"`` (ICE) | ``"both"``
    """
    from sklearn.inspection import partial_dependence as _pd

    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        else:
            feature_names = [
                f"feature_{i}" for i in range(X.shape[1] if hasattr(X, "shape") else 0)
            ]

    # Resolve string feature to index for sklearn (which prefers ints).
    def _to_idx(f: str | int) -> int:
        if isinstance(f, str):
            try:
                return feature_names.index(f)
            except ValueError as exc:
                raise ValueError(f"Feature {f!r} not in feature_names.") from exc
        return int(f)

    if isinstance(feature, tuple) and len(feature) == 2:
        idx_pair = (_to_idx(feature[0]), _to_idx(feature[1]))
        names = (feature_names[idx_pair[0]], feature_names[idx_pair[1]])
        result = _pd(estimator, X, [idx_pair], grid_resolution=grid_resolution, kind="average")
        # 2-D contour plot.
        # `result["average"]` shape: (1, len(grid0), len(grid1))
        z = np.asarray(result["average"][0])
        xv = np.asarray(result["grid_values"][0])
        yv = np.asarray(result["grid_values"][1])
        fig = go.Figure(
            data=go.Heatmap(
                z=z.T,
                x=xv,
                y=yv,
                colorscale=[[0, PALETTE["bg"]], [1, PALETTE["accent"]]],
                colorbar={"thickness": 12, "outlinewidth": 0, "title": "PDP"},
                hovertemplate=f"{names[0]}: %{{x}}<br>{names[1]}: %{{y}}<br>PDP: %{{z:.3f}}<extra></extra>",
            )
        )
        return apply_layout(
            fig,
            title=title or f"Partial dependence: {names[0]} × {names[1]}",
            xaxis_title=names[0],
            yaxis_title=names[1],
            showlegend=False,
        )

    # 1-D case.
    idx = _to_idx(feature if not isinstance(feature, tuple) else feature[0])
    name = feature_names[idx]
    result = _pd(estimator, X, [idx], grid_resolution=grid_resolution, kind=kind)
    grid = np.asarray(result["grid_values"][0])

    fig = go.Figure()
    if kind in ("individual", "both"):
        # ICE traces.
        ice = np.asarray(result["individual"][0])  # (n_rows, len(grid))
        for i in range(ice.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=ice[i],
                    mode="lines",
                    line={"color": PALETTE["muted"], "width": 1},
                    opacity=0.15,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    if kind in ("average", "both"):
        avg = np.asarray(result["average"][0])
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=avg,
                mode="lines",
                name="PDP (mean)",
                line={"color": PALETTE["accent"], "width": 2.5},
                hovertemplate=f"{name}: %{{x:.3f}}<br>PDP: %{{y:.3f}}<extra></extra>",
            )
        )

    return apply_layout(
        fig,
        title=title or f"Partial dependence: {name}",
        xaxis_title=name,
        yaxis_title="Partial dependence",
        showlegend=kind == "both",
    )


def ice_curve(
    estimator: Any,
    X: Any,
    feature: str | int,
    *,
    feature_names: list[str] | None = None,
    grid_resolution: int = 50,
    centered: bool = False,
    sample: int | None = 100,
    random_state: int = 0,
    title: str | None = None,
) -> go.Figure:
    """Individual Conditional Expectation: per-row PDP trajectories.

    More transparent than a simple PDP because it shows interactions —
    if ICE curves are non-parallel the model has interaction effects on
    that feature.

    Parameters
    ----------
    centered : if True, subtract each curve's first value (c-ICE).
    sample : take this many random rows; ``None`` for all.
    """
    from sklearn.inspection import partial_dependence as _pd

    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)

    rng = np.random.default_rng(random_state)
    if sample is not None and hasattr(X, "shape") and X.shape[0] > sample:
        sample_idx = rng.choice(X.shape[0], size=sample, replace=False)
        if hasattr(X, "iloc"):
            X_sub = X.iloc[sample_idx]
        else:
            X_sub = X[sample_idx]
    else:
        X_sub = X

    if isinstance(feature, str):
        idx = (feature_names or list(range(X.shape[1]))).index(feature)
    else:
        idx = int(feature)
    name = (feature_names or [f"feature_{idx}"])[idx]

    result = _pd(estimator, X_sub, [idx], grid_resolution=grid_resolution, kind="both")
    grid = np.asarray(result["grid_values"][0])
    ice = np.asarray(result["individual"][0])  # (n_rows, len(grid))
    avg = np.asarray(result["average"][0])

    if centered:
        ice = ice - ice[:, :1]
        avg = avg - avg[0]

    fig = go.Figure()
    for i in range(ice.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=ice[i],
                mode="lines",
                line={"color": PALETTE["accent"], "width": 0.8},
                opacity=0.18,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=avg,
            mode="lines",
            name="Mean",
            line={"color": PALETTE["danger"], "width": 2.5},
            hovertemplate=f"{name}: %{{x:.3f}}<br>Mean: %{{y:.3f}}<extra></extra>",
        )
    )

    yaxis_title = "Centered prediction" if centered else "Prediction"
    return apply_layout(
        fig,
        title=title or f"ICE: {name}{' (centered)' if centered else ''}",
        xaxis_title=name,
        yaxis_title=yaxis_title,
        showlegend=False,
    )


# ---- SHAP plots: optional, soft-dep --------------------------------------


def _require_shap():
    try:
        import shap  # noqa: F401

        return shap
    except ModuleNotFoundError as exc:
        raise ImportError(
            "shap is not installed. Install with `pip install shap` (or "
            "`pip install pycaret[explain]` once the extras are wired in 4.0.0a4)."
        ) from exc


def shap_summary(
    estimator: Any,
    X: Any,
    *,
    feature_names: list[str] | None = None,
    sample: int | None = 200,
    random_state: int = 0,
    title: str | None = "SHAP feature importance",
) -> go.Figure:
    """Mean |SHAP value| per feature — bar chart.

    Soft-dep on ``shap``. Falls back to a clear ImportError if shap
    isn't available; UI side should hide / disable the chart card in
    that case.
    """
    shap = _require_shap()

    bare = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    rng = np.random.default_rng(random_state)
    if sample is not None and hasattr(X, "shape") and X.shape[0] > sample:
        idx = rng.choice(X.shape[0], size=sample, replace=False)
        X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_sub = X

    explainer = shap.Explainer(bare, X_sub)
    sv = explainer(X_sub)
    values = np.asarray(sv.values)
    if values.ndim == 3:
        # Multiclass — take mean across classes.
        values = np.abs(values).mean(axis=(0, 2))
    else:
        values = np.abs(values).mean(axis=0)

    if feature_names is None and hasattr(X_sub, "columns"):
        feature_names = list(X_sub.columns)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(values))]

    order = np.argsort(values)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = values[order]

    fig = go.Figure(
        data=go.Bar(
            x=sorted_vals[::-1],
            y=sorted_names[::-1],
            orientation="h",
            marker={"color": PALETTE["accent"]},
            hovertemplate="%{y}<br>mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    return apply_layout(
        fig,
        title=title,
        xaxis_title="mean |SHAP value|",
        yaxis_title="Feature",
        showlegend=False,
        height=max(360, 24 * len(sorted_names) + 60),
    )


def shap_beeswarm(
    estimator: Any,
    X: Any,
    *,
    feature_names: list[str] | None = None,
    sample: int | None = 200,
    random_state: int = 0,
    top_n: int = 12,
    title: str | None = "SHAP beeswarm",
) -> go.Figure:
    """Per-row SHAP values colored by feature value, sorted by importance.

    Soft-dep on ``shap``.
    """
    shap = _require_shap()

    bare = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    rng = np.random.default_rng(random_state)
    if sample is not None and hasattr(X, "shape") and X.shape[0] > sample:
        idx = rng.choice(X.shape[0], size=sample, replace=False)
        X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_sub = X

    explainer = shap.Explainer(bare, X_sub)
    sv = explainer(X_sub)
    values = np.asarray(sv.values)
    data = np.asarray(sv.data)
    if values.ndim == 3:
        # Multiclass — collapse to class 1 for the visualization.
        values = values[..., -1]

    if feature_names is None and hasattr(X_sub, "columns"):
        feature_names = list(X_sub.columns)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(values.shape[1])]

    # Order features by mean |SHAP|.
    order = np.argsort(np.abs(values).mean(axis=0))[::-1][:top_n]
    fig = go.Figure()
    rng2 = np.random.default_rng(random_state)
    for plot_y, feat_idx in enumerate(order):
        # Jitter y so points don't stack vertically on a single line.
        jitter = rng2.normal(scale=0.12, size=values.shape[0])
        feat_vals = data[:, feat_idx]
        # Normalize feature values to [0, 1] for colorscale.
        if np.issubdtype(feat_vals.dtype, np.number):
            v = feat_vals.astype(float)
            lo, hi = np.nanmin(v), np.nanmax(v)
            color_vals = (v - lo) / (hi - lo + 1e-9)
        else:
            color_vals = np.zeros(len(feat_vals))
        fig.add_trace(
            go.Scatter(
                x=values[:, feat_idx],
                y=plot_y + jitter,
                mode="markers",
                marker={
                    "size": 5,
                    "color": color_vals,
                    "colorscale": [[0, PALETTE["accent"]], [1, PALETTE["danger"]]],
                    "showscale": plot_y == 0,
                    "colorbar": (
                        {"title": "Feature value", "thickness": 10, "outlinewidth": 0}
                        if plot_y == 0
                        else None
                    ),
                },
                opacity=0.7,
                hovertemplate=(f"{feature_names[feat_idx]}<br>SHAP: %{{x:.3f}}<extra></extra>"),
                showlegend=False,
            )
        )

    fig.add_vline(x=0, line={"color": PALETTE["muted"], "width": 1, "dash": "dash"})

    return apply_layout(
        fig,
        title=title,
        xaxis_title="SHAP value (impact on model output)",
        yaxis_title="",
        showlegend=False,
        height=max(360, 30 * len(order) + 80),
    ).update_yaxes(
        tickmode="array",
        tickvals=list(range(len(order))),
        ticktext=[feature_names[i] for i in order],
        autorange="reversed",
    )


__all__.extend(["shap_summary", "shap_beeswarm"])
# Reuse for hover; suppress unused warning.
_ = categorical_sequence
