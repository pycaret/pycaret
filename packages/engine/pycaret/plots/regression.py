"""Plotly-native regression diagnostics.

Each function takes a fitted regressor (or ``Pipeline``) plus a holdout
set and returns a ``plotly.graph_objects.Figure``.

Usage::

    from pycaret.plots import regression as plots
    fig = plots.residuals(pipeline, X_test, y_test)
    fig.to_dict()  # API JSON payload
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from pycaret.plots._base import PALETTE, apply_layout, unwrap_estimator

__all__ = [
    "feature_importance",
    "learning_curve",
    "prediction_error",
    "residuals",
    "residuals_distribution",
]


def residuals(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    title: str | None = "Residuals vs. predicted",
) -> go.Figure:
    """Scatter of residuals against predicted values.

    A well-fit model shows a horizontal cloud around zero with no pattern.
    Patterns indicate heteroscedasticity / missed structure.
    """
    y_true = np.asarray(y, dtype=float)
    y_pred = np.asarray(estimator.predict(X), dtype=float)
    resid = y_true - y_pred

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=resid,
            mode="markers",
            name="Residuals",
            marker={"color": PALETTE["accent"], "size": 6, "opacity": 0.6},
            hovertemplate="Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
        )
    )
    # Zero reference line.
    fig.add_hline(y=0, line={"color": PALETTE["muted"], "width": 1, "dash": "dash"})

    return apply_layout(
        fig,
        title=title,
        xaxis_title="Predicted value",
        yaxis_title="Residual (y - ŷ)",
        showlegend=False,
    )


def residuals_distribution(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    nbins: int = 30,
    title: str | None = "Residuals distribution",
) -> go.Figure:
    """Histogram + KDE of residuals; ideal is roughly normal centered at 0."""
    y_true = np.asarray(y, dtype=float)
    y_pred = np.asarray(estimator.predict(X), dtype=float)
    resid = y_true - y_pred

    fig = go.Figure(
        data=go.Histogram(
            x=resid,
            nbinsx=nbins,
            marker={
                "color": PALETTE["accent"],
                "line": {"color": PALETTE["accent_alt"], "width": 1},
            },
            opacity=0.85,
            name="Residuals",
            hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    # Mean indicator.
    mean = float(np.mean(resid))
    fig.add_vline(
        x=mean,
        line={"color": PALETTE["danger"], "width": 1.5, "dash": "dash"},
        annotation_text=f"mean = {mean:.3f}",
        annotation_position="top right",
    )

    return apply_layout(
        fig, title=title, xaxis_title="Residual", yaxis_title="Count", showlegend=False
    )


def prediction_error(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    title: str | None = "Predicted vs. actual",
) -> go.Figure:
    """Predicted vs. actual scatter with the y=x reference line.

    The closer points cluster on the diagonal, the better the fit.
    R² is annotated when computable.
    """
    from sklearn.metrics import r2_score

    y_true = np.asarray(y, dtype=float)
    y_pred = np.asarray(estimator.predict(X), dtype=float)

    try:
        r2 = r2_score(y_true, y_pred)
        title_full = f"{title}  (R² = {r2:.3f})" if title else f"R² = {r2:.3f}"
    except Exception:  # noqa: BLE001
        title_full = title

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker={"color": PALETTE["accent"], "size": 6, "opacity": 0.6},
            hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>",
        )
    )

    # Reference y=x line.
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Perfect prediction",
            line={"color": PALETTE["muted"], "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )
    )

    return apply_layout(
        fig,
        title=title_full,
        xaxis_title="Actual",
        yaxis_title="Predicted",
        showlegend=False,
    )


def learning_curve(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    cv: int = 5,
    train_sizes: np.ndarray | list | None = None,
    scoring: str = "r2",
    title: str | None = "Learning curve",
) -> go.Figure:
    """Train + validation score as a function of training set size.

    Uses sklearn's ``learning_curve`` helper. The estimator is cloned per
    train-size so the input pipeline isn't mutated.
    """
    from sklearn.model_selection import learning_curve as _lc

    if train_sizes is None:
        train_sizes_arr = np.linspace(0.1, 1.0, 8)
    else:
        train_sizes_arr = np.asarray(train_sizes)

    sizes, train_scores, val_scores = _lc(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes_arr,
        scoring=scoring,
        n_jobs=1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig = go.Figure()
    # Training score band.
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=train_mean + train_std,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=train_mean - train_std,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(91,141,239,0.15)",
            line={"width": 0},
            name="Training ±σ",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=train_mean,
            mode="lines+markers",
            name="Training",
            line={"color": PALETTE["accent"], "width": 2},
            marker={"size": 8},
        )
    )
    # Validation score band.
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=val_mean + val_std,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=val_mean - val_std,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,138,101,0.15)",
            line={"width": 0},
            name="Validation ±σ",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=val_mean,
            mode="lines+markers",
            name="Validation",
            line={"color": PALETTE["series_2"], "width": 2},
            marker={"size": 8},
        )
    )

    return apply_layout(
        fig,
        title=title,
        xaxis_title="Training set size",
        yaxis_title=scoring,
    )


def feature_importance(
    estimator: Any,
    feature_names: list[str] | None = None,
    *,
    top_n: int = 20,
    title: str | None = "Feature importance",
) -> go.Figure:
    """Horizontal bar chart of feature importances.

    Reads ``.feature_importances_`` (tree models) or ``.coef_`` (linear
    models). Pipelines are unwrapped to the final estimator.
    """
    bare = unwrap_estimator(estimator)

    importances = None
    source = "importance"
    if hasattr(bare, "feature_importances_"):
        importances = np.asarray(bare.feature_importances_, dtype=float)
    elif hasattr(bare, "coef_"):
        coef = np.asarray(bare.coef_, dtype=float)
        if coef.ndim == 2:
            # Multiclass: sum |coef| across classes.
            importances = np.abs(coef).sum(axis=0)
        else:
            importances = np.abs(coef)
        source = "|coef|"
    if importances is None:
        raise ValueError(
            f"{type(bare).__name__} exposes neither feature_importances_ nor coef_; "
            "use the SHAP-based plots in pycaret.plots.feature instead."
        )

    n_features = len(importances)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    order = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    fig = go.Figure(
        data=go.Bar(
            x=sorted_importances[::-1],
            y=sorted_names[::-1],
            orientation="h",
            marker={"color": PALETTE["accent"]},
            hovertemplate="%{y}<br>" + source.title() + ": %{x:.4f}<extra></extra>",
        )
    )
    return apply_layout(
        fig,
        title=title,
        xaxis_title=source,
        yaxis_title="Feature",
        showlegend=False,
        height=max(360, 24 * len(sorted_names) + 60),
    )
