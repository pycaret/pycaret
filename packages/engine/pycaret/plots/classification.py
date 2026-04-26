"""Plotly-native classification diagnostics.

Each function takes a fitted classifier (or ``Pipeline``) plus a holdout
set and returns a ``plotly.graph_objects.Figure``. No file I/O, no
matplotlib — figures travel as JSON to the React dashboard.

Usage::

    from pycaret.plots import classification as plots

    fig = plots.confusion_matrix(pipeline, X_test, y_test)
    fig.show()           # notebook
    fig.to_dict()        # API JSON payload
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from pycaret.plots._base import PALETTE, apply_layout, categorical_sequence

__all__ = [
    "calibration_curve",
    "class_distribution",
    "confusion_matrix",
    "gain_curve",
    "lift_curve",
    "pr_curve",
    "roc_curve",
    "threshold_curve",
]


def _coerce_labels(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make sure ``y_true`` and ``y_pred`` live in the same label space.

    PyCaret 4.0 wraps a ``LabelEncoder`` outside the model Pipeline, so
    ``pipeline.predict(X)`` returns integer-encoded class indices while
    ``y_test`` from the experiment retains its original labels (e.g.
    "CH" / "MM"). This helper detects the dtype mismatch and best-effort
    encodes ``y_true`` into the same integer space ``y_pred`` uses.
    """
    if y_true.dtype == y_pred.dtype:
        return y_true, y_pred
    # Try to align: build a fresh LabelEncoder from the union of values.
    if y_pred.dtype.kind in ("i", "u") and y_true.dtype.kind not in ("i", "u"):
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        # Fit on sorted unique to make 0 → first class, 1 → second, …
        # (matches how the supervised native setup encodes the target).
        try:
            le.fit(sorted(set(y_true.tolist())))
            return le.transform(y_true), y_pred
        except Exception:  # noqa: BLE001
            pass
    return y_true, y_pred


def _predict_proba(estimator: Any, X: Any) -> np.ndarray | None:
    """Best-effort probability extraction. Returns None if estimator is hard.

    Order of preference: ``predict_proba`` → ``decision_function`` → None.
    For decision_function, applies a sigmoid to map back to [0, 1].
    """
    if hasattr(estimator, "predict_proba"):
        try:
            return np.asarray(estimator.predict_proba(X))
        except Exception:  # noqa: BLE001
            pass
    if hasattr(estimator, "decision_function"):
        try:
            scores = np.asarray(estimator.decision_function(X))
            # Two-class case: return [1-p, p] so callers can index uniformly.
            if scores.ndim == 1:
                proba_pos = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - proba_pos, proba_pos])
            # Multiclass: softmax.
            scores = scores - scores.max(axis=1, keepdims=True)
            exp = np.exp(scores)
            return exp / exp.sum(axis=1, keepdims=True)
        except Exception:  # noqa: BLE001
            pass
    return None


def confusion_matrix(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    labels: list | None = None,
    normalize: str | None = None,
    title: str | None = "Confusion matrix",
) -> go.Figure:
    """Per-class confusion matrix as a heatmap.

    Parameters
    ----------
    estimator : fitted classifier or Pipeline
    X, y : holdout features + true labels
    labels : optional ordered class labels; otherwise derived from y
    normalize : ``None`` | ``"true"`` (per-row) | ``"pred"`` (per-col) | ``"all"``
    """
    from sklearn.metrics import confusion_matrix as _cm

    y_true = np.asarray(y)
    y_pred = np.asarray(estimator.predict(X))
    y_true, y_pred = _coerce_labels(y_true, y_pred)

    if labels is None:
        labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    cm = _cm(y_true, y_pred, labels=labels, normalize=normalize)

    text = [[f"{v:.2f}" if normalize else f"{int(v)}" for v in row] for row in cm]
    label_strs = [str(label) for label in labels]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=label_strs,
            y=label_strs,
            colorscale=[[0, PALETTE["bg"]], [1, PALETTE["accent"]]],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13, "color": PALETTE["ink"]},
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
            showscale=True,
            colorbar={"thickness": 12, "outlinewidth": 0},
        )
    )
    return apply_layout(
        fig,
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        showlegend=False,
    )


def roc_curve(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    title: str | None = "ROC curve",
) -> go.Figure:
    """ROC curve(s), one per class for multiclass."""
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve as _roc
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y)
    proba = _predict_proba(estimator, X)
    if proba is None:
        raise ValueError("ROC requires an estimator with `predict_proba` or `decision_function`.")
    # Align y_true with the space estimator.classes_ uses.
    if hasattr(estimator, "classes_"):
        est_classes = np.asarray(estimator.classes_)
        if est_classes.dtype != y_true.dtype:
            y_true, _ = _coerce_labels(y_true, est_classes)

    classes = sorted(set(np.unique(y_true).tolist()))
    fig = go.Figure()

    if len(classes) == 2:
        # Binary: one curve, positive class is the second column.
        fpr, tpr, _ = _roc(y_true, proba[:, 1], pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"AUC = {roc_auc:.3f}",
                line={"color": PALETTE["accent"], "width": 2},
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            )
        )
    else:
        # Multiclass: one-vs-rest.
        y_bin = label_binarize(y_true, classes=classes)
        colors = categorical_sequence(len(classes))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = _roc(y_bin[:, i], proba[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{cls} (AUC = {roc_auc:.3f})",
                    line={"color": colors[i], "width": 2},
                    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
                )
            )

    # Random baseline.
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": PALETTE["muted"], "width": 1, "dash": "dash"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    return apply_layout(
        fig,
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )


def pr_curve(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    title: str | None = "Precision-Recall curve",
) -> go.Figure:
    """Precision-recall curve(s), one per class for multiclass."""
    from sklearn.metrics import auc, precision_recall_curve
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y)
    proba = _predict_proba(estimator, X)
    if proba is None:
        raise ValueError("PR curve requires probability scores.")
    if hasattr(estimator, "classes_"):
        est_classes = np.asarray(estimator.classes_)
        if est_classes.dtype != y_true.dtype:
            y_true, _ = _coerce_labels(y_true, est_classes)

    classes = sorted(set(np.unique(y_true).tolist()))
    fig = go.Figure()

    if len(classes) == 2:
        precision, recall, _ = precision_recall_curve(y_true, proba[:, 1], pos_label=classes[1])
        ap = auc(recall, precision)
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"AP = {ap:.3f}",
                line={"color": PALETTE["accent"], "width": 2},
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
            )
        )
    else:
        y_bin = label_binarize(y_true, classes=classes)
        colors = categorical_sequence(len(classes))
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
            ap = auc(recall, precision)
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"{cls} (AP = {ap:.3f})",
                    line={"color": colors[i], "width": 2},
                    hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
                )
            )

    return apply_layout(fig, title=title, xaxis_title="Recall", yaxis_title="Precision")


def calibration_curve(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    n_bins: int = 10,
    title: str | None = "Calibration curve",
) -> go.Figure:
    """Reliability diagram: predicted probability vs observed frequency.

    Binary classification only. For multiclass, run one-vs-rest yourself.
    """
    from sklearn.calibration import calibration_curve as _cal

    y_true = np.asarray(y)
    if hasattr(estimator, "classes_"):
        est_classes = np.asarray(estimator.classes_)
        if est_classes.dtype != y_true.dtype:
            y_true, _ = _coerce_labels(y_true, est_classes)

    classes = sorted(set(np.unique(y_true).tolist()))
    if len(classes) != 2:
        raise ValueError("calibration_curve currently supports binary classification only.")

    proba = _predict_proba(estimator, X)
    if proba is None:
        raise ValueError("Calibration requires probability scores.")
    pos_proba = proba[:, 1]
    y_binary = (y_true == classes[1]).astype(int)

    frac_positive, mean_predicted = _cal(y_binary, pos_proba, n_bins=n_bins)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mean_predicted,
            y=frac_positive,
            mode="lines+markers",
            name="Model",
            line={"color": PALETTE["accent"], "width": 2},
            marker={"size": 8},
            hovertemplate="Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly calibrated",
            line={"color": PALETTE["muted"], "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )
    )

    return apply_layout(
        fig,
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed frequency",
    )


def threshold_curve(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    title: str | None = "Threshold sweep",
) -> go.Figure:
    """Precision / recall / F1 / accuracy as a function of the decision
    threshold. Binary classification only.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = np.asarray(y)
    if hasattr(estimator, "classes_"):
        est_classes = np.asarray(estimator.classes_)
        if est_classes.dtype != y_true.dtype:
            y_true, _ = _coerce_labels(y_true, est_classes)
    classes = sorted(set(np.unique(y_true).tolist()))
    if len(classes) != 2:
        raise ValueError("threshold_curve supports binary classification only.")

    proba = _predict_proba(estimator, X)
    if proba is None:
        raise ValueError("threshold_curve requires probability scores.")
    pos_proba = proba[:, 1]
    y_binary = (y_true == classes[1]).astype(int)

    thresholds = np.linspace(0.05, 0.95, 19)
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        y_pred = (pos_proba >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": precision_score(y_binary, y_pred, zero_division=0),
                "recall": recall_score(y_binary, y_pred, zero_division=0),
                "f1": f1_score(y_binary, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_binary, y_pred),
            }
        )

    fig = go.Figure()
    metric_colors = {
        "precision": PALETTE["series_1"],
        "recall": PALETTE["series_2"],
        "f1": PALETTE["series_3"],
        "accuracy": PALETTE["series_4"],
    }
    for metric, color in metric_colors.items():
        fig.add_trace(
            go.Scatter(
                x=[r["threshold"] for r in rows],
                y=[r[metric] for r in rows],
                mode="lines+markers",
                name=metric.title(),
                line={"color": color, "width": 2},
                marker={"size": 6},
                hovertemplate=f"Threshold: %{{x:.2f}}<br>{metric.title()}: %{{y:.3f}}<extra></extra>",
            )
        )
    return apply_layout(
        fig, title=title, xaxis_title="Decision threshold", yaxis_title="Metric value"
    )


def lift_curve(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    title: str | None = "Lift curve",
) -> go.Figure:
    """Cumulative lift over random by decile of predicted probability.

    Binary classification only.
    """
    y_true = np.asarray(y)
    if hasattr(estimator, "classes_"):
        est_classes = np.asarray(estimator.classes_)
        if est_classes.dtype != y_true.dtype:
            y_true, _ = _coerce_labels(y_true, est_classes)
    classes = sorted(set(np.unique(y_true).tolist()))
    if len(classes) != 2:
        raise ValueError("lift_curve supports binary classification only.")

    proba = _predict_proba(estimator, X)
    if proba is None:
        raise ValueError("lift_curve requires probability scores.")
    pos_proba = proba[:, 1]
    y_binary = (y_true == classes[1]).astype(int)

    n = len(y_binary)
    base_rate = y_binary.mean() or 1e-9
    order = np.argsort(-pos_proba)
    sorted_y = y_binary[order]
    cum_pos = np.cumsum(sorted_y)
    population_pct = np.arange(1, n + 1) / n
    cum_response_rate = cum_pos / np.arange(1, n + 1)
    lift = cum_response_rate / base_rate

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=population_pct,
            y=lift,
            mode="lines",
            name="Lift",
            line={"color": PALETTE["accent"], "width": 2},
            hovertemplate="Pop. percentile: %{x:.2f}<br>Lift: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[1, 1],
            mode="lines",
            name="Random",
            line={"color": PALETTE["muted"], "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    return apply_layout(
        fig, title=title, xaxis_title="Population percentile", yaxis_title="Lift over random"
    )


def gain_curve(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    title: str | None = "Gain curve",
) -> go.Figure:
    """Cumulative gains: % positives captured by top X% of population.

    Binary classification only.
    """
    y_true = np.asarray(y)
    if hasattr(estimator, "classes_"):
        est_classes = np.asarray(estimator.classes_)
        if est_classes.dtype != y_true.dtype:
            y_true, _ = _coerce_labels(y_true, est_classes)
    classes = sorted(set(np.unique(y_true).tolist()))
    if len(classes) != 2:
        raise ValueError("gain_curve supports binary classification only.")

    proba = _predict_proba(estimator, X)
    if proba is None:
        raise ValueError("gain_curve requires probability scores.")
    pos_proba = proba[:, 1]
    y_binary = (y_true == classes[1]).astype(int)

    n = len(y_binary)
    n_pos = max(int(y_binary.sum()), 1)
    order = np.argsort(-pos_proba)
    sorted_y = y_binary[order]
    cum_pos = np.cumsum(sorted_y)
    population_pct = np.arange(1, n + 1) / n
    captured_pct = cum_pos / n_pos

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=population_pct,
            y=captured_pct,
            mode="lines",
            name="Model",
            line={"color": PALETTE["accent"], "width": 2},
            hovertemplate="Pop. percentile: %{x:.2f}<br>Captured: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": PALETTE["muted"], "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    return apply_layout(
        fig, title=title, xaxis_title="Population percentile", yaxis_title="% positives captured"
    )


def class_distribution(
    y: Any,
    *,
    title: str | None = "Class distribution",
) -> go.Figure:
    """Histogram of class counts in the target. Useful for spotting imbalance."""
    y_arr = np.asarray(y)
    unique, counts = np.unique(y_arr, return_counts=True)
    colors = categorical_sequence(len(unique))
    fig = go.Figure(
        data=go.Bar(
            x=[str(u) for u in unique],
            y=counts,
            marker={"color": colors},
            text=counts,
            textposition="outside",
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        )
    )
    return apply_layout(
        fig, title=title, xaxis_title="Class", yaxis_title="Count", showlegend=False
    )


# `make_subplots` is exported here in case higher-level dashboards need
# to compose multi-panel diagnostics.
__all__.append("make_subplots")
