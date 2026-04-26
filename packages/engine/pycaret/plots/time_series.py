"""Plotly-native time-series diagnostics.

Functions:

- ``forecast`` — point predictions with optional confidence band against
  the actuals.
- ``decomposition`` — additive/multiplicative trend / seasonal / residual
  via statsmodels' ``seasonal_decompose``.
- ``residual_diagnostics`` — 4-panel: residuals over time, histogram,
  ACF, PACF.
- ``acf`` — autocorrelation function bar chart.
- ``pacf`` — partial autocorrelation function bar chart.
- ``cv_split_visualizer`` — visual of how an ``ExpandingWindowSplitter`` /
  ``SlidingWindowSplitter`` carves the series into train/test folds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pycaret.plots._base import PALETTE, apply_layout, categorical_sequence

__all__ = [
    "acf",
    "cv_split_visualizer",
    "decomposition",
    "forecast",
    "pacf",
    "residual_diagnostics",
]


def forecast(
    y_true: Any,
    y_pred: Any,
    *,
    lower: Any | None = None,
    upper: Any | None = None,
    history: Any | None = None,
    title: str | None = "Forecast",
) -> go.Figure:
    """Forecast vs. actuals with optional CI band.

    Parameters
    ----------
    y_true : actuals over the forecast horizon (or None for no ground truth)
    y_pred : point predictions
    lower / upper : prediction interval bounds (same index as y_pred)
    history : training-period series shown before the forecast (optional)
    """
    fig = go.Figure()

    def _idx(s: Any) -> Any:
        if hasattr(s, "index"):
            return s.index
        return list(range(len(s)))

    if history is not None:
        fig.add_trace(
            go.Scatter(
                x=_idx(history),
                y=np.asarray(history, dtype=float),
                mode="lines",
                name="History",
                line={"color": PALETTE["muted"], "width": 1.5},
                hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
            )
        )

    # Confidence band (drawn first so it sits behind the lines).
    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=_idx(upper),
                y=np.asarray(upper, dtype=float),
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=_idx(lower),
                y=np.asarray(lower, dtype=float),
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(91,141,239,0.18)",
                line={"width": 0},
                name="Prediction interval",
                hoverinfo="skip",
            )
        )

    if y_true is not None:
        fig.add_trace(
            go.Scatter(
                x=_idx(y_true),
                y=np.asarray(y_true, dtype=float),
                mode="lines+markers",
                name="Actual",
                line={"color": PALETTE["ink"], "width": 1.5},
                marker={"size": 6},
                hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=_idx(y_pred),
            y=np.asarray(y_pred, dtype=float),
            mode="lines+markers",
            name="Forecast",
            line={"color": PALETTE["accent"], "width": 2},
            marker={"size": 6},
            hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
        )
    )

    return apply_layout(fig, title=title, xaxis_title="Time", yaxis_title="Value")


def decomposition(
    y: Any,
    *,
    period: int | None = None,
    model: str = "additive",
    title: str | None = "Seasonal decomposition",
) -> go.Figure:
    """Trend / seasonal / residual decomposition via statsmodels.

    Parameters
    ----------
    y : series with a regular time index
    period : seasonal period (e.g. 12 for monthly with yearly seasonality);
        inferred from ``y.index.freqstr`` when omitted
    model : ``"additive"`` (default) or ``"multiplicative"``
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    if period is None:
        # Heuristic: monthly→12, quarterly→4, weekly→52, daily→7.
        try:
            from pycaret.utils.time_series import get_sp_from_str

            period = get_sp_from_str(y.index.freqstr) if y.index.freqstr else 12
        except Exception:  # noqa: BLE001
            period = 12

    series = y if isinstance(y, pd.Series) else pd.Series(y)
    result = seasonal_decompose(series, period=period, model=model, extrapolate_trend="freq")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
    )
    series_list = [
        ("Observed", series, PALETTE["ink"]),
        ("Trend", result.trend, PALETTE["accent"]),
        ("Seasonal", result.seasonal, PALETTE["series_3"]),
        ("Residual", result.resid, PALETTE["series_2"]),
    ]
    for i, (_label, data, color) in enumerate(series_list, start=1):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                mode="lines",
                line={"color": color, "width": 1.5},
                showlegend=False,
                hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        height=620,
        title=title,
        font={"family": "Inter, system-ui, sans-serif", "size": 11, "color": PALETTE["ink"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 56, "r": 24, "t": 56, "b": 40},
        showlegend=False,
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=PALETTE["grid"], row=i, col=1)
        fig.update_yaxes(gridcolor=PALETTE["grid"], row=i, col=1)
    return fig


def acf(
    y: Any,
    *,
    nlags: int = 40,
    alpha: float = 0.05,
    title: str | None = "Autocorrelation",
) -> go.Figure:
    """Autocorrelation function bar chart with confidence interval band.

    Use to spot seasonality (peaks at lag = period) and residual
    correlation (residuals shouldn't have peaks).
    """
    from statsmodels.tsa.stattools import acf as _acf

    arr = np.asarray(y, dtype=float)
    arr = arr[~np.isnan(arr)]
    nlags = min(nlags, max(1, len(arr) // 2 - 1))
    values, confint = _acf(arr, nlags=nlags, alpha=alpha)

    lags = list(range(len(values)))
    upper = confint[:, 1] - values
    lower = values - confint[:, 0]

    fig = go.Figure()
    # CI band (centered around 0 for context).
    fig.add_trace(
        go.Scatter(
            x=lags + lags[::-1],
            y=list(upper) + list(-lower[::-1]),
            fill="toself",
            fillcolor="rgba(91,141,239,0.12)",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Stems.
    for lag, val in zip(lags, values, strict=True):
        fig.add_shape(
            type="line",
            x0=lag,
            x1=lag,
            y0=0,
            y1=val,
            line={"color": PALETTE["accent"], "width": 1.5},
        )
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=values,
            mode="markers",
            marker={"size": 6, "color": PALETTE["accent"]},
            hovertemplate="Lag: %{x}<br>ACF: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line={"color": PALETTE["muted"], "width": 1})

    return apply_layout(
        fig,
        title=title,
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        showlegend=False,
    )


def pacf(
    y: Any,
    *,
    nlags: int = 40,
    alpha: float = 0.05,
    title: str | None = "Partial autocorrelation",
) -> go.Figure:
    """Partial autocorrelation — same shape as ``acf`` but conditioning
    on the intermediate lags.
    """
    from statsmodels.tsa.stattools import pacf as _pacf

    arr = np.asarray(y, dtype=float)
    arr = arr[~np.isnan(arr)]
    nlags = min(nlags, max(1, len(arr) // 2 - 1))
    values, confint = _pacf(arr, nlags=nlags, alpha=alpha)
    lags = list(range(len(values)))
    upper = confint[:, 1] - values
    lower = values - confint[:, 0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lags + lags[::-1],
            y=list(upper) + list(-lower[::-1]),
            fill="toself",
            fillcolor="rgba(255,138,101,0.12)",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    for lag, val in zip(lags, values, strict=True):
        fig.add_shape(
            type="line",
            x0=lag,
            x1=lag,
            y0=0,
            y1=val,
            line={"color": PALETTE["series_2"], "width": 1.5},
        )
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=values,
            mode="markers",
            marker={"size": 6, "color": PALETTE["series_2"]},
            hovertemplate="Lag: %{x}<br>PACF: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line={"color": PALETTE["muted"], "width": 1})

    return apply_layout(
        fig,
        title=title,
        xaxis_title="Lag",
        yaxis_title="Partial autocorrelation",
        showlegend=False,
    )


def residual_diagnostics(
    y_true: Any,
    y_pred: Any,
    *,
    nlags: int = 30,
    title: str | None = "Residual diagnostics",
) -> go.Figure:
    """4-panel residual diagnostics for forecasts:

    1. Residuals over time.
    2. Residuals histogram with mean line.
    3. ACF of residuals.
    4. PACF of residuals.

    A well-behaved forecast has near-zero mean residuals, no time
    pattern, and ACF/PACF within the CI band.
    """
    from statsmodels.tsa.stattools import acf as _acf
    from statsmodels.tsa.stattools import pacf as _pacf

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    resid = y_true_arr - y_pred_arr
    resid_idx = y_true.index if hasattr(y_true, "index") else np.arange(len(resid))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Residuals over time", "Distribution", "ACF", "PACF"),
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Scatter(
            x=resid_idx,
            y=resid,
            mode="lines+markers",
            line={"color": PALETTE["accent"], "width": 1.5},
            marker={"size": 5},
            showlegend=False,
            hovertemplate="%{x}<br>Residual: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line={"color": PALETTE["muted"], "width": 1, "dash": "dash"}, row=1, col=1)

    fig.add_trace(
        go.Histogram(
            x=resid,
            nbinsx=20,
            marker={"color": PALETTE["accent"]},
            opacity=0.85,
            showlegend=False,
            hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    nlags_eff = min(nlags, max(1, len(resid) // 2 - 1))
    acf_vals = _acf(resid, nlags=nlags_eff)
    pacf_vals = _pacf(resid, nlags=nlags_eff)
    lags = list(range(len(acf_vals)))
    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_vals,
            marker={"color": PALETTE["series_3"]},
            showlegend=False,
            hovertemplate="Lag: %{x}<br>ACF: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=lags,
            y=pacf_vals,
            marker={"color": PALETTE["series_2"]},
            showlegend=False,
            hovertemplate="Lag: %{x}<br>PACF: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=560,
        title=title,
        font={"family": "Inter, system-ui, sans-serif", "size": 11, "color": PALETTE["ink"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 56, "r": 24, "t": 56, "b": 40},
        showlegend=False,
    )
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(gridcolor=PALETTE["grid"], row=r, col=c)
            fig.update_yaxes(gridcolor=PALETTE["grid"], row=r, col=c)
    return fig


def cv_split_visualizer(
    fold_generator: Any,
    y: Any,
    *,
    title: str | None = "CV folds",
) -> go.Figure:
    """Visualize how an sktime splitter carves the training series.

    Each fold is a row; train indices are accent-colored, test indices
    are red. Useful for spotting splitter misconfigurations.
    """
    arr_index = y.index if hasattr(y, "index") else np.arange(len(y))
    fig = go.Figure()

    folds = list(fold_generator.split(y))
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        fig.add_trace(
            go.Scatter(
                x=arr_index[train_idx],
                y=[fold_i] * len(train_idx),
                mode="markers",
                marker={"color": PALETTE["accent"], "size": 6, "symbol": "square"},
                name="Train" if fold_i == 0 else None,
                showlegend=fold_i == 0,
                hovertemplate=f"Fold {fold_i}<br>Train<br>%{{x}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=arr_index[test_idx],
                y=[fold_i] * len(test_idx),
                mode="markers",
                marker={"color": PALETTE["danger"], "size": 8, "symbol": "diamond"},
                name="Test" if fold_i == 0 else None,
                showlegend=fold_i == 0,
                hovertemplate=f"Fold {fold_i}<br>Test<br>%{{x}}<extra></extra>",
            )
        )

    return apply_layout(
        fig,
        title=title,
        xaxis_title="Time",
        yaxis_title="Fold #",
        height=140 + 40 * max(1, len(folds)),
    )


# Suppress unused imports warning.
_ = categorical_sequence
