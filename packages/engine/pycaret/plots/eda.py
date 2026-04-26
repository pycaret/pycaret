"""Plotly-native exploratory data analysis (EDA) plots.

Functions:

- ``column_distribution`` — histogram (numeric) / bar (categorical) for
  a single column.
- ``correlation_heatmap`` — Pearson correlation matrix as a heatmap.
- ``missingness_map`` — per-column missing-rate bar.
- ``target_vs_feature`` — joint distribution of a feature and the
  target (boxplot for numeric→categorical, scatter for numeric→numeric,
  count-bar for categorical→categorical).
- ``profile_summary`` — per-column data-quality summary table figure.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from pycaret.plots._base import PALETTE, apply_layout, categorical_sequence

__all__ = [
    "column_distribution",
    "correlation_heatmap",
    "missingness_map",
    "profile_summary",
    "target_vs_feature",
]


def column_distribution(
    data: Any,
    column: str,
    *,
    bins: int = 30,
    title: str | None = None,
) -> go.Figure:
    """Histogram (numeric) or count-bar (categorical) for one column."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not found.")
    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        fig = go.Figure(
            data=go.Histogram(
                x=series.dropna(),
                nbinsx=bins,
                marker={"color": PALETTE["accent"]},
                opacity=0.9,
                hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>",
            )
        )
        # Mean + median annotations.
        mean = float(series.mean())
        median = float(series.median())
        fig.add_vline(
            x=mean,
            line={"color": PALETTE["danger"], "width": 1.2, "dash": "dash"},
            annotation_text=f"mean = {mean:.3f}",
            annotation_position="top right",
        )
        fig.add_vline(
            x=median,
            line={"color": PALETTE["success"], "width": 1.2, "dash": "dot"},
            annotation_text=f"median = {median:.3f}",
            annotation_position="top left",
        )
        return apply_layout(
            fig,
            title=title or f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            showlegend=False,
        )

    # Categorical / object.
    counts = series.value_counts(dropna=False)
    colors = categorical_sequence(len(counts))
    fig = go.Figure(
        data=go.Bar(
            x=[str(v) for v in counts.index],
            y=counts.values,
            marker={"color": colors},
            text=counts.values,
            textposition="outside",
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        )
    )
    return apply_layout(
        fig,
        title=title or f"{column} (top values)",
        xaxis_title=column,
        yaxis_title="Count",
        showlegend=False,
    )


def correlation_heatmap(
    data: Any,
    *,
    method: str = "pearson",
    annotate: bool = True,
    title: str | None = "Correlation matrix",
) -> go.Figure:
    """Numeric correlation matrix as a heatmap.

    ``method``: ``pearson`` (default), ``spearman``, ``kendall``.
    """
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        raise ValueError("Need ≥ 2 numeric columns for a correlation heatmap.")
    corr = numeric.corr(method=method).round(3)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[
                [0.0, PALETTE["danger"]],
                [0.5, PALETTE["bg"]],
                [1.0, PALETTE["accent"]],
            ],
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar={"thickness": 12, "outlinewidth": 0, "title": method.title()},
            text=corr.values if annotate else None,
            texttemplate="%{text:.2f}" if annotate else None,
            textfont={"size": 10, "color": PALETTE["ink"]},
            hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    return apply_layout(
        fig,
        title=title,
        showlegend=False,
        height=max(360, 28 * len(corr.columns) + 80),
    )


def missingness_map(
    data: Any,
    *,
    title: str | None = "Missing data per column",
) -> go.Figure:
    """Per-column missing-rate horizontal bar chart."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    miss = df.isna().mean().sort_values(ascending=False)
    if miss.max() == 0:
        # No missing values — render a flat empty chart with a subtitle.
        fig = go.Figure(
            data=go.Bar(
                x=[0.0] * len(miss),
                y=miss.index.tolist(),
                orientation="h",
                marker={"color": PALETTE["success"]},
                hovertemplate="%{y}<br>0 missing<extra></extra>",
            )
        )
        return apply_layout(
            fig,
            title=f"{title} (no missing values)",
            xaxis_title="Missing rate",
            yaxis_title="Column",
            showlegend=False,
            height=max(280, 24 * len(miss) + 60),
        )

    fig = go.Figure(
        data=go.Bar(
            x=miss.values * 100,
            y=miss.index.tolist(),
            orientation="h",
            marker={"color": PALETTE["danger"], "opacity": 0.8},
            text=[f"{v * 100:.1f}%" for v in miss.values],
            textposition="outside",
            hovertemplate="%{y}<br>Missing: %{x:.2f}%<extra></extra>",
        )
    )
    return apply_layout(
        fig,
        title=title,
        xaxis_title="Missing %",
        yaxis_title="Column",
        showlegend=False,
        height=max(280, 24 * len(miss) + 60),
    )


def target_vs_feature(
    data: Any,
    feature: str,
    target: str,
    *,
    title: str | None = None,
) -> go.Figure:
    """Joint plot of feature × target.

    - numeric × numeric → scatter with trend line
    - numeric × categorical → boxplot grouped by target
    - categorical × numeric → boxplot grouped by feature
    - categorical × categorical → grouped bar (count)
    """
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    for col in (feature, target):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found.")
    f_num = pd.api.types.is_numeric_dtype(df[feature])
    t_num = pd.api.types.is_numeric_dtype(df[target])

    if f_num and t_num:
        # Scatter.
        fig = go.Figure(
            data=go.Scatter(
                x=df[feature],
                y=df[target],
                mode="markers",
                marker={"size": 5, "color": PALETTE["accent"], "opacity": 0.55},
                hovertemplate=f"{feature}: %{{x}}<br>{target}: %{{y}}<extra></extra>",
            )
        )
        return apply_layout(
            fig,
            title=title or f"{target} vs. {feature}",
            xaxis_title=feature,
            yaxis_title=target,
            showlegend=False,
        )

    if f_num and not t_num:
        # Box plot grouped by target.
        fig = go.Figure()
        groups = sorted(df[target].dropna().unique().tolist(), key=str)
        colors = categorical_sequence(len(groups))
        for color, g in zip(colors, groups, strict=True):
            sub = df[df[target] == g][feature].dropna()
            fig.add_trace(
                go.Box(
                    y=sub,
                    name=str(g),
                    marker={"color": color},
                    boxmean=True,
                )
            )
        return apply_layout(
            fig,
            title=title or f"{feature} grouped by {target}",
            xaxis_title=target,
            yaxis_title=feature,
        )

    if not f_num and t_num:
        # Box plot grouped by feature.
        fig = go.Figure()
        groups = sorted(df[feature].dropna().unique().tolist(), key=str)[:30]
        colors = categorical_sequence(len(groups))
        for color, g in zip(colors, groups, strict=True):
            sub = df[df[feature] == g][target].dropna()
            fig.add_trace(
                go.Box(
                    y=sub,
                    name=str(g),
                    marker={"color": color},
                    boxmean=True,
                )
            )
        return apply_layout(
            fig,
            title=title or f"{target} grouped by {feature}",
            xaxis_title=feature,
            yaxis_title=target,
        )

    # Both categorical — grouped bar count.
    pivot = df.groupby([feature, target], observed=True).size().unstack(fill_value=0)
    fig = go.Figure()
    target_levels = pivot.columns.tolist()
    colors = categorical_sequence(len(target_levels))
    for color, lvl in zip(colors, target_levels, strict=True):
        fig.add_trace(
            go.Bar(
                x=[str(v) for v in pivot.index],
                y=pivot[lvl].values,
                name=str(lvl),
                marker={"color": color},
                hovertemplate=f"{feature}: %{{x}}<br>{target}: {lvl}<br>Count: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    return apply_layout(
        fig,
        title=title or f"{feature} × {target} (counts)",
        xaxis_title=feature,
        yaxis_title="Count",
    )


def profile_summary(
    data: Any,
    *,
    title: str | None = "Column profile",
) -> go.Figure:
    """Per-column data-quality summary as a Plotly table figure.

    Columns: Name, Type, Non-null %, Unique, Mean, Std, Min, Max.
    Useful as the EDA screen's "headline" panel.
    """
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    rows: list[list[Any]] = []
    for col in df.columns:
        s = df[col]
        non_null = float(s.notna().mean()) * 100
        unique_count = int(s.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(s):
            rows.append(
                [
                    col,
                    str(s.dtype),
                    f"{non_null:.1f}%",
                    unique_count,
                    f"{s.mean():.3f}",
                    f"{s.std():.3f}",
                    f"{s.min():.3f}",
                    f"{s.max():.3f}",
                ]
            )
        else:
            top = s.value_counts(dropna=True).head(1)
            top_value = f"{top.index[0]} ({int(top.values[0])})" if len(top) else "–"
            rows.append(
                [
                    col,
                    str(s.dtype),
                    f"{non_null:.1f}%",
                    unique_count,
                    "–",
                    "–",
                    f"top: {top_value}",
                    "–",
                ]
            )

    headers = ["Column", "Type", "Non-null", "Unique", "Mean", "Std", "Min", "Max"]
    fig = go.Figure(
        data=go.Table(
            header={
                "values": headers,
                "fill_color": PALETTE["accent"],
                "font": {"color": "white", "size": 12},
                "align": "left",
            },
            cells={
                "values": list(map(list, zip(*rows, strict=True))) if rows else [[]] * len(headers),
                "fill_color": [
                    [
                        PALETTE["bg"] if i % 2 == 0 else "rgba(91,141,239,0.04)"
                        for i in range(len(rows))
                    ]
                ]
                * len(headers),
                "font": {"color": PALETTE["ink"], "size": 11},
                "align": "left",
                "height": 26,
            },
        )
    )
    fig.update_layout(
        title=title,
        font={"family": "Inter, system-ui, sans-serif"},
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        height=80 + 26 * (len(rows) + 1),
    )
    return fig
