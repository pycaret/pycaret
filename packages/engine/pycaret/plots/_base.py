"""Shared Plotly figure utilities for ``pycaret.plots``.

Centralizes the design tokens (color palette, font, layout defaults) so
every chart looks consistent in the dashboard. Aligned with the React
UI's Tailwind tokens (`ink-*`, `accent-*`, `success-*`, `danger-*`,
`warn-*`).
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# Palette — keep in sync with `apps/web/src/index.css` design tokens.
# Hex values pulled from the existing UI; muted tones suit dark + light bg.
PALETTE: dict[str, str] = {
    # Primary action / data series.
    "accent": "#5B8DEF",
    "accent_alt": "#3F6FD9",
    # Categorical series — uses a colorblind-safe sequence (Okabe-Ito-ish).
    "series_1": "#5B8DEF",
    "series_2": "#FF8A65",
    "series_3": "#26A69A",
    "series_4": "#AB47BC",
    "series_5": "#FFCA28",
    "series_6": "#EF5350",
    "series_7": "#42A5F5",
    "series_8": "#7E57C2",
    # Status colors.
    "success": "#22C55E",
    "danger": "#EF4444",
    "warn": "#F59E0B",
    "muted": "#64748B",
    # Backgrounds + grid.
    "bg": "#FFFFFF",
    "bg_dark": "#0F172A",
    "grid": "#E2E8F0",
    "grid_dark": "#1E293B",
    "ink": "#1E293B",
    "ink_dark": "#F1F5F9",
}


def categorical_sequence(n: int) -> list[str]:
    """Return ``n`` categorical colors cycling through the series palette."""
    seq = [PALETTE[f"series_{i}"] for i in range(1, 9)]
    if n <= len(seq):
        return seq[:n]
    # Cycle if more series than palette entries.
    return [seq[i % len(seq)] for i in range(n)]


def default_layout(
    *,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    showlegend: bool = True,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    """Standard layout dict applied to every chart.

    Themed for the React dashboard's light surface; use Plotly's
    ``template`` swap on the client side for dark mode.
    """
    layout: dict[str, Any] = {
        "title": {"text": title} if title else None,
        "font": {"family": "Inter, system-ui, sans-serif", "size": 12, "color": PALETTE["ink"]},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 56, "r": 24, "t": 48 if title else 24, "b": 48},
        "showlegend": showlegend,
        "legend": {
            "orientation": "h",
            "y": -0.18,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
        "xaxis": {
            "title": {"text": xaxis_title} if xaxis_title else None,
            "gridcolor": PALETTE["grid"],
            "zerolinecolor": PALETTE["grid"],
            "linecolor": PALETTE["grid"],
            "ticks": "outside",
            "tickcolor": PALETTE["grid"],
            "automargin": True,
        },
        "yaxis": {
            "title": {"text": yaxis_title} if yaxis_title else None,
            "gridcolor": PALETTE["grid"],
            "zerolinecolor": PALETTE["grid"],
            "linecolor": PALETTE["grid"],
            "ticks": "outside",
            "tickcolor": PALETTE["grid"],
            "automargin": True,
        },
        "hoverlabel": {
            "bgcolor": "white",
            "bordercolor": PALETTE["grid"],
            "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
        },
    }
    if height is not None:
        layout["height"] = height
    if width is not None:
        layout["width"] = width
    # Strip None values so Plotly doesn't choke.
    return {k: v for k, v in layout.items() if v is not None}


def apply_layout(fig: go.Figure, **layout_kwargs: Any) -> go.Figure:
    """Apply ``default_layout(**kwargs)`` to ``fig`` in-place and return it."""
    fig.update_layout(**default_layout(**layout_kwargs))
    return fig


def unwrap_estimator(estimator: Any) -> Any:
    """Extract the bare model from a sklearn / sktime ``Pipeline`` if needed.

    Most chart functions need the final estimator (for ``predict_proba``,
    ``feature_importances_``, etc.) but accept a fitted pipeline as input
    for convenience.
    """
    if hasattr(estimator, "steps"):
        return estimator.steps[-1][1]
    return estimator
