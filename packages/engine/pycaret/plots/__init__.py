"""PyCaret 4.0 plot library — Plotly-native, framework-agnostic.

Replaces the legacy ``plot_model`` API (deleted in s46). Each plot
function takes a fitted estimator (typically a sklearn / sktime
``Pipeline``) plus data and returns a ``plotly.graph_objects.Figure``.

The figure is the universal currency: notebooks call ``.show()``, the
FastAPI layer calls ``.to_dict()`` for HTTP transport, the React UI
consumes the JSON via ``react-plotly.js``.

Module layout:

- ``classification`` — confusion matrix, ROC, PR, calibration, threshold,
  lift, gain.
- ``regression`` — residuals, prediction-error, learning curve.
- ``feature`` — feature importance + SHAP (s48).
- ``clustering`` — silhouette, elbow, embedding (s49).
- ``anomaly`` — score histogram, anomaly map (s50).
- ``time_series`` — decomposition, forecast fan, residual diagnostics (s51).
- ``eda`` — distributions, correlation, missingness (s52).

Design tokens align with the React UI's ``ink-*`` / ``accent-*`` /
``success-*`` / ``danger-*`` palette so plots look at home in the
dashboard without per-screen restyling.
"""

from __future__ import annotations

from pycaret.plots import classification, regression
from pycaret.plots._base import PALETTE, default_layout

__all__ = [
    "PALETTE",
    "classification",
    "default_layout",
    "regression",
]
