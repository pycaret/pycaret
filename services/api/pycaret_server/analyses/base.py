"""Analysis result envelope + base error type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class AnalysisProcedureError(RuntimeError):
    """Raised when a procedure can't compute (missing column, empty
    group, singular matrix, …). The API layer converts this to a
    400 with the message body."""


@dataclass
class AnalysisResult:
    """Uniform envelope returned by every analysis procedure.

    Every field is JSON-serialisable so the result writes straight onto
    the Run row's ``metrics`` JSON. The Plotly ``figure`` mirrors the
    shape ``PlotlyFigure`` in the frontend types.
    """

    # Headline numbers — required.
    test_statistic: float | None
    p_value: float | None
    # Optional descriptors.
    effect_size: float | None = None
    effect_size_name: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    # Tabular detail (cohort means, ANOVA table, coefficient rows).
    table: list[dict[str, Any]] = field(default_factory=list)
    # Plain-English interpretation rendered above the table.
    interpretation: str = ""
    # Plotly figure dict (data + layout).
    figure: dict[str, Any] | None = None
    # Free-form per-procedure extras (e.g. diagnostic test results).
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_statistic": _safe_float(self.test_statistic),
            "p_value": _safe_float(self.p_value),
            "effect_size": _safe_float(self.effect_size),
            "effect_size_name": self.effect_size_name,
            "ci_low": _safe_float(self.ci_low),
            "ci_high": _safe_float(self.ci_high),
            "table": self.table,
            "interpretation": self.interpretation,
            "figure": self.figure,
            "extra": self.extra,
        }


def _safe_float(v: float | None) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # NaN / Inf would blow up JSON serialisation.
    if f != f or f in (float("inf"), float("-inf")):  # NaN check
        return None
    return f
