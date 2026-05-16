"""Analysis procedure registry + dispatcher."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from pycaret_server.analyses.base import (
    AnalysisProcedureError,
    AnalysisResult,
)
from pycaret_server.analyses.procedures import (
    run_anova_oneway,
    run_arima,
    run_chi2,
    run_cox_ph,
    run_kaplan_meier,
    run_kruskal,
    run_logrank,
    run_mannwhitney,
    run_ols,
    run_paired_ttest,
    run_prophet,
    run_ttest,
)


ProcedureFn = Callable[[pd.DataFrame, dict[str, Any]], AnalysisResult]


_PROCEDURES: dict[str, ProcedureFn] = {
    "ttest": run_ttest,
    "welch_ttest": run_ttest,  # alias — equal_var=False is the default
    "paired_ttest": run_paired_ttest,
    "mannwhitney": run_mannwhitney,
    "anova_oneway": run_anova_oneway,
    "kruskal": run_kruskal,
    "chi2": run_chi2,
    "ols": run_ols,
    "kaplan_meier": run_kaplan_meier,
    "logrank": run_logrank,
    "cox_ph": run_cox_ph,
    "arima": run_arima,
    "prophet": run_prophet,
}


def list_kinds() -> list[str]:
    return sorted(_PROCEDURES)


def run_analysis(
    kind: str, df: pd.DataFrame, params: dict[str, Any]
) -> AnalysisResult:
    fn = _PROCEDURES.get(kind)
    if fn is None:
        raise AnalysisProcedureError(
            f"unknown analysis kind {kind!r}; available: {list_kinds()}"
        )
    return fn(df, params)
