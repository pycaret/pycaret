"""Phase 11 statistical-computing package.

Each procedure is a typed module under this package exposing a single
``run(df, params) -> AnalysisResult`` function. The dispatcher
(:mod:`pycaret_server.analyses.factory`) resolves by ``Analysis.kind``;
adding a new procedure is one new module + one entry.

The result envelope is uniform: every procedure returns a
``test_statistic``, ``p_value``, ``effect_size`` (when applicable),
``ci_low`` / ``ci_high`` (when applicable), a plain-English
``interpretation``, and a Plotly figure dict for the result card.

v1 procedures:

- ``ttest`` — independent two-sample t-test (Welch by default).
- ``paired_ttest`` — paired-sample t-test.
- ``mannwhitney`` — Mann-Whitney U.
- ``anova_oneway`` — one-way ANOVA + omega-squared effect size.
- ``kruskal`` — Kruskal-Wallis.
- ``chi2`` — chi-square independence + Cramér's V.
- ``ols`` — OLS regression with diagnostic suite (residuals, Q-Q,
  leverage, VIF, Cook's distance, Durbin-Watson).
- ``kaplan_meier`` — non-parametric survival curve.
- ``logrank`` — two-group log-rank test.
- ``cox_ph`` — Cox proportional hazards.
- ``arima`` — Box-Jenkins ARIMA point forecast.
- ``prophet`` — Facebook Prophet forecast (when ``prophet`` extra installed).

Heavy stats deps (``statsmodels``, ``lifelines``, ``prophet``) import
lazily inside each module so the base server install stays slim.
"""

from pycaret_server.analyses.base import (
    AnalysisProcedureError,
    AnalysisResult,
)
from pycaret_server.analyses.factory import (
    list_kinds,
    run_analysis,
)

__all__ = [
    "AnalysisProcedureError",
    "AnalysisResult",
    "list_kinds",
    "run_analysis",
]
