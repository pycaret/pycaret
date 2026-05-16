"""The v1 library of statistical procedures.

Each ``run_*`` function takes ``(df, params)`` and returns an
``AnalysisResult``. Heavy deps (``statsmodels``, ``lifelines``,
``prophet``) import lazily so the base install stays light.

Conventions:

- All p-values use the two-sided test unless otherwise documented.
- Effect-size names follow the procedure's literature (``cohens_d``,
  ``eta_squared``, ``omega_squared``, ``cramers_v``, ``r2``,
  ``concordance``…).
- Plotly figures are built with raw data dicts (no plotly-py
  dependency required) so the result JSON drops straight into the
  frontend's ``<PlotlyFigure>``.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from pycaret_server.analyses.base import (
    AnalysisProcedureError,
    AnalysisResult,
)


# ─────────────────────────────────────────── helpers


def _require_column(df: pd.DataFrame, col: str, role: str) -> pd.Series:
    if col not in df.columns:
        raise AnalysisProcedureError(f"{role} column {col!r} not found in data")
    return df[col]


def _coerce_numeric(s: pd.Series, col: str) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").dropna()
    if out.empty:
        raise AnalysisProcedureError(
            f"column {col!r} has no numeric values after coercion"
        )
    return out


def _bar_figure(labels: list[str], values: list[float], y_title: str) -> dict:
    """A trivial bar chart used by t-test / ANOVA result cards."""
    return {
        "data": [
            {
                "type": "bar",
                "x": labels,
                "y": values,
                "marker": {"color": "#3b82f6"},
            }
        ],
        "layout": {
            "title": "",
            "xaxis": {"title": ""},
            "yaxis": {"title": y_title},
            "margin": {"l": 50, "r": 20, "t": 30, "b": 50},
        },
    }


def _line_figure(x: list, y: list, name: str, y_title: str) -> dict:
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": x,
                "y": y,
                "name": name,
                "line": {"color": "#3b82f6"},
            }
        ],
        "layout": {
            "xaxis": {"title": ""},
            "yaxis": {"title": y_title},
            "margin": {"l": 60, "r": 20, "t": 30, "b": 50},
        },
    }


# ─────────────────────────────────────────── t-tests


def run_ttest(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Independent two-sample t-test (Welch by default)."""
    from scipy import stats  # type: ignore[import-untyped]

    group_col = params["grouping_column"]
    measure_col = params["measure_column"]
    equal_var = bool(params.get("equal_var", False))
    alpha = float(params.get("alpha", 0.05))

    grp = _require_column(df, group_col, "grouping")
    msr = _require_column(df, measure_col, "measure")
    groups = list(grp.dropna().unique())
    if len(groups) != 2:
        raise AnalysisProcedureError(
            f"grouping column must have exactly 2 levels, got {len(groups)}"
        )
    a = _coerce_numeric(msr[grp == groups[0]], measure_col)
    b = _coerce_numeric(msr[grp == groups[1]], measure_col)
    res = stats.ttest_ind(a, b, equal_var=equal_var)
    # Cohen's d.
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / pooled if pooled else None
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = stats.t.ppf(1 - alpha / 2, df=len(a) + len(b) - 2)
    return AnalysisResult(
        test_statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=d,
        effect_size_name="cohens_d",
        ci_low=diff - z * se,
        ci_high=diff + z * se,
        table=[
            {"group": str(groups[0]), "n": int(len(a)), "mean": float(a.mean()), "std": float(a.std(ddof=1))},
            {"group": str(groups[1]), "n": int(len(b)), "mean": float(b.mean()), "std": float(b.std(ddof=1))},
        ],
        interpretation=_interpret_pvalue(res.pvalue, alpha)
        + f" Mean difference = {diff:.4g} (Cohen's d = {d:.3g})."
        if d is not None
        else "",
        figure=_bar_figure(
            [str(groups[0]), str(groups[1])],
            [float(a.mean()), float(b.mean())],
            f"mean({measure_col})",
        ),
    )


def run_paired_ttest(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Paired-sample t-test. Pass two measure columns (``column_a``, ``column_b``)."""
    from scipy import stats

    a_col = params["column_a"]
    b_col = params["column_b"]
    a = _coerce_numeric(_require_column(df, a_col, "column_a"), a_col)
    b = _coerce_numeric(_require_column(df, b_col, "column_b"), b_col)
    if len(a) != len(b):
        # Align by index — pandas does this if both came from same df.
        joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        a = joined["a"]
        b = joined["b"]
    res = stats.ttest_rel(a, b)
    diff = (a - b).mean()
    sd = (a - b).std(ddof=1)
    d = diff / sd if sd else None
    return AnalysisResult(
        test_statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=d,
        effect_size_name="cohens_d_paired",
        table=[
            {"column": a_col, "n": int(len(a)), "mean": float(a.mean())},
            {"column": b_col, "n": int(len(b)), "mean": float(b.mean())},
        ],
        interpretation=_interpret_pvalue(res.pvalue)
        + f" Paired mean diff = {diff:.4g}.",
        figure=_bar_figure([a_col, b_col], [float(a.mean()), float(b.mean())], "mean"),
    )


def run_mannwhitney(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Mann-Whitney U test (non-parametric two-group comparison)."""
    from scipy import stats

    group_col = params["grouping_column"]
    measure_col = params["measure_column"]
    grp = _require_column(df, group_col, "grouping")
    msr = _require_column(df, measure_col, "measure")
    levels = list(grp.dropna().unique())
    if len(levels) != 2:
        raise AnalysisProcedureError("Mann-Whitney needs exactly 2 levels")
    a = _coerce_numeric(msr[grp == levels[0]], measure_col)
    b = _coerce_numeric(msr[grp == levels[1]], measure_col)
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial r.
    n1, n2 = len(a), len(b)
    r = 1 - (2 * res.statistic) / (n1 * n2) if (n1 * n2) else None
    return AnalysisResult(
        test_statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=r,
        effect_size_name="rank_biserial_r",
        table=[
            {"group": str(levels[0]), "n": int(n1), "median": float(a.median())},
            {"group": str(levels[1]), "n": int(n2), "median": float(b.median())},
        ],
        interpretation=_interpret_pvalue(res.pvalue),
        figure=_bar_figure(
            [str(levels[0]), str(levels[1])],
            [float(a.median()), float(b.median())],
            f"median({measure_col})",
        ),
    )


# ─────────────────────────────────────────── ANOVA / Kruskal


def run_anova_oneway(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """One-way ANOVA with omega-squared effect size."""
    from scipy import stats

    group_col = params["grouping_column"]
    measure_col = params["measure_column"]
    grp = _require_column(df, group_col, "grouping")
    msr = _require_column(df, measure_col, "measure")
    groups = [
        _coerce_numeric(msr[grp == lvl], measure_col)
        for lvl in grp.dropna().unique()
    ]
    if len(groups) < 2:
        raise AnalysisProcedureError("ANOVA needs at least 2 groups")
    res = stats.f_oneway(*groups)
    # Omega² = (SS_between - df_between * MS_within) / (SS_total + MS_within).
    grand = pd.concat(groups)
    ss_total = float(((grand - grand.mean()) ** 2).sum())
    ss_between = float(
        sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
    )
    ss_within = ss_total - ss_between
    df_between = len(groups) - 1
    df_within = sum(len(g) - 1 for g in groups)
    ms_within = ss_within / df_within if df_within else float("nan")
    omega2 = (
        (ss_between - df_between * ms_within) / (ss_total + ms_within)
        if (ss_total + ms_within)
        else None
    )
    levels = list(grp.dropna().unique())
    return AnalysisResult(
        test_statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=omega2,
        effect_size_name="omega_squared",
        table=[
            {
                "group": str(lvl),
                "n": int(len(g)),
                "mean": float(g.mean()),
                "std": float(g.std(ddof=1)),
            }
            for lvl, g in zip(levels, groups, strict=False)
        ],
        interpretation=_interpret_pvalue(res.pvalue)
        + f" Between-group SS = {ss_between:.4g}, within = {ss_within:.4g}.",
        figure=_bar_figure(
            [str(lvl) for lvl in levels],
            [float(g.mean()) for g in groups],
            f"mean({measure_col})",
        ),
    )


def run_kruskal(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Kruskal-Wallis H test (non-parametric one-way)."""
    from scipy import stats

    group_col = params["grouping_column"]
    measure_col = params["measure_column"]
    grp = _require_column(df, group_col, "grouping")
    msr = _require_column(df, measure_col, "measure")
    groups = [
        _coerce_numeric(msr[grp == lvl], measure_col)
        for lvl in grp.dropna().unique()
    ]
    res = stats.kruskal(*groups)
    levels = list(grp.dropna().unique())
    return AnalysisResult(
        test_statistic=float(res.statistic),
        p_value=float(res.pvalue),
        table=[
            {"group": str(lvl), "n": int(len(g)), "median": float(g.median())}
            for lvl, g in zip(levels, groups, strict=False)
        ],
        interpretation=_interpret_pvalue(res.pvalue),
        figure=_bar_figure(
            [str(lvl) for lvl in levels],
            [float(g.median()) for g in groups],
            f"median({measure_col})",
        ),
    )


# ─────────────────────────────────────────── chi-square / Fisher / Cramér's V


def run_chi2(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Pearson's chi-square independence + Cramér's V."""
    from scipy import stats

    a_col = params["column_a"]
    b_col = params["column_b"]
    a = _require_column(df, a_col, "column_a")
    b = _require_column(df, b_col, "column_b")
    table = pd.crosstab(a, b)
    res = stats.chi2_contingency(table)
    chi2, p, dof, _exp = res
    n = int(table.values.sum())
    cramers = math.sqrt(chi2 / (n * (min(table.shape) - 1))) if n else None
    return AnalysisResult(
        test_statistic=float(chi2),
        p_value=float(p),
        effect_size=cramers,
        effect_size_name="cramers_v",
        table=[
            {"_index": str(idx), **{str(c): int(v) for c, v in row.items()}}
            for idx, row in table.iterrows()
        ],
        interpretation=_interpret_pvalue(p) + f" dof = {dof}.",
        figure=_bar_figure(
            [str(idx) for idx in table.index],
            [int(table.loc[idx].sum()) for idx in table.index],
            "row total",
        ),
    )


# ─────────────────────────────────────────── OLS regression diagnostics


def run_ols(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """OLS regression with the standard diagnostic suite.

    Body: ``{response, predictors: [colA, colB], add_intercept?}``.
    Returns coefficient table + residual diagnostics + a residuals-vs-
    fitted scatter for the figure.
    """
    import numpy as np
    import statsmodels.api as sm  # type: ignore[import-untyped]
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore[import-untyped]
    from statsmodels.stats.stattools import durbin_watson  # type: ignore[import-untyped]

    response = params["response"]
    predictors: list[str] = list(params.get("predictors") or [])
    if not predictors:
        raise AnalysisProcedureError("predictors list must be non-empty")
    add_intercept = bool(params.get("add_intercept", True))

    y = _coerce_numeric(_require_column(df, response, "response"), response)
    X = df[predictors].copy()
    for c in predictors:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna()
    y = y.loc[X.index]
    if add_intercept:
        X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # VIF — only for the predictors, skip the intercept.
    vifs: dict[str, float] = {}
    if X.shape[1] > 1:
        for i, col in enumerate(X.columns):
            if col == "const":
                continue
            try:
                vifs[col] = float(variance_inflation_factor(X.values, i))
            except Exception:  # noqa: BLE001
                pass
    coef_rows = []
    for name in model.params.index:
        coef_rows.append(
            {
                "term": str(name),
                "estimate": float(model.params[name]),
                "std_err": float(model.bse[name]),
                "t": float(model.tvalues[name]),
                "p_value": float(model.pvalues[name]),
                "ci_low": float(model.conf_int().loc[name][0]),
                "ci_high": float(model.conf_int().loc[name][1]),
                "vif": vifs.get(name),
            }
        )
    dw = float(durbin_watson(model.resid))
    fig = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": list(map(float, model.fittedvalues)),
                "y": list(map(float, model.resid)),
                "marker": {"color": "#3b82f6", "size": 6, "opacity": 0.6},
                "name": "residuals",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [float(model.fittedvalues.min()), float(model.fittedvalues.max())],
                "y": [0, 0],
                "line": {"color": "#94a3b8", "dash": "dash"},
                "name": "zero",
            },
        ],
        "layout": {
            "xaxis": {"title": "Fitted"},
            "yaxis": {"title": "Residual"},
            "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
        },
    }
    return AnalysisResult(
        test_statistic=float(model.fvalue) if not np.isnan(model.fvalue) else None,
        p_value=float(model.f_pvalue) if not np.isnan(model.f_pvalue) else None,
        effect_size=float(model.rsquared),
        effect_size_name="r_squared",
        table=coef_rows,
        interpretation=(
            f"R² = {model.rsquared:.4g}, adj R² = {model.rsquared_adj:.4g}, "
            f"F = {model.fvalue:.4g}, p = {model.f_pvalue:.4g}, "
            f"Durbin-Watson = {dw:.3g}."
        ),
        figure=fig,
        extra={
            "rsquared": float(model.rsquared),
            "rsquared_adj": float(model.rsquared_adj),
            "durbin_watson": dw,
            "n_obs": int(model.nobs),
        },
    )


# ─────────────────────────────────────────── survival


def run_kaplan_meier(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Kaplan-Meier non-parametric survival.

    Body: ``{time_column, event_column, group_column?}``.
    """
    from lifelines import KaplanMeierFitter  # type: ignore[import-untyped]

    time_col = params["time_column"]
    event_col = params["event_column"]
    group_col = params.get("group_column")
    t = _coerce_numeric(_require_column(df, time_col, "time"), time_col)
    e = _require_column(df, event_col, "event").astype(int)
    e = e.loc[t.index]

    kmf = KaplanMeierFitter()
    if group_col:
        traces = []
        rows = []
        for level, sub in df.groupby(group_col):
            tg = pd.to_numeric(sub[time_col], errors="coerce").dropna()
            eg = sub.loc[tg.index, event_col].astype(int)
            kmf.fit(tg, eg, label=str(level))
            sf = kmf.survival_function_
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [float(x) for x in sf.index],
                    "y": [float(v) for v in sf.iloc[:, 0]],
                    "name": str(level),
                }
            )
            rows.append(
                {
                    "group": str(level),
                    "n": int(len(tg)),
                    "events": int(eg.sum()),
                    "median_survival": (
                        float(kmf.median_survival_time_)
                        if kmf.median_survival_time_ is not None
                        else None
                    ),
                }
            )
        figure = {
            "data": traces,
            "layout": {"yaxis": {"title": "S(t)"}, "xaxis": {"title": "time"}},
        }
        return AnalysisResult(
            test_statistic=None,
            p_value=None,
            table=rows,
            interpretation="Survival curves by group; median rows in the table.",
            figure=figure,
        )
    kmf.fit(t, e)
    sf = kmf.survival_function_
    return AnalysisResult(
        test_statistic=None,
        p_value=None,
        table=[
            {
                "n": int(len(t)),
                "events": int(e.sum()),
                "median_survival": (
                    float(kmf.median_survival_time_)
                    if kmf.median_survival_time_ is not None
                    else None
                ),
            }
        ],
        interpretation="Non-parametric survival curve.",
        figure=_line_figure(
            [float(x) for x in sf.index],
            [float(v) for v in sf.iloc[:, 0]],
            "S(t)",
            "S(t)",
        ),
    )


def run_logrank(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Two-sample log-rank test."""
    from lifelines.statistics import logrank_test  # type: ignore[import-untyped]

    time_col = params["time_column"]
    event_col = params["event_column"]
    group_col = params["group_column"]
    df = df.dropna(subset=[time_col, event_col, group_col])
    levels = list(df[group_col].unique())
    if len(levels) != 2:
        raise AnalysisProcedureError("log-rank needs exactly 2 groups")
    a = df[df[group_col] == levels[0]]
    b = df[df[group_col] == levels[1]]
    res = logrank_test(
        pd.to_numeric(a[time_col]),
        pd.to_numeric(b[time_col]),
        event_observed_A=a[event_col].astype(int),
        event_observed_B=b[event_col].astype(int),
    )
    return AnalysisResult(
        test_statistic=float(res.test_statistic),
        p_value=float(res.p_value),
        table=[
            {"group": str(levels[0]), "n": int(len(a)), "events": int(a[event_col].sum())},
            {"group": str(levels[1]), "n": int(len(b)), "events": int(b[event_col].sum())},
        ],
        interpretation=_interpret_pvalue(res.p_value),
        figure=None,
    )


def run_cox_ph(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Cox proportional hazards.

    Body: ``{time_column, event_column, covariates: [colA, colB]}``.
    """
    from lifelines import CoxPHFitter  # type: ignore[import-untyped]

    time_col = params["time_column"]
    event_col = params["event_column"]
    covs: list[str] = list(params.get("covariates") or [])
    cols = [time_col, event_col, *covs]
    sub = df[cols].dropna()
    for c in covs:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col=time_col, event_col=event_col)
    summary = cph.summary
    coef_rows = []
    for term in summary.index:
        coef_rows.append(
            {
                "term": str(term),
                "coef": float(summary.loc[term, "coef"]),
                "hazard_ratio": float(summary.loc[term, "exp(coef)"]),
                "se": float(summary.loc[term, "se(coef)"]),
                "p_value": float(summary.loc[term, "p"]),
                "ci_low": float(summary.loc[term, "exp(coef) lower 95%"]),
                "ci_high": float(summary.loc[term, "exp(coef) upper 95%"]),
            }
        )
    return AnalysisResult(
        test_statistic=None,
        p_value=None,
        effect_size=float(cph.concordance_index_),
        effect_size_name="concordance",
        table=coef_rows,
        interpretation=(
            f"Concordance index = {cph.concordance_index_:.3g}. "
            f"Log-likelihood = {cph.log_likelihood_:.4g}."
        ),
        figure=None,
        extra={"concordance": float(cph.concordance_index_)},
    )


# ─────────────────────────────────────────── forecasting


def run_arima(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Box-Jenkins ARIMA point forecast.

    Body: ``{column, order: [p,d,q], horizon}``. ``order`` defaults
    to (1,1,1); ``horizon`` defaults to 10.
    """
    import statsmodels.api as sm

    col = params["column"]
    order = tuple(params.get("order") or (1, 1, 1))
    horizon = int(params.get("horizon", 10))
    y = _coerce_numeric(_require_column(df, col, "value"), col)
    model = sm.tsa.ARIMA(y, order=order).fit()
    forecast = model.get_forecast(steps=horizon)
    mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)
    fig_x = list(range(len(y))) + list(range(len(y), len(y) + horizon))
    fig_y_actual = list(map(float, y.tolist())) + [None] * horizon
    fig_y_pred = [None] * len(y) + list(map(float, mean.tolist()))
    return AnalysisResult(
        test_statistic=None,
        p_value=None,
        effect_size=float(model.aic),
        effect_size_name="aic",
        table=[
            {
                "step": i + 1,
                "forecast": float(mean.iloc[i]),
                "ci_low": float(ci.iloc[i, 0]),
                "ci_high": float(ci.iloc[i, 1]),
            }
            for i in range(horizon)
        ],
        interpretation=(
            f"ARIMA{order} fit. AIC = {model.aic:.4g}, BIC = {model.bic:.4g}."
        ),
        figure={
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": fig_x,
                    "y": fig_y_actual,
                    "name": "actual",
                    "line": {"color": "#3b82f6"},
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": fig_x,
                    "y": fig_y_pred,
                    "name": "forecast",
                    "line": {"color": "#f59e0b", "dash": "dash"},
                },
            ],
            "layout": {
                "yaxis": {"title": col},
                "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
            },
        },
        extra={"aic": float(model.aic), "bic": float(model.bic)},
    )


def run_prophet(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:
    """Prophet forecast.

    Body: ``{date_column, value_column, horizon_days?}``. Requires
    the optional ``prophet`` extra.
    """
    try:
        from prophet import Prophet  # type: ignore[import-not-found]
    except ImportError as exc:  # noqa: BLE001
        raise AnalysisProcedureError(
            "prophet not installed; pip install pycaret-server[prophet]"
        ) from exc

    date_col = params["date_column"]
    value_col = params["value_column"]
    horizon = int(params.get("horizon_days", 30))
    sub = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    sub["ds"] = pd.to_datetime(sub["ds"])
    sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
    sub = sub.dropna()
    m = Prophet()
    m.fit(sub)
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)
    fc_tail = forecast.tail(horizon)
    return AnalysisResult(
        test_statistic=None,
        p_value=None,
        table=[
            {
                "ds": row.ds.isoformat() if hasattr(row.ds, "isoformat") else str(row.ds),
                "yhat": float(row.yhat),
                "yhat_lower": float(row.yhat_lower),
                "yhat_upper": float(row.yhat_upper),
            }
            for row in fc_tail.itertuples()
        ],
        interpretation=f"Prophet {horizon}-day forecast.",
        figure={
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [str(d) for d in sub["ds"]],
                    "y": [float(v) for v in sub["y"]],
                    "name": "actual",
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [str(d) for d in forecast["ds"]],
                    "y": [float(v) for v in forecast["yhat"]],
                    "name": "forecast",
                    "line": {"color": "#f59e0b", "dash": "dash"},
                },
            ],
            "layout": {"margin": {"l": 60, "r": 20, "t": 20, "b": 50}},
        },
    )


# ─────────────────────────────────────────── interpretation helper


def _interpret_pvalue(p: float, alpha: float = 0.05) -> str:
    if p is None or (isinstance(p, float) and (p != p or p == float("inf"))):
        return "p-value not computable."
    if p < alpha:
        return f"Reject H0 at α={alpha} (p={p:.4g})."
    return f"Fail to reject H0 at α={alpha} (p={p:.4g})."
