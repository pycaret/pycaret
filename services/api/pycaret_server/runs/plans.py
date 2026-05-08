"""Run "plans" — named sequences of engine-verb calls.

A `RunPlan` tells the orchestrator what to actually do after `.fit(data)`:

- ``"setup"``   — just fit; no model training.
- ``"create"``  — fit + `create_model(model_id, **plan_params)`.
- ``"compare"`` — fit + `compare_models(**plan_params)`; returns leaderboard.

Each plan returns a `PlanOutcome` dataclass that the orchestrator converts
into DB rows (leaderboard on Run, artifact rows, fold metrics).

Kept separate from `orchestrator.py` so unit tests can exercise plans against
an in-memory experiment without a threadpool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pycaret.core.experiment import Experiment

PlanName = Literal["setup", "create", "compare", "search"]


@dataclass
class PlanOutcome:
    """What a plan returns to the orchestrator."""

    leaderboard: pd.DataFrame | None = None
    best_model: Any | None = None  # the fitted estimator, if any
    extra: dict[str, Any] = field(default_factory=dict)


def execute_plan(
    exp: Experiment,
    plan: PlanName,
    *,
    model_id: str | None = None,
    plan_params: dict[str, Any] | None = None,
) -> PlanOutcome:
    """Run the named plan against an already-fit `Experiment`. Returns a PlanOutcome."""
    params = dict(plan_params or {})

    if plan == "setup":
        return PlanOutcome()

    if plan == "create":
        if not model_id:
            raise ValueError("plan='create' requires a model_id")
        result = exp.create_model(model_id, **params)
        return PlanOutcome(
            leaderboard=getattr(result, "metrics", None),
            best_model=getattr(result, "pipeline", None),
        )

    if plan == "compare":
        result = exp.compare_models(**params)
        return PlanOutcome(
            leaderboard=getattr(result, "leaderboard", None),
            best_model=getattr(result, "best", None),
            extra={"ranked_ids": list(getattr(result, "ranked_ids", []) or [])},
        )

    if plan == "search":
        raise RuntimeError(
            "plan='search' must be handled by the orchestrator (it iterates "
            "preprocessing variants by re-fitting the experiment)."
        )

    raise ValueError(f"unknown plan {plan!r}")


def execute_search_plan(
    *,
    build_exp,
    df: pd.DataFrame,
    target: str | None,
    base_setup_params: dict[str, Any],
    plan_params: dict[str, Any] | None,
    checkpoint=lambda: None,
) -> PlanOutcome:
    """Iterate preprocessing variants, run ``compare_models`` for each, aggregate.

    ``plan_params`` keys:
      * ``variants``: list of dicts. Each is a ``setup_params`` *override*
        merged on top of ``base_setup_params``. If empty, defaults to
        ``[{}, {"normalize": True}, {"normalize": True, "transformation": True}]``.
      * ``compare_params``: dict passed to ``compare_models`` for every variant.
      * ``optimize``: metric name passed as ``compare_models(sort=...)``.

    Returns a single ``PlanOutcome`` whose ``leaderboard`` is the
    concatenation of every variant's leaderboard with a ``Variant`` column,
    and whose ``best_model`` is the globally-best fitted pipeline.
    """
    params = dict(plan_params or {})
    variants_in = params.get("variants")
    if not variants_in:
        variants = [
            {},
            {"normalize": True},
            {"normalize": True, "transformation": True},
        ]
    else:
        variants = list(variants_in)

    compare_params = dict(params.get("compare_params") or {})
    if "optimize" in params and "sort" not in compare_params:
        compare_params["sort"] = params["optimize"]

    aggregated: list[pd.DataFrame] = []
    best_overall = None
    best_score: float | None = None
    best_lower_is_better = False

    for idx, override in enumerate(variants):
        checkpoint()
        merged = {**base_setup_params, **(override or {})}
        exp = build_exp(merged, target)
        exp.fit(df)
        result = exp.compare_models(**compare_params)
        lb = getattr(result, "leaderboard", None)
        if lb is None or lb.empty:
            continue
        lb = lb.copy()
        lb.insert(0, "Variant", idx)
        aggregated.append(lb)

        candidate = getattr(result, "best", None)
        if candidate is None:
            continue
        # Use the first numeric column as the ranking key (mirrors
        # ``CompareResult.leaderboard`` ordering, which is already sorted by
        # the optimize metric).
        first_row = lb.iloc[0]
        sort_col = compare_params.get("sort") or _first_metric_col(lb)
        if sort_col not in lb.columns:
            continue
        score = float(first_row[sort_col])
        lower_is_better = sort_col.lower() in {"mae", "mse", "rmse", "rmsle", "mape"}
        if best_overall is None or _is_better(
            score, best_score, lower_is_better
        ):
            best_overall = candidate
            best_score = score
            best_lower_is_better = lower_is_better

    if not aggregated:
        return PlanOutcome()

    final = pd.concat(aggregated, ignore_index=True)
    sort_col = compare_params.get("sort") or _first_metric_col(final)
    if sort_col in final.columns:
        final = final.sort_values(
            by=sort_col, ascending=best_lower_is_better
        ).reset_index(drop=True)

    return PlanOutcome(
        leaderboard=final,
        best_model=best_overall,
        extra={
            "n_variants": len(variants),
            "best_score": best_score,
            "sort_metric": sort_col,
        },
    )


def _first_metric_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c in ("Variant", "Model"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return "Model"


def _is_better(score: float, current_best: float | None, lower_is_better: bool) -> bool:
    if current_best is None:
        return True
    return score < current_best if lower_is_better else score > current_best


# -------------------------------------------------------------- data loaders


def load_sklearn_dataset(name: str) -> tuple[pd.DataFrame, str]:
    """Return (df, target_col) for a tiny built-in sklearn dataset.

    These are handy for tests + tutorials because they don't require network
    access. Classification: iris, wine, breast_cancer. Regression: diabetes.
    """
    import sklearn.datasets as sk

    loaders = {
        "iris": sk.load_iris,
        "wine": sk.load_wine,
        "breast_cancer": sk.load_breast_cancer,
        "diabetes": sk.load_diabetes,
    }
    if name not in loaders:
        raise ValueError(f"unknown sklearn dataset {name!r}; pick from {list(loaders)}")
    bundle = loaders[name](as_frame=True)
    df = bundle.frame.copy()
    target = bundle.target.name if hasattr(bundle.target, "name") else "target"
    if target not in df.columns:
        df[target] = bundle.target
    return df, target


def load_inline(rows: list[dict]) -> pd.DataFrame:
    """Convert an inline list-of-dicts into a DataFrame."""
    if not rows:
        raise ValueError("data_inline must contain at least one row")
    return pd.DataFrame.from_records(rows)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV uploaded via a DataSource. Path comes from DataSource.config."""
    import pathlib

    p = pathlib.Path(path)
    if not p.is_file():
        raise ValueError(f"uploaded CSV no longer exists at {path!r}")
    return pd.read_csv(p)
