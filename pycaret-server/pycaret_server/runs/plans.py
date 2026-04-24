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

PlanName = Literal["setup", "create", "compare"]


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

    raise ValueError(f"unknown plan {plan!r}")


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
