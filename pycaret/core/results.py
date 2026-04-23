"""Typed result objects returned from every Experiment operation.

Every PyCaret 4.0 verb (`compare_models`, `create_model`, `tune_model`, ...)
returns a frozen dataclass. This gives us three things simultaneously:

1. **Notebook compatibility** — the result's `.pipeline` (or `.models` /
   `.best`) is still a sklearn Pipeline you can use directly, and the `pull()`
   functional still returns the leaderboard DataFrame.
2. **Agent / UI friendliness** — all fields are introspectable, typed, and
   serializable (except the sklearn Pipeline itself, which carries its own
   pickling contract via joblib/cloudpickle).
3. **Event trace** — every result carries the `Event` list produced during
   its operation so a UI can replay progress after the fact.

Fields that a React UI / LLM agent cares about are first-class; convenience
fields (DataFrames) are derived or side-channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.pipeline import Pipeline

    from pycaret.logging.events import Event


# Each result holds `events` as a list — populated by the operation itself.
# We use `field(default_factory=list)` so the dataclass is still usable even
# when events were not recorded (e.g. unit tests constructing a result by hand).


@dataclass(frozen=True)
class CreateResult:
    """Returned by `Experiment.create_model(...)`.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted preprocessing + estimator pipeline. Return value of the
        legacy `create_model(...)` call in 3.x for drop-in compatibility.
    model_id : str
        The pycaret model id that was created (e.g. "lr", "rf").
    metrics : pandas.DataFrame
        The cross-validated metrics table for this fit (one row per fold +
        a mean / std row).
    params : dict
        The estimator's fitted parameters (from `get_params()`), handy for
        logging/UI display.
    events : list[Event]
    """

    pipeline: Pipeline
    model_id: str
    metrics: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)


@dataclass(frozen=True)
class CompareResult:
    """Returned by `Experiment.compare_models(...)`."""

    best: Pipeline
    models: list[Pipeline]
    leaderboard: pd.DataFrame
    ranked_ids: list[str]
    events: list[Event] = field(default_factory=list)

    def __iter__(self):  # pragma: no cover — ergonomic: `for m in result`
        return iter(self.models)

    def __getitem__(self, idx):  # pragma: no cover — ergonomic slicing
        return self.models[idx]


@dataclass(frozen=True)
class TuneResult:
    """Returned by `Experiment.tune_model(...)`."""

    pipeline: Pipeline
    best_params: dict[str, Any]
    search: BaseCrossValidator | None  # the underlying search object
    cv_results: pd.DataFrame
    metrics: pd.DataFrame
    events: list[Event] = field(default_factory=list)


@dataclass(frozen=True)
class EnsembleResult:
    """Returned by `Experiment.ensemble_model(...)` (bagging / boosting)."""

    pipeline: Pipeline
    method: str  # "Bagging" or "Boosting"
    metrics: pd.DataFrame
    events: list[Event] = field(default_factory=list)


@dataclass(frozen=True)
class BlendResult:
    """Returned by `Experiment.blend_models(...)`."""

    pipeline: Pipeline
    metrics: pd.DataFrame
    events: list[Event] = field(default_factory=list)


@dataclass(frozen=True)
class StackResult:
    """Returned by `Experiment.stack_models(...)`."""

    pipeline: Pipeline
    metrics: pd.DataFrame
    events: list[Event] = field(default_factory=list)


@dataclass(frozen=True)
class CalibrateResult:
    """Returned by `Experiment.calibrate_model(...)`."""

    pipeline: Pipeline
    method: str  # "sigmoid" or "isotonic"
    metrics: pd.DataFrame
    events: list[Event] = field(default_factory=list)


@dataclass(frozen=True)
class FinalizeResult:
    """Returned by `Experiment.finalize_model(...)` — the full-data refit."""

    pipeline: Pipeline
    events: list[Event] = field(default_factory=list)


@dataclass(frozen=True)
class PredictResult:
    """Returned by `Experiment.predict_model(...)`.

    The notebook-familiar return shape is `predictions` — a DataFrame with the
    original columns plus `prediction_label` and (for classification) a
    `prediction_score` column.
    """

    predictions: pd.DataFrame
    metrics: pd.DataFrame | None = None  # None when ground truth is not provided
    events: list[Event] = field(default_factory=list)
