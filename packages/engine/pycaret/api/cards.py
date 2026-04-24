"""Serializable descriptors for models, metrics, and parameters.

These dataclasses are the primary contract between the PyCaret engine and
anything that needs to describe its capabilities to a human (via a React form,
a help text, a tutorial generator) or to an LLM (as structured context).

All fields are plain-Python types — strings, numbers, lists, dicts — so a
`dataclasses.asdict(card)` call round-trips through `json.dumps` without any
custom encoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ParameterKind(StrEnum):
    """What kind of widget a React UI should render for this parameter."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    ENUM = "enum"  # one-of a fixed choice list
    LIST = "list"  # free-form list
    COLUMN = "column"  # a column of the dataset (autocomplete)
    COLUMNS = "columns"  # a multi-select of columns
    MODEL_ID = "model_id"  # dropdown populated from list_models()
    METRIC_ID = "metric_id"  # dropdown populated from list_metrics()
    UNKNOWN = "unknown"  # last-resort; render as freeform text


@dataclass(frozen=True)
class ParameterCard:
    """Describes a single named parameter (e.g. `setup(normalize=...)` or a
    model hyperparameter).

    Attributes
    ----------
    name : str
        The parameter name as it appears in the function signature.
    kind : ParameterKind
        The render hint.
    default : Any
        JSON-safe default value (may be `None`).
    description : str
        Short human-readable blurb.
    choices : list, optional
        Populated when `kind == ENUM`.
    minimum / maximum : number, optional
        Populated for bounded numeric parameters.
    required : bool
        If True, the UI must collect a value before invoking the operation.
    group : str, optional
        Section label for grouping related parameters in the UI (e.g.
        "Preprocessing", "Training", "Tuning").
    """

    name: str
    kind: ParameterKind
    default: Any = None
    description: str = ""
    choices: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    required: bool = False
    group: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass(frozen=True)
class ModelCard:
    """Describes a single model available to a PyCaret experiment.

    Attributes
    ----------
    id : str
        The pycaret model id (e.g. "lr", "rf", "xgboost").
    name : str
        Human-readable name (e.g. "Logistic Regression").
    task : str
        The task this model is registered for ("classification", "regression", ...).
    description : str
        Short blurb pointing at the underlying estimator and its strengths.
    library : str
        The upstream package providing the estimator ("sklearn", "xgboost",
        "lightgbm", "catboost", "sktime", ...).
    gpu_enabled : bool
        True if the container can route to a GPU-accelerated backend.
    is_turbo : bool
        True if the model is included in a small/fast `compare_models` default set.
    is_available : bool
        False when the model's package is not installed in the current env.
    hyperparameters : list[ParameterCard]
        Tunable hyperparameters. Populated on demand by `describe_model`.
    tags : list[str]
        Free-form labels for filtering (e.g. "linear", "tree", "boosted",
        "probabilistic").
    """

    id: str
    name: str
    task: str
    description: str = ""
    library: str = ""
    gpu_enabled: bool = False
    is_turbo: bool = True
    is_available: bool = True
    hyperparameters: list[ParameterCard] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hyperparameters"] = [p.to_dict() for p in self.hyperparameters]
        return d


@dataclass(frozen=True)
class MetricCard:
    """Describes a single metric available in a task.

    Attributes
    ----------
    id : str
        The pycaret metric id (e.g. "Accuracy", "AUC", "R2").
    name : str
        Human-readable name.
    task : str
    greater_is_better : bool
        If True, higher values are better (used by leaderboards / optimizers).
    description : str
    is_default : bool
        If True, this is the default optimizer metric for its task.
    is_available : bool
        False when the metric requires an optional package that isn't installed.
    """

    id: str
    name: str
    task: str
    greater_is_better: bool = True
    description: str = ""
    is_default: bool = False
    is_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
