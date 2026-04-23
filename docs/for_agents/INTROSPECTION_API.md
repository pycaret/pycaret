# Introspection API

`pycaret.api` is the typed, side-effect-free view of the engine. Every function returns dataclasses (or lists of them). Every dataclass has a `.to_dict()` that round-trips through `json.dumps` without a custom encoder.

Use this surface to:

- Populate a React dropdown of available models for a task.
- Generate a dynamic form for `fit`/`setup` parameters.
- Give an LLM agent structured context about what PyCaret can do.
- Auto-generate documentation.

## Functions

```python
from pycaret.api import (
    list_models,                 # (task) -> list[ModelCard]
    describe_model,              # (task, id) -> ModelCard
    list_metrics,                # (task) -> list[MetricCard]
    describe_setup_params,       # (task) -> SetupParamSchema
    list_available_models,       # (experiment) -> list[ModelCard]  (runtime-aware)
)
```

All `task` parameters accept either a `TaskType` enum value or a plain string (`"classification"`, `"regression"`, `"clustering"`, `"anomaly"`, `"time_series"`).

## `ModelCard`

```python
@dataclass(frozen=True)
class ModelCard:
    id: str                              # "lr", "rf", "xgboost", ...
    name: str                            # "Logistic Regression"
    task: str                            # "classification"
    description: str                     # one-line blurb
    library: str                         # "sklearn", "xgboost", "lightgbm", ...
    gpu_enabled: bool                    # has a GPU backend?
    is_turbo: bool                       # part of the default compare_models set?
    is_available: bool                   # True unless the backing package is missing
    hyperparameters: list[ParameterCard] # tunable parameters (populated on demand)
    tags: list[str]                      # ["linear", "probabilistic"], ...
```

## `MetricCard`

```python
@dataclass(frozen=True)
class MetricCard:
    id: str                              # "Accuracy", "AUC", "R2", ...
    name: str                            # "Area Under the Curve"
    task: str
    greater_is_better: bool
    description: str
    is_default: bool                     # the default optimizer metric?
    is_available: bool
```

## `ParameterCard`

```python
@dataclass(frozen=True)
class ParameterCard:
    name: str                            # the kwarg name as it appears in the signature
    kind: ParameterKind                  # render hint for a UI
    default: Any                         # JSON-safe default
    description: str
    choices: list[Any] | None            # populated when kind == ENUM
    minimum: float | None
    maximum: float | None
    required: bool
    group: str                           # UI grouping hint ("Preprocessing", ...)
```

## `ParameterKind`

```python
class ParameterKind(str, Enum):
    BOOL        = "bool"
    INT         = "int"
    FLOAT       = "float"
    STRING      = "string"
    ENUM        = "enum"       # one-of a fixed choice list
    LIST        = "list"
    COLUMN      = "column"     # autocomplete from the dataset's columns
    COLUMNS     = "columns"    # multi-select of dataset columns
    MODEL_ID    = "model_id"   # dropdown from list_models()
    METRIC_ID   = "metric_id"  # dropdown from list_metrics()
    UNKNOWN     = "unknown"    # fall back to freeform text
```

## `SetupParamSchema`

```python
@dataclass(frozen=True)
class SetupParamSchema:
    task: str
    parameters: list[ParameterCard]      # iterate to render a form
    groups: list[str]                    # ordered group names for tab layout
```

## Usage examples

### Build a model-picker dropdown

```python
from pycaret.api import list_models

models = list_models("classification")
options = [{"label": m.name, "value": m.id} for m in models if m.is_available]
```

### Render a `setup(...)` form

```python
from pycaret.api import describe_setup_params
import json

schema = describe_setup_params("classification")
json_for_react = json.dumps(schema.to_dict())
# Each parameters[i]["kind"] tells the UI what widget to render.
# Group by `group` for tab layout.
```

### Runtime-aware list (flags missing-library models)

```python
from pycaret.tasks import RegressionExperiment
from pycaret.api import list_available_models

exp = RegressionExperiment(target="y")
cards = list_available_models(exp)
missing = [c.name for c in cards if not c.is_available]
```

### Feed context to an LLM agent

```python
from pycaret.api import list_models, list_metrics, describe_setup_params
import json

task = "classification"
context = {
    "models": [m.to_dict() for m in list_models(task)],
    "metrics": [m.to_dict() for m in list_metrics(task)],
    "setup_params": describe_setup_params(task).to_dict(),
}
prompt = f"PyCaret {task} available options:\n{json.dumps(context, indent=2)}"
```
