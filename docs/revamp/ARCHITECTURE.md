# PyCaret 4.0 — Engine Architecture

*Authored: 2026-04-22 (session 2) · Part of the PyCaret 4.0 revamp.*

## Why this exists

PyCaret 1.x was a 100% functional API, designed at a workshop pace. PyCaret 3.0 bolted an OOP layer on top without removing the functional plumbing. The result (in 3.4.0) is:

- `internal/pycaret_experiment/supervised_experiment.py` — **5,855 LOC god-class**.
- `internal/pycaret_experiment/tabular_experiment.py` — 2,894 LOC.
- `classification/functional.py` — 3,323 LOC mostly re-declared signatures.
- `classification/oop.py` — 3,446 LOC mostly re-declared signatures.
- Module-level global `_current_experiment` mutated by `setup()`.
- Nested `InternalPipeline` subclass that is *almost* a `sklearn.pipeline.Pipeline`.
- Mixed responsibilities: data loading, preprocessing, model training, metric calculation, plotting, logging, and serialization all collapsed into two base classes.

The user correctly called this "a college project." The 4.0 revamp replaces it with a **proper sklearn-composable engine** designed to power:

1. The legacy notebook golden path (`setup → compare_models → ...`) **unchanged at the call site**.
2. A forthcoming React UI that talks to the engine in-process.
3. LLM agents that introspect the engine and drive it programmatically.

## Core design principles

### 1. The engine is a `BaseEstimator`.

`Experiment` is a proper sklearn-compatible object — it implements `get_params`, `set_params`, `__sklearn_tags__`, `__sklearn_is_fitted__`. This is not cosmetic; it means:

- An `Experiment` can be pickled cleanly.
- An `Experiment` can be nested inside sklearn's `Pipeline` / `GridSearchCV` when that makes sense.
- Anyone who knows sklearn *already knows* 80% of the pycaret API.
- `clone(exp)` works; immutable config is preserved, fitted state is dropped.

### 2. `fit()` is the setup. `setup()` is a functional alias.

```python
# OOP (the real API):
exp = ClassificationExperiment(target="Purchase", session_id=42)
exp.fit(data)                 # runs preprocessing, splits, caches
best = exp.compare_models()   # returns a sklearn Pipeline

# Functional (thin adapter, unchanged notebook UX):
from pycaret.classification import setup, compare_models
exp = setup(data, target="Purchase", session_id=42)
best = compare_models()
```

The functional API uses a `contextvars.ContextVar` holding the current experiment. Thread-safe, async-safe, explicit — not a module-level global.

### 3. Every method returns a typed dataclass.

No more `pycaret.pull()` as the canonical way to get metrics. Each operation returns a dataclass like `CompareResult(models: list[Pipeline], leaderboard: DataFrame, best: Pipeline, events: list[Event])`. The DataFrame is still there for notebook users — it's a property. Agents and the UI consume the structured fields.

### 4. Build on `sklearn.pipeline.Pipeline`, don't replace it.

`create_model` returns a `sklearn.pipeline.Pipeline` with the preprocessor + fitted estimator. `tune_model`, `ensemble_model`, `calibrate_model` all return the same shape. `predict_model(pipeline, X)` is `pipeline.predict(X)` plus output formatting. No custom pipeline class.

### 5. Preprocessing is a `ColumnTransformer` + `Pipeline`, not a bespoke graph.

`PreprocessorBuilder` composes sklearn's own transformers (imputers, encoders, scalers, feature selectors). The custom pycaret pieces (iterative imputer adaptations, rare-category encoder) are single-purpose transformers that follow the sklearn protocol, not god-class methods.

### 6. Tuning uses canonical sklearn searches + optuna, nothing custom.

`GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`, `HalvingRandomSearchCV`, `optuna.integration.OptunaSearchCV`. The `Tuner` abstraction picks one and drives it. No reimplementation of CV loops.

### 7. Logging is an event stream, not a tracker adapter.

`pycaret.logging` emits structured `Event` dataclasses (`ExperimentStarted`, `PreprocessorFitted`, `ModelCompared`, `ModelTuned`, …) through a `BaseLogger` interface. The default logger is in-memory + file; the React UI will consume the same stream over websocket. mlflow/comet/wandb are not mentioned anywhere in the core.

### 8. No prints. No interactive input. No hidden state.

Every long-running operation emits events that a UI renders as progress. The engine never writes to stdout directly.

## New package layout

```
pycaret/
├── __init__.py                  # version, public re-exports
├── core/                        # NEW — engine primitives
│   ├── __init__.py
│   ├── experiment.py            # Experiment (BaseEstimator subclass)
│   ├── results.py               # CompareResult, CreateResult, TuneResult, PredictResult, etc.
│   ├── tasks.py                 # TaskType enum
│   ├── state.py                 # ContextVar for current-experiment (functional API)
│   └── errors.py                # PyCaretError hierarchy
├── api/                         # NEW — introspection surface for UI + agents
│   ├── __init__.py
│   ├── cards.py                 # ModelCard, MetricCard, ParameterCard
│   ├── schemas.py               # SetupParamSchema (drives React dynamic forms)
│   └── describe.py              # list_models(task), describe_model(id), ...
├── logging/                     # NEW — event-stream logger
│   ├── __init__.py
│   ├── base.py                  # BaseLogger (no-op)
│   ├── events.py                # Event dataclasses
│   └── memory.py                # In-memory + file-backed logger, UI-ready
├── tasks/                       # NEW — task-specific Experiment subclasses
│   ├── __init__.py
│   ├── classification.py        # ClassificationExperiment
│   ├── regression.py            # (future session)
│   ├── clustering.py            # (future session)
│   ├── anomaly.py               # (future session)
│   └── time_series.py           # (future session)
├── classification/              # PRESERVED — thin adapter over tasks.classification
│   ├── __init__.py
│   ├── functional.py            # setup(), compare_models(), ... delegate to ContextVar
│   └── oop.py                   # re-exports ClassificationExperiment
├── regression/                  # same pattern (session 3+)
├── clustering/
├── anomaly/
├── time_series/
├── internal/                    # LEGACY — kept until operations are migrated
│   ├── pycaret_experiment/      # god-class, delegated-to during transition
│   ├── preprocess/
│   └── ...
├── containers/                  # LEGACY — model/metric registries, migrating to pycaret/models + pycaret/metrics
├── datasets.py
└── utils/
```

**The key insight for this multi-session migration:** `pycaret.tasks.ClassificationExperiment` wraps an instance of the legacy `pycaret.internal.pycaret_experiment.supervised_experiment._SupervisedExperiment` during the transition. Each verb (`compare_models`, `tune_model`, ...) starts as a thin delegation (`return self._legacy.compare_models(...)`) and is progressively rewritten in-place. The public API never breaks; the god-class is drained one method at a time.

## Interface contracts

### `Experiment(BaseEstimator)`

```python
class Experiment(BaseEstimator):
    # Configuration — all `__init__` parameters are stored verbatim.
    # No preprocessing, no data loading, no side effects during construction.
    def __init__(self, *, task, target=None, session_id=42, fold=10, ...): ...

    # Sklearn-canonical fit — runs setup, splits, preprocessing.
    def fit(self, X, y=None) -> "Experiment": ...

    # Returns raw transformed data — useful for introspection and UIs.
    def transform(self, X) -> pd.DataFrame: ...

    # Operations — each returns a typed result dataclass.
    def compare_models(self, **kwargs) -> CompareResult: ...
    def create_model(self, model_id, **kwargs) -> CreateResult: ...
    def tune_model(self, pipeline, **kwargs) -> TuneResult: ...
    def ensemble_model(self, pipeline, **kwargs) -> EnsembleResult: ...
    def blend_models(self, pipelines, **kwargs) -> BlendResult: ...
    def stack_models(self, pipelines, **kwargs) -> StackResult: ...
    def calibrate_model(self, pipeline, **kwargs) -> CalibrateResult: ...
    def finalize_model(self, pipeline) -> Pipeline: ...
    def predict_model(self, pipeline, data=None) -> PredictResult: ...
    def plot_model(self, pipeline, kind, **kwargs) -> "plotly.graph_objects.Figure": ...

    # Persistence
    def save_model(self, pipeline, path) -> None: ...
    @staticmethod
    def load_model(path) -> Pipeline: ...

    # Introspection (also exposed as module-level functions under pycaret.api)
    def list_models(self) -> list[ModelCard]: ...
    def list_metrics(self) -> list[MetricCard]: ...
    def describe_model(self, model_id) -> ModelCard: ...
    def describe_setup_params(self) -> SetupParamSchema: ...

    # Config access for notebook users (replaces get_config/set_config pattern)
    @property
    def X_train(self) -> pd.DataFrame: ...
    @property
    def X_test(self) -> pd.DataFrame: ...
    @property
    def y_train(self) -> pd.Series: ...
    @property
    def y_test(self) -> pd.Series: ...
    @property
    def pipeline(self) -> Pipeline: ...     # the fitted preprocessor
    @property
    def logger(self) -> BaseLogger: ...
    @property
    def events(self) -> list[Event]: ...    # the event stream, for UI replay

    # Sklearn tag surface
    def __sklearn_tags__(self) -> Tags: ...
    def __sklearn_is_fitted__(self) -> bool: ...
```

### `ClassificationExperiment(Experiment)`

```python
class ClassificationExperiment(Experiment):
    def __init__(self, *, target=None, ...):
        super().__init__(task=TaskType.CLASSIFICATION, target=target, ...)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
```

Same pattern for `RegressionExperiment`, `ClusteringExperiment`, `AnomalyExperiment`, `TimeSeriesExperiment`.

### Result dataclasses (`pycaret.core.results`)

All frozen dataclasses. All JSON-serializable (except the fitted pipeline, which carries its own pickling). Fields that every UI/agent cares about are first-class.

```python
@dataclass(frozen=True)
class CompareResult:
    best: Pipeline                             # top-ranked fitted pipeline
    models: list[Pipeline]                     # top-N by score, in rank order
    leaderboard: pd.DataFrame                  # score table, notebook-friendly
    ranked_ids: list[str]                      # ordered pycaret model ids
    events: list[Event]                        # the per-model timings + scores

@dataclass(frozen=True)
class TuneResult:
    pipeline: Pipeline                         # the tuned fitted pipeline
    params: dict                               # best params found
    search: BaseSearchCV                       # the underlying search object, for power users
    cv_results: pd.DataFrame                   # the full CV grid
    events: list[Event]
```

### Event stream (`pycaret.logging.events`)

```python
@dataclass(frozen=True)
class Event:
    kind: str                                  # "experiment.started", "model.created", ...
    timestamp: datetime
    duration_ms: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
```

Concrete kinds include `experiment.started`, `preprocessor.fitted`, `model.created`, `model.compared`, `model.tuned`, `model.predicted`, `error.raised`. The React UI subscribes to the stream and renders progress; LLM agents reason over the events as trace data.

### Introspection surface (`pycaret.api`)

```python
# Zero-argument introspection — works without an experiment, for static docs
def list_models(task: TaskType, include_extras: bool = True) -> list[ModelCard]: ...
def describe_model(task: TaskType, model_id: str) -> ModelCard: ...
def list_metrics(task: TaskType) -> list[MetricCard]: ...
def describe_setup_params(task: TaskType) -> SetupParamSchema: ...

# With an experiment — runtime state included
def list_available_models(experiment: Experiment) -> list[ModelCard]: ...  # filters by installed extras
```

`ModelCard`, `MetricCard`, `ParameterCard`, `SetupParamSchema` are serializable dataclasses that carry enough structure for a React form to render them directly (field types, enums, ranges, dependencies).

## What lands when

| Layer | Session 2 | Session 3+ |
|---|---|---|
| `ARCHITECTURE.md` design doc | ✅ written | — |
| `pycaret/core/` primitives | ✅ skeleton | grow as verbs are migrated |
| `pycaret/api/` introspection | ✅ implemented for classification | extend to all tasks |
| `pycaret/logging/` event stream | ✅ implemented | real UI consumption tests |
| `pycaret/tasks/classification.py` | ✅ delegating implementation | progressively absorb god-class methods |
| `pycaret/classification/` functional + oop | ✅ thinned to adapter | — |
| Regression/clustering/anomaly/time_series tasks | — | session 3 |
| Preprocessor rewrite as native `ColumnTransformer` | — | session 4 |
| Full god-class retirement | — | session 5+ |

## Non-goals for this session

- We are **not** deleting `internal/pycaret_experiment/` yet. Its existence is what lets the golden path keep working while the new surface grows.
- We are **not** rewriting preprocessing, metric, or model containers yet.
- We are **not** implementing Phase 3 (Plotly plot rewrite) yet.
