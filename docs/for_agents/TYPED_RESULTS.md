# Typed results — every verb's return type

Every public `Experiment` verb returns a frozen dataclass. An agent / UI can destructure it without defensive `isinstance` checks.

All dataclasses live in `pycaret.core.results`. All have an `events: list[Event]` field (may be empty) and all are JSON-serializable except for the fitted pipeline (which carries its own pickling contract via joblib).

| Verb | Returns |
|---|---|
| `create_model(id, **kwargs)` | `CreateResult` |
| `compare_models(**kwargs)` | `CompareResult` |
| `tune_model(model, **kwargs)` | `TuneResult` |
| `ensemble_model(model, **kwargs)` | `EnsembleResult` |
| `blend_models(models, **kwargs)` | `BlendResult` |
| `stack_models(models, **kwargs)` | `StackResult` |
| `calibrate_model(model, **kwargs)` | `CalibrateResult` |
| `finalize_model(model)` | `FinalizeResult` |
| `predict_model(model, data=None)` | `PredictResult` |

## `CreateResult`

```python
@dataclass(frozen=True)
class CreateResult:
    pipeline: sklearn.pipeline.Pipeline   # the fitted preprocessor+estimator
    model_id: str                          # "lr", "rf", "xgboost", ...
    metrics: pd.DataFrame                  # CV metrics (rows = folds + mean/std)
    params: dict[str, Any]                 # estimator.get_params(deep=False)
    events: list[Event]
```

## `CompareResult`

```python
@dataclass(frozen=True)
class CompareResult:
    best: Pipeline                         # top-ranked fitted pipeline
    models: list[Pipeline]                 # top N in rank order (N = n_select)
    leaderboard: pd.DataFrame              # full score table; notebook-friendly
    ranked_ids: list[str]                  # pycaret model ids in rank order
    events: list[Event]
```

`CompareResult` implements `__iter__` and `__getitem__` on `models` so `for m in compare_result` and `compare_result[:3]` work like 3.x.

## `TuneResult`

```python
@dataclass(frozen=True)
class TuneResult:
    pipeline: Pipeline                     # the tuned, fitted pipeline
    best_params: dict[str, Any]            # winning hyperparameters
    search: BaseCrossValidator | None      # the underlying search object
    cv_results: pd.DataFrame               # the full CV grid (one row per fold×param)
    metrics: pd.DataFrame                  # summary metrics of the tuned model
    events: list[Event]
```

## `EnsembleResult` / `BlendResult` / `StackResult` / `CalibrateResult`

All identical shape:

```python
@dataclass(frozen=True)
class EnsembleResult:
    pipeline: Pipeline
    method: str                            # e.g. "Bagging", "Boosting", "sigmoid", "isotonic"
    metrics: pd.DataFrame
    events: list[Event]
```

(`BlendResult` and `StackResult` omit `method`.)

## `FinalizeResult`

```python
@dataclass(frozen=True)
class FinalizeResult:
    pipeline: Pipeline                     # refit on train+test; ready for deploy
    events: list[Event]
```

## `PredictResult`

```python
@dataclass(frozen=True)
class PredictResult:
    predictions: pd.DataFrame              # original columns + prediction_label + prediction_score (cls only)
    metrics: pd.DataFrame | None           # None when ground truth not supplied
    events: list[Event]
```

---

## Idioms

```python
# Classic 3.x call shape:
best = exp.compare_models().best

# The full leaderboard as a DataFrame:
leaderboard = exp.compare_models().leaderboard

# List of top-3 in rank order:
top3 = exp.compare_models(n_select=3).models

# Tune → predict:
tuned = exp.tune_model(best).pipeline
preds = exp.predict_model(tuned).predictions

# Get the event trace to render progress later:
events = exp.compare_models().events
```
