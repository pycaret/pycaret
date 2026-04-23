# Engine walkthrough — `fit → compare_models → predict_model`

What actually happens when you run the canonical 4 lines. Written for an agent generating code or explaining PyCaret to a user.

```python
from pycaret.tasks import ClassificationExperiment
exp = ClassificationExperiment(target="y", session_id=42).fit(df)
result = exp.compare_models()
preds = exp.predict_model(result.best)
```

---

## `ClassificationExperiment(target="y", ...)`

1. `ClassificationExperiment.__init__` calls `super().__init__(task=TaskType.CLASSIFICATION, target="y", ...)` and `Experiment.__init__` stores every parameter verbatim onto `self`. No work happens.
2. sklearn convention: `get_params()` will now return all 15 parameters; `clone(exp)` will produce a fresh instance with the same config and no fitted state.
3. `exp.__sklearn_is_fitted__()` returns `False`.

## `exp.fit(df)`

`fit` is defined on `Experiment` and does these things in order:

1. **Normalize call shape.** Supervised tasks: either a DataFrame containing the target column, or `(X, y)`. `_coerce_supervised_fit_inputs` merges them into a single DataFrame `data`.
2. **Generate an experiment_id.** A `uuid4` string stamped on every emitted event.
3. **Build the legacy delegate.** `self._legacy = self._build_legacy_experiment()` — for `ClassificationExperiment` this is an instance of the 3.x `_NonTSSupervisedExperiment`-backed god-class.
4. **Install a logger** if one wasn't provided. `log_experiment=True` → `MemoryLogger`; else `NullLogger`.
5. **Emit `EventKind.EXPERIMENT_STARTED`.**
6. **Call `self._legacy.setup(...)`** with a kwarg mapping produced by `_build_legacy_setup_kwargs()`. This is where preprocessing, train/test split, and metric registration actually happen (inside the god-class during the transition period).
7. **Set `self._fitted = True`.**
8. **Emit `EventKind.EXPERIMENT_FITTED`** with the wall-clock duration.
9. **Return `self`** — chainable.

After fit, these properties are populated by delegation through `self._legacy`:
`X`, `X_train`, `X_test`, `y`, `y_train`, `y_test`, `preprocess_pipeline`.

## `result = exp.compare_models(...)`

Defined on `SupervisedExperiment`:

1. **Raise `NotFittedError`** if the experiment is unfitted.
2. **Emit `EventKind.MODEL_COMPARE_STARTED`.**
3. **Call `self._legacy.compare_models(...)`** — the god-class runs cross-validation on every turbo model, ranks them, and returns a list (or single model if `n_select=1`).
4. **Pull the leaderboard** via `self._safe_pull()` — this reads the DataFrame the legacy engine stored as its most recent output.
5. **Emit `EventKind.MODEL_COMPARE_FINISHED`** with the duration and `n_select`.
6. **Wrap into `CompareResult`** with `best`, `models`, `leaderboard`, `ranked_ids`, `events`.

`CompareResult` implements `__iter__` and `__getitem__` so the 3.x `top3 = compare_models(n_select=3)` list-indexing idiom still works: `top3 = exp.compare_models(n_select=3)[:3]`.

## `preds = exp.predict_model(model)`

Defined on `Experiment` (task-agnostic):

1. **Raise `NotFittedError`** if unfitted.
2. **Call `self._legacy.predict_model(...)`** — the god-class applies the fitted preprocessor, runs `model.predict(X_test)` (and `predict_proba` for classification), and assembles the DataFrame.
3. **Pull scoring metrics** if the test data had ground truth.
4. **Emit `EventKind.MODEL_PREDICTED`** with the row count.
5. **Wrap into `PredictResult`** with `predictions` (DataFrame with `prediction_label` / `prediction_score` columns) and `metrics` (DataFrame or `None`).

---

## Where the legacy god-class lives

`pycaret/internal/pycaret_experiment/supervised_experiment.py` (~5,800 LOC) and `tabular_experiment.py` (~2,900 LOC) hold the current `setup`, `compare_models`, `tune_model`, `predict_model` implementations. `Experiment._legacy` points at one of these.

**Phase 5** of the revamp (see `docs/revamp/ROADMAP.md`) is *draining* this god-class: each verb is progressively rewritten natively on top of `sklearn.pipeline.Pipeline` + `sklearn.model_selection`, the delegation call in `core/experiment.py` or `core/supervised.py` is replaced with the native body, and the corresponding legacy method is deleted.

The public API never changes during this migration. A user writing against PyCaret 4.0 today will see identical behavior after the god-class is fully drained.

---

## Why the transition pattern works

- **No migration debt** — the public API is already the final one.
- **Incremental risk** — one verb at a time; the test suite catches regressions.
- **Every intermediate commit is shippable** — the notebook golden path runs green at every step.
- **Drain order is flexible** — agents can pick up any verb; they don't have to go in order.

See `docs/for_developers/DRAINING_THE_GODCLASS.md` for the step-by-step playbook.
