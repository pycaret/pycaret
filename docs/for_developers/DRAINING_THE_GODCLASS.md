# Draining the god-class — the verb-migration playbook

The 4.0 engine has a clean public API but still delegates most verbs to the 3.x god-class in `pycaret/internal/pycaret_experiment/`. Phase 5 of the roadmap is progressively rewriting each verb natively on top of `sklearn.pipeline.Pipeline` + `sklearn.model_selection` and deleting the delegation call.

This is the playbook for picking up a single verb and migrating it.

## Pick a verb

Recommended order (easiest → hardest):

1. **`save_model` / `load_model`** — already stateless top-level; `Experiment.save_model` is a thin delegate. Replace with direct `joblib.dump` via the preprocess_pipeline wrap.
2. **`predict_model`** — given a fitted pipeline and new data, apply and return a DataFrame. No CV, no training.
3. **`create_model`** — train one estimator with cross-validation. Mostly a `cross_validate` call.
4. **`tune_model`** — wrap `GridSearchCV` / `HalvingRandomSearchCV` / `optuna.integration.OptunaSearchCV`.
5. **`ensemble_model`** — `BaggingClassifier` / `AdaBoostClassifier` wrap.
6. **`blend_models`** — `VotingClassifier` / `VotingRegressor` wrap.
7. **`stack_models`** — `StackingClassifier` / `StackingRegressor` wrap.
8. **`calibrate_model`** — `CalibratedClassifierCV` wrap.
9. **`compare_models`** — loop over the model registry calling `create_model` then rank. The heaviest (most user-customization to replicate).
10. **`finalize_model`** — refit pipeline on full data.

## Recipe

### 1. Write the native implementation

Find the verb on `Experiment` / `SupervisedExperiment` / `UnsupervisedExperiment`. Replace the body that calls `self._legacy.<verb>(...)` with a native implementation.

Before:

```python
def create_model(self, estimator, *args, **kwargs) -> CreateResult:
    self._require_fitted()
    t0 = time.perf_counter()
    self.logger.log(EventKind.MODEL_CREATE_STARTED, payload={"estimator": ...})
    model = self._legacy.create_model(estimator, *args, **kwargs)
    metrics = self._safe_pull()
    self.logger.log(EventKind.MODEL_CREATED, duration_ms=..., payload={...})
    return CreateResult(pipeline=model, model_id=..., metrics=metrics, params=...)
```

After:

```python
def create_model(self, estimator, *args, **kwargs) -> CreateResult:
    self._require_fitted()
    t0 = time.perf_counter()
    self.logger.log(EventKind.MODEL_CREATE_STARTED, payload={"estimator": ...})

    # --- native implementation ---
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import Pipeline

    estimator_obj = self._resolve_model_spec(estimator)
    pipeline = Pipeline([("prep", self.preprocess_pipeline), ("model", estimator_obj)])
    cv_results = cross_validate(
        pipeline, self.X_train, self.y_train,
        cv=self._resolve_cv(), scoring=self._resolve_scoring(),
        return_estimator=True, n_jobs=self.n_jobs,
    )
    fitted = cv_results["estimator"][-1]  # or refit on full train
    metrics = self._format_cv_metrics(cv_results)
    # --- end native implementation ---

    self.logger.log(EventKind.MODEL_CREATED, duration_ms=..., payload={...})
    return CreateResult(pipeline=fitted, model_id=..., metrics=metrics, params=...)
```

### 2. Keep the public API unchanged

- Function signature: identical.
- Return type: still `CreateResult`.
- Error raised on unfit: still `NotFittedError`.
- Events emitted: still `MODEL_CREATE_STARTED` / `MODEL_CREATED`.
- DataFrame shape in `CreateResult.metrics`: same columns, same rows, same dtypes.

### 3. Update the test

The existing e2e test in `tests/test_e2e_oop.py` already exercises the verb. It should continue passing without changes. If it doesn't, your implementation has a drift — fix the implementation, not the test.

Add at least one new test in `tests/test_core_architecture.py` for the unit-level shape:

```python
def test_create_model_returns_fitted_pipeline():
    from pycaret.datasets import get_data
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="Purchase").fit(get_data("juice"))
    result = exp.create_model("lr", cross_validation=False)
    assert result.pipeline is not None
    assert result.model_id == "lr"
    assert len(result.metrics) >= 1
```

### 4. Delete the legacy method

Once your native implementation is green on `tests/test_e2e_oop.py`:

1. Find the legacy method in `pycaret/internal/pycaret_experiment/supervised_experiment.py` (or wherever).
2. Delete the method body.
3. Run the test suite again — if anything still calls it, they should now call the new native path via the public API.

### 5. Document

- Release-notes entry: `CHANGED` + `INTERNAL` (external API unchanged). Describe the LOC dropped.
- Update `docs/revamp/STATUS.md` delta table.
- If the verb's semantics subtly changed (e.g. you now use a different CV default), flag in `DECISIONS.md`.

## Hard constraints

- **Never break the notebook golden path.** `setup → compare_models → predict_model` on `juice` must stay green at every commit.
- **Never change the public signature.**
- **Never change the `EventKind` emitted.**
- **Never add a print statement** inside the engine.

## When the god-class is empty

After all verbs are drained:

1. `pycaret/internal/pycaret_experiment/` contains only empty stub methods.
2. Delete the entire directory.
3. Delete `Experiment._legacy` and `_build_legacy_experiment()`.
4. Victory lap in `release_notes_pycaret4.md`. 🏁
