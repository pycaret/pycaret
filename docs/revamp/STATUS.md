# PyCaret 4.0 Revamp — Status

*Updated: 2026-04-25, end of session 34*

## Session 34 — Fix sklearn 1.6+ `squared=` deprecation in regression metrics — ✅

After session 33 closed the public-surface drain, the test output was still flooded with ~70 `DeprecationWarning`s per regression run because the legacy `RMSEMetricContainer` passed `args={"squared": False}` to `mean_squared_error` — sklearn 1.6+ deprecated that path in favor of a dedicated `root_mean_squared_error`. Session 34 fixes the registry to use the new function when available + falls back to a `sqrt(mse)` shim for older sklearn. Same fix pattern applied to `RMSLEMetricContainer`.

### What landed

- **`packages/engine/pycaret/containers/metrics/regression.py`**:
  - `RMSEMetricContainer` — `score_func = metrics.root_mean_squared_error` when available; otherwise a local `rmse_func` shim that calls `sqrt(mean_squared_error(...))`. `args` no longer carries `squared`.
  - `RMSLEMetricContainer` — prefers `metrics.root_mean_squared_log_error` when available; otherwise the existing `sqrt(msle)` shim with `np.abs()` for negative-input safety.
- **`packages/engine/tests/test_session34_metric_warnings.py`** — 4 tests:
  - RMSE's `score_func` is *not* the deprecated `mean_squared_error` reference.
  - RMSE's `args` dict no longer carries `squared`.
  - Calling RMSE's `score_func` with sklearn ≥1.8-shaped kwargs succeeds without `DeprecationWarning`.
  - Full regression CV via `create_model` doesn't emit any deprecation warnings.

### Headline metrics

| | Session 33 end | Session 34 end |
|---|---|---|
| Engine tests (fast + slow) | 141 | **145** (+4) |
| **Combined tests** | **287** | **291** |
| Test-output warnings | ~500 | **~431** (-69, eliminated all `squared=`) |
| Real bugfixes shipped during drain | 2 (add_metric, set_config) | **3** (+ RMSE deprecation) |

### Side note: 4.0.0a2 release

Still pending PyPI Trusted Publishing config (user account reset still in progress).

---

## Session 33 — `get_config` / `set_config` drain — ✅

Last drainable secondary verbs done. **Every drainable verb on the public surface is now native.** What remains on `_legacy`:

- `setup()` inside `fit()` — biggest piece, requires native preprocessing chain (3-5 sessions, post-release).
- `plot_model` / `evaluate_model` — Phase 3 Plotly-native rewrite (separate timeline).

### What landed

- **`Experiment.get_config(variable=None)`** — drained. Reads from `self._fit_state` (data accessors, transformed splits, registries) + constructor params on `self`. `get_config(None)` returns the sorted list of accessible names. Supports legacy aliases: `seed` → `session_id`, `pipeline` → `preprocess_pipeline`. Raises `ValueError` for unknown names.
- **`Experiment.set_config(variable=None, value=None, **kwargs)`** — drained with a deliberately tight allowlist. `_SETTABLE_CONFIG_KEYS` = `{session_id, n_jobs, verbose, fold, log_experiment}`. Anything else raises `ValueError` with a pointer to re-creating the Experiment instead. Both single (`set_config("n_jobs", 4)`) and bulk (`set_config(n_jobs=4, verbose=True)`) call shapes work; mixing them raises.
- **Why the tight allowlist on `set_config`**: in 3.x, you could mutate `target` / `train_size` post-fit but the changes wouldn't propagate to the snapshot. That's a real footgun. The 4.0 version refuses + tells the user what to do instead. `session_id` / `n_jobs` / `verbose` / `fold` are safe because they only affect new model creations, not the existing snapshot.
- **`packages/engine/tests/test_session33_config.py`** — 10 tests:
  - `get_config(None)` returns the full names list (drain-locked against `legacy.get_config`).
  - `get_config("X_train")` returns the snapshot reference (object identity).
  - `seed` / `pipeline` aliases work.
  - Unknown variable → `ValueError`.
  - `set_config("n_jobs", 4)` mutates (drain-locked).
  - Bulk `set_config(verbose=True, fold=5)` works.
  - Non-settable param → `ValueError` with helpful message.
  - Underscore-prefixed → `ValueError`.
  - Mixing positional + kwargs → `ValueError`.
  - Both raise `NotFittedError` pre-fit.

### Headline metrics

| | Session 32 end | Session 33 end |
|---|---|---|
| Drainable secondary verbs still on `_legacy` | 2 (`get_config`, `set_config`) | **0** ✅ |
| Engine tests (fast + slow) | 131 | **141** (+10) |
| **Combined tests** | **277** | **287** |

### Drain status — comprehensive

```
✅ All 16 modeling verbs            (sessions 22-28)
✅ User-facing data accessors       (session 29)
✅ Internal training state          (session 30)
✅ pull / models / get_metrics      (session 31)
✅ add_metric / remove_metric       (session 32)
✅ get_config / set_config          (session 33)  ← this session
```

### What remains (for `4.0.0` non-alpha)

- **`setup()` drain** — replace `self._legacy.setup(...)` inside `fit()` with a native preprocessing chain. ~3-5 sessions of work.
- **`plot_model` / `evaluate_model`** — Phase 3 roadmap (separate workstream — Plotly registry, not part of the drain).
- **Delete `pycaret/internal/pycaret_experiment/`** once `setup()` is native.

### Recommendation

**Ship `4.0.0a3` now.** The drain has delivered the 4.0 design promise on the public surface. `setup()` is internal — users don't see it. Remaining work is internal cleanup + `plot_model` Plotly rewrite. A few more breaking changes via `4.0.0a3` build community trust before the non-alpha cut.

Alternative: continue the `setup()` drain across 3-5 more sessions, ship `4.0.0` non-alpha at the end. Bigger announcement, longer wait.

### Side note: 4.0.0a2 release

Still pending PyPI Trusted Publishing config (user account reset in progress). Once unblocked, `gh run rerun 24917779816 --failed` ships a2 — but with sessions 30-33 in the bag, it's worth bumping to a3 first to give users the latest drain work.

---

## Session 32 — `add_metric` / `remove_metric` drain + per-Experiment metric registry — ✅

After session 31, the metric registry was being read directly from the global container helpers (`pycaret.containers.metrics.<task>.get_all_metric_containers({}, ...)`). Custom metrics added via `add_metric` lived in the legacy holder and never reached the native CV path. Session 32 fixes that by promoting the metric registry to a per-Experiment dict at `self._fit_state["metric_registry"]` — and **a custom metric registered via `add_metric` now actually shows up in subsequent CV results**.

### What landed

- **`Experiment._get_metric_registry()`** — single source of truth. Lazily builds the registry from the task helper on first call, caches in `_fit_state["metric_registry"]`. Works post-fit (cached, mutable) and pre-fit / fit-sentinel (fresh build, no caching). Returns `None` for time-series (which falls back to legacy).
- **All 6 metric-registry callsites consolidated** to use `_get_metric_registry()` in `_compute_predict_metrics`, `_cross_validate_supervised`, and `get_metrics()`. The 4-way classification/regression/clustering/anomaly task switching now lives only in the helper.
- **`Experiment.add_metric(id, name, score_func, target='pred', greater_is_better=True, args=None, is_multiclass=True)`** — drained. Builds the right `<Task>MetricContainer` for the current task and inserts it into the snapshot. Falls back to legacy for time-series.
- **`Experiment.remove_metric(name_or_id)`** — drained. Pops from the snapshot. Accepts ID or display name (legacy semantics). Raises `ValueError` if no match.
- **`packages/engine/tests/test_session32_metric_registry.py`** — 10 tests:
  - **The killer test**: `add_metric(...)` → next `create_model` includes the new column in CV metrics.
  - `add_metric` shows up in `get_metrics()` with `Custom=True`.
  - `remove_metric` drops from CV.
  - `remove_metric` accepts display name (matches "Accuracy" → drops the `acc` entry).
  - `remove_metric` unknown → `ValueError`.
  - Drain-locks for both verbs (poison `legacy.add_metric` / `legacy.remove_metric`).
  - Custom metric persists across `create_model` → `tune_model` → `compare_models`.
  - Regression `add_metric` works.
  - `NotFittedError` pre-fit on both.

### Headline metrics

| | Session 31 end | Session 32 end |
|---|---|---|
| Drainable secondary verbs still on `_legacy` | 2 (`add_metric`, `remove_metric`) | **0** ✅ |
| Engine tests (fast + slow) | 121 | **131** (+10) |
| **Combined tests** | **267** | **277** |

### Drain status

```
✅ All 16 modeling verbs            (sessions 22-28)
✅ User-facing data accessors       (session 29)
✅ Internal training state          (session 30)
✅ pull / models / get_metrics      (session 31)
✅ add_metric / remove_metric       (session 32)  ← this session
```

The only remaining `_legacy` callsites in `core/`:
- `self._legacy.setup(...)` inside `fit()` — biggest remaining piece.
- `get_config` / `set_config` — escape hatches for advanced users; small drain.
- `plot_model` / `evaluate_model` — Phase 3 roadmap (Plotly-native rewrite).
- TS-task fallback paths.

### Path to 4.0.0 non-alpha

The remaining setup() drain is genuinely 3-5 sessions of work to replicate 100+ legacy preprocessing options (normalize / transformation / remove_outliers / feature_selection / target encoding / etc.). A pragmatic path:

1. **Drain `get_config` / `set_config`** — small. Either snapshot the configurable knobs or document them as legacy-only.
2. **Ship `4.0.0a3` or `4.0.0` non-alpha** with the current architecture: public API fully native, `_legacy` still loaded as an internal preprocessing engine. Document this in release notes.
3. **Post-release**: tackle the native preprocessing chain incrementally.

This is honest engineering. The drain has delivered the 4.0 design promise (sklearn Pipeline-in / Pipeline-out, no god-class on the public surface) without months of preprocessing rewrite.

### Side note: 4.0.0a2 release

Still pending PyPI Trusted Publishing config. User account reset in progress.

---

## Session 31 — Secondary-verb drain: pull / models / get_metrics — ✅

After session 30 finished the internal-state drain, only `setup()` and a handful of advisory verbs remained on `self._legacy`. Session 31 drains the three secondary verbs that have a clean native equivalent.

### What landed

- **`Experiment.pull()`** reads from `self._fit_state["last_metrics"]`. Every native modeling verb (`create_model`, `tune_model`, `compare_models`) now updates that slot before returning via a new `_set_last_metrics()` helper. Falls back to `self._legacy.pull()` when no native verb has run yet (e.g. inside the TS-fallback path).
- **`Experiment.models()`** builds the public DataFrame (`Name` / `Reference` / `Turbo`, indexed by ID) directly from the snapshot's `model_registry`. The `internal=True` flag still delegates — that view exposes the full `ModelContainer` row with engine-internal fields, which we preserve for advanced callers.
- **`Experiment.get_metrics()`** reads directly from the task's metric registry helper (`pycaret.containers.metrics.<task>.get_all_metric_containers`). Output schema mirrors the legacy one (`Name` / `Display Name` / `Score Function` / `Scorer` / `Target` / `Args` / `Greater is Better` / `Multiclass` / `Custom`). Time-series falls back to legacy.
- **`packages/engine/tests/test_session31_secondary_verbs.py`** — 8 tests:
  - `pull()` returns `CreateResult.metrics` after `create_model`, drain-locked against `legacy.pull`.
  - `pull()` tracks the `compare_models` leaderboard.
  - `pull()` tracks `tune_model.metrics`.
  - `models()` returns the native DataFrame from the snapshot, drain-locked against `legacy.models`.
  - `models(internal=True)` keeps delegating (preserves the richer view).
  - `get_metrics()` reads from the metric registry, drain-locked against `legacy.get_metrics`.
  - Regression `get_metrics()` includes MAE / R2.
  - All 3 raise `NotFittedError` pre-fit.

### Headline metrics

| | Session 30 end | Session 31 end |
|---|---|---|
| Drainable secondary verbs still on `self._legacy` | 3 | **0** ✅ |
| Engine tests (fast + slow) | 113 | **121** (+8) |
| **Combined tests** | **259** | **267** |

### Drain progress

```
✅ All 16 modeling verbs (sessions 22-28)
✅ User-facing data accessors (session 29)
✅ Internal training state (session 30)
✅ pull / models / get_metrics (session 31)
```

The only remaining `_legacy` callsites in `core/` are:
- `self._legacy.setup(...)` inside `fit()` — last drain target (replace with native preprocessing chain).
- 6 advisory verbs that need bigger registry-side refactors: `add_metric`, `remove_metric`, `get_config`, `set_config`, `plot_model`, `evaluate_model`. Not in the predict / tune / compare path.
- TS-task fallback paths (every native verb has a `_<verb>_legacy` for time-series).

### Path to 4.0.0 non-alpha

1. **Native preprocessing chain** to replace `setup()` — biggest remaining piece, ~1-2 sessions.
2. **Optional**: drain the 6 advisory verbs (registry refactor). Could ship without — they're stable + advisory.
3. **Delete `pycaret/internal/pycaret_experiment/`** once `setup()` no longer needs it.
4. **Tag `4.0.0`** to PyPI.

### Side note: 4.0.0a2 release

Still pending PyPI Trusted Publishing config. Wheel + sdist + smoke matrix all passed; only blocker is OIDC trusted-publisher entry on PyPI's end (user account reset in progress).

---

## Session 30 — Internal-state drain: transformed splits + fold generator + model registry — ✅

After session 29 promoted user-facing accessors to `self._fit_state`, six **internal** legacy reads remained inside the drained verbs (the post-preprocessing splits used for CV, the fold generator, and the model-container registry). Session 30 captures all of them in `_fit_state` at fit time. **Every drained verb now reads training data, the CV generator, and the model registry from the snapshot.**

### What landed

- **`packages/engine/pycaret/core/experiment.py` — `_snapshot_fit_state()`** extended with 6 new slots:
  - `X_transformed`, `X_train_transformed`, `y_transformed`, `y_train_transformed` — post-preprocessor splits used by `create_model`'s CV loop and `finalize_model`'s full-data refit.
  - `fold_generator` — pre-built `StratifiedKFold` / similar instance used by every CV-running verb.
  - `model_registry` — `dict(getattr(legacy, "_all_models_internal", {}))` snapshot of the model containers (so we hold references but the dict itself is detached from the legacy holder).
- **13 callsites drained** across `core/experiment.py`, `core/supervised.py`, and `core/unsupervised.py`. Every `self._legacy.X_train_transformed` / `_all_models_internal` / `fold_generator` read in the drained verbs is now `self._fit_state[...]`.
- **`packages/engine/tests/test_session30_internal_state_drain.py`** — 5 tests:
  - **Drain-lock pattern** generalised: a `_PoisonedAttrAccess` sentinel raises on `__getattr__`, `__getitem__`, `__contains__`, `__iter__`, `__len__`. We poison every legacy attr we drained then run `create_model` / `tune_model` / `finalize_model` / clustering `create_model` and assert success.
  - `test_tune_model_uses_snapshot_for_search_space` — drops `legacy._all_models_internal` to `{}` after fit; tune_model still finds `lr` in the snapshot's registry copy and produces non-empty `best_params`.
  - `test_fit_state_holds_all_internal_keys` — sanity check that all 13 keys are populated post-fit.

### Headline metrics

| | Session 29 end | Session 30 end |
|---|---|---|
| Internal `_legacy.<X_train_transformed/X_transformed/...>` reads in drained verbs | 13 callsites | **0** ✅ |
| Engine tests (fast + slow) | 108 | **113** (+5) |
| **Combined tests** | **254** | **259** |

### Drain progress

The only `self._legacy.<x>` reads that remain in `core/`:
- `self._legacy.setup(...)` — the only call inside `fit()`. The actual experiment fitting. Will be drained when we replace it with a native preprocessing chain (penultimate step before deleting `pycaret/internal/pycaret_experiment/`).
- Secondary verbs (`pull`, `models`, `get_metrics`, `add_metric`, `remove_metric`, `get_config`, `set_config`, `plot_model`, `evaluate_model`) still delegate to `self._legacy.<verb>` — these are advisory / introspection helpers, not in the predict/tune/compare path.

After the secondary verbs + `setup()` are drained, `pycaret/internal/pycaret_experiment/` becomes deletable + we cut `4.0.0` non-alpha.

### Side note: 4.0.0a2 release

Tag pushed; build + smoke matrix all green; **publish-pypi blocked on PyPI Trusted Publishing config** (PyPI account reset in progress on the user's side). The wheel + sdist are valid; the only blocker is the OIDC trusted-publisher entry on PyPI's end. Will retry once PyPI reset completes.

---

## Session 29 — Property drain: data accessors — ✅

The user-facing data accessor properties — `X`, `X_train`, `X_test`, `y`, `y_train`, `y_test`, `preprocess_pipeline` — no longer dispatch to `self._legacy.<attr>` on every access. They read from a snapshot taken at the end of `fit()` and stored in `self._fit_state`.

Why this matters: with the modeling-verb drain done (sessions 22-28), the public API surface is now reachable without a single `self._legacy.<attr>` lookup. The legacy holder is still populated by `setup()` and used internally by the drained verbs (transformed splits, fold generator, model registry), but **no user-facing call now requires `self._legacy` to exist**. That's the architectural prereq for deleting `pycaret/internal/pycaret_experiment/` once the internal state-holder migration lands.

### What landed

- **`packages/engine/pycaret/core/experiment.py`** — added `_snapshot_fit_state()` called once at the end of `fit()`. It captures references to the legacy state in a `dict` on `self`. The 7 data-accessor properties now read from there:
  - `self.X`, `X_train`, `X_test`, `y`, `y_train`, `y_test`, `preprocess_pipeline`
  - Defensive `getattr(legacy, name, None)` lets the same code path work for tasks that don't have all attributes (clustering / anomaly have no `y_test`).
  - We hold *references*, not deep copies — mutating `exp.X_train` still propagates to the underlying frame, matching legacy semantics.
- **`packages/engine/tests/test_session29_property_drain.py`** — 4 tests:
  - `test_data_properties_do_not_call_legacy_after_fit` — the drain-lock. Replaces every `self._legacy.<accessor>` with a raise-on-read sentinel after fit; the 7 properties continue to return correct values.
  - `test_data_properties_clustering_y_is_none` — clustering's `y/y_train/y_test` come back as `None` (correct for unsupervised), `X` and `preprocess_pipeline` are present.
  - `test_data_properties_require_fit` — every accessor raises `NotFittedError` pre-fit.
  - `test_fit_state_returns_equivalent_data_to_legacy` — sanity check that `_fit_state` and the underlying legacy data match shape + columns.

### Headline metrics

| | Session 28 end | Session 29 end |
|---|---|---|
| User-facing API surface still touching `self._legacy` | 7 (data accessors) + 6 (verbs not yet drained) | **0 + 6** |
| Engine tests (fast + slow) | 104 | **108** (+4) |
| **Combined tests** | **250** | **254** |

### What's next

Remaining `_legacy` reads inside drained verbs (these are *internal*, not user-facing):
- `self._legacy.X_train_transformed` / `X_transformed` / `y_train_transformed` / `y_transformed` — preprocessed splits used inside `create_model`'s CV loop.
- `self._legacy.fold_generator` — pre-built CV strategy.
- `self._legacy._all_models_internal` — model registry.

Plus the still-delegating verbs (`plot_model`, `evaluate_model`, `pull`, `models`, `get_metrics`, `add_metric`, `remove_metric`, `get_config`, `set_config`).

Path to `4.0.0`:
1. Promote the 4 transformed-state attributes + `fold_generator` to `_fit_state` snapshots — same drain pattern as session 29.
2. Refactor `pycaret.containers.metrics.*` and `pycaret.containers.models.*` to take an `Experiment` directly instead of a `_legacy` instance.
3. Drain the secondary verbs (`plot_model`, `pull`, `models`, etc.) — most of these have native equivalents in `pandas` / our metric registry already.
4. Drop `setup()` from `fit()` — replace with a native preprocessing chain. Last drain.
5. Delete `pycaret/internal/pycaret_experiment/` entirely.
6. **Cut `4.0.0` non-alpha to PyPI**.

A pragmatic intermediate milestone: **ship `4.0.0a2`** at the current state. The public API is fully native; the internal `_legacy` holder is an implementation detail that doesn't leak to users. That's a good shipping point for community feedback while the internal migration finishes.

---

## Session 28 — God-class drain: unsupervised verbs — ✅

**The OOP drain is essentially complete.** With session 28, both `UnsupervisedExperiment.create_model` and `UnsupervisedExperiment.assign_model` for clustering + anomaly run natively. The only `_legacy` callsite that remains for the *modeling* surface is the time-series experiment subclass (which has different shapes — `fh`, `seasonal_period`, no fold generator in the usual sense — and warrants its own session).

### What landed

- **`packages/engine/pycaret/core/unsupervised.py`** — drained both unsupervised verbs:
  - **`UnsupervisedExperiment.create_model`** — resolves a registry ID (KMeans / DBSCAN / Birch / IForest / LOF / etc.) into an instance, fits on `self._legacy.X_transformed`, and returns a `CreateResult` whose `.pipeline` is a real sklearn Pipeline. Accepts `num_clusters=` for clustering (translates to `n_clusters`) and `fraction=` for anomaly (translates to `contamination`). Falls through to the registry default kwargs if the constructor rejects a forwarded one (`AffinityPropagation` doesn't accept `n_clusters`, etc.).
  - **`UnsupervisedExperiment.assign_model`** — unwraps Pipeline → bare model, reads `model.labels_` (and `model.decision_scores_` for anomaly), decorates a copy of `self.X` with `Cluster`/`Anomaly`/`Anomaly_Score` columns. `transformation=True` returns rows from `X_transformed` instead. `score=False` skips the score column.
- **CBLOF retry preserved**. The `cluster` (CBLOF) anomaly detector can fail when the default `n_clusters` yields a degenerate small/large cluster split. Native `create_model` mirrors the legacy retry: catch the `ValueError`, set `n_clusters=12`, refit. Identical behavior to 3.x — the `test_model_equality_anomaly` test passes without modification.
- **`packages/engine/pycaret/core/experiment.py`** — `predict_model`'s transitional bare-estimator branch comment updated. The branch is dead for both supervised AND unsupervised tasks now (both `create_model` paths return a real Pipeline). The branch lives on as a belt-and-braces fallback for callers passing in their own bare estimators directly.
- **`packages/engine/tests/test_session28_unsupervised.py`** — 11 new tests:
  - Clustering: KMeans Pipeline shape, `assign_model` decorates with `Cluster` column, drain-locks for both `create_model` + `assign_model`, predict-chain for KMeans (which supports `.predict`).
  - Anomaly: IForest Pipeline shape, `assign_model` adds `Anomaly` + `Anomaly_Score`, `score=False` omits the score column, drain-lock for `create_model`.
  - Cross-task: unknown ID → `ConfigurationError`, all four verbs require fit → `NotFittedError`.

### Headline metrics

| | Session 27 end | Session 28 end |
|---|---|---|
| OOP verbs still on `self._legacy` (clf/reg/clu/anomaly) | 0 / 0 / 2 / 2 | **0 / 0 / 0 / 0** ✅ |
| Engine tests (fast + slow) | 93 | **104** (+11) |
| **Combined tests** | **239** | **250** |

### Drain progress

```
Supervised (sessions 22-27): ALL 13 verbs ✓
Unsupervised  (session 28):  ALL  3 verbs ✓ (create_model, predict_model via core/experiment, assign_model)
Time-series   (pending):     legacy delegation remains
```

### What's next (session 29+)

The remaining work to ship `4.0.0`:

1. **Drain the time-series Experiment subclass** — `_create_model_legacy` fallback in `core/experiment.py` is still active for TS. The TS hierarchy in `pycaret/internal/pycaret_experiment/ts_supervised_experiment.py` has its own setup, fold generator, and forecasting verbs (forecast / plot_forecasts). One session.
2. **Refactor model + metric registry helpers** to read from `Experiment` directly instead of `self._legacy`. The `get_all_metric_containers` / `get_all_model_containers` functions take an `experiment` arg + read fields like `seed`, `gpu_param`, `num_classes`. Migrating them to read from the `Experiment` (or a thin context object) lets us delete the legacy holder entirely.
3. **Read-only properties** (`X`, `X_train`, `X_test`, `y`, `y_train`, `y_test`, `preprocess_pipeline`) currently delegate to `self._legacy`. Move them onto the `Experiment` directly so `_legacy` is no longer referenced post-fit.
4. **Delete `pycaret/internal/pycaret_experiment/`** entirely once steps 1-3 are done.
5. **Cut `4.0.0` non-alpha to PyPI**.

---

## Session 27 — God-class drain: ensemble / blend / stack / calibrate / finalize — ✅

**The supervised drain is COMPLETE.** All 13 OOP verbs (4 persistence + 9 model) on classification + regression now run without ever calling `self._legacy.<verb>`. The remaining 5 verbs landed in one batch this session — each is a thin sklearn-meta-estimator wrapper that reuses the already-drained `create_model` to assemble Pipelines + run CV.

### What landed

- **`packages/engine/pycaret/core/supervised.py`** — drained the final 5 supervised verbs:
  - **`ensemble_model`** — `method="Bagging"` → `BaggingClassifier`/`BaggingRegressor`. `method="Boosting"` → `AdaBoostClassifier`/`AdaBoostRegressor`. Returns a Pipeline named `Bagging[<base_id>]` or `AdaBoost[<base_id>]`.
  - **`blend_models`** — wraps `VotingClassifier` / `VotingRegressor`. Classification `method="auto"` picks `"soft"` when every base model has `predict_proba`, else `"hard"`. Each base estimator is added with a unique name (`{model_id}_{i}`) so sklearn's named-step constraint holds even if the same model is included twice.
  - **`stack_models`** — wraps `StackingClassifier` / `StackingRegressor`. Default meta-learner: `LogisticRegression(max_iter=1000)` for classification, `LinearRegression()` for regression. `meta_model=` overrides. Returns a Pipeline named `Stacking[<meta_id>]`.
  - **`calibrate_model`** — wraps `CalibratedClassifierCV` (classification only — raises `ValueError` for regression). `method` is `"sigmoid"` (Platt) or `"isotonic"`.
  - **`finalize_model`** — re-fits on `X_transformed` + `y_transformed` (the FULL dataset, train + holdout combined). Returns a fresh fitted Pipeline; the input is left untouched.
- **Two new helpers**: `_unwrap_estimator(obj)` returns `(bare_model, model_id)` from a Pipeline / registry-ID / bare estimator (single source of truth for unwrapping). `_wrap_in_pipeline(model, name)` is the canonical Pipeline-assembly helper used by `finalize_model` (and conceptually mirrored inside `create_model`).
- **`packages/engine/tests/test_session27_combine.py`** — 13 tests, ~30s total. One drain-lock per verb (poison `self._legacy.<verb>` + assert native success), happy-path classification + regression where applicable, edge cases (`calibrate_model` rejects regression). All 13 verbs predict-chain back through `predict_model` cleanly.

### Headline metrics

| | Session 26 end | Session 27 end |
|---|---|---|
| Supervised OOP verbs still on `self._legacy` | 1 | **0** ✅ |
| Engine tests (fast + slow) | 80 | **93** (+13) |
| **Combined tests** | **226** | **239** |
| `pycaret/internal/pycaret_experiment/` deletable for supervised | no | **yes** (pending TS / clustering / anomaly drain) |

### Drain progress: ALL 13 supervised verbs done

```
[✓] save_model          (session 22)
[✓] load_model          (session 22)
[✓] save_experiment     (session 22)
[✓] load_experiment     (session 22)
[✓] predict_model       (session 23)
[✓] create_model        (session 24)
[✓] tune_model          (session 25)
[✓] compare_models      (session 26)
[✓] ensemble_model      (session 27)  ← THIS BATCH
[✓] blend_models        (session 27)  ← THIS BATCH
[✓] stack_models        (session 27)  ← THIS BATCH
[✓] calibrate_model     (session 27)  ← THIS BATCH
[✓] finalize_model      (session 27)  ← THIS BATCH
```

### What's next

The supervised pipeline is fully native. The remaining work to ship `4.0.0` non-alpha:

1. **Drain unsupervised + time-series verbs** (`create_model` / `predict_model` / `assign_model` for clustering & anomaly; the time-series `Experiment` subclass). Smaller scope — clustering/anomaly already use sklearn-native flows; mostly removing the `_legacy` wrappers.
2. **Strip the transitional bare-estimator branch in `predict_model`** — now that `create_model` returns a real Pipeline, the bare-estimator path can be deleted for supervised tasks (kept only for clustering/anomaly until those drain).
3. **Delete `pycaret/internal/pycaret_experiment/`** entirely once nothing reads from it. The model + metric registries (`pycaret.containers.*`) need a small refactor to read from `Experiment` directly instead of from the legacy object.
4. **Cut `4.0.0` to PyPI** — non-alpha release.

---

## Session 26 — God-class drain: `compare_models` (supervised) — ✅

The heart of the AutoML loop is now native. **`SupervisedExperiment.compare_models`** for classification + regression no longer delegates to `self._legacy.compare_models`. It iterates the engine's model registry, calls the (already-drained) `self.create_model` for each candidate, and assembles the leaderboard from each model's `Mean` metrics row.

This unlocks the full **`compare_models` → `tune_model` → `predict_model`** notebook flow on the native path with no transitional branches. The 4.0 invariant "every CreateResult/TuneResult/CompareResult `.pipeline` (or `.best`) is a real sklearn Pipeline" now holds across all the major supervised verbs.

### What landed

- **`packages/engine/pycaret/core/supervised.py`** — rewrote `compare_models` into a task-aware dispatcher:
  - Supervised → `_compare_models_supervised_native`.
  - Time-series / clustering / anomaly → `_compare_models_legacy`.
- Native path:
  1. Build the candidate list: `include` if given, else every active registry entry (`is_special=False`). Drop `exclude`. Drop `_TURBO_EXCLUDE = {"rbfsvm", "gpc", "mlp"}` if `turbo=True`.
  2. Pick default `sort` metric: `"Accuracy"` for classification, `"R2"` for regression.
  3. Per-candidate loop: call `self.create_model(cand, fold=, cross_validation=, fit_kwargs=, round=, verbose=False)`. On per-model exception with `errors="ignore"`: skip; with `"raise"`: propagate.
  4. Assemble leaderboard rows from each model's `Mean` row.
  5. Sort the leaderboard. Detects ascending-vs-descending automatically: error metrics (`MAE`, `MSE`, `RMSE`, `MAPE`, `RMSLE`, `neg_*`) sort ascending / "smaller is better"; everything else descending.
  6. Return `CompareResult(best, models[:n_select], leaderboard, ranked_ids)`.
- **Signature slim-down** (BREAKING, supervised): kept `include`, `exclude`, `fold`, `cross_validation`, `sort`, `n_select`, `turbo`, `errors`, `fit_kwargs`, `round`, `verbose`. Dropped 3.x cruft: `budget_time`, `experiment_custom_tags`, `probability_threshold`, `groups`, `caller_params`.
- **`packages/engine/tests/test_session26_compare.py`** — 10 new tests:
  - Top-N Pipelines + leaderboard contains both candidates.
  - Default sort = `Accuracy` (classification) / `R2` (regression), descending.
  - `sort="MAE"` → ascending order (error metric).
  - `exclude=["dt"]` drops a model.
  - `turbo=True` skips `rbfsvm` / `gpc` / `mlp` even when explicitly included.
  - **Drain-lock**: `self._legacy.compare_models` poisoned to raise; native path still succeeds.
  - End-to-end chain: `compare_models → predict_model` on `result.best` (no transitional branches).
  - `errors="ignore"` skips a bogus model id without sinking the whole run.
  - `NotFittedError` on unfit experiment.

### Headline metrics

| | Session 25 end | Session 26 end |
|---|---|---|
| Supervised OOP verbs still on `self._legacy` | 2 | **1** |
| Engine tests (fast + slow) | 70 | **80** (+10) |
| **Combined tests** | **216** | **226** |

### Drain progress: 8 of 10 supervised verbs done

```
[✓] save_model          (session 22)
[✓] load_model          (session 22)
[✓] save_experiment     (session 22)
[✓] load_experiment     (session 22)
[✓] predict_model       (session 23)
[✓] create_model        (session 24)
[✓] tune_model          (session 25)
[✓] compare_models      (session 26)
[ ] ensemble_model      (session 27 target)
[ ] blend_models / stack_models / calibrate_model / finalize_model
```

After ensemble_model + the four remaining verbs land, `pycaret/internal/pycaret_experiment/` becomes deletable + **`4.0.0`** ships non-alpha to PyPI.

---

## Session 25 — God-class drain: `tune_model` (supervised) — ✅

Fourth drain. **`SupervisedExperiment.tune_model`** for classification + regression no longer delegates to `self._legacy.tune_model`. It runs `sklearn.model_selection.RandomizedSearchCV` on the base estimator pulled from the registry's `tune_grid`, then assembles the best Pipeline (preprocessor + tuned model). The shared `_cross_validate_supervised` helper (built in session 24) gives the returned `TuneResult.metrics` an identical schema to `CreateResult.metrics`.

### What landed

- **`packages/engine/pycaret/core/supervised.py`** — rewrote `tune_model` into a task-aware dispatcher:
  - Supervised → `_tune_model_supervised_native`.
  - Time-series / clustering / anomaly → `_tune_model_legacy` (their drains are later sessions).
- Native path:
  1. Unwrap the estimator: a Pipeline (last step = bare model + `model_id`), a registry ID string (resolved through the helper from session 24), or a bare estimator.
  2. Pick search space: `custom_grid=` > registry's `tune_grid` (explicit dict[str, list]). Registry's `tune_distribution` uses a custom PyCaret distribution type that sklearn's `RandomizedSearchCV` can't sample from — adapting it to scipy distributions is a polish item; for now `tune_grid` is the source of truth.
  3. Resolve scoring: maps PyCaret metric names (``"Accuracy"`` / ``"AUC"`` / ``"MAE"`` / ``"R2"``) to sklearn scorer strings (``"accuracy"`` / ``"roc_auc"`` / ``"neg_mean_absolute_error"`` / ``"r2"``).
  4. Run `RandomizedSearchCV(deepcopy(bare_model), ..., refit=True)` over `X_train_transformed`, `y_train_transformed` with `cv=fold or self._legacy.fold_generator`.
  5. Re-assemble the Pipeline = `deepcopy(self.preprocess_pipeline).steps + [(model_id, search.best_estimator_)]`.
  6. Compute per-fold metrics for the winning estimator via the shared `_cross_validate_supervised` helper so `TuneResult.metrics` has the same shape as `CreateResult.metrics`.
- **`packages/engine/tests/test_session25_tune.py`** — 9 new tests:
  - Pipeline shape (last step is the tuned bare estimator under `model_id` name).
  - `cv_results` (sklearn's `cv_results_` as a DataFrame, length = `n_iter`) + `metrics` (Fold/Mean/Std rows, classification metric columns).
  - `custom_grid={"C": [0.1, 1.0, 10.0]}` overrides the registry default.
  - `optimize="AUC"` maps to the sklearn `"roc_auc"` scorer.
  - Regression default is `"r2"`.
  - **Drain-lock**: monkeypatch `self._legacy.tune_model` to raise; native path still succeeds.
  - End-to-end chain: `create_model` → `tune_model` → `predict_model` all on a sklearn Pipeline, no transitional branches.
  - Registry ID directly: `tune_model("lr", ...)` works without a prior `create_model`.
  - `NotFittedError` on unfit experiment.

### Headline metrics

| | Session 24 end | Session 25 end |
|---|---|---|
| Supervised OOP verbs still on `self._legacy` | 3 | **2** |
| Engine tests (fast + slow) | 61 | **70** (+9) |
| **Combined tests** | **207** | **216** |

### What's next (session 26+)

Next on the drain: **`compare_models`** — the heart of the AutoML loop. It iterates over the registry, runs `create_model` for each, ranks them by an `optimize` metric, and returns the top-K. Native version uses the already-drained `create_model` plus a per-model loop. After that: `ensemble_model` / `blend_models` / `stack_models` / `calibrate_model` / `finalize_model`.

Drain progress: **7 of 10** verbs drained.

```
[✓] save_model          (session 22)
[✓] load_model          (session 22)
[✓] save_experiment     (session 22)
[✓] load_experiment     (session 22)
[✓] predict_model       (session 23)
[✓] create_model        (session 24, supervised)
[✓] tune_model          (session 25, supervised)
[ ] compare_models      (session 26 target)
[ ] ensemble_model
[ ] blend_models
[ ] stack_models
[ ] calibrate_model
[ ] finalize_model
```

---

## Session 24 — God-class drain: `create_model` (supervised) — ✅

Third drain in the session-22→32 sweep. **`Experiment.create_model` on classification + regression** no longer delegates to `self._legacy.create_model`. It resolves the estimator from the engine's model registry, runs k-fold CV manually with the task's metric registry, refits on the full training set, and assembles a **real sklearn `Pipeline`** (preprocessor + trained model).

This unlocks a big 4.0 invariant: **`CreateResult.pipeline` is now a real Pipeline** for supervised tasks. Downstream verbs (`predict_model`, soon-to-be-drained `tune_model`, `ensemble_model`, etc.) can consume it directly.

### What landed

- **`packages/engine/pycaret/core/experiment.py`** — rewrote `create_model` into a task-aware dispatcher:
  - Supervised (classification / regression) → native path (`_create_model_supervised_native`).
  - Time-series / clustering / anomaly → still delegates via `_create_model_legacy` (their drains are later sessions).
- Native path:
  1. Resolve `estimator` — registry ID string (e.g. `"lr"`) → `container.class_def(**container.args, **user_kwargs)`, OR a user-built sklearn estimator → use as-is.
  2. Pull `X_train_transformed` / `y_train_transformed` from the (still-live) legacy state.
  3. If `cross_validation=True`: run k-fold CV manually — deep-copy the model per fold, fit on train-fold, predict on val-fold, compute metrics via the shared `pycaret.utils.generic.calculate_metrics` + task-specific `get_all_metric_containers`. Aggregate with `Mean` + `Std` rows. Any per-fold failure is swallowed so CV degrades gracefully instead of crashing.
  4. Final fit on the full training set.
  5. Assemble the returned `Pipeline` = `deepcopy(self.preprocess_pipeline).steps + [(model_id, trained_model)]`.
- **Signature slim-down** — dropped 3.x cruft: `probability_threshold`, `experiment_custom_tags`, `refit`, `return_train_score`, `groups`, `predict`. Kept the meaningful ones: `fold`, `cross_validation`, `fit_kwargs`, `round`, `verbose`, plus `**estimator_kwargs` forwarded to the constructor when `estimator` is a registry ID.
- **`packages/engine/tests/test_session24_create_model.py`** — 10 new tests covering:
  - Pipeline shape (last step = model under `model_id` name, preprocessing steps before).
  - CV metrics DataFrame (`Fold 0..N-1`, `Mean`, `Std`; classification + regression metric registries pull the right columns).
  - `cross_validation=False` → metrics None, pipeline still fitted.
  - Unknown registry ID raises `ConfigurationError`.
  - Pre-constructed estimator is accepted + user hyperparameters survive (`C=2.0`).
  - Chained `create_model → predict_model` works without touching `self.preprocess_pipeline` — a monkeypatch test proves the transitional bare-estimator branch in `predict_model` is dead for supervised tasks.
  - Drain-lock: `self._legacy.create_model` is poisoned to raise; native `create_model` still returns a valid `CreateResult`.
  - Clustering fallback still works (legacy delegation).
  - `NotFittedError` on unfit experiment.
- **`packages/engine/tests/test_models.py`** — updated the existing `check_exp` helper to unwrap the final step from the returned Pipeline before calling each registry container's `Equality` predicate. The predicates check `isinstance(obj, <class>)` against the bare class; the Pipeline wrapper made them fail. Now 5/5 test_models tests green on the drain.

### Headline metrics

| | Session 23 end | Session 24 end |
|---|---|---|
| OOP verbs still delegating to `self._legacy` (supervised) | 5 | **4** |
| Supervised `CreateResult.pipeline` type | bare estimator | **sklearn `Pipeline`** |
| Engine tests (fast + slow) | 51 | **61** (+10) |
| **Combined tests** | **197** | **207** |

### What's next (session 25+)

Next on the drain list: **`tune_model`**. It does hyperparameter search (grid/random/optuna) on top of the already-fitted model. The native version will build a proper `GridSearchCV` / `RandomizedSearchCV` over the Pipeline + return the best Pipeline. No more `self._legacy.tune_model`.

Remaining: `tune_model` → `ensemble_model` → `blend_models` → `stack_models` → `calibrate_model` → `compare_models` → `finalize_model` → then delete `pycaret/internal/pycaret_experiment/` + ship `4.0.0`.

---

## Session 23 — God-class drain: `predict_model` — ✅

Second pass of the drain. `Experiment.predict_model` no longer calls `self._legacy.predict_model`. It dispatches directly on the estimator + handles the 4 task shapes (classification / regression / clustering / anomaly) natively.

### What landed

- **`packages/engine/pycaret/core/experiment.py`** — rewrote `predict_model` (~170 LoC). Key design choices:
  - Accepts either a fitted sklearn `Pipeline` (the clean 4.0 shape, what create_model will return post-drain) OR a bare fitted estimator (the current transitional reality — `CreateResult.pipeline` today is a `LogisticRegression`, not a Pipeline). For the bare case, we apply `self.preprocess_pipeline` to transform X first. This accommodation is flagged in the docstring as transitional and will collapse once session 24 drains `create_model`.
  - Task-specific output columns dispatched on `self.task`:
    - Classification binary → `prediction_label` + `prediction_score` (positive-class prob).
    - Classification multiclass, `raw_score=False` → `prediction_label` + `prediction_score` (winning-class prob).
    - Classification multiclass, `raw_score=True` → `prediction_label` + `prediction_score_<class>` per class (summing to ~1 per row).
    - Regression → `prediction_label` only.
    - Clustering → `Cluster` column with `"Cluster {i}"` labels.
    - Anomaly → `Anomaly` + `Anomaly_Score` (when `decision_function` exists).
  - Metrics DataFrame computed on the holdout when y is known; reuses the existing metric registry via `pycaret.utils.generic.calculate_metrics` + `pycaret.containers.metrics.{classification,regression}.get_all_metric_containers`. Falls through to `None` on any registry hiccup — metrics are advisory, a predict must never fail because of them.
- **Parameter cleanup** (all 3.x cruft removed):
  - Dropped `probability_threshold` (was a binary-classification hack; callers can do the same thresholding on `prediction_score` directly).
  - Dropped `encoded_labels` (label encoding is handled by the preprocessor already; users wanting integer labels can `map` the column).
  - Dropped `preprocess` / `ml_usecase` (both were v3.x internal dispatch).
- **`packages/engine/tests/test_session23_predict.py`** — 12 new tests. Split into two tiers:
  - **Fast (7 tests, ~3s total)** — fabricate a tiny fitted Pipeline + a fit-sentinel Experiment and exercise the raw predict dispatch: rejects non-estimator input, binary/multiclass/regression output shapes, multiclass `raw_score` per-class columns sum to ~1, NotFittedError raised without fit, metrics present when data has target / absent when data has only features.
  - **Slow (5 tests, ~15s each)** — full engine E2E on `juice` + `boston` datasets. Includes the drain-lock test (`test_predict_model_does_not_call_legacy_predict_model`) that monkeypatches `exp._legacy.predict_model` to raise + then calls `exp.predict_model` and asserts it succeeds.

### Headline metrics

| | Session 22 end | Session 23 end |
|---|---|---|
| OOP verbs still delegating to `self._legacy` | 6 | **5** |
| Engine tests (fast + slow) | 35 | **51** (+16) |
| **Combined tests** | **181** | **197** |
| Engine-side 3.x params still in `predict_model` signature | 6 | **0** (all 3.x cruft dropped) |

### What's next (session 24+)

Next on the drain list: **`create_model`**. More invasive than the previous two — it has to materialise the right sklearn estimator from the engine's model registry (`get_all_model_containers`), wrap it in a Pipeline with the preprocessor, cross-validate it, and populate the results container. That work will also unlock a clean "CreateResult.pipeline is always a sklearn Pipeline" invariant, at which point the transitional bare-estimator path in `predict_model` can be deleted.

Remaining: `create_model` → `tune_model` → `ensemble_model` → `blend_models` → `stack_models` → `calibrate_model` → `compare_models` → `finalize_model`.

---

## Session 22 — God-class drain kickoff: persistence verbs — ✅

First pass of engine Phase 5 — the "10 OOP verbs still delegate to `self._legacy`" starts getting drained one verb at a time. This session targets the 4 **persistence** verbs on `Experiment` (`save_model`, `load_model`, `save_experiment`, `load_experiment`) — the simplest of the bunch, since a fitted sklearn Pipeline is just a picklable object and there's no PyCaret-specific payload.

### What landed

- **`packages/engine/pycaret/core/experiment.py`** — drained all 4 persistence verbs. They no longer call `self._legacy.*`; they delegate to the stateless helpers in `pycaret.persistence` (which already existed but were only wired up at the module level).
  - `exp.save_model(model, path)` → `Path` — works **with or without fit**; writing a pipeline doesn't depend on experiment state.
  - `exp.load_model(path)` — straight `joblib.load` on the file.
  - `exp.save_experiment(path)` — pickle `self`. Requires fit (an unfit Experiment is just constructor kwargs you already have).
  - `Experiment.load_experiment(path)` — classmethod; re-hydrates an Experiment, raises `TypeError` if the file contained a plain model (and the message steers the caller to `load_model`).
- **`packages/engine/tests/test_session22_persistence.py`** — 7 new unit tests, all run in ~2s:
  - Round-trip predictions match via `exp.save_model` + `exp.load_model`.
  - **Crucially: `test_save_model_does_not_touch_legacy`** — constructs an unfit Experiment (so `self._legacy` doesn't exist), calls save/load, asserts no `_legacy` was lazily created. This test locks the drain against regressions.
  - `save_model` accepts both `Path` and `str`.
  - `MODEL_SAVED` event is emitted on the logger with the absolute path.
  - `save_experiment` raises `NotFittedError` when called on an unfit Experiment.
  - `load_experiment` rejects a plain model file with a helpful `TypeError`.
  - Top-level `pycaret.save_model` / `pycaret.load_model` remain exposed.
- **~200 LoC of dependency surface removed** — the old code path ran cloud-credential injection (AWS S3 / GCP / Azure), MLflow artifact logging, and a 3.x-era metadata header. All of that is out of scope for 4.0 (cloud serving is Control Plane territory; artifact logging is per-logger plugin territory).

### Headline metrics

| | Session 21 end | Session 22 end |
|---|---|---|
| OOP verbs still delegating to `self._legacy` | 10 | **6** |
| Engine fast tests | 28 | **35** (+7) |
| **Combined tests** | **174** | **181** (35 engine + 80 server + 62 web + 4 E2E slow) |
| Engine code paths carrying 3.x-era persistence logic | 1 (tabular + internal) | **0** (fully drained) |

### What's next (session 23+)

Next verb on the drain list, in the order called out in the roadmap:

1. ✅ **`save_model` + `load_model`** (session 22, this one).
2. **`predict_model`** (session 23 target) — straightforward delegation; new impl should just call `pipeline.predict(X)` + optional decorated output (probabilities, prediction_label column).
3. **`create_model`** — more invasive; needs to materialise the right sklearn estimator from the engine's model registry + wrap it in a Pipeline with the preprocessor.
4. `tune_model` → `ensemble_model` → `blend_models` → `stack_models` → `calibrate_model` → `compare_models` → `finalize_model`.

Once all 10 are drained, `pycaret/internal/pycaret_experiment/` is deleted entirely and **`4.0.0`** ships non-alpha on PyPI.

---

## Session 21 — Drift analyst + audit logs — ✅

The 6th and final LLM copilot lands (drift analyst) alongside the last enterprise-readiness item on the MVP-2 punch list (audit logs). After this session there's nothing left on the platform roadmap before the god-class drain — session 22 pivots to engine work → `4.0.0` non-alpha.

### What landed — backend

- **`DriftReport` model** (SPEC § 4.12) — new `drift_reports` table: window_start / window_end, overall `drift_score` (0..1), bucketed `drift_status` (`none | mild | moderate | severe`), per-feature `feature_drift_json` (`{feature: {score, kind}}` where kind ∈ PSI / KS / chi² / missing_rate), `prediction_drift_json` (JS divergence), optional `sample_size`, FK to owning `Deployment`.
- **3 drift CRUD routes** under `/api/v1/`:
  - `POST /deployments/{id}/drift-reports` — create a snapshot. Server buckets `drift_status` from `drift_score` (thresholds 0.10 / 0.25 / 0.40, aligned with PSI convention). Guards `window_end >= window_start`.
  - `GET /deployments/{id}/drift-reports` — list (newest first).
  - `GET /drift-reports/{id}` — single report.
- **`drift_analysis` consultation** (6th LLM copilot) — reads a `DriftReport` + its Deployment + the owning Pipeline, asks the LLM for a verdict prefixed with `RETRAIN NOW` / `INVESTIGATE` / `MONITOR` / `NO ACTION`. Same verdict-string classifier pattern as the deployment reviewer — UI tone-codes with `.startsWith()`.
- **`POST /llm/analyze-drift`** — body `{drift_report_id}`. Uses the shared `ConsultationContext` + `get_router().consult()` path, so it's free-riding on the existing audit trail + provider routing.
- **`AuditLog` model** (SPEC § 17.4) — new `audit_logs` table, append-only. Columns: workspace_id (nullable for global events), user_id (nullable for unauthenticated/failed-auth calls), action (dotted `{namespace}.{verb}`), method, path, target_type, target_id, status_code, payload (JSON, scrubbed), ip_address, user_agent, created_at. Intentionally no `updated_at` — rows are immutable.
- **`AuditLogMiddleware`** — FastAPI middleware that records one row per `POST/PATCH/PUT/DELETE` on `/api/v1/*`. Captures + re-injects the request body (so route handlers still read it), scrubs sensitive fields (`password`, `api_key`, `token`, `refresh_token`, `access_token`, `api_key_encrypted`, `plaintext_token`, `password_hash`), derives the `{entity}.{verb}` action from the path + method, extracts `workspace_id` from `/workspaces/{id}/…` URLs. Best-effort — never blocks or fails the request. Reads + `/auth/refresh` + heartbeats are skipped.
- **`get_current_user` stashes the resolved user** onto `request.state.audit_user` so the middleware can attribute rows without re-resolving the header.
- **2 audit viewer routes**:
  - `GET /admin/audit-logs` — installation-wide, superuser only (via `require_admin` dependency). Filters on action, user_id, workspace_id, target_type, target_id, since, until, plus limit/offset.
  - `GET /workspaces/{id}/audit-logs` — workspace-scoped, workspace admin or superuser. Same filter surface minus workspace_id.
- **1 new Alembic migration** (`0cd9d5ea2e17`) adds both tables in one revision.
- **12 new integration tests** in `services/api/tests/test_session21.py` covering:
  - Drift CRUD: create-buckets-status, bucket-boundaries (none/mild/severe), list + get, window-end-before-start rejection.
  - Drift analyst: happy-path runs the LLM, 404 on unknown report.
  - Audit logs: mutating requests are recorded + attributed, password is scrubbed on bootstrap, workspace-scoped viewer requires admin, admin route requires superuser, action filter works, workspace-scoped viewer filters by workspace.

### What landed — frontend

- **`<DriftAnalysisModal>`** — opens on "✨ Analyze" click on any row in the drift-reports list. Auto-fires the consultation on open (same pattern as `<DeploymentReviewModal>` + `<AnalyzeDatasetModal>`). Tone-codes the 4 verdict prefixes (`RETRAIN NOW` → danger, `INVESTIGATE` → warn, `MONITOR` → ink-200, `NO ACTION` → success). Shows the feature-drift snapshot sorted by score desc (dominant features at the top) + the LLM reasoning + risk flags.
- **`<DriftReportsCard>`** — inline card on `/deployments/:id` below the PredictTester. Lists existing reports with window / score / status / sample columns + a "✨ Analyze" button per row. "Record snapshot" button opens an inline form that accepts `drift_score` + optional `sample_size` + pasted `feature_drift_json` / `prediction_drift_json`. Client-side JSON parsing + 0–1 range guard on score. Empty-state copy nudges toward the POST-from-CI pattern.
- **`<AuditLogViewer>`** at `/admin/audit` — superuser-gated screen. Table with When / Action / Method / Path / Status / User columns; click a row to expand the scrubbed payload + workspace_id / target / IP / user-agent inline. Filter bar (action + target_type + limit). Status codes tone-coded (5xx red, 4xx amber).
- **Wiring**:
  - `DeploymentDetail` gains the `<DriftReportsCard>` at the bottom of the left column.
  - `App.tsx` registers `/admin/audit`.
  - `Layout` top nav gains an "Audit log" link that only renders for superusers.
- **API bindings** — new `driftApi` (list/create/get) + `auditApi` (listAdmin/listForWorkspace).
- **10 new Vitest tests** — 3 for DriftAnalysisModal (inert-when-closed, danger tone on `RETRAIN NOW` + feature rows sorted, success tone on `NO ACTION`), 4 for DriftReportsCard (empty state, list + open modal, create form submit, client-side score range validation), 3 for AuditLogViewer (row expand shows scrubbed payload, non-superuser sees forbidden message, filter bar wires through).

### Headline metrics

| | Session 20 end | Session 21 end |
|---|---|---|
| LLM copilots shipped (of 6 in spec) | 5 | **6** (all six) |
| API routes (under `/api/v1/`) | ~58 | **~63** (+5) |
| Server integration tests | 68 | **80** (+12) |
| UI components | 12 | **14** (+ DriftAnalysisModal, DriftReportsCard) |
| UI screens | 15 | **16** (+ AuditLogViewer) |
| UI tests | 52 | **62** (+10) |
| **Combined tests** | **148** | **174** (32 engine + 80 server + 62 web) |
| Production bundle (gz) | 99 kB | **101 kB** (+2 kB) |

### AI at every stage of the product loop

```
Upload CSV            → ✨ AI          (dataset consultant — session 17)
New Experiment        → ✨ Ask AI      (experiment designer — session 18)
Run succeeds          → ✨ Explain     (run explainer — session 18)
Run fails             → ✨ Diagnose    (failure debugger — session 19)
Deploy a Pipeline     → ✨ Review      (deployment reviewer — session 19)
Drift detected        → ✨ Analyze     (drift analyst — session 21)  ← NEW
```

All 6 copilots in SPEC § 12.2 are now live.

### What's next (session 22+)

**God-class drain (engine Phase 5)** — the 10 OOP verbs that still delegate to `self._legacy` migrate to native sklearn one at a time. ~10 sessions worth of work. Order: `save_model → predict_model → create_model → tune_model → ensemble_model → blend_models → stack_models → calibrate_model → compare_models → finalize_model`. Once drained, cut `4.0.0` non-alpha to PyPI.

---

## Session 20 — Workspace members + programmatic API-key auth — ✅

Multi-user collaboration lands. Admins can now invite + manage members from a dedicated workspace screen, and programmatic callers (CLI tools, CI jobs, notebooks) can authenticate with `X-PyCaret-Key` headers instead of JWT. Drift analyst + audit logs slipped to session 21 to keep this session's scope honest.

### What landed — backend

- **`services/api/pycaret_server/api/members.py`** — 4 routes under `/workspaces/{workspace_id}/members`:
  - `GET /` — list members with role + activity status. Open to all workspace members.
  - `POST /` — invite an existing user (by email). v1 does not send emails — if no user exists with that email, returns 404 with a hint. Email-invite-with-account-creation is V2.
  - `PATCH /{user_id}` — change a member's role. Last-admin guard: refuses to demote the sole admin (400).
  - `DELETE /{user_id}` — remove a member. Last-admin guard mirrors PATCH.
- **`X-PyCaret-Key` middleware** — `get_current_user` dependency now accepts `Authorization: Bearer …` (JWT) OR `X-PyCaret-Key: pck_…` (API key). JWT wins when both are present (common dev pattern: long-lived key in env + short-lived UI session). Revoked / expired / unknown keys all return 401.
- **Role model** — v1 restricts to `admin | member` literal. SPEC § 17.2 proposes a richer 6-role set; rolled forward when SSO lands.
- **14 new integration tests** in `services/api/tests/test_session20.py` covering:
  - Member CRUD: list, invite-existing, invite-unknown-404, non-admin-can't-invite, promote, demote, last-admin-guard (both demote + remove), remove.
  - API-key auth: happy-path auth, revoked rejected, bogus rejected, expired rejected (forges `expires_at` backwards), JWT-takes-precedence, missing-both-rejected.

### What landed — frontend

- **`<WorkspaceMembers>`** screen at `/workspaces/:wsId/members` — full CRUD UI for admins, read-only for members:
  - Invite form (admins only): email + role select + submit → fires `membersApi.invite`.
  - Members table with inline role select + Remove button per row.
  - Last-admin guard mirrored in UI: sole admin's role select is disabled (with tooltip) + Remove button disabled.
  - Non-admins see the list without invite form + without action column.
  - Own row flagged `(you)` next to the display name.
- **Wiring**:
  - `WorkspaceDetail` gains a "Members" button in the header nav alongside Pipelines / Deployments / LLM.
  - New route `/workspaces/:wsId/members` in `App.tsx`.
- **API bindings** — new `membersApi` module (list / invite / changeRole / remove) + `MemberRead` / `InviteRequest` / `PatchRoleRequest` / `WorkspaceRole` types.
- **4 new Vitest tests** — admin view (shows invite form, can change role), last-admin disables both select + remove, non-admin hides both invite + action column, invite submit fires API with chosen role.

### Headline metrics

| | Session 19 end | Session 20 end |
|---|---|---|
| API routes (under `/api/v1/`) | ~54 | **~58** (+4 members CRUD) |
| Server integration tests | 54 | **68** (+14) |
| Auth methods | JWT only | **JWT + X-PyCaret-Key** |
| UI screens | 14 | **15** (+ WorkspaceMembers) |
| UI tests | 48 | **52** (+4) |
| **Combined tests** | **134** | **148** (32 engine + 68 server + 52 web) |
| Production bundle (gz) | 98 kB | **99 kB** (+1 kB) |

### What's next (session 21)

- **Drift analyst** — 7th / 6th copilot (SPEC § 12.2). Needs `DriftReport` model (SPEC § 4.12) + scheduled `drift_detection_job` + a monitoring surface.
- **Audit logs** — SPEC § 17.4. Cross-cutting table + middleware that records every mutating call + `/admin/audit` viewer.

### And then session 22+

God-class drain (engine Phase 5) → 4.0.0 non-alpha release on PyPI.

---

## Session 19 — Failure debugger + Deployment reviewer + API keys — ✅

Two more LLM advisories (5 of 6 copilots in SPEC § 12.2 now live) + the first admin surface: personal API keys. What's left in session 20: drift analyst + audit logs + workspace member management.

### What landed — LLM advisories

- **`llm/consultations/failure_debugging.py`** (5th copilot) — reads a failed Run's error + event tail. System prompt classifies the cause as DATA / CONFIG / ENGINE, proposes a minimal config change that would unblock a retry, and flags uncertainty when there are multiple candidate causes.
- **`llm/consultations/deployment_risk_review.py`** (6th-ish copilot — drift is deferred) — reads a Pipeline + origin Run + leaderboard. System prompt tells the LLM to return a verdict starting with one of `APPROVE` / `APPROVE WITH CAVEATS: …` / `DO NOT DEPLOY: …`, walking through overfit / tiny-margin / small-sample / missing-imputer / missing-encoder / version-skew risks explicitly. UI tone-codes the verdict accordingly.
- **2 new routes**:
  - `POST /api/v1/llm/debug-run` — body `{run_id}`. Only `status='failed'` runs accepted (400 otherwise: `"debug is for failed runs only"`).
  - `POST /api/v1/llm/review-deployment` — body `{pipeline_id}`. Pulls the origin Run + leaderboard, persists the consultation correlated to `origin_run_id`.
- **6 new integration tests** in `services/api/tests/test_session19.py` covering the happy paths + the "only failed" + "only succeeded explains" cross-guard + 404 on unknown pipeline.

### What landed — API keys

- **`services/api/pycaret_server/api/api_keys.py`** — 3 routes:
  - `POST /auth/api-keys` — mint. Returns plaintext **once** (`pck_` prefix + `secrets.token_urlsafe(32)`). Hash + prefix stored; plaintext never.
  - `GET /auth/api-keys` — list the caller's keys. Never exposes plaintext.
  - `DELETE /auth/api-keys/{id}` — revoke (soft delete — `revoked_at` set; audit trail preserved). Can only revoke your own keys unless superuser.
- **Key format**: `pck_` recognisable prefix so leaks are greppable in logs + GitHub secret scanners; 32-byte url-safe-b64 body; total ~47 chars.
- **Middleware that accepts `X-PyCaret-Key` for programmatic traffic is session-20 work** — this session just ships the CRUD surface.

### What landed — frontend

- **`<FailureDebuggerCard>`** — inline card on `/runs/:id` when `status === 'failed'`. Red-tinted border (`border-danger-500/30`). Same opt-in pattern as `<RunExplainerCard>` — button fires the consultation on click, not on mount. Button flips "Diagnose" → "Re-diagnose" after first success.
- **`<DeploymentReviewModal>`** — modal on `/pipelines/:id`. Opens on "✨ Review" button click in the deploy sidebar. Auto-fires on open. Verdict text tone-coded: `DO NOT DEPLOY` → red, `APPROVE WITH CAVEATS` → warn-amber, `APPROVE` → success-green. UI does NOT block the Deploy button — the reviewer is advisory per SPEC § 12.3.
- **`<ApiKeysScreen>`** at `/account/api-keys` — list with status column (active / revoked / expired), per-row revoke button with confirm prompt, "New API key" form, and a one-time plaintext-display panel that appears once on successful creation with a Copy button + big warning.
- **Wiring**:
  - `RunDetail` now splits terminal-state rendering: `succeeded` → `<RunExplainerCard>`, `failed` → `<FailureDebuggerCard>` (was single "terminal" branch before).
  - `PipelineDetail` deploy sidebar gains an "✨ Review" button alongside Deploy; opens the modal pre-keyed to the current pipeline.
  - `Layout` top nav gains an **"API keys"** link.
- **API bindings** — `llmApi.debugRun` + `llmApi.reviewDeployment`; new `apiKeysApi` with list / create / revoke.
- **7 new Vitest tests** — 2 for FailureDebuggerCard, 2 for DeploymentReviewModal (verdict tone-coding verified), 3 for ApiKeysScreen (empty state, create flow with one-time plaintext, active/revoked status column).

### Headline metrics

| | Session 18 end | Session 19 end |
|---|---|---|
| LLM consultation types shipped (of 6 in spec § 12.2) | 3 | **5** (+ failure_debugging, deployment_risk_review) |
| API routes (under `/api/v1/`) | ~49 | **~54** |
| Server integration tests | 45 | **54** (+9) |
| UI shared components | 10 | **12** (+ FailureDebuggerCard, DeploymentReviewModal) |
| UI screens | 13 | **14** (+ ApiKeysScreen) |
| UI tests | 41 | **48** (+7) |
| **Combined tests** | **118** | **134** (32 engine + 54 server + 48 web) |
| Production bundle (gz) | 96 kB | **98 kB** (+2 kB) |

### AI at every stage of the product loop

```
Upload CSV            → ✨ AI          (dataset consultant — session 17)
New Experiment        → ✨ Ask AI      (experiment designer — session 18)
Run succeeds          → ✨ Explain     (run explainer — session 18)
Run fails             → ✨ Diagnose    (failure debugger — session 19)  ← NEW
Deploy a Pipeline     → ✨ Review      (deployment reviewer — session 19)  ← NEW
```

Only `drift_analysis` remains unimplemented — that's session 20, alongside the drift-report infrastructure it depends on.

### What's next (session 20)

- **Workspace members** — invite / list / role changes / remove. Needed before multi-user workflows land.
- **Drift analyst** — backend + `DriftReport` model (SPEC § 4.12) + scheduled `drift_detection_job`. UI surface on the Deployment detail screen or a dedicated `/monitoring` page.
- **`X-PyCaret-Key` auth middleware** — accept API keys as an alternative to JWT on all `/api/v1/*` routes.
- **Audit logs** — SPEC § 17.4. Cross-cutting table + middleware + viewer screen.

### And then session 21+

God-class drain (engine Phase 5) → 4.0.0 non-alpha release on PyPI.

---

## Session 18 — Experiment designer + Run explainer advisories — ✅

The **three classic copilots** the spec asks for (§ 12.2) are now live. Session 17 shipped the dataset consultant + router infrastructure; session 18 completes the trio:

- **Dataset consultant** (session 17) — "what's this dataset, what task fits, what risks are hiding"
- **Experiment designer** (session 18) — "given this dataset + this goal, design a full experiment"
- **Run explainer** (session 18) — "this run finished, explain what happened + what to try next"

All three route through the same `LLMRouter`, persist identical `LLMConsultation` rows, and hand the user an `LLMAdvice` envelope with `suggested_config_json` + `suggested_action` + `reasoning_summary` + `risk_flags`. The safety contract (SPEC § 12.3) holds: LLM proposes, user approves, deterministic engine executes.

### What landed — backend

- **2 new consultation modules** (~180 LOC Python):
  - **`llm/consultations/experiment_design.py`** — reads a CSV profile + a free-text user goal, serialises as JSON, asks the LLM for a RunConfig-shaped proposal (`task_type`, `target`, `train_size`, `fold`, `primary_metric`, `preprocessing`, `model_shortlist`, `class_imbalance_strategy`). System prompt tells the model to ground every choice in the profile + never invent columns.
  - **`llm/consultations/run_explanation.py`** — reads a completed Run's snapshot + leaderboard + full event stream, asks the LLM for a plain-prose explanation + prioritised next experiments. Event stream truncated to head-5 + tail-45 with a `__truncated__` marker so the prompt stays bounded.
- **2 new API routes** in `api/llm.py`:
  - `POST /api/v1/llm/design-experiment` — body `{workspace_id, data_source_id, goal}`. Same CSV-only / workspace-match guards as `analyze-dataset`. Pydantic `min_length=1` on `goal` fires 422 on empty input.
  - `POST /api/v1/llm/explain-run` — body `{run_id}`. Walks `run → experiment → project → workspace` for access control. Rejects non-terminal runs with 400 (`"wait for a terminal state before explaining"`). Correlates the consultation to its run via the `run_id`/`experiment_id`/`project_id` FKs on `LLMConsultation`.
- **6 new integration tests** (`services/api/tests/test_llm_advisories.py`): designer happy path + required-goal guard + non-csv-400; explainer happy path (actually runs a create-LR on iris → waits for succeeded → explains) + non-terminal guard + requires-configured-LLM.

### What landed — frontend

- **`<ExperimentDesignerModal>`** — opens from the New Experiment wizard. CSV picker (CSV-only; S3/Postgres filtered) + free-text goal textarea. On submit, renders the `LLMAdvice` envelope: suggested action, reasoning, risk-flag chips, suggested RunConfig as pretty-printed JSON, provider/model/latency footer.
- **`<RunExplainerCard>`** — sits inline on `/runs/:id`, only on terminal runs (`succeeded | failed | cancelled`). Button is opt-in (doesn't auto-fire on mount — explanations cost tokens; they shouldn't happen on every run view). After the LLM responds: plain-prose explanation, "ideas to try" list extracted from `suggested_config_json.next_actions`, risk-flag chips, re-explain button for follow-ups.
- **Wiring**:
  - `NewExperiment` header gains an **"✨ Ask AI"** button → opens the designer modal.
  - `RunDetail` drops `<RunExplainerCard runId={runId} />` between the Leaderboard and Promote sections, guarded on `terminal === true`.
- **API bindings** — `llmApi.designExperiment` + `llmApi.explainRun` added to `endpoints.ts`.
- **5 new Vitest tests** — 2 for RunExplainerCard (opt-in behaviour, click-to-fire + render), 3 for ExperimentDesignerModal (inert when closed, loads CSV-only options + keeps submit disabled, fires with correct args + renders advice).

### Headline metrics

| | Session 17 end | Session 18 end |
|---|---|---|
| LLM consultation types (of 6 planned in spec § 12.2) | 1 (dataset_analysis) | **3** (+ experiment_design, run_summary) |
| API routes (under `/api/v1/`) | ~47 | **~49** (+2) |
| Server integration tests | 39 | **45** (+6) |
| UI shared components | 8 | **10** (+ ExperimentDesignerModal, RunExplainerCard) |
| UI tests | 36 | **41** (+5) |
| **Combined tests** | **107** | **118** (32 engine + 45 server + 41 web) |
| Production bundle (gz) | 95 kB | **96 kB** (+1 kB) |

### The "beautiful product loop" now has AI at every stage

```
Upload CSV                     → ✨ AI  "analyze this dataset"    (session 17)
New Experiment                 → ✨ AI  "design an experiment"    (session 18)
Run completes                  → ✨ AI  "explain this run"        (session 18)
Promote + deploy + /predict                                        (session 16)
```

### What's next (session 19)

- **Admin screens** — users list, workspace membership + roles, API keys, audit logs. V2 foundation work, per SPEC § 17.
- **3 remaining copilot types** from § 12.2: `failure_debugging`, `deployment_risk_review`, `drift_analysis` — land alongside their UI surfaces (RunDetail on failure, DeploymentDetail, monitoring pages).

### What's next (session 20+)

God-class drain (engine Phase 5) → 4.0.0 (non-alpha) release.

---

## Session 17 — LLM router (Claude + OpenAI) + dataset consultant — ✅

The **AI-native** half of the Control Plane lands. From a browser, a user can configure their workspace's LLM provider (Claude or OpenAI), test the connection, and hit an "✨ AI" button next to any uploaded CSV to get a consultant's opinion on task type, target column, preprocessing strategy, and risk flags.

Per [`DECISIONS.md § 2026-04-24 · session-13 · 3`](DECISIONS.md), the router is **provider-agnostic from day one**: Anthropic and OpenAI are both first-class backends; adding Google / Azure / Ollama later is a one-class + one-factory-entry operation.

Per [`CONTROL_PLANE_SPEC.md § 12.3`](CONTROL_PLANE_SPEC.md#123-important-constraint), the LLM is **advisory**: every consultation returns `suggested_config_json` + `suggested_action` + `reasoning_summary` + `risk_flags`. The deterministic engine executes what the user approves; the LLM never triggers a side effect.

### What landed — backend

- **2 new DB tables** (Alembic migration `d582b350c276`):
  - `llm_provider_settings` — per-workspace provider config. `UniqueConstraint(workspace_id, provider)` so a workspace can retain an Anthropic + OpenAI history side-by-side; the `enabled` flag picks which one runs.
  - `llm_consultations` — append-only audit of every advisory call. Stores prompt, raw response, normalised `LLMAdvice`, latency, error. Optional FKs to project / experiment / run correlate consultations to the domain object that triggered them.
- **`services/api/pycaret_server/llm/`** module (~600 LOC Python):
  - `schemas.py` — Pydantic models. `LLMAdvice` is the canonical envelope; `LLMProviderSettingRead` deliberately drops `api_key_encrypted` + adds `has_api_key: bool` so the browser never sees plaintext.
  - `providers/base.py` — `LLMProvider` Protocol (one method: `complete(system, user, output_schema) -> dict`).
  - `providers/anthropic_provider.py` — Claude via tool-use. Declares an inline tool wrapping `output_schema`; consumes the first `tool_use` content block.
  - `providers/openai_provider.py` — OpenAI structured-output via `response_format={"type": "json_schema", ...}`. Works against native OpenAI API, Azure OpenAI, and any OpenAI-compatible endpoint (Ollama, vLLM) via `base_url`.
  - `providers/fake.py` — deterministic stand-in for tests + local dev, with a `canned_response` override.
  - `providers/__init__.py` — registry + `register_fake_for_tests()` helper that installs the fake under every provider name.
  - `router.py` — `LLMRouter`. `consult(session, ctx)` runs: load active setting → build provider → call → normalise to `LLMAdvice` → persist `LLMConsultation` (even on failure) → return. `test_connection(setting)` does a lightweight round-trip.
  - `consultations/dataset_analysis.py` — the dataset consultant. Reads the CSV's first 200 rows + total row count + column types + cardinality, serialises as JSON, asks the LLM for a RunConfig-shaped suggestion. Strict `additionalProperties: false` on top-level keys so the model can't invent fields.
- **`services/api/pycaret_server/api/llm.py`** — 5 paths / 6 operations:
  - `GET /api/v1/workspaces/{id}/llm/settings`
  - `PUT /api/v1/workspaces/{id}/llm/settings` (admin-gated; switching providers auto-disables the previous one)
  - `POST /api/v1/workspaces/{id}/llm/test-connection`
  - `POST /api/v1/llm/analyze-dataset` (body: `{workspace_id, data_source_id, task_type_hint?}`)
  - `GET /api/v1/workspaces/{id}/llm/consultations` (history, newest first, cap 500)
  - `GET /api/v1/llm/consultations/{id}`
- **App lifespan** now also resets the LLM router on shutdown (matches orchestrator + deployment registry).
- **`pyproject.toml` extras**: new `llm-anthropic`, `llm-openai`, `llm` (both). Neither SDK is required for the base install; `FakeLLMProvider` backs tests.
- **9 integration tests** (`services/api/tests/test_llm.py`) cover: settings empty state, upsert + API-key not leaked, unknown-provider 400, switching-providers disables previous, test-connection ok path, test-connection 400 when unconfigured, **analyze-dataset happy path** (end-to-end — upload CSV → configure LLM → analyze → list + get from history), analyze-dataset requires configured LLM (400), analyze-dataset rejects non-CSV source (400).

### What landed — frontend

- **New route `/workspaces/:wsId/llm`** — `LLMSettings.tsx` screen. Provider picker (6 options; Anthropic + OpenAI supported, 4 more disabled "(coming later)"), model name (auto-suggests defaults per provider), API key as `type="password"` (never round-tripped back via `GET /settings`), optional base_url, enabled toggle. "Test connection" button runs the lightweight round-trip.
- **`<AnalyzeDatasetModal>`** — opens with a `dataSourceId`, fires `llmApi.analyzeDataset`, renders the `LLMAdvice` envelope: suggested action as headline, reasoning as paragraph, risk flags as tone-coded chips, suggested config as pretty-printed JSON block, provider/model/latency in a footer. Esc-to-close + click-outside-to-close.
- **`<DataSourcesCard>`** — each CSV row now has an **"✨ AI"** button next to the delete button; clicking opens `<AnalyzeDatasetModal>` for that dataset.
- **`<WorkspaceDetail>`** header — third nav button ✨ LLM alongside Pipelines + Deployments, linking to the settings screen.
- **3 new Vitest tests** (`AnalyzeDatasetModal.test.tsx`): modal is inert when `open=false`, auto-fires the mutation on open and renders the advice envelope, close-button callback.

### Headline metrics

| | Session 16 end | Session 17 end |
|---|---|---|
| DB tables | 16 (14 app + 2 Alembic) | **18** (+ `llm_provider_settings`, `llm_consultations`) |
| Alembic migrations | 1 (baseline) | **2** |
| API routes (under `/api/v1/`) | ~42 | **~47** |
| Server integration tests | 30 | **39** (+9) |
| UI shared components | 7 | **8** (+ AnalyzeDatasetModal) |
| UI screens | 12 | **13** (+ LLMSettings) |
| UI routes | 12 | **13** |
| UI tests | 33 | **36** (+3) |
| **Combined tests** | **95** | **107** (32 engine + 39 server + 36 web) |
| Production bundle (gz) | 93 kB | **95 kB** (+2 kB) |

### Live-verified E2E

Against the real backend + FakeLLMProvider registered under every provider name:

```
[llm settings]     provider=anthropic model=claude-sonnet-4-5 has_api_key=True
[test connection]  ok=True latency=0ms
[csv upload]       iris.csv, 150 rows
[analyze]          provider=anthropic latency=0ms
  suggested_action: "Run a classification compare on iris with fold=5."
  risk_flags:       ['small_sample']
  suggested_config_json keys: ['task_type', 'target', 'primary_metric', 'preprocessing']
[history]          1 consultation(s); type=dataset_analysis
```

### What's next (session 18)

Spec § 12.2 lists 6 advisory features. Session 17 ships 1 (dataset_analysis). Session 18 adds the next two:

- **Experiment designer** — takes a dataset + user goal → proposes a full `RunConfig`. UI surface: a new "✨ Ask AI" button on the New Experiment wizard that pre-fills the dynamic form.
- **Run explainer** — reads a completed run's leaderboard + events → explains why the best model won + suggests next experiments. UI surface: a collapsible card on `/runs/:id`.

Plus: **admin screens** (users + API keys + audit logs — V2 foundation) start queuing up for session 19.

### What's next (session 20+)

Engine-side god-class drain → 4.0.0 (non-alpha) release.

---

## Session 16 — Pipelines, Deployments, CSV upload — closes the serving loop — ✅

The full Control Plane product loop is now live in the UI — **from a raw CSV upload through a promoted pipeline deployed behind a slug answering live predictions**, with no Python required.

### What landed

- **4 new screens** wired into the nav:
  - **`/workspaces/:wsId/pipelines`** (`Pipelines.tsx`) — workspace-scoped registry of promoted pipelines. Table with name, model_id, SHA-256 prefix, tags, created date.
  - **`/workspaces/:wsId/pipelines/:pipelineId`** (`PipelineDetail.tsx`) — pipeline metadata + a sidebar deploy-form (slug validator regex `[a-z0-9][a-z0-9-]{1,62}[a-z0-9]`, auth-mode selector) + a live-metrics table of every deployment backed by this pipeline.
  - **`/workspaces/:wsId/deployments`** (`Deployments.tsx`) — workspace-level deployments list with p50/p95 latency, inference count, error count, last-hit timestamp. Polls every 5 s so metrics stay fresh.
  - **`/deployments/:deploymentId`** (`DeploymentDetail.tsx`) — single-deployment view. Four stat cards (predictions / errors / p50 / p95) over a live `PredictTester`. Polls every 3 s. Sidebar shows deployment / workspace / pipeline IDs for copy-paste. Delete button with confirmation prompt (can also reach pipeline via link back).
- **2 new components**:
  - **`<PredictTester>`** — a monospace JSON-array textarea pre-seeded with an iris-shaped payload. Live-validates JSON as the user types (hint turns red, submit disables). On submit, renders a predictions table + latency + request-id chip. Pastes cleanly for bulk predictions.
  - **`<DataSourcesCard>`** — lives in the `WorkspaceDetail` sidebar. Lists existing CSV uploads with row count / file size / column count. File-picker + name input + submit wired to `dataSourcesApi.uploadCsv` (multipart). Per-row delete with confirmation.
- **API + types**:
  - `pipelinesApi` (list / get / remove) and `deploymentsApi` (list / get / create / remove / **predict**). `PredictRequest` + `PredictResponse` types mirror the backend contract.
  - `Deployment` type now imported in the endpoints module for `deploymentsApi` return types.
- **Nav**:
  - `WorkspaceDetail` header now has **Pipelines** + **Deployments** buttons at the top-right.
  - `RunDetail` post-promote hint now links directly to the pipeline detail page.
  - Runs-table rows in `ExperimentDetail` were already clickable (session 15).

### Headline metrics

| | Session 15 end | Session 16 end |
|---|---|---|
| UI screens | 8 | **12** (+ Pipelines / PipelineDetail / Deployments / DeploymentDetail) |
| UI shared components | 5 | **7** (+ PredictTester + DataSourcesCard) |
| UI routes | 8 | **12** |
| UI tests | 27 | **33** (+6: 3 PredictTester + 3 DataSourcesCard) |
| Combined tests | 89 | **95** (32 engine + 30 server + 33 web) |
| UI LOC | ~2,950 | **~3,800** (+850) |
| Production bundle (gz) | 89 kB | **93 kB** (+4 kB) |

### End-to-end, in 8 clicks — zero Python

1. `/setup` → bootstrap admin
2. `/` → pick workspace
3. Workspace sidebar → **upload CSV** (iris.csv, 150 rows, parsed + SHA-256'd)
4. Click project → **"New experiment"** → dynamic form from engine
5. Experiment screen sidebar → **plan=create, model=lr, source=iris.csv** → Submit
6. Run row clickable → `/runs/:id` → watch live WebSocket events, leaderboard materialises
7. **Promote** → land on `/workspaces/:wsId/pipelines/:id`
8. Sidebar deploy form → slug `iris-v1` → **Deploy** → `/deployments/:id` → **Send request** (PredictTester) → predictions + 0.9 ms latency

Live-verified against the real backend. 3-row predict on a freshly-deployed iris pipeline: latency = 0.9 ms, `inference_count` ticks to 3, `p50 = 0.9`, `p95 = 0.9`.

### What's next (session 17)

- **LLM router** (Anthropic Claude + OpenAI) + first 2 advisory endpoints:
  - **Dataset consultant** — reads a CSV's profile + returns a suggested task type, target column, preprocessing strategy, risk flags.
  - **Experiment designer** — takes a dataset + user goal → returns a proposed `RunConfig` the user reviews + approves.
- Both surface as panels in the UI: "Ask the AI" button on `WorkspaceDetail` / `NewExperiment` → modal with the advisory response.

### What's next (session 18+)

- **Admin screens** — users, API keys, audit logs (V2 foundations).
- **Monitoring + drift screens**.
- **God-class drain** (engine Phase 5) → 4.0.0 (non-alpha) release.

---

## Session 15 — Run detail + live WebSocket event stream — ✅

The final missing piece of the beautiful product loop. A user can now click any row in the experiment's runs table and land on a dedicated run-detail screen that shows engine events in real time, the sortable leaderboard, a cancel button while pending, and a promote-to-pipeline form on success.

### What landed

- **`<EventStream>` component** (`apps/web/src/components/EventStream.tsx`). Full WebSocket lifecycle: connects to `/api/v1/runs/:id/events/ws?token=<jwt>` using the current access token, parses each JSON message as a `WsEvent`, caps rendered history at 500 events (oldest dropped), auto-reconnects once on unexpected close (not on 4401/4403 — those are auth failures that shouldn't silently retry), resets state on run-id change, and renders events as a card list with a status indicator (connecting → live → closed / error), per-event timestamp, tone-coded kind text (started = teal, finished = green, failed = red, warning = amber), and optional duration.
- **`<Leaderboard>` component** (`apps/web/src/components/Leaderboard.tsx`). Renders any JSON-table shape the engine emits — zero hard-coded metric names. First-row column order is preserved. Click-to-sort per column (numeric sort for number-valued cells, string sort otherwise). Number formatter: integers stay bare, floats get 4 decimals, very small values get exponential notation. Empty state fallback until `Run.leaderboard` materialises.
- **`/runs/:runId`** screen (`apps/web/src/pages/RunDetail.tsx`). Status header with tone-coded label + ID + duration + error pre-block if failed. Cancel button (shown only while `queued` / `running`). Full-width live event stream. Leaderboard section. Promote-to-pipeline form (shown only on `succeeded`). Complete request snapshot at the bottom for reproducibility. Polls the run row every 2 s while pending; polling stops on terminal state.
- **Upgraded `ExperimentDetail`** sidebar:
  - **Model picker** — replaces the free-text `model_id` field with a `<select>` driven by `describeApi.models(task)`. Task-specific, with `is_available` flag propagated (unavailable models render as disabled `<option>`s with "(install required)" suffix).
  - **Data-source picker** — single combo-valued `<select>` mixing the workspace's CSV uploads (preferred, at the top) with the built-in sklearn sample datasets (useful fallback for a fresh install demo). Submit dispatches to either `data_source_id` or `sklearn_dataset` based on the selected value's prefix (`sklearn:` vs. UUID).
- **Runs table rows** in `ExperimentDetail` are now clickable — they link to the new `/runs/:id` screen.
- **API + type bindings** — `runsApi` (list for experiment / submit / get / events / cancel / wait / promote), `dataSourcesApi` (list / get / remove / **uploadCsv** with multipart `FormData`). Types: `DataSource`, `DataSourceKind`, `RunPlan`, `RunCreate`, `Pipeline`, `Deployment`, `WsEvent`.
- **8 new Vitest tests** — 4 for `<Leaderboard>` (empty state, column order preservation, numeric formatting, numeric sort round-trip) + 4 for `<EventStream>` with a controllable `FakeWebSocket` (connects to correct URL with token, renders live events, handles `run.closed` sentinel, surfaces auth-failure close codes).

### Headline metrics

| | Session 14 end | Session 15 end |
|---|---|---|
| UI screens | 7 | **8** (+ RunDetail) |
| UI shared components | 3 | **5** (+ EventStream + Leaderboard) |
| UI routes | 7 | **8** (+ `/runs/:runId`) |
| UI tests | 19 | **27** (+8) |
| Combined tests | 81 | **89** (32 engine + 30 server + 27 web) |
| UI LOC | ~2,100 | **~2,950** (+850) |
| Production bundle (gz) | 86 kB | **89 kB** (+3 kB) |

### The beautiful product loop, end-to-end

All in one session of UI work, with zero Python required:

```
1. /setup               → bootstrap admin
2. /login               → sign in
3. /                    → pick a workspace, or create one
4. /workspaces/:id      → pick a project, or create one
5. .../projects/:id     → click "New experiment"
6. .../experiments/new  → fill wizard (dynamic form from describe_setup_params)
7. .../experiments/:id  → pick plan (compare), sklearn:iris, click Submit
8. /runs/:id            → watch live events stream in, leaderboard render,
                          click "Promote" when it succeeds
```

Verified E2E against the live backend: a `create` run on `sklearn:iris` emits 4 events, produces a 4-row leaderboard with 7 metric columns, and promotes into a `Pipeline` row with a SHA-256 checksum. 19 classification models exposed for the picker.

### What's next (session 16)

- **Pipelines + Deployments screens** — `/pipelines/:id` and `/deployments/:id`. List, promote already runs; the missing piece is the UI for *deploying* a promoted pipeline behind a slug, plus the `/predict` test-form + request-log view.
- **CSV upload UI** — a small card on `WorkspaceDetail` or a new `/datasets` screen, using the `dataSourcesApi.uploadCsv` binding already shipped this session.

### What's next (session 17+)

- **LLM router** (Claude + OpenAI) + first 2 advisory endpoints (dataset analyst + experiment designer).
- **Admin screens** — users + API keys + audit logs.
- **God-class drain** → 4.0.0 (non-alpha) release.

---

## Session 14 — Project detail + Experiment wizard (100% data-driven dynamic form) — ✅

The centerpiece of MVP 3: a data scientist can now bootstrap → pick a workspace → pick a project → **configure a full experiment through a dynamic form that the UI has never heard of**, then submit runs against it. Zero hard-coded parameter names in the UI — the engine's `describe_setup_params(task)` is the single source of truth.

### What landed

- **Dynamic form infrastructure** — two new files that between them are the load-bearing contract from the engine to the UI:
  - **`apps/web/src/components/DynamicForm.tsx`** — `<ParamInput>` dispatches on `kind` (bool / int / float / enum / column / string) and returns the right native HTML input with validation hints (min/max, required, choices). `<DynamicForm>` groups params by `group` in the order declared by `schema.groups` and preserves user input as the form re-renders.
  - **`apps/web/src/components/DynamicForm.helpers.ts`** — pure helpers: `applyDefaults(schema, values)` seeds missing fields from schema defaults without clobbering user input; `stripDefaults(schema, values)` removes values equal to defaults so the API payload captures *user intent* only (engine owns defaults).
- **Three new screens**:
  - **`/workspaces/:wsId/projects/:projectId`** (`ProjectDetail.tsx`) — project header, tags, experiments list, "New experiment" button. Breadcrumb: Workspaces / {workspace} / {project}.
  - **`/workspaces/:wsId/projects/:projectId/experiments/new`** (`NewExperiment.tsx`) — two-card wizard. Card 1: name + task dropdown + target column (shown only for supervised tasks). Card 2: the dynamic form, seeded with schema defaults, reloaded whenever the task changes. Submits `POST /projects/{id}/experiments` with stripped (user-intent-only) `setup_params`.
  - **`/workspaces/:wsId/projects/:projectId/experiments/:experimentId`** (`ExperimentDetail.tsx`) — two-column layout. Main: config overview (param diff vs. engine defaults) + runs table (status-coloured + auto-polls every 2s while any run is queued/running). Sidebar: "New run" form — plan (setup|create|compare), model id (for create), sklearn sample dataset selector. Status column colour-coded via `STATUS_COLOR` map.
- **API + type bindings**:
  - `apps/web/src/api/types.ts` — new types: `SetupParam`, `SetupParamSchema`, `ModelCard`, `MetricCard`, `ExperimentCreate`.
  - `apps/web/src/api/endpoints.ts` — new `experimentsApi` (list / get / create / remove) and `describeApi` (setupParams / models / metrics).
- **Route wiring** — 3 new authenticated routes in `App.tsx`. `WorkspaceDetail.tsx` projects are now clickable links through the new hierarchy.
- **Tests** — 13 new vitest tests lock in the dynamic-form contract:
  - `<ParamInput>` renders the correct input type per `kind` (bool → checkbox, int/float → number with step, enum → select, column with columns → select, column without → text).
  - `applyDefaults` / `stripDefaults` round-trip correctly.
  - `<DynamicForm>` groups preserve `schema.groups` order; `hide` works; `onChange` bubbles merged values; empty schema doesn't crash.

### Headline metrics

| | Session 13 end | Session 14 end |
|---|---|---|
| UI screens | 4 (Setup / Login / Workspaces / WorkspaceDetail) | **7** (+ ProjectDetail + NewExperiment + ExperimentDetail) |
| UI components | 2 (AuthGate + Layout) | **3** (+ DynamicForm) |
| UI tests | 6 | **19** (+ 13 for DynamicForm / ParamInput / helpers) |
| UI LOC | ~1,300 | **~2,100** (+800) |
| Production bundle | 83 kB gz | **86 kB gz** (+3 kB) |
| Combined tests | 68 | **81** (32 engine + 30 server + 19 web) |

### What works today

The first beautiful product loop is about to be real. From a fresh clone, in two terminals:

```bash
# terminal 1
uv run --package pycaret-server pycaret-server serve --reload
# terminal 2
cd apps/web && npm run dev
```

Then in a browser:

1. http://localhost:3000/setup → bootstrap admin
2. Sign in → see workspaces → click a workspace
3. Click a project (or create one)
4. **"New experiment"** → pick classification, target=`target`, tune `fold=5` + `normalize=true` via the dynamic form → submit
5. Land on the experiment detail → pick `plan=compare`, `dataset=iris` in the sidebar → **"Submit run"**
6. Watch the runs table auto-refresh; status flips `queued` → `running` → `succeeded` with the duration filled in.

All without typing Python.

### Zero hard-coded parameter names

This is the design principle session 14 locks in: the UI has never heard of `normalize`, `fold`, `train_size`, etc. The engine's `describe_setup_params` is rendered to a form via a single `kind → JSX` dispatcher. Tomorrow the engine can add `transformation_method: "quantile" | "yeo-johnson"` (enum, group "Preprocessing") and the form picks it up with zero UI changes.

Verified end-to-end against the live backend:

```
setup-params: 13 params in 6 groups
  groups: ['Data', 'Experiment', 'Cross-Validation', 'Preprocessing', 'Compute', 'Logging']
experiment created: task=classification, target=target
  stored setup_params: {'fold': 5, 'normalize': True, 'session_id': 42}
```

### What's next (session 15)

- **`/runs/:id`** — dedicated run detail screen with **live WebSocket event stream** (every engine `Event` rendered in real time), leaderboard table with sortable columns, artifact download, promote-to-pipeline button, cancel button.
- **Data source integration** in the New Run form — replace the "sklearn sample dataset" picker with a proper `data_source_id` selector (drives against the existing CSV upload endpoint).
- **Better model picker** — replace the free-text `model_id` with a dropdown driven by `describeApi.models(task)`.

### What's next (session 16+)

- Dataset upload UI + profile screen.
- LLM router + first 2 advisory endpoints (dataset analyst + experiment designer).
- Admin screens.
- God-class drain → 4.0.0 release.

---

## Session 13 — Monorepo restructure + Control Plane vision lock-in — ✅

Largest structural change since the Part-2 platform kickoff. The flat layout (`pycaret/`, `pycaret-server/`, `pycaret-ui/`, `docker/` all at root) is gone; replaced by the canonical `apps/` + `services/` + `packages/` + `infra/` layout from the Control Plane spec. All 68 tests remain green.

Also: the product vision got materially bigger. The owner's side-research produced a comprehensive "PyCaret Control Plane" technical spec (24 sections, ~300 planned endpoints, full LLM + monitoring + drift + Kubernetes + multi-cloud story). We accepted it as the canonical scope and updated every relevant doc.

### What landed — structure

```
BEFORE                          AFTER
pycaret/                        packages/engine/pycaret/
tests/                          packages/engine/tests/
pycaret-server/                 services/api/
pycaret-ui/                     apps/web/
docker/                         infra/docker/

(+ new empty stubs)
                                apps/desktop/           (V2 Electron)
                                services/worker/        (V2 job runner)
                                services/deployment-runtime/  (V2 serving)
                                packages/sdk-python/    (V2 Python client)
                                packages/shared-schemas/ (V2 JSON schemas)
                                infra/helm/             (V2 K8s chart)
                                infra/terraform/aws|gcp|azure  (V2 IaC)
```

Root `pyproject.toml` is now a **pure workspace manifest** — no package metadata, just `[tool.uv.workspace]` + shared ruff defaults. Engine metadata moved to `packages/engine/pyproject.toml` alongside the source. Root `tests/` folder absorbed into `packages/engine/tests/` (the server already had its own under `services/api/tests/`; the UI under `apps/web/src/*.test.tsx`).

All Python package names are unchanged: `import pycaret` + `import pycaret_server` work identically. `pip install pycaret` still builds from `packages/engine/`. PyPI + notebook users are unaffected.

### What landed — docs

- **`CONTROL_PLANE_SPEC.md`** (new) — owner's 24-section spec checked in verbatim. Canonical product scope.
- **`VISION.md`** (new) — 1-page product statement distilled from the spec.
- **`ARCHITECTURE.md`** (rewritten) — full system architecture: monorepo layout, service topology, engine/backend/frontend/infra breakdown, LLM router plan, RunConfig contract. The previous engine-internal content moved to `ARCHITECTURE_ENGINE.md` (preserved for history).
- **`ROADMAP.md`** (rewritten) — restructured around MVP 1 (engine) / MVP 2 (backend) / MVP 3 (UI) / MVP 4 (self-hosted) / V2 / V3. Every already-shipped phase mapped into its MVP bucket; forward work laid out through session ~20.
- **`DECISIONS.md`** — 4 new entries: (1) restructure now, (2) Electron deferred to V2, (3) LLM **router** supporting Claude + OpenAI from day one (not single-provider), (4) product name = "PyCaret" + UI brand = "PyCaret Control Plane".
- **`AGENTS.md`** (rewritten) — new 60-second briefing, new repo map, new "which phase am I in?" decision tree, new common-task playbooks for backend routes / frontend screens / LLM features.
- **`CONTRIBUTING.md`** (rewritten) — new local setup flow (uv + npm dual pipeline), new test commands, new PR checklist.
- **`README.md`** (rewritten) — repositioned as the platform's landing page (not just an engine README). Three deployment-mode table. Both notebook quickstart + Control Plane quickstart side by side.
- **`PLATFORM_QUICKSTART.md`** — all paths updated to new structure.
- **11 new scaffolded stub READMEs** — every empty future directory has a README explaining its future role so the structure is self-documenting.

### What landed — code

- Root `pyproject.toml` restructured; `packages/engine/pyproject.toml` + `packages/engine/README.md` written.
- `infra/docker/Dockerfile.api` updated: `COPY packages/engine/...` + `COPY services/api/...` + `uv pip install -e ./packages/engine -e ./services/api`.
- `infra/docker/Dockerfile.ui` updated: `COPY apps/web/...`.
- `infra/docker/docker-compose.yml` updated: build context `../..`, service renamed `ui` → `web`, image `pycaret-web:dev`.
- `.github/workflows/test.yml` updated: ruff paths, pytest paths, UI job `working-directory: apps/web`, cache path `apps/web/package-lock.json`, UI job name "Web (…)".
- 4 ruff import-order auto-fixes applied during the first check on the new paths.

### Headline metrics (unchanged by restructure)

| | Session 12 end | Session 13 end |
|---|---|---|
| Monorepo packages | 3 | **3** (structure only) |
| Total tests | 68 | **68** (32 engine + 30 server + 6 web) |
| Top-level dirs with real code | 5 (engine + server + ui + docker + tests) | **4** (`apps/`, `services/`, `packages/`, `infra/`) |
| Doc count in `docs/revamp/` | 9 | **11** (+ VISION, + CONTROL_PLANE_SPEC; ARCHITECTURE split into 2) |
| Forward-roadmap scope | ~5 sessions (Phase 10 finish) | **~8 sessions to full MVP + multi-session V2 backlog** |

### What's next (session 14)

Per the refreshed roadmap:

- **Session 14** — `/projects/:id` + `/experiments/:id` experiment wizard (dynamic form from `describe_setup_params`, 4 config modes: manual / assisted / auto / expert).
- **Session 15** — `/runs/:id` with live WebSocket event stream + leaderboard + artifact actions.
- **Session 16** — Trial entity + Model Library DB sync.
- **Session 17** — LLM router (Claude + OpenAI providers) + first 2 advisory endpoints.
- **Session 18** — Dataset upload UI + profile screen.
- **Session 19** — Admin screens + API keys + audit logs (V2 foundations).
- **Session 20+** — God-class drain → 4.0.0 (non-alpha) release.

---

## Session 12 — Frontend scaffold + bootstrap flow (Phase 10 start) — ✅

The platform finally has a face. A user can navigate to `http://localhost:3000`, bootstrap their admin account, sign in, create workspaces, create projects — all against the same `pycaret-server` we finished in session 11.

### What landed

- **`pycaret-ui/` — new monorepo sibling** (~1,300 LOC TSX + config). Vite 5 + React 18 + TypeScript 5 (strict, `verbatimModuleSyntax`) + Tailwind 3 (dark-mode first) + TanStack Query + Zustand + React Router 6 + axios.
- **Typed API client** in `src/api/` — hand-written mirrors of the Pydantic schemas (`types.ts`) + per-route axios methods (`endpoints.ts`). `npm run gen:api` regenerates `schema.ts` from a live `/openapi.json` for when the surface grows.
- **Auth layer**:
  - `useAuthStore` (Zustand) — single source of truth for `{accessToken, refreshToken, user}`. Refresh token persisted to `localStorage` so page reloads don't kick the user back to `/login`.
  - axios interceptor — single-flight `refresh()` on 401 (no thundering-herd if N requests 401 at once). Access token never touches `localStorage`; it's restored from the refresh token at load time.
  - `<AuthGate>` — guards authenticated routes; shows a "Restoring session…" flash during the one-shot refresh, then either renders children or redirects to `/login` with `state.from` set.
- **4 screens**, all live against the backend:
  - `/setup` — first-run wizard. Disabled if server is already bootstrapped.
  - `/login` — sign in. Redirects to `/setup` if server isn't bootstrapped yet.
  - `/` — workspace list + "New workspace" side-card.
  - `/workspaces/:id` — workspace header + project list + "New project" side-card (with comma-separated tag input).
- **Design system primitives** in `src/index.css`: `.btn-primary/.btn-secondary/.btn-ghost/.btn-danger`, `.input`, `.field`, `.card`, `.hint`, `.error`, `.kbd`. Slate-leaning palette, teal accent.
- **Tests** (Vitest + Testing Library, jsdom env):
  - `auth.test.ts` — localStorage persistence + clear + no-op refresh without token.
  - `AuthGate.test.tsx` — redirects to `/login` when no tokens; renders children when authed.
  - `Setup.test.tsx` — renders form + submit-disabled-until-password-valid.
- **Build pipeline** — typecheck (`tsc -b`), lint (ESLint flat config, 0 warnings), test (Vitest), production build (Vite). Current bundle: **254 kB raw / 83 kB gzipped**.
- **Docker**:
  - `docker/Dockerfile.ui` — two-stage (Node 22-alpine build → nginx 1.27-alpine runtime), non-root `nginx` user, healthchecked.
  - `docker/nginx.ui.conf` — SPA history fallback, `/api/` + `/healthz` reverse proxy to `api:8000`, WebSocket upgrade on `/api/v1/runs/*` with 1h timeouts for long runs.
  - `docker-compose.yml` now has a `ui` service depending on `api:service_healthy`, exposing port 3000.
- **CI** — new `ui` job (typecheck + lint + test + build) on every push. Wired into `ci-status`. Uses Node 22 + npm cache.

### Headline metrics

| | Session 11 end | Session 12 end |
|---|---|---|
| Monorepo packages | 2 (pycaret + pycaret-server) | **3** (+ pycaret-ui) |
| Total tests | 62 | **68** (+6 UI) |
| LOC | engine ~49k + server ~3.6k | **+ ui ~1.3k TSX** |
| Docker images | 1 (API) | **2** (API + UI) |
| CI jobs | 3 (lint, test, notebooks) | **4** (+ ui) |

### What works today

```bash
# Terminal 1 — backend
cd pycaret-server && uv run pycaret-server serve --reload

# Terminal 2 — frontend
cd pycaret-ui && npm install && npm run dev

# Open http://localhost:3000/setup → bootstrap → sign in → click around
```

Or with Docker:

```bash
docker compose -f docker/docker-compose.yml up --build
# http://localhost:3000  — full stack
```

### What's next (session 13)

4 remaining screens to close Phase 10:

1. **`/workspaces/:id/projects/:id`** — project detail: experiment list + "New experiment" button.
2. **`/projects/:id/experiments/:id`** — experiment setup form rendered **100% from `describe_setup_params`** (the single most important UX principle — zero UI code hard-codes a parameter name).
3. **`/runs/:id`** — live event stream via WebSocket + leaderboard table + artifact download + promote-to-pipeline button.
4. **Admin** — user management + workspace settings (single screen, admin-only).

Plus polish: light-mode, error boundaries, toast system for non-form errors, keyboard shortcuts.

Phase 10 is likely 2-3 more sessions before it's beta-ready.

---

## Session 11 — Phase 9 finish: data sources, deployments, cancel, alembic — ✅

Closes Phase 9. The backend is now feature-complete for Part-2's API surface — a client can upload real data, train a model, promote it, deploy it behind a slug, and serve predictions through the same process — all under migration control.

### What landed

- **Data-source module** (`pycaret_server/api/data_sources.py`, ~220 LOC)
  - `POST /api/v1/workspaces/{id}/data-sources/upload` — streaming multipart CSV with 64 MB cap, on-the-fly SHA-256, quick `pd.read_csv(nrows=5)` sample for column metadata, uploaded file stored under `${ARTIFACT_DIR}/data-sources/<uuid>.csv`.
  - `POST /api/v1/workspaces/{id}/data-sources` — register S3 or Postgres connector config (no connectivity check yet).
  - `GET /api/v1/workspaces/{id}/data-sources`, `GET /api/v1/data-sources/{id}`, `DELETE /api/v1/data-sources/{id}` (cleans the uploaded file).
  - Run submit now accepts `data_source_id` + optional `target` override. The orchestrator resolves the CSV path at dispatch time; unsupported kinds reject early with 400.
- **Serving module** (`pycaret_server/serving.py` + `api/deployments.py`, ~400 LOC combined)
  - `DeploymentRegistry` — process-local LRU caching fitted pipelines keyed by slug, with rolling 100-sample latency window → p50/p95.
  - `POST /api/v1/runs/{id}/promote` — promote a succeeded Run's `pipeline_pickle` artifact to a workspace-scoped `pipelines` row.
  - Pipeline CRUD: `GET /workspaces/{id}/pipelines`, `GET/DELETE /pipelines/{id}` (409 if deployments still reference it).
  - `POST /api/v1/pipelines/{id}/deployments` — create a `Deployment` with `endpoint_slug` (lowercased slug regex), `auth_mode` (workspace|api-key|public).
  - `GET /api/v1/workspaces/{id}/deployments`, `GET/DELETE /api/v1/deployments/{id}`.
  - **`POST /api/v1/deployments/{slug}/predict`** — slug → load → predict, updates inference_count + last_inference_at + p50/p95 on the row. Errors tick `error_count`.
- **Run cancellation** (`pycaret_server/runs/orchestrator.py`, diff ~40 LOC)
  - `RunOrchestrator.cancel(run_id)` sets a per-run `threading.Event`.
  - Worker polls the event via `_checkpoint()` at every stage boundary (pre-load, post-load, post-fit, post-plan). Raises `_CancelledError` → `Run.status = "cancelled"`.
  - `POST /api/v1/runs/{id}/cancel` returns the current row; terminal states are a no-op.
- **Alembic baseline** (`pycaret-server/alembic.ini`, `pycaret_server/migrations/`)
  - 1 revision (`9f9b7c770df0_baseline_schema`) capturing all 14 app tables + all indexes + all unique constraints.
  - `pycaret_server/db/bootstrap.py::ensure_schema` replaces lifespan's `create_all`. Auto-migrates empty SQLite (dev); demands explicit migration on Postgres/MySQL (prod).
  - **`pycaret-server migrate [--url ... --revision head]`** CLI subcommand for ops.
  - A legacy `create_all`-seeded DB is detected (`users` table present, no `alembic_version`) and auto-stamped to baseline, so upgrading existing deployments is transparent.
- **App factory** tears down the `DeploymentRegistry` alongside the `RunOrchestrator` on shutdown so reload mode doesn't carry stale pipelines across processes.

### Headline metrics

| | Session 10 end | Session 11 end |
|---|---|---|
| Total tests | 52 (32 engine + 20 server) | **62** (32 engine + 30 server) |
| API routes (under /api/v1) | 26 + 1 WS | **39** + 1 WS |
| pycaret-server LOC | ~2,400 | **~3,600** |
| Alembic revisions | 0 | **1 (baseline)** |
| Platform phases done | 🟢 9 core | ✅ **Phase 9 fully complete, Phase 8 fully complete** |

### What works today — end-to-end demo flow

```bash
export TOKEN=...  # from /api/v1/auth/login
# 1. upload a CSV
curl -sX POST .../data-sources/upload \
  -H "authorization: bearer $TOKEN" \
  -F "name=iris.csv" -F "file=@iris.csv"
# 2. submit a run from it
curl -sX POST .../experiments/$EXP/runs \
  -d '{"plan":"create","model_id":"lr","data_source_id":"'$DS'","target":"target"}'
# 3. wait until done
curl -sX POST .../runs/$RUN/wait?timeout_s=120
# 4. promote the fitted pipeline
curl -sX POST .../runs/$RUN/promote -d '{"name":"iris-v1"}'
# 5. deploy it
curl -sX POST .../pipelines/$PIPE/deployments -d '{"endpoint_slug":"iris-v1"}'
# 6. SERVE predictions
curl -sX POST .../deployments/iris-v1/predict \
  -d '{"rows":[{"sepal length (cm)":5.1,"sepal width (cm)":3.5,...}]}'
```

### What's next (session 12)

Two credible paths:

- **Phase 10 start — Frontend (React UI).** 8 screens: setup / login / workspaces / project / experiment / run / admin-users / admin-workspace. Vite + React 18 + TanStack Query + Plotly.js. First session scaffolds the Vite app, typed API client from `/openapi.json`, auth + bootstrap + workspace screens; subsequent sessions do experiment / run / deploy.
- **Phase 5 — God-class drain.** 10 verbs on `pycaret/core/experiment.py` still delegate to `self._legacy`. Migrate them onto `sklearn.pipeline.Pipeline` directly, in `save_model → predict_model → create_model → tune_model → ensemble_model → blend_models → stack_models → calibrate_model → compare_models → finalize_model` order. Each verb = ~1 session.

Either route is independent; the frontend can consume the current API immediately.

---

## Session 10 — Run execution + event stream (Phase 9 core complete) — ✅

The scaffold from session 9 gets a heart: `POST /api/v1/experiments/{id}/runs` now actually runs a PyCaret experiment and streams events back to any client that asks.

### What landed

- **`pycaret_server/runs/` subsystem** — 4 new modules, ~580 LOC:
  - `broker.py` — `EventBroker`, a thread-safe fan-out that bridges worker-thread event emission to asyncio-consumer WebSocket handlers via `loop.call_soon_threadsafe`.
  - `logger_bridge.py` — `DBEventLogger(pycaret.logging.BaseLogger)` that persists every engine `Event` as an `events` row and republishes through the broker.
  - `plans.py` — pure "plan executor": `setup` | `create` | `compare` mapped onto engine verbs, plus a `load_sklearn_dataset(name)` helper that pulls tiny iris / wine / breast_cancer / diabetes frames from sklearn (no network required).
  - `orchestrator.py` — `RunOrchestrator` with a 2-thread `ThreadPoolExecutor`, full lifecycle transitions (queued → running → succeeded|failed), pipeline pickling to `${PYCARET_ARTIFACT_DIR}/runs/<run_id>/pipeline.pkl`, SHA-256 checksums, leaderboard → JSON on the Run row, `Artifact` row written for every fitted pipeline.
- **`pycaret_server/api/runs.py`** — 5 HTTP routes + 1 WebSocket:
  - `POST /api/v1/experiments/{id}/runs` → 202 + queued Run.
  - `GET /api/v1/experiments/{id}/runs` → list.
  - `GET /api/v1/runs/{id}` → status + leaderboard + metrics summary.
  - `GET /api/v1/runs/{id}/events?limit=&after_id=` → paginated replay.
  - `POST /api/v1/runs/{id}/wait?timeout_s=30` → block until terminal (notebook + test convenience).
  - `WS /api/v1/runs/{id}/events/ws?token=<jwt>` → replays stored events then live-streams until `run.closed`.
- **Request snapshot** — every Run stores the full submit payload (task, target, setup params, plan, data source) on `Run.snapshot` for reproducibility.
- **App lifespan** now tears down the orchestrator cleanly on shutdown so worker threads stop between tests.
- **6 new integration tests** — submit validation (3 bad shapes), setup-only lifecycle, create-plan + artifact persistence, list-by-experiment, WebSocket replay, WebSocket 4401 on missing token. All green.

### Headline metrics

| | Session 9 end | Session 10 end |
|---|---|---|
| Total tests | 46 (32 engine + 14 server) | **52** (32 engine + 20 server) |
| API routes (under `/api/v1`) | 21 | **26** + 1 WebSocket |
| pycaret-server LOC | ~1,800 | **~2,400** |
| Platform phases | 🟡 9 partial, 🟡 11 partial | 🟢 **Phase 9 core complete** |

### What works today

```bash
# 1. bootstrap + login
curl -sX POST localhost:8000/api/v1/setup/bootstrap \
  -H 'content-type: application/json' \
  -d '{"email":"me@x","password":"supersecret","workspace_name":"demo"}' | jq -r .access_token
# 2. create workspace -> project -> experiment (classification on iris)
# 3. submit a run
curl -sX POST localhost:8000/api/v1/experiments/$EXP/runs \
  -H "authorization: bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"plan":"create","model_id":"lr","sklearn_dataset":"iris"}'
# 4. wait until done
curl -sX POST "localhost:8000/api/v1/runs/$RUN/wait?timeout_s=60" \
  -H "authorization: bearer $TOKEN" | jq .status
```

### What's next (session 11)

- `/api/v1/deployments/*` + in-house serving (catch-all `/predict` route; promote a `Pipeline` row to a `Deployment` row).
- Data-source connectors (CSV upload + S3 + Postgres); move the `sklearn_dataset` field into a fallback-only niche.
- Alembic baseline migration — stop relying on `Base.metadata.create_all` in lifespan.
- Fold-metric extraction — unpack `leaderboard` into per-fold × per-model × per-metric rows.
- Run cancellation (cooperative `threading.Event` consumed by a periodic check hook in the engine).

---

## Session 9 — Backend scaffolding (Phase 8 + Phase 9 + Phase 11 partial) — ✅

Major Part-2 milestone. With engine 4.0.0a1 shipped (41-dep lean install, sklearn 1.8 / NumPy 2.4 / pandas 3.0), the backend comes online as a monorepo sibling.

### What landed

- **`pycaret-server/` monorepo sibling** — new package with its own `pyproject.toml`, installed via uv workspace.
- **14 SQLAlchemy tables** (matches `PLATFORM_PLAN.md § 3`): `users`, `workspaces`, `workspace_members`, `projects`, `data_sources`, `experiments`, `runs`, `events`, `artifacts`, `fold_metrics`, `pipelines`, `pipeline_project_links`, `deployments`, `api_keys`, `sessions`. Full relationships mapped; delete cascades in place.
- **Auth** — bcrypt password hashing + JWT access-token (60 min default) + rotating refresh-token (30 d, session-row storage, hashed server-side).
- **29 routes** mounted at `/api/v1/*`:
  - `setup/{status,bootstrap}` — first-run flow
  - `auth/{login,refresh,logout,me}` — JWT auth
  - `describe/{models,models/{id},metrics,setup-params}` — engine introspection proxy
  - `workspaces/*` CRUD
  - `workspaces/{id}/projects/*` CRUD
  - `projects/{id}/experiments/*` CRUD
- **FastAPI app factory** with CORS + lifespan that auto-creates SQLite tables on first boot.
- **CLI** — `pycaret-server serve [--reload]` starts uvicorn.
- **Multi-stage Dockerfile** (Python 3.13-slim + uv + non-root runtime user + healthcheck).
- **`docker-compose.yml`** for local dev (SQLite + artifact volume at `./data/`).
- **14 integration tests** (pytest + httpx TestClient) — green in ~8 s.
- **CI updated** to test both engine (32 tests) and server (14 tests).
- **docs/revamp/PLATFORM_QUICKSTART.md** — 5-min clone-to-running walkthrough.

### Headline metrics

| | Session 6 end | Session 9 end |
|---|---|---|
| Packages in the monorepo | 1 (pycaret) | **2** (pycaret + pycaret-server) |
| Total tests | 32 (engine) | **46** (32 engine + 14 server) |
| SQLAlchemy tables | 0 | **14** |
| API routes | 0 | **29** (8 meta, 21 under `/api/v1`) |
| Docker artifacts | — | Dockerfile.api + compose |
| Core platform phases | 🔴 0/6 not started | ✅ Phase 8 complete, 🟡 Phase 9 mostly done, 🟡 Phase 11 partial |

### What works today

Clone the repo, `uv sync --all-packages --all-extras`, `uv run --package pycaret-server pycaret-server serve --reload`, open http://localhost:8000/docs, POST to `/api/v1/setup/bootstrap`, get a token, create workspaces / projects / experiments through the Swagger UI. The engine's `list_models` / `describe_model` / `describe_setup_params` are exposed as live endpoints that a React form can render from.

### What's next (session 10)

- `POST /api/v1/experiments/{id}/runs` → background-worker dispatch to `pycaret.tasks.*Experiment` (thread-based for v1).
- `GET /ws/runs/{id}/events` WebSocket fan-out from the engine's `BaseLogger`.
- `/api/v1/deployments/*` + in-house serving (catch-all `/predict` route).
- Data-source connectors (CSV upload + S3 + Postgres).
- Alembic baseline migration replacing boot-time `create_all`.

---

*Session 6 status (previous engine cleanup + platform plan):*

## Session 6 — Cleanup pass 2 + Platform-Plan authored — ✅

Two distinct efforts landed this session.

### A. Engine cleanup pass 2

| Metric | Session 5 end | Session 6 end | Δ |
|---|---:|---:|---:|
| `pycaret/` source LOC | 51,976 | **50,544** | **−1,432** |
| Zero-import leaf files | 3 present | **0** (all deleted) | − |
| Killed-verb methods still in codebase | 15 | **0** | **−15** |
| cuml GPU-fallback shim | present (143 LOC) | **0** (deleted) | − |
| Full test suite | 32/32 green, 2:07 | **32/32 green, 1:37** | −30s |

Breakdown:
- Deleted `pycaret/distributions.py`, `pycaret/internal/cloudpickle_compat.py` — both had zero callers.
- Deleted `pycaret/internal/cuml_wrappers.py` + stubbed the 6 GPU-fallback call sites in the 4 model-container files (unreachable anyway with default `gpu_param=False`).
- Deleted the `pycaret/loggers/` shim package; re-pointed 7 `BaseLogger` import sites to `pycaret.logging.base` directly.
- Deleted 9 killed-verb methods wholesale across the god-class + 5 task oop files: `check_fairness`, `check_drift`, `dashboard`, `create_api`, `create_docker`, `create_app`, `convert_model`, `deploy_model`, `eda`. 15 method definitions × ~77 LOC avg = 1,156 LOC gone. Zero behaviour change (public API didn't expose them).

### B. Application-platform plan authored

User laid out Part-2 vision: PyCaret as an enterprise-grade open-source AutoML platform — credible alternative to DataRobot / H2O.ai. Detailed design captured in [`PLATFORM_PLAN.md`](PLATFORM_PLAN.md).

Headline:
- Monorepo: `pycaret` (library) + `pycaret-server` (FastAPI) + `pycaret-ui` (React) + `pycaret-cli` (CLI).
- Hierarchy: Workspace → Project → Experiment → Run → Pipeline (11 SQLAlchemy tables).
- SQLite default; Postgres/MySQL opt-in via `DATABASE_URL`.
- First-run self-service admin setup.
- `docker compose up` from fresh clone → running app at http://localhost:3000, no config.
- JWT auth, admin/member roles.
- WebSocket fan-out of the engine's event stream to the UI.
- React setup form rendered from `pycaret.api.describe_setup_params` (zero hardcoded param names).

6 new phases added to the roadmap (7-12). **Gated on Phase 5 — `pycaret==4.0.0alpha0` being released to PyPI** — so the library stays laser-focused on shipping first.

---

*Session 4 status (repo restructure + issue triage):*

## Session 4 — Repo restructure + dev/agent docs + issue triage — ✅ DONE

User ask: "clear the folder, restructure for dev contributions, get rid of old stuff, one notebook per use-case fully working, MD files for agents, download all open issues, start cleaning them up."

### What shipped


## Session 4 — Repo restructure + dev/agent docs + issue triage — ✅ DONE

User ask: "clear the folder, restructure for dev contributions, get rid of old stuff, one notebook per use-case fully working, MD files for agents, download all open issues, start cleaning them up."

### What shipped

- **Purged dead weight:** `Docker_files/`, `docs/source/` (Sphinx), `docs/{Makefile,make.bat,make.sh,logs.log}`, `tutorials/{legacy_v3,time_series,translations}/`, `tutorials/pycaret_cheat-sheet_in_excel.xlsx`, root-level `logs.log`, `.readthedocs.yml`, `.slugignore`. Rewrote `.gitignore` to be 4.0-clean.
- **Renamed `tutorials/` → `notebooks/`** (modern naming).
- **5 working, executed notebooks** — one per task — under `notebooks/`. Generated by `scripts/build_notebooks.py`, executed end-to-end on Python 3.13, outputs persisted in the `.ipynb` JSON so GitHub renders them.
- **`/AGENTS.md` at repo root** — 60-second briefing for AI coding agents (TL;DR, rules, conventions, repo map, common-task recipes).
- **`docs/for_agents/` — 5 deep-dive files:** engine walkthrough, typed results, event stream, introspection API, verb×task cheatsheet.
- **`docs/for_developers/` — 5 dev-onboarding files:** setup, testing, god-class-draining playbook, coding style, release process.
- **`CONTRIBUTING.md` rewritten** for 4.0.
- **All 388 open GitHub issues downloaded and triaged:**
  - 8 (2%) — fixed in 4.0 → close
  - 92 (24%) — out of scope per kill-list → close
  - 123 (32%) — stale (no update since 2023) → auto-ping, close after 30d
  - 58 (15%) — still-relevant bugs → Phase 5 queue
  - 107 (28%) — still-relevant enhancements → per-item decision
  - **224 of 388 (58%) can be closed or auto-pinged without further triage.**
- **`scripts/triage_issues.py`** + **`scripts/build_notebooks.py`** — two maintenance scripts, re-runnable.
- **NumPy 2 compat fix** in `pycaret/internal/patches/sklearn.py` (`np.product` → `np.prod`) surfaced during notebook exec.

### Final repo layout

```
pycaret/
├── README.md                 README.md
├── AGENTS.md                 briefing for AI coding agents (NEW)
├── CONTRIBUTING.md           rewritten for 4.0 (UPDATED)
├── CODE_OF_CONDUCT.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore                (UPDATED)
├── pycaret/                  engine source (~49K LOC)
├── tests/                    4 test files (32 tests, 100% green)
├── notebooks/                5 executed end-to-end notebooks (NEW)
├── datasets/                 bundled sample CSVs
├── scripts/                  maintenance scripts (NEW)
└── docs/
    ├── images/               logo etc.
    ├── revamp/               engineering narrative (8 top-level docs)
    │   ├── ARCHITECTURE.md
    │   ├── AUDIT.md
    │   ├── DECISIONS.md
    │   ├── KILL_LIST.md
    │   ├── README.md
    │   ├── ROADMAP.md
    │   ├── STATUS.md
    │   ├── release_notes_pycaret4.md
    │   ├── github_issues/    issue snapshot + triage (NEW)
    │   └── thinking/         intermediate rationale
    ├── for_agents/           agent-facing deep dives (NEW, 5 files)
    └── for_developers/       dev onboarding (NEW, 5 files)
```

---

*Session 3 status (functional API kill):*

## Session 3 — Functional API killed; 4.0 is OOP-only — ✅ DONE

The user made the final call: "nobody will migrate 3→4, 4 is a totally new thing, I really want to get rid of 90% tech debt now." This session deletes the module-level functional API entirely. PyCaret 4.0 has exactly one canonical way to use it: the `Experiment` classes.

### Before → after

| Metric | Session 2 end | Session 3 end | Δ |
|---|---:|---:|---:|
| Source LOC in `pycaret/` | ~60,700 | ~49,400 | **−11,300** |
| Test files | 45 | 4 | **−41** |
| Full-suite pass rate | 77% (568/734) | **100% (32/32)** | +23pp |
| Public module-level functions | 145 | **0** | **−145** |
| Canonical API surfaces | 2 (functional + OOP) | **1 (OOP)** | **−1** |
| Module-level mutable state | 5 ContextVars / globals | **0** | **−5** |

### What's now the canonical 4.0 API

```python
from pycaret.tasks import (
    ClassificationExperiment, RegressionExperiment,
    ClusteringExperiment, AnomalyExperiment, TimeSeriesExperiment,
)
from pycaret import save_model, load_model

exp = ClassificationExperiment(target="y", session_id=42).fit(df)
best = exp.compare_models().best
preds = exp.predict_model(best).predictions
save_model(best, "model.pkl")
```

### What was deleted / thinned

- **5 `functional.py` files** totalling 11,333 LOC — gone.
- **41 test files** coupled to the functional API — gone. Replaced by 4 OOP-native test files (32 tests, 100% green in ~2 min).
- **`pycaret/core/state.py`** (ContextVar machinery) — gone. No more implicit "current experiment."
- **6 task module `__init__.py`s** — collapsed from 40-entry re-export lists (~90 LOC each) to thin docstring + single-line import (~15 LOC each).
- **`TSForecastingExperiment`** class name → **`TimeSeriesExperiment`** (cleaner, matches the task module name).
- **README.md** fully rewritten for the 4.0 positioning.
- **Tutorials** moved to `tutorials/legacy_v3/`; `tutorials/README.md` documents the 4.0 OOP pattern for all 5 tasks.

### What's new

- **`pycaret.tasks`** now exports all 5 task subclasses: `ClassificationExperiment`, `RegressionExperiment`, `ClusteringExperiment`, `AnomalyExperiment`, `TimeSeriesExperiment`.
- **`pycaret.core.SupervisedExperiment` / `UnsupervisedExperiment`** — the two intermediate bases. Supervised verbs live on `SupervisedExperiment` only; unsupervised tasks don't inherit verbs they can't implement.
- **`pycaret.persistence`** — stateless `save_model(model, path)` / `load_model(path)` utilities, also re-exported as `pycaret.save_model` / `pycaret.load_model`.
- **`tests/test_e2e_oop.py`** — end-to-end smoke tests for all 5 tasks.

### What's still in play

- The 3.x god-class in `pycaret/internal/pycaret_experiment/` is still alive as `Experiment._legacy`. Verbs still delegate to it. **This is deliberate** — it keeps the public API stable while each verb is rewritten natively verb-by-verb on top of `sklearn.pipeline.Pipeline` in subsequent sessions.
- Tutorial notebooks are preserved as references in `tutorials/legacy_v3/` but not yet re-authored as 4.0 OOP notebooks.

---

*Session 2 status (Phase 4 architecture kickoff):*

## Session 2 — Phase 4 Engine Architecture — 🟡 ARCHITECTURE LANDED

The 3.x "functional API + OOP afterthought" design has been replaced by a real sklearn-composable engine. `Experiment` is now a `BaseEstimator` subclass; task subclasses preconfigure it; every verb returns a typed dataclass; events flow through a `BaseLogger`.

### What's new this session

| New package | What it provides |
|---|---|
| `pycaret.core` | `Experiment` base class (sklearn-compatible), `TaskType` enum, 9 typed result dataclasses, `ContextVar`-backed current-experiment state, `PyCaretError` hierarchy |
| `pycaret.logging` | Structured event stream: `Event` / `EventKind` (22 kinds) / `BaseLogger` / `MemoryLogger` (thread-safe, file-teeing). Subscribers for React UI fan-out |
| `pycaret.api` | JSON-serializable model/metric/parameter introspection: `list_models`, `describe_model`, `list_metrics`, `describe_setup_params`, `list_available_models` |
| `pycaret.tasks` | `ClassificationExperiment(Experiment)` — the first task subclass, end-to-end green |

### Headline validation

- **End-to-end green on `juice` dataset:** `ClassificationExperiment(target="Purchase").fit(df).compare_models().predict_model(best)` — returns typed dataclasses, emits 5 structured events.
- **Sklearn compatibility verified:** `get_params()` returns 15 params, `sklearn.base.clone(exp)` works, `__sklearn_tags__().estimator_type == "classifier"`.
- **17/17 new-architecture unit tests pass in 0.2s.**
- **No regression on the legacy subset** (23/23 pass on `test_models.py` + `test_datasets.py` + `test_core_architecture.py`).
- **JSON round-trip proven:** `json.dumps(describe_setup_params('classification').to_dict())` produces a valid React-form schema (13 params across 6 groups).

### What's still in play

- Legacy `pycaret/classification/{functional.py,oop.py}` both still exist and work; notebook users see no change.
- Phase 5 rewires the functional API's `setup/compare_models/...` to construct a `pycaret.tasks.ClassificationExperiment` and drive it through the new core, closing the loop.

---

*Session 1 status (Phase 0 + most of Phase 1):*

## Phase 0 (Groundwork) — ✅ COMPLETE

| Task | State | Evidence |
|---|---|---|
| Clone upstream repo | ✅ | `C:\Users\moezs\pycaret\pycaret\` |
| Install `uv` | ✅ | `uv 0.11.7` |
| Determine Python / sklearn target | ✅ | Python 3.13 primary; sklearn 1.7 transitional (see DECISIONS.md) |
| Scaffold `docs/revamp/` | ✅ | README, AUDIT, KILL_LIST, ROADMAP, DECISIONS, STATUS, thinking/ |
| Write v4 `pyproject.toml` | ✅ | Hatchling backend, uv lockfile, lean deps, no mlflow/comet/parallel/yellowbrick |
| Create uv venv on target Python | ✅ | `.venv/` on Python 3.13.13 |
| Package imports after amputation | ✅ | All 6 public submodules import |
| End-to-end smoke test | ✅ | `setup → compare_models → predict_model` on `juice` dataset |
| Full test-suite run captured | ✅ | 568 passed / 158 failed / 8 skipped in 34:26 · see `thinking/phase0_failure_landscape.md` |

## What was amputated in Phase 1 (done opportunistically during Phase 0)

Deleted from source tree:
- `pycaret/parallel/` (fugue backend)
- `pycaret/internal/parallel/`
- `pycaret/loggers/{mlflow,comet,wandb,dagshub,dashboard}_logger.py`
- `pycaret/internal/patches/yellowbrick.py`
- `pycaret/internal/plots/yellowbrick.py`
- 11 test files (parallel / mlflow / create_{api,app,docker} / dashboard / drift / fairness / sklearn-intelex)

Rewired in source:
- `pycaret/loggers/__init__.py` reduced to `BaseLogger` only
- `compare_models` signatures: `parallel` argument removed from 7 files
- `_parallel_compare_models` method deleted from `supervised_experiment.py`
- `MlflowLogger/CometLogger/WandbLogger/DagshubLogger/show_yellowbrick_plot/skplt` stubbed in `tabular_experiment.py` with `NotImplementedError`-raising placeholders (will be replaced by Plotly-native plots + built-in logger in Phases 2-3)
- `pycaret/internal/patches/yellowbrick` module-import side effects replaced with `contextlib.nullcontext()` in `tabular_experiment.py` plot dispatch
- `pycaret/utils/_dependencies.py` — dropped `distutils.LooseVersion` (removed in Python 3.12), now uses `packaging.version.Version` and stdlib `importlib.metadata`
- `FastMemory.__init__` — joblib 1.4+ removed `bytes_limit` kwarg from `Memory.__init__`; now forwarded to `reduce_size()` per new API
- `np.NaN` → `np.nan` (NumPy 2.0 compat)
- BATS / TBATS containers now try-import their tbats backend and mark themselves inactive if missing (keeps the `numpy<2` tbats constraint out of the default install)
- `plotly_resampler` (two display-format paths in time_series/forecasting/oop.py) stubbed with `NotImplementedError`
- `scikitplot` import removed from `internal/plots/helper.py` (was just a thin matplotlib re-export)

## What's in `pyproject.toml` now

| Category | Packages |
|---|---|
| Core (30 → **19**) | numpy, pandas, scipy, scikit-learn, joblib, cloudpickle, lightgbm, category-encoders, imbalanced-learn, plotly, kaleido, matplotlib (transitional), ipython, ipywidgets, tqdm, jinja2, requests, psutil, nbformat, xxhash |
| `models` extra | xgboost, catboost, kmodes, mlxtend |
| `tuners` extra | optuna, optuna-integration, scikit-optimize, hyperopt |
| `analysis` extra | shap, interpret, umap-learn |
| `anomaly` extra | pyod, numba |
| `timeseries` extra | statsmodels, sktime, pmdarima (tbats/statsforecast dropped) |
| `prophet` extra | prophet |
| `dev` / `test` groups | ruff, mypy, pre-commit / pytest, pytest-xdist, pytest-cov, nbval |

Gone from deps entirely (kill list satisfied):
- mlflow, comet-ml, wandb, dagshub
- fugue, dask, distributed, ray[tune], tune-sklearn
- yellowbrick, mljar-scikit-plot, schemdraw, plotly-resampler
- evidently, fairlearn, ydata-profiling, explainerdashboard
- gradio, fastapi, uvicorn, boto3, m2cgen, moto
- flask, Werkzeug, dash[testing]
- scikit-learn-intelex, trio

## Headline metrics

- **Source tree LOC (baseline):** 62,164
- **Source tree LOC (after Phase 1 amputation):** ~60,700 *(small — we cut a lot of deps but the big god-classes still sit in `internal/pycaret_experiment/`; Phase 2-3 is where the real LOC drop comes)*
- **Tests:** 815 collected (down from ~900 due to kill-list deletions), 0 collection errors. First run: **568 passed / 158 failed / 8 skipped (77.4% pass on first pass, see `thinking/phase0_failure_landscape.md` for the root-cause clustering).** Three more engine-only test files deleted after the run.
- **In-session fixes already applied after the baseline run:**
  - Logger regression (`'bool' object has no attribute 'log_experiment'`) — fixed by rewriting `BaseLogger` as a no-op hook surface and having `_convert_log_experiment` always return an instance.
  - sklearn 1.7 `_check_reg_targets` signature change in the custom MAPE container.
- **uv venv install time:** ~2 minutes for `--all-extras`
- **End-to-end smoke:** setup + compare_models (3 models) + predict_model on `juice` dataset → LogisticRegression selected; predictions shape (321, 21)

## Next up — Phase 2 (Modernization)

The failure landscape (see `thinking/phase0_failure_landscape.md`) gives Phase 2 a concrete ROI-ordered punch list:

1. **`internal/preprocess/iterative_imputer.py`** — swap `self._validate_data` for the new sklearn helpers. Unblocks 13 tests in one file.
2. **Time-series test-harness `PeriodIndex` name drift** — hunt-and-replace `'Period'` → `'period[M]'` in TS test expectations. Unblocks ~90 tests if the pattern is consistent.
3. **Add `__sklearn_tags__` to `internal/tunable.py`** — unblocks the ~6 tunable-estimator tests and future-proofs custom user subclasses.
4. **Delete `test_convert_model.py`** (m2cgen feature is killed, file is dead).
5. Sweep the remaining ~10 scattered failures in `test_multiclass.py`, `test_overflow.py`, `test_utils.py`, etc.

**Projected pass rate after (1)–(4): ~92%.** That's the exit gate for calling Phase 2 "done" for supervised modules.

## Open questions for the user

1. Should the 4.0 work commit directly on `main` or on a `v4` branch? (Nothing has been committed yet; workspace is dirty.)
2. Keep a `v3.4.0` branch / tag of the 3.x line before merging 4.0? Recommended yes.
3. Target first `4.0.0-alpha` tag — this session, or after Phase 2 repairs?
