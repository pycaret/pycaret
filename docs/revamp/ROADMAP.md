# PyCaret 4.0 — Phased Roadmap

A phase is not "done" until its exit criteria are met and a `DECISIONS.md` entry is written.

## Phase 0 — Groundwork (this session) — ✅ COMPLETE

- [x] Clone upstream, install `uv`
- [x] Verify sklearn support matrix ⇒ Python 3.13 as primary dev target (see DECISIONS.md for the 3.14/PEP-649 finding)
- [x] Scaffold `docs/revamp/` (this directory)
- [x] Complete baseline audit (`AUDIT.md`) and kill list (`KILL_LIST.md`)
- [x] Write new `pyproject.toml` v4 (lean deps, uv-first, hatchling)
- [x] Create `.venv` via `uv sync`, verify `import pycaret` + all 6 public submodules import
- [x] End-to-end smoke (`setup → compare_models → predict_model`) green on `juice` dataset
- [x] Full test-suite run captured in `thinking/phase0_pytest_run1.log`

**Exit criteria met:** Repo builds under `uv`, all public submodules import, a smoke test passes end-to-end, a test-failure inventory exists.

## Phase 1 — Amputation (remove the kill list) — ✅ MOSTLY COMPLETE

Done opportunistically during Phase 0 rather than waiting for a separate pass.

1. [x] Removed `pycaret/parallel/` and `pycaret/internal/parallel/`; dropped `parallel` args from 7 files.
2. [x] Removed `loggers/{mlflow,comet,wandb,dagshub,dashboard}_logger.py`. Deferred: the new `pycaret/logging/` module with an in-memory + file-backed structured event stream — Phase 4 owns this (it's coupled to the UI event design).
3. [ ] Remove `create_api`, `create_app`, `create_docker`, `dashboard`, `check_drift`, `check_fairness`, `eda`, `convert_model`, `deploy_model` **methods** from all experiment classes. *(Tests for these are already deleted; methods still exist in `*/functional.py` and `*/oop.py`. Leave until Phase 2 so we don't compound churn while debugging sklearn 1.7 drift.)*
4. [x] Removed `internal/patches/yellowbrick.py`; plot branches in `tabular_experiment.py` point at `_v4_removed` stubs.
5. [x] Deleted 11 dead test files and the mlflow-custom-tag methods inside the 4 supervised-module test files.

**Exit criteria met:** Package imports; 815 tests collect cleanly; smoke test green.

## Phase 2 — Modernization (compat with current sklearn / NumPy / pandas / Python)

Goal: remove all upper-bound pins, get on Python 3.14 / sklearn 1.8 / NumPy 2.x / pandas 2.2+.

1. Upgrade transformers in `internal/preprocess/*` to new sklearn tagging (`__sklearn_tags__`), `set_output` API, and any dropped private API.
2. Replace deprecated `np.X` / `pd.X` calls (audit `FutureWarning` output from the test run).
3. Update `internal/pipeline.py` and `internal/meta_estimators.py` for new estimator protocol.
4. Unpin `sktime` and absorb its API changes in `time_series/`.

**Exit criteria:** `pytest tests/test_classification.py tests/test_regression.py tests/test_clustering.py tests/test_anomaly.py` fully green on Python 3.14 + sklearn 1.8.

## Phase 3 — Plotly plot rewrite

Goal: replace the yellowbrick plot dispatcher with a clean Plotly module.

1. New `pycaret/plots/` (flat, no `internal/plots/`). One file per plot family:
   - `classification_curves.py` (ROC, PR, threshold)
   - `classification_matrix.py` (confusion, class prediction error, classification report)
   - `regression_diagnostics.py` (residuals, prediction error, Cook's distance)
   - `clustering.py` (elbow, silhouette, intercluster distance)
   - `feature.py` (RadViz, manifold)
   - `model_selection.py` (learning curve, validation curve, RFECV)
2. Each function returns a `plotly.graph_objects.Figure`; `plot_model` dispatches via a registry `dict[str, Callable]`. No giant if/elif chain.
3. Visual polish — unified theme (Plotly Template), consistent colour palette, dark-mode-friendly.

**Exit criteria:** `pytest tests/test_classification_plots.py tests/test_regression_plots.py` green; `yellowbrick`, `mljar-scikit-plot`, `schemdraw` no longer in `pyproject.toml`.

## Phase 3.5 — Functional API killed, OOP-only — ✅ COMPLETE (session 3)

- [x] All 5 `functional.py` files deleted (~11,300 LOC).
- [x] `pycaret.tasks` exports all 5 task subclasses (`ClassificationExperiment`, `RegressionExperiment`, `ClusteringExperiment`, `AnomalyExperiment`, `TimeSeriesExperiment`).
- [x] `pycaret.core.SupervisedExperiment` / `UnsupervisedExperiment` intermediate bases.
- [x] `pycaret.save_model` / `pycaret.load_model` stateless top-level utilities.
- [x] `pycaret/core/state.py` deleted (no ContextVar, no implicit state).
- [x] All 6 task-module `__init__.py`s collapsed to thin re-exports.
- [x] 41 functional-API-coupled tests deleted; 4 OOP-native test files remain (32/32 pass).
- [x] README rewritten for 4.0 positioning; tutorials doc updated; 3.x notebooks archived under `tutorials/legacy_v3/`.

**Exit criteria met:** `pycaret.classification.setup` raises `AttributeError`; the OOP API is the only canonical surface; `pytest tests/` is 100% green.

## Phase 4 — API for agents / React UI — 🟡 ARCHITECTURE LANDED (session 2)

Goal: make the engine introspectable and scriptable by an external process.

1. [x] Public `pycaret.api` submodule: `list_models(task)` (19 classification + 26 regression cards), `describe_model(task, id)`, `list_metrics(task)`, `describe_setup_params(task)` returning JSON-serializable `ModelCard` / `MetricCard` / `ParameterCard` / `SetupParamSchema` dataclasses.
2. [x] Typed return objects for every verb — `CompareResult`, `CreateResult`, `TuneResult`, `EnsembleResult`, `BlendResult`, `StackResult`, `CalibrateResult`, `FinalizeResult`, `PredictResult` in `pycaret.core.results`.
3. [x] Streaming events through `pycaret.logging.MemoryLogger` — every verb emits typed `Event`s with durations and payloads; `BaseLogger.subscribe(callback)` fans out to a React UI / agent.
4. [ ] No prints to stdout — legacy code paths still print progress bars; new `Experiment` surface doesn't, but audit remains for Phase 5.
5. [x] `Experiment(BaseEstimator)` in `pycaret.core.experiment` — sklearn-compatible (get_params, set_params, __sklearn_tags__, __sklearn_is_fitted__, clone-safe).
6. [x] `ClassificationExperiment(Experiment)` in `pycaret.tasks.classification` — end-to-end green on `juice` dataset.

**Exit criteria progress:** `json.dumps(describe_setup_params('classification').to_dict())` produces a valid dynamic-form schema (13 params across 6 groups). ✅

**Remaining work for full Phase 4:** extend `pycaret.tasks` with `RegressionExperiment`, `ClusteringExperiment`, `AnomalyExperiment`, `TimeSeriesExperiment` (session 3). Rewire the notebook-functional API (`setup`, `compare_models`, ...) to construct and drive a `pycaret.tasks.*` subclass rather than a legacy god-class (session 5).

## Phase 5 — Docs, notebooks, README, release

1. Re-run all 6 tutorial notebooks end-to-end on 4.0; commit with fresh outputs.
2. Rewrite `README.md` with the 4.0 positioning and install-with-uv quickstart.
3. Full test matrix in CI: Python 3.11 / 3.12 / 3.13 / 3.14.
4. Cut 4.0.0 release.

**Exit criteria:** Green CI across the matrix; notebooks run clean; README reflects 4.0.

## Out of scope (explicit non-goals)

- Multi-GPU / distributed training (no parallel).
- Hosted experiment tracking (no mlflow/comet).
- Deploy helpers (no boto3/docker/gradio/fastapi in core).
- Backward compatibility with 3.x *internal* APIs — only notebook golden path preserved.
