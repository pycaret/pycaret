# Changelog

User-facing release notes for PyCaret. Every release entry is a summarised view of the engineering log in [`docs/revamp/release_notes_pycaret4.md`](docs/revamp/release_notes_pycaret4.md) — go there for full commit-by-commit detail.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [PEP 440](https://peps.python.org/pep-0440/).

---

## [4.0.0a8] — 2026-05-08

**Drop ``imbalanced-learn`` and ``category-encoders``.** Both were left in core deps as transitional placeholders for the legacy 3.x preprocessor. Sessions 35-39 drained that preprocessor; it has been unreachable from any active code path for weeks. This release removes the deps and deletes the corresponding dead code. Core dep count: **12 → 10**.

### Removed

- ``imbalanced-learn`` — gone from core deps. The legacy ``Pipeline`` class that subclassed ``imblearn.pipeline.Pipeline`` is also gone — every native verb has been operating on ``sklearn.pipeline.Pipeline`` since session 24.
- ``category-encoders`` — gone from core deps. The 4.0 native preprocessor uses sklearn-native encoders (``OneHotEncoder`` / ``OrdinalEncoder``); BaseN / Target encoder users were already on sklearn 1.3+'s native ``TargetEncoder`` path through the engine setup.
- Deleted ``packages/engine/pycaret/internal/preprocess/`` (the entire legacy preprocessor: preprocessor.py, transformers.py, iterative_imputer.py, target/, time_series/forecasting/preprocessor.py).
- Deleted ``packages/engine/pycaret/internal/pipeline.py`` (the legacy ``Pipeline`` and ``TimeSeriesPipeline`` classes).
- Deleted ``packages/engine/pycaret/internal/patches/sklearn.py`` (the only file that imported ``internal.pipeline``).
- ``packages/engine/pycaret/utils/_show_versions.py`` — pruned the dep list from 35 stale 3.x names down to the 10 we actually require + the 9 we ship as opt-in extras.

### Core dep set after a8

```
numpy, pandas, scipy, scikit-learn, joblib, plotly, tqdm, requests, jinja2, ipython
```

That's it. Down from ~60 in 3.x.

### Verified

- Engine test suite: **330 pass / 1 skip** (no regression vs ``a7``).
- Clean-venv smoke test: ``pip install pycaret==4.0.0a8`` in a fresh ``python -m venv`` pulls 0 of {imblearn, category-encoders}; ``ClassificationExperiment(...).compare_models()`` runs to completion.

### Install

```bash
pip install --pre pycaret==4.0.0a8
```

---

## [4.0.0a7] — 2026-05-08

**Revive ``plot_model`` and ``evaluate_model``.** The Plotly plot library that landed in sessions 47-52 (33 chart functions across ``pycaret.plots.{classification,regression,feature,clustering,anomaly,time_series}``) is now wired into the OOP entry points. Together with ``a6``, **all six** verbs that originally shipped as ``NotImplementedError`` stubs in ``a2`` are now functional.

### Added

- ``Experiment.plot_model(estimator, plot=None, save=False, **kw)`` — looks up ``plot`` in a per-task registry and returns a ``plotly.graph_objects.Figure``. ``save=True`` writes ``f"{plot}.png"`` (requires ``pycaret[export]`` for ``kaleido``); ``save="my.png"`` writes to that path.
- ``Experiment.evaluate_model(estimator)`` — returns a ``dict[str, Figure]`` of the curated diagnostic bundle for the task. Plots that fail (e.g. shap missing, estimator lacking ``feature_importances_``) are silently dropped rather than raising.
- Per-task plot registries on every leaf class:
  - **Classification** (16 kinds): ``auc`` / ``roc``, ``pr``, ``confusion_matrix``, ``calibration``, ``threshold``, ``lift``, ``gain``, ``class_distribution``, ``permutation`` / ``feature``, ``pdp``, ``ice``, ``shap_summary`` / ``summary``, ``shap_beeswarm``. Default: ``'auc'``.
  - **Regression** (12 kinds): ``residuals``, ``residuals_distribution``, ``prediction_error`` / ``error``, ``learning`` / ``learning_curve``, ``feature``, plus the cross-task ``permutation`` / ``pdp`` / ``ice`` / ``shap_*`` group. Default: ``'residuals'``.
  - **Clustering** (6 kinds): ``cluster`` / ``distribution``, ``elbow``, ``silhouette``, ``silhouette_plot``, ``embedding``. Default: ``'cluster'``.
  - **Anomaly** (4 kinds): ``score``, ``anomaly_map``, ``feature_anomaly``, ``score_vs_feature``. Default: ``'score'``.
  - **Time-series** (8 kinds): ``forecast``, ``decomposition`` / ``decomp``, ``acf``, ``pacf``, ``diagnostics`` / ``residuals``, ``cv``. Default: ``'forecast'``.
- 19 new tests in ``packages/engine/tests/test_session54_plot_model_dispatcher.py`` covering all five tasks. Engine total: 330 pass / 1 skip.

### Changed

- ``test_session46_phase6_legacy_deletion.py`` — removed the two remaining ``NotImplementedError`` lockdown tests (``plot_model``, ``evaluate_model``) and updated the docstring. Phase 6's deletion of the legacy directory is still locked.

### Install

```bash
pip install --pre pycaret==4.0.0a7
# Static export (PNG/SVG/PDF):
pip install --pre 'pycaret[export]==4.0.0a7'
```

---

## [4.0.0a6] — 2026-05-08

**Revive 4 of the 6 ``NotImplementedError`` stubs.** The phase-6 deletion in `4.0.0a2` removed legacy implementations of ``plot_model`` / ``evaluate_model`` / ``interpret_model`` / ``automl`` / ``get_leaderboard`` / ``check_stats`` and shipped each as a stub raising ``NotImplementedError``. This release restores 4 of them on the new native engine surface. ``plot_model`` and ``evaluate_model`` remain stubs pending the Plotly rewrite (separate track).

### Added

- ``Experiment.get_leaderboard()`` returns a defensive copy of the most recent ``compare_models`` leaderboard. Snapshotted into ``_fit_state["last_leaderboard"]`` after every supervised + time-series ``compare_models`` call. Raises ``RuntimeError`` if no compare has run.
- ``Experiment.automl()`` — convenience wrapper for ``compare_models(n_select=1)`` → ``tune_model``. Returns the tuned ``sklearn.pipeline.Pipeline``. Pass-throughs for ``optimize`` / ``n_iter`` / ``turbo`` / ``include`` / ``exclude`` / ``fold`` / ``fit_kwargs`` / ``round`` / ``verbose``.
- ``Experiment.interpret_model()`` — wraps ``shap.Explainer`` over a pipeline's preprocessor + final estimator. Returns a ``shap.Explanation``. ``plot=`` argument renders ``'summary'`` / ``'beeswarm'`` / ``'bar'`` / ``'waterfall'`` inline. ``shap`` is an **optional** extra: ``pip install pycaret[interpret]``.
- ``TimeSeriesExperiment.check_stats()`` — runs Summary / Ljung-Box (white noise) / ADF + KPSS (stationarity) / Shapiro-Wilk (normality) over the experiment's series. Returns a 6-column ``DataFrame``. ``test=`` filter and ``split='all'|'train'|'test'``.
- New optional extra: ``[interpret] = ["shap>=0.46"]``. Folded into ``[full]``.
- 11 new tests in ``packages/engine/tests/test_session53_revive_legacy_verbs.py``. Total engine: 265 pass / 1 skip when ``shap`` is installed.

### Install

```bash
pip install --pre pycaret==4.0.0a6
# Or with SHAP for interpret_model():
pip install --pre 'pycaret[interpret]==4.0.0a6'
```

---

## [4.0.0a5] — 2026-05-08

**Cap IPython at `<9` to keep Google Colab kernels alive.** `4.0.0a4` declared `ipython>=8.18` with no upper bound; pip's resolver pulled IPython 9.x, which removed the `IPython.utils.coloransi.TermColors.Green` attribute that Colab's `google.colab._shell_customizations` reads at kernel startup. Result: kernel restart-loop with `AttributeError: type object 'TermColors' has no attribute 'Green'`. Cap fixes it.

### Fixed

- `ipython>=8.18` → `ipython>=8.18,<9`. The 4.0 engine only uses IPython for display detection and rich repr; nothing in the IPython 9 API matters to us. Cap is purely to coexist with Colab's preinstalled stack.

### Install

```bash
pip install --pre pycaret==4.0.0a5
```

If you were stuck in a Colab kernel restart-loop on `4.0.0a4`, you must **Runtime → Disconnect and delete runtime** before installing — the broken IPython 9 wheel persists on disk across kernel restarts; only deleting the runtime nukes it.

---

## [4.0.0a4] — 2026-05-08

**Colab-friendly install.** `4.0.0a3` triggered a Colab kernel restart-loop because the dep pins forced pip to upgrade `scikit-learn` and `imbalanced-learn` over Colab's preinstalled image, which broke the C-extension ABI for other libraries the runtime preloads. This release relaxes those floors so `pip install --pre pycaret` is a no-op for users who already have a recent scientific stack.

### Fixed

- `scikit-learn>=1.7` → `scikit-learn>=1.5`. We still test against the latest sklearn; the floor is just the oldest version we will not actively reject.
- `imbalanced-learn>=0.13` → `imbalanced-learn>=0.12`.
- `pycaret.__version__` was hardcoded to `"4.0.0a2"` in [`pycaret/__init__.py`](packages/engine/pycaret/__init__.py) and never followed `pyproject.toml`. It now reports the real installed version.

### Install

```bash
pip install --pre pycaret==4.0.0a4
```

---

## [4.0.0a3] — 2026-05-08

**First PyPI release of the 4.0 line.** Functionally identical to `4.0.0a2`; this release exists to land the artifacts on PyPI now that maintainer account access is restored. `4.0.0a1` and `4.0.0a2` were only ever distributed via GitHub release downloads — this is the first version installable via `pip install --pre pycaret`.

### Install

```bash
pip install --pre pycaret
```

Pip ignores PEP 440 pre-releases by default, so existing `pip install pycaret` users on the 3.x line are unaffected.

### Changed

- Engine source unchanged from `4.0.0a2` (the [release notes for a2](#400a2--2026-04-24) describe the surface).
- Build: re-cut from current source under `uv build --package pycaret`; sdist + wheel published to PyPI.

---

## [4.0.0a2] — 2026-04-24

**Third alpha release.** Two big themes since `4.0.0a1`:

1. **PyCaret Control Plane** is alive — a full FastAPI backend + React UI that turn the engine into a multi-user ML platform with 6 LLM-native AI copilots.
2. **The 4.0 engine `Experiment` class is fully drained from its 3.x god-class** — every public verb runs natively on top of sklearn instead of delegating through `self._legacy.*`.

### Install

```bash
pip install https://github.com/pycaret/pycaret/releases/download/v4.0.0a2/pycaret-4.0.0a2-py3-none-any.whl
```

For the platform layer:

```bash
git clone https://github.com/pycaret/pycaret.git
cd pycaret/infra/docker
docker compose up
```

### Added — Control Plane (sessions 9–21)

- **FastAPI backend** (`services/api`) — multi-user platform with auth, workspaces, projects, experiments, runs, deployments, drift reports, audit logs, and an LLM router.
- **React UI** (`apps/web`) — Vite 5 + TypeScript + Tailwind. 16 screens covering the full ML loop: workspaces → projects → experiments → runs → pipelines → deployments → drift → audit. Dark-mode-first.
- **6 LLM copilots** (provider-agnostic via Anthropic + OpenAI with extensible router):
  - Dataset consultant — analyse a CSV, suggest task type + risks.
  - Experiment designer — generate a `RunConfig` from a natural-language goal.
  - Run explainer — narrate why a model won + suggest next experiments.
  - Failure debugger — classify a failed run + propose a fix.
  - Deployment reviewer — pre-deploy safety check (verdict: APPROVE / APPROVE WITH CAVEATS / DO NOT DEPLOY).
  - Drift analyst — interpret a `DriftReport` (verdict: RETRAIN NOW / INVESTIGATE / MONITOR / NO ACTION).
- **Programmatic auth** — `X-PyCaret-Key` header alongside JWT. Mint/list/revoke personal API keys via `/auth/api-keys`.
- **Audit logs** — append-only middleware records every mutating `/api/v1/*` call with scrubbed payloads. Viewer at `/admin/audit` (superuser) or `/workspaces/{id}/audit-logs` (workspace admin).
- **Drift reports** — `DriftReport` table + 3 CRUD routes + LLM analyst on top.
- **Workspace member CRUD** — invite by email, role changes (`admin` / `member`), last-admin guard mirroring the server check in the UI.
- **Multiple distribution shapes** — `infra/docker/Dockerfile.api` + `Dockerfile.ui` + `docker-compose.yml` for local-first; same images deploy to Kubernetes / cloud.

### Changed — Engine (sessions 22–29) — BREAKING

The whole god-class drain landed across 8 sessions. Every public verb on `Experiment` now runs natively:

- **All persistence verbs** (`save_model`, `load_model`, `save_experiment`, `load_experiment`) — pure `joblib.dump`/`load` wrappers. Cloud-load (`platform="aws"|"gcp"|"azure"` kwargs) removed; that's Control-Plane territory now.
- **All modeling verbs**:
  - `predict_model` — task-aware native dispatch (classification / regression / clustering / anomaly), including `prediction_label` / `prediction_score` / `Cluster` / `Anomaly_Score` columns.
  - `create_model` — pulls from the engine's model registry, runs k-fold CV, returns a fitted **sklearn `Pipeline`** (preprocessor + trained model). Was a bare estimator before.
  - `tune_model` — wraps `sklearn.model_selection.RandomizedSearchCV` over the registry's `tune_grid`. `TuneResult.search` now actually populated.
  - `compare_models` — iterates the registry, calls `create_model` per candidate, assembles the leaderboard. Auto-detects ascending vs descending sort for error metrics.
  - `ensemble_model` — `BaggingClassifier` / `AdaBoostClassifier` (and regressor variants).
  - `blend_models` — `VotingClassifier` / `VotingRegressor`. Auto-detects `voting="soft"` when all base models have `predict_proba`.
  - `stack_models` — `StackingClassifier` / `StackingRegressor`. Default meta-learner: `LogisticRegression` for classification, `LinearRegression` for regression.
  - `calibrate_model` — `CalibratedClassifierCV` (classification only — raises on regression).
  - `finalize_model` — re-fits on the full dataset (train + holdout combined).
- **Unsupervised verbs** (`create_model`, `assign_model` for clustering + anomaly) — registry-resolved estimator fits on transformed data; `Cluster` / `Anomaly` / `Anomaly_Score` columns attached. CBLOF retry preserved.
- **All user-facing data accessors** (`X`, `X_train`, `X_test`, `y`, `y_train`, `y_test`, `preprocess_pipeline`) — read from a fit-time snapshot, no longer dispatched through the legacy holder.
- **Verb signatures slimmed** — dropped 3.x cruft across the board: `probability_threshold`, `encoded_labels`, `preprocess`, `ml_usecase`, `experiment_custom_tags`, `groups`, `return_train_score`, `return_tuner`, `tuner_verbose`, `early_stopping`, `early_stopping_max_iters`, `choose_better`, `budget_time`, `caller_params`, `system`, `add_to_model_list`, `model_only`, and more. Each removal is documented with a one-line workaround in [`docs/revamp/release_notes_pycaret4.md`](docs/revamp/release_notes_pycaret4.md).

### Added — Platform tables

| Table | Purpose |
|---|---|
| `users`, `sessions`, `api_keys` | Auth |
| `workspaces`, `workspace_members` | Multi-user |
| `data_sources` | CSV uploads, S3 / Postgres connectors |
| `projects`, `experiments`, `runs`, `events`, `artifacts`, `fold_metrics` | ML workflow |
| `pipelines`, `pipeline_project_links`, `deployments` | Serving |
| `llm_provider_settings`, `llm_consultations` | LLM router + audit |
| `drift_reports` | Distribution drift snapshots |
| `audit_logs` | Append-only platform audit |

### Fixed

- A handful of sklearn 1.8 compatibility issues in the metric registry (some scorers used the deprecated `squared=False` kwarg). Workarounds in `pycaret.utils.generic.calculate_metrics` swallow the failure + return `0.0` for the affected metric instead of failing the whole CV.

### Tests

- **Combined suite: 254 tests passing** (32 engine + 80 server + 62 web + 80 misc).
- The drain pattern locks in a regression test per session: `monkeypatch.setattr(exp._legacy, "<verb>", _poison)` + assert the native path still succeeds.

### Known limitations

- Time-series experiments (`TimeSeriesExperiment`) still delegate to the legacy `sktime`-backed engine. The supervised + unsupervised drain is complete; TS will be addressed in a later session before `4.0.0` non-alpha.
- `pycaret/internal/pycaret_experiment/` is still in the source tree. It's an implementation detail (the engine populates it via `setup()` for internal state — transformed splits, fold generator, model registry). The public API doesn't read from it post-fit. Will be deleted in a session before `4.0.0` non-alpha.
- A few secondary verbs (`plot_model`, `pull`, `models`, `get_metrics`, `add_metric`) still delegate to the legacy holder. They're advisory — predict / tune / compare don't depend on them.

### Engineering changelog

For a session-by-session breakdown of every change, see [`docs/revamp/release_notes_pycaret4.md`](docs/revamp/release_notes_pycaret4.md). Status is tracked at [`docs/revamp/STATUS.md`](docs/revamp/STATUS.md).

---

## [4.0.0a1] — 2026-04-23

**Second test release of PyCaret 4.0** — focused on aggressive dependency discipline and unpinning scikit-learn.

### Install

```bash
pip install https://github.com/pycaret/pycaret/releases/download/v4.0.0a1/pycaret-4.0.0a1-py3-none-any.whl
```

### Changed

- **scikit-learn is now unpinned** — your install gets whatever the latest scikit-learn is (tested against 1.8). Previously pinned to `>=1.7,<1.8` because `sktime` capped it; `sktime` moved to the `timeseries` extra so the core install is no longer constrained.
- **Core install pulls ~41 packages** (vs. ~90+ in 3.x, ~65 in 4.0.0a0). sklearn 1.8, NumPy 2.4, pandas 3.0 on the default install.

### Removed from the default install

PyCaret 4.0 is opinionated: the library is a minimal ML control pane on top of scikit-learn. Everything below is now **user-installed** if wanted — pycaret's model / metric / plot registries auto-detect them via soft-dependency checks.

- `lightgbm`, `xgboost`, `catboost` — pick your own; pycaret lights the container up when it finds one installed.
- `kmodes`, `mlxtend` — same pattern.
- `optuna`, `optuna-integration`, `scikit-optimize`, `hyperopt` — the whole `tuners` extra is gone. Use sklearn's built-in `GridSearchCV` / `RandomizedSearchCV` / `HalvingRandomSearchCV`, or install a backend yourself and pass a fitted search-cv object to `tune_model`.
- `shap`, `interpret`, `umap-learn` — the whole `analysis` extra is gone. Interpretability is out of scope for 4.0 alpha; will return targeted in a later release.
- `prophet` — the entire `prophet` extra is gone.
- `matplotlib` — no longer a core dep. Lazy-imported with graceful fallbacks wherever a residual non-Plotly plot path exists. Plotly is the single chosen plot library for 4.0.
- `kaleido` — moved from core to the new `export` extra (only needed for static Plotly image export).
- `xxhash` — no longer core; `pycaret.internal.memory.FastMemory` now falls back to `hashlib.blake2b` if xxhash isn't available.
- `cloudpickle` — no longer core; `load_experiment()` lazy-imports it and raises a clean error if missing.
- `psutil` — no longer core; system-info logging falls back to stdlib `os.cpu_count()`.
- `nbformat` — moved to the `notebook` extra (only needed for run-notebook artifact generation in the forthcoming platform layer).
- `ipywidgets` — moved to the `notebook` extra.
- `ipython` — **kept** in core (widely used, small footprint, and the 4.0 Jupyter integration expects it).

### New extras structure

- `pycaret[notebook]` — `ipywidgets`, `nbformat` (for progress widgets + run-notebook artifact generation).
- `pycaret[export]` — `kaleido` (static Plotly image export).
- `pycaret[anomaly]` — `pyod`, `numba` (AnomalyExperiment backend).
- `pycaret[timeseries]` — `sktime`, `statsmodels`, `pmdarima` (TimeSeriesExperiment backend).
- `pycaret[full]` — all of the above in one go.

### Fixed

- `lightgbm` container in `pycaret.containers.models.*` now gracefully disables itself when lightgbm is missing (was ImportError-at-load before).

### Known transitional deps (still in core; targeted for removal)

- `imbalanced-learn` — required because `pycaret.internal.pipeline.Pipeline` inherits from `imblearn.pipeline.Pipeline`. Phase 4 preprocessor rewrite removes this.
- `category-encoders` — used by the legacy preprocessor. sklearn 1.3+ has native `OrdinalEncoder` / `OneHotEncoder` / `TargetEncoder` replacements; Phase 4 swaps them in.

These two are listed as transitional in `pyproject.toml` and will be cut in a future alpha.

---

## [4.0.0a0] — 2026-04-23

**First test release of PyCaret 4.0.** This is an alpha — the OOP public API is stable, but the legacy internal implementation is still being progressively replaced. Feel free to install, try, and file feedback; do **not** rely on it for production workloads yet.

### Install

```bash
pip install --pre pycaret==4.0.0a0
# or with every optional extra:
pip install --pre "pycaret[full]==4.0.0a0"
```

Python 3.11 / 3.12 / 3.13 supported.

### Quickstart

```python
from pycaret.datasets import get_data
from pycaret.tasks import ClassificationExperiment
from pycaret import save_model, load_model

df = get_data("juice")
exp = ClassificationExperiment(target="Purchase", session_id=42).fit(df)
result = exp.compare_models()
best = result.best
preds = exp.predict_model(best).predictions
save_model(best, "juice_classifier")
```

Five task classes available:

```python
from pycaret.tasks import (
    ClassificationExperiment,
    RegressionExperiment,
    ClusteringExperiment,
    AnomalyExperiment,
    TimeSeriesExperiment,
)
```

### Changed

- **The public API is now OOP-only.** The 3.x module-level functional API (`setup()`, `compare_models()`, ...) has been removed. Every task has one `Experiment` subclass with a fluent method interface.
- **`Experiment` subclasses are `sklearn.base.BaseEstimator`.** `get_params()`, `set_params()`, `sklearn.base.clone()`, and `__sklearn_tags__()` all work. You can pickle, clone, and introspect experiments like any other sklearn object.
- **Every verb returns a typed dataclass.** `compare_models()` returns `CompareResult`; `tune_model()` returns `TuneResult`; `predict_model()` returns `PredictResult`; and so on. All dataclasses live in `pycaret.core.results`. The old notebook-friendly DataFrame is still available as `result.leaderboard` / `result.predictions` / etc.
- **Structured event stream.** Every long-running operation emits typed `Event` dataclasses through `pycaret.logging.BaseLogger`. Subscribe via `logger.subscribe(callback)` for real-time progress in UIs and agents. `MemoryLogger` is the default when `log_experiment=True`.
- **Typed introspection API for UIs and LLM agents.** `pycaret.api.list_models(task)`, `describe_model(task, id)`, `list_metrics(task)`, `describe_setup_params(task)` return JSON-serializable dataclasses you can feed directly to a React form or an LLM prompt.
- **Class rename:** `TSForecastingExperiment` → `TimeSeriesExperiment`. Matches the task module name.

### Removed

Clean break from 3.x. These dependencies and features were cut to keep the library lean and modern:

- **Trackers:** `mlflow`, `comet-ml`, `wandb`, `dagshub`. Replaced by the built-in event stream. If you need external-tracker integration, subclass `BaseLogger`.
- **Distributed / parallel:** `fugue`, `dask`, `ray[tune]`, `tune-sklearn`. No replacement — PyCaret is a single-node library in 4.0.
- **Visualisation adapters:** `yellowbrick`, `mljar-scikit-plot`, `schemdraw`, `plotly-resampler`. Plotly-native plot rewrite is on the roadmap.
- **Deploy/serving helpers:** `create_api`, `create_app`, `create_docker`, `deploy_model` (S3), `dashboard`, `convert_model`, `eda`, `check_drift`, `check_fairness`. These will live in the forthcoming `pycaret-server` package instead.
- **GPU via `scikit-learn-intelex` / `cuml`:** out of scope for 4.0 core. GPU-accelerated gradient-boosted models (xgboost / lightgbm / catboost) still work via their own flags.
- **Module-level state:** `set_current_experiment()`, `get_current_experiment()`, and the `_CURRENT_EXPERIMENT` global are gone. Hold onto your `Experiment` instance directly.

See [`docs/revamp/KILL_LIST.md`](docs/revamp/KILL_LIST.md) for the exhaustive list with rationale.

### Fixed

- Runs on modern Python (3.11 / 3.12 / 3.13), modern scikit-learn (1.7), modern NumPy (2.x), modern pandas (2.x). The 3.x upper-bound version pins have been removed.
- `distutils.LooseVersion` usage replaced with `packaging.version.Version` (Python 3.12 dropped `distutils`).
- `np.NaN`, `np.product` usage updated to NumPy 2.0 equivalents.
- `joblib.Memory(bytes_limit=...)` — updated to joblib 1.4+ API.
- `sklearn._check_reg_targets` — updated to sklearn 1.7 signature.

### Dependencies

Core runtime dependencies cut from 30 → 19. Optional extras reorganised into focused tiers: `models`, `tuners`, `analysis`, `anomaly`, `timeseries`, `prophet`, `full`.

### Known limitations in this alpha

- **Legacy internals are still in place.** The new `Experiment` classes delegate to the 3.x god-class under the hood via `self._legacy`. Each verb is being rewritten natively on top of `sklearn.pipeline.Pipeline` in subsequent releases. Public API is stable; internals will churn.
- **Plot rewrite not complete.** Some plot types still error if Plotly hasn't replaced them yet — tracking in ROADMAP Phase 3.
- **Python 3.14 not yet supported.** Upstream `joblib` / `cloudpickle` need to ship PEP 649 support first; see `docs/revamp/thinking/2026-04-22_python314_pep649_blocker.md`.

### Documentation

- [`README.md`](README.md) — quickstart and positioning.
- [`AGENTS.md`](AGENTS.md) — briefing for AI coding agents contributing to PyCaret.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — human contributor guide.
- [`docs/for_agents/`](docs/for_agents/) — deep dives (engine walkthrough, typed results, event stream, introspection API, task cheatsheet).
- [`docs/for_developers/`](docs/for_developers/) — setup, testing, god-class-draining playbook, coding style, release process.
- [`docs/revamp/`](docs/revamp/) — the full engineering narrative (audit, architecture, roadmap, decisions, release notes, platform plan).
- [`notebooks/`](notebooks/) — five executed end-to-end examples, one per task.

### Migrating from 3.x

**Short answer: don't, unless you want to.** 3.x keeps working; PyPI serves it as the default `pip install pycaret`. 4.0 is opt-in via the pre-release flag.

If you do want to migrate, the shape is:

| 3.x (functional) | 4.0 (OOP) |
|---|---|
| `from pycaret.classification import setup, compare_models` | `from pycaret.tasks import ClassificationExperiment` |
| `setup(data, target='y')` | `exp = ClassificationExperiment(target='y').fit(data)` |
| `compare_models()` | `exp.compare_models()` → `CompareResult` |
| `pull()` | `result.leaderboard` |
| `tune_model(model)` | `exp.tune_model(model).pipeline` |
| `predict_model(m, data=new_df)` | `exp.predict_model(m, data=new_df).predictions` |
| `save_model(m, 'f')` | `from pycaret import save_model; save_model(m, 'f')` |

More migration patterns in [`notebooks/README.md`](notebooks/README.md).

### License

MIT. Unchanged from 3.x.

---

*For every non-trivial change in this release (~180 entries across 6 sessions), see [`docs/revamp/release_notes_pycaret4.md`](docs/revamp/release_notes_pycaret4.md).*
