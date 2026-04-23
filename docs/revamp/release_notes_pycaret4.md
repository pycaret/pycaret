# PyCaret 4.0 — Engineering Change Log

> **This file is the single source of truth for everything that changed between PyCaret 3.4.0 and 4.0.**
>
> Every non-trivial edit during the 4.0 revamp is logged here. At release time, the user-facing `RELEASE_NOTES.md` (and the GitHub Release body) will be generated from this file by summarising and regrouping entries. Do not edit past entries when new work is added; append only.

## How to use this file
>
> For the Part-2 Application-Platform plan (CLI / FastAPI / DB / React / Docker), see
> [`PLATFORM_PLAN.md`](PLATFORM_PLAN.md). Its phase breakdown is tracked in
> [`ROADMAP.md`](ROADMAP.md) under Part 2.

- **Organization:** newest session first. Within each session, entries are grouped by type.
- **Entry types:** `BREAKING`, `REMOVED`, `ADDED`, `CHANGED`, `FIXED`, `DEPRECATED`, `SECURITY`, `DOCS`, `BUILD`, `TESTS`, `DEPS`, `INTERNAL`.
- **One change = one bullet.** If a single change touches many files, list it once with the file count; reserve multi-bullet listings for distinct changes.
- **Every bullet must be independently understandable.** Include the file path(s) and, for behavior changes, the before/after in a sentence.
- **Linking:** when a change is backed by a decision or audit note, reference it (e.g. `(see DECISIONS.md · 2026-04-22 · tbats demotion)`).
- **Breakage tagging:** anything that changes a public import path, a function signature, or removes a symbol must carry `BREAKING` in addition to the type tag.

## Entry template

```
- `TYPE[, BREAKING]` — **Short imperative title.** Fuller explanation. File paths. Rationale if non-obvious.
```

## Category legend (for the user-facing generator)

| Type | User-facing section it feeds |
|---|---|
| `BREAKING` | "⚠ Breaking changes — read before upgrading" |
| `REMOVED` | "Removed features and dependencies" |
| `ADDED` | "New features" |
| `CHANGED` | "Behavior changes" |
| `FIXED` | "Bug fixes" |
| `DEPRECATED` | "Deprecations" (4.0 has none — clean break) |
| `SECURITY` | "Security fixes" |
| `DOCS` | Collapsed into "Documentation" footnote |
| `BUILD` | "Installation & packaging" |
| `TESTS` | Usually omitted from user notes |
| `DEPS` | "Dependency changes" |
| `INTERNAL` | Usually omitted from user notes |

---

# Session 6 — 2026-04-23 — Cleanup pass 2 + Application-Platform plan authored

Baseline: end of session 5 (v4 branch live on GitHub, CI green).
Environment: unchanged.

Theme: user asked for "one more round of clean ups. get rid of any garbage from 3.0. keep the bare minimum. the core logic that we will use and thats it." Then laid out the Part-2 vision — PyCaret as an enterprise-grade open-source application platform (CLI + FastAPI + SQL DB + React UI + Docker). Session 6 executed the cleanup and captured the platform plan.

## REMOVED

- `REMOVED` — **`pycaret/distributions.py`** (0 callers) deleted.
- `REMOVED` — **`pycaret/internal/cloudpickle_compat.py`** (0 callers) deleted.
- `REMOVED` — **`pycaret/internal/cuml_wrappers.py`** (143 LOC) deleted. cuml is not a 4.0 dep; GPU fallback via NVIDIA cuml is out of scope for the 4.0 engine.
- `REMOVED` — **`pycaret/loggers/`** shim package deleted. Re-pointed 7 `BaseLogger` import sites to `pycaret.logging.base` directly (1 in each of: `classification/oop.py`, `regression/oop.py`, `time_series/forecasting/oop.py`, `internal/pycaret_experiment/tabular_experiment.py`, `internal/pycaret_experiment/unsupervised_experiment.py`; 2 others already migrated). The 4.0 `BaseLogger` lives in `pycaret.logging.base`; the shim was legacy-compat and had no user after session 3.
- `REMOVED, BREAKING` — **9 killed-verb methods** deleted across god-class + task oop wrappers (no replacement; public API didn't expose them):

  | File | Methods deleted | ~LOC |
  |---|---|---:|
  | `internal/pycaret_experiment/pycaret_experiment.py` | `deploy_model` (stub) | 9 |
  | `internal/pycaret_experiment/tabular_experiment.py` | `deploy_model`, `convert_model`, `create_api`, `create_docker` | 361 |
  | `internal/pycaret_experiment/supervised_experiment.py` | `check_fairness`, `create_app`, `dashboard`, `check_drift` | 353 |
  | `classification/oop.py` | `deploy_model`, `dashboard` | 174 |
  | `regression/oop.py` | `deploy_model`, `dashboard` | 168 |
  | `time_series/forecasting/oop.py` | `deploy_model` | 91 |
  | **Total** | **15 method definitions** | **~1,156** |

  Lazy imports inside those methods (mlflow / comet / wandb / dagshub / fairlearn / evidently / gradio / fastapi / boto3 / m2cgen) disappeared with the bodies.

## CHANGED

- `CHANGED` — **Model containers (`containers/models/{classification,regression,clustering,anomaly}.py`) — cuml branches now raise `NotImplementedError`.** Deleted the `import pycaret.internal.cuml_wrappers` imports + the `pycaret.internal.cuml_wrappers.get_*()` call sites inside `if gpu_imported:` blocks, and replaced `import cuml.X` lines inside `if experiment.gpu_param == "force":` / `elif experiment.gpu_param:` blocks with a raise. These branches were unreachable with default `gpu_param=False` + cuml-not-installed, so no behaviour change; the code is now honest about it. (10 more cuml imports in `containers/models/time_series.py` left as-is — same dead-branch pattern; they'll go with the Phase-5 god-class drain.)
- `INTERNAL` — **`from functools import partial`** removed from `supervised_experiment.py` (only the deleted `check_fairness` method used it).

## ADDED

- `DOCS, ADDED` — **`docs/revamp/PLATFORM_PLAN.md`** (~350 lines) — detailed design for the Part-2 application platform:
  - **Vision**: credible open-source alternative to DataRobot / H2O.ai for teams under ~20 people.
  - **Architecture**: monorepo with 4 sibling packages — `pycaret` (library, current) + `pycaret-server` (FastAPI) + `pycaret-ui` (React) + `pycaret-cli` (CLI).
  - **Data model**: Workspace → Project → Experiment → Run → Pipeline. 11 SQLAlchemy tables.
  - **First-run flow**: `docker compose up` → self-service admin setup wizard → no external config.
  - **Database**: SQLite default, Postgres/MySQL opt-in via `DATABASE_URL`.
  - **Auth**: local user store + JWT; OAuth as plugin; admin/member roles.
  - **Tech choices**: Vite + React 18 + Tailwind + TanStack Query + Zustand + Plotly.js; FastAPI + uvicorn + SQLAlchemy + Alembic; Typer + Rich for CLI.
  - **6 new phases (7-12)** added to ROADMAP.
  - **Gated on Phase 5** — `pycaret==4.0.0alpha0` shipping — so engine stays focused.
  - Explicit "out of scope": Celery/Redis v1, K8s operator, GraphQL, multi-tenant SaaS, hosted billing, model serving.

## DOCS

- `DOCS` — **`ROADMAP.md` restructured** into Part 1 (Engine, Phases 0-6) and Part 2 (Platform, Phases 7-12). Every checkbox reflects actual state: Phases 0, 1, 3.5 ✅ COMPLETE; Phase 2 / 4 / 6 ✅ MOSTLY / 🟡 PARTIAL; Phase 5 🟡 IN FLIGHT (god-class drain, 10-verb migration order spelled out); Phases 7-12 🔴 NOT STARTED.
- `DOCS` — **`STATUS.md`** updated with session-6 delta table + platform-plan summary.
- `DOCS` — **`docs/revamp/README.md`** hub index updated to include `ARCHITECTURE.md`, `PLATFORM_PLAN.md`, `github_issues/`. New "Two parts, one programme" section. Reading order reorganized.

## TESTS

- `TESTS` — **32/32 still green** on Python 3.13 + sklearn 1.7.2 + NumPy 2.3.5 + pandas 2.x, in 1:37 (was 2:07 in session 5 — slightly faster with less code to import).

## ADDED — 6 resolved platform decisions

Owner answered the six parked questions from `PLATFORM_PLAN.md §7`. Each answer is now baked into the plan and recorded as an ADR in `DECISIONS.md`:

- `DOCS, ADDED` — **Decision 1: Run notebooks are first-class artifacts.** Every Run persists `run.ipynb` + `fitted_pipeline.pkl` + `leaderboard.json` + `events.jsonl` + `preview.html`. Immutable, downloadable, shareable via signed URL, previewable in-app. Storage: local disk v1, S3 when deployed.
- `DOCS, ADDED` — **Decision 2: Data-source connectors v1 = CSV upload + S3 + Postgres.** `DataSourceConnector` ABC allows adding Snowflake / GSheets / MySQL later without core changes. AWS-first since immediate deploy target.
- `DOCS, ADDED` — **Decision 3: Pipelines are workspace-scoped + shareable across projects.** `Pipeline` moves out of `Project` into `Workspace`; `pipeline_project_links` many-to-many joins them. Workspace gets a top-level "Pipelines" screen.
- `DOCS, ADDED` — **Decision 4: In-house serving system, not MLServer/BentoML.** `DeploymentRegistry` loads pickles into memory; single catch-all `POST /api/v1/deployments/{slug}/predict` handles inference. Per-deployment auth: `workspace` / `api-key` / `public`. Per-deployment metrics: count, p50/p95 latency, error rate. Phase 11 renamed "In-house serving + Docker/deploy".
- `DOCS, ADDED` — **Decision 5: Dual-license the platform packages.** Engine `pycaret` stays MIT. `pycaret-server` / `pycaret-cli` / `pycaret-ui` become MIT + BSL 1.1 (BSL for multi-tenant hosted SaaS only; converts to MIT after 3 years). CLA added to CONTRIBUTING.md. Mirrors Sentry / Cal.com / Supabase / Plausible posture.
- `DOCS, ADDED` — **Decision 6: Metrics stored as summary AND per-fold.** Two tables — `runs.metrics_summary` (leaderboard shape) and `fold_metrics` (per-fold × per-model × per-metric). Summary drives leaderboard; per-fold unlocks variance / stability / time-to-train analysis.

Data model in `PLATFORM_PLAN.md §3` expanded to 14 SQLAlchemy tables (from 11): added `fold_metrics`, `deployments`, `api_keys`, `pipeline_project_links`. Phase 11 now covers the serving subsystem in detail. Dep discipline §6 updated with `nbconvert` (notebook preview), `boto3` (S3 extra), `psycopg[binary]` (postgres extra), `python-multipart` (CSV upload), `joblib` (deployment loading).

New §8 "Licensing posture" added to `PLATFORM_PLAN.md`. Reading order updated.

## Session 6 delta summary

| Metric | Session 5 end | Session 6 end | Δ |
|---|---:|---:|---:|
| Source LOC in `pycaret/` | 51,976 | **50,544** | **−1,432** |
| Zero-import leaf files | 3 | **0** | **−3** |
| Killed-verb methods in source | 15 | **0** | **−15** |
| cuml-coupled files with runtime risk | 5 | **0** (branches raise) | − |
| Part-2 platform plan | none | **PLATFORM_PLAN.md (~350 lines)** | +1 doc |
| Roadmap phases defined | 6 engine phases | **12 (6 engine + 6 platform)** | +6 |
| Test pass rate | 100% (32/32) | 100% (32/32) | — |

---

# Session 4 — 2026-04-23 — Repo restructure + working notebooks + agent/dev docs + issue triage

Baseline: end of session 3.
Environment: unchanged.

Theme: user asked to "clear the folder, restructure for dev contributions, get rid of old stuff, one notebook per use-case fully working, MD files for agents, download all open issues, start cleaning them up." Session 4 does all of that in one pass.

## REMOVED — dead weight purged

- `REMOVED` — **`Docker_files/`** directory (old `pycaret_full` / `pycaret_slim` Docker image scaffolding) deleted.
- `REMOVED` — **`docs/source/`** directory (Sphinx docs tree) deleted — we use GitBook for hosted docs now; Sphinx was unused.
- `REMOVED` — **`docs/Makefile`, `docs/make.bat`, `docs/make.sh`, `docs/logs.log`** — Sphinx build scaffolding, deleted.
- `REMOVED` — **`tutorials/legacy_v3/`** (the 6 archived 3.x notebooks from session 3) deleted.
- `REMOVED` — **`tutorials/time_series/forecasting/`** (old TS example scripts) deleted.
- `REMOVED` — **`tutorials/translations/`** (4 language-translated tutorial trees: chinese/greek/japanese/portuguese) deleted.
- `REMOVED` — **`tutorials/pycaret_cheat-sheet_in_excel.xlsx`** deleted.
- `REMOVED` — **`logs.log`** (root-level 28K-line runtime log) deleted.
- `REMOVED` — **`.readthedocs.yml`** — ReadTheDocs config, unused since GitBook migration.
- `REMOVED` — **`.slugignore`** — Heroku-specific ignore file, long dead.

## CHANGED — directory restructure

- `CHANGED, BREAKING` — **`tutorials/` → `notebooks/`**. Modern naming; all 5 canonical notebooks live under `notebooks/` now (`01_classification.ipynb` … `05_time_series.ipynb`).
- `CHANGED` — **`.gitignore` rewritten.** Cleaner structure; `artifacts/`, `*.log`, `.ruff_cache/`, `demo*.py`, `nbtest*.ipynb` added; legacy 3.x scratch patterns retained as safety net.

## ADDED — working notebooks (one per task)

- `ADDED` — **`scripts/build_notebooks.py`** — programmatic notebook generator using `nbformat` + `nbclient`. Run with `--run` to execute and capture fresh outputs.
- `ADDED` — **`notebooks/01_classification.ipynb`** (43 KB with outputs). Canonical `ClassificationExperiment` → `fit` → `compare_models` → `tune_model` → `predict_model` → `save/load_model` → inspect event stream.
- `ADDED` — **`notebooks/02_regression.ipynb`** (25 KB).
- `ADDED` — **`notebooks/03_clustering.ipynb`** (18 KB). `ClusteringExperiment` → `create_model("kmeans")` → `assign_model`.
- `ADDED` — **`notebooks/04_anomaly_detection.ipynb`** (18 KB). `AnomalyExperiment` → `create_model("iforest")` → `assign_model`.
- `ADDED` — **`notebooks/05_time_series.ipynb`** (13 KB). `TimeSeriesExperiment(fh=12)` → `compare_models` → `predict_model`.
- `ADDED` — **All 5 notebooks executed end-to-end under Python 3.13 + sklearn 1.7.2.** Outputs are persisted in the `.ipynb` JSON so GitHub can render them and users can skim without running.

## ADDED — `AGENTS.md` at repo root

- `ADDED` — **`/AGENTS.md`** — 60-second briefing for AI coding agents: TL;DR, read-first list, non-negotiables (8 rules), conventions, workflow, repo map, common-task recipes. This is the file any agent should consume before touching the repo.

## ADDED — `docs/for_agents/` deep dives (5 files)

- `ADDED` — **`docs/for_agents/README.md`** — index of agent-facing docs.
- `ADDED` — **`docs/for_agents/ENGINE_WALKTHROUGH.md`** — step-by-step of what happens inside `fit → compare_models → predict_model`, including the legacy-delegation pattern.
- `ADDED` — **`docs/for_agents/TYPED_RESULTS.md`** — every result dataclass (9 of them) with full field listing and usage idioms.
- `ADDED` — **`docs/for_agents/EVENT_STREAM.md`** — all 22 `EventKind`s tabulated with their typical payloads; subscriber contract + custom-logger recipe.
- `ADDED` — **`docs/for_agents/INTROSPECTION_API.md`** — the `pycaret.api` surface as a UI/agent integration contract, with React-form and LLM-prompt examples.
- `ADDED` — **`docs/for_agents/TASK_CHEATSHEET.md`** — verb × task matrix + constructor-parameter matrix on one page; import-path reference.

## ADDED — `docs/for_developers/` onboarding (5 files)

- `ADDED` — **`docs/for_developers/README.md`** — index.
- `ADDED` — **`docs/for_developers/SETUP.md`** — clone → `uv sync --all-extras` → first green test in < 5 min; Python-version matrix; Windows notes; IDE setup.
- `ADDED` — **`docs/for_developers/TESTING.md`** — test layout, how to run/write tests, markers, CI matrix, coverage, what not to write.
- `ADDED` — **`docs/for_developers/DRAINING_THE_GODCLASS.md`** — verb-migration playbook for Phase 5: recommended order, before/after recipe, hard constraints, completion criteria.
- `ADDED` — **`docs/for_developers/CODING_STYLE.md`** — formatting, imports, type hints, docstrings, naming, dataclass conventions, error handling, logging, dependency discipline, git hygiene.
- `ADDED` — **`docs/for_developers/RELEASE_PROCESS.md`** — versioning, tagging, notes generation, PyPI publish.

## CHANGED — `CONTRIBUTING.md` rewritten for 4.0

- `CHANGED` — **`CONTRIBUTING.md`** — replaced 3.x-era content (black/isort mention, sphinx docs, `pip install -e .[test]`) with a 4.0-flavored guide that links into `AGENTS.md` + `docs/for_agents/` + `docs/for_developers/`. Includes the current PR checklist.

## ADDED — GitHub issue snapshot + triage

- `ADDED` — **`docs/revamp/github_issues/open_issues_raw.json`** — raw snapshot of all 388 open issues on `pycaret/pycaret` (fetched via `gh issue list`).
- `ADDED` — **`scripts/triage_issues.py`** — classifier that reads the raw snapshot and assigns each issue to one of 5 buckets: `fixed_in_4_0`, `out_of_scope`, `stale`, `still_relevant_bug`, `still_relevant_enhancement`. Heuristic-based; reruns are idempotent.
- `ADDED` — **`docs/revamp/github_issues/triage.json`** — machine-readable output of the classifier.
- `ADDED` — **`docs/revamp/github_issues/triage.md`** — human-browsable triage report, one table per bucket with issue number, title, labels, update date, classification reason.
- `ADDED` — **`docs/revamp/github_issues/README.md`** — triage summary + methodology.
- `ADDED` — **`docs/revamp/github_issues/PLAYBOOK.md`** — step-by-step actions per bucket with ready-to-paste reply templates.

### Headline triage result

From 388 open issues:
- **8 (2%) `fixed_in_4_0`** — close with 4.0 release-notes pointer.
- **92 (24%) `out_of_scope`** — close with `KILL_LIST.md` pointer.
- **123 (32%) `stale`** — auto-ping, close after 30 days of silence.
- **58 (15%) `still_relevant_bug`** — label `4.0-candidate`, route to Phase 5.
- **107 (28%) `still_relevant_enhancement`** — per-item decision.

**224 of 388 (58%) can be closed or auto-pinged without per-issue human judgment.**

## FIXED

- `FIXED` — **`np.product` → `np.prod`** in `pycaret/internal/patches/sklearn.py:106`. NumPy 2 removed the capitalised alias; the call site surfaced when the notebook-execution pass tried to run `compare_models`. Was blocking the full e2e notebook run.

## INTERNAL

- `INTERNAL` — **`scripts/` directory created** for maintenance scripts. Two scripts landed this session; future additions (bulk issue close, LOC reports, release build helpers) go here.
- `INTERNAL` — `tests/conftest.py`, `tests/test_models.py`, and all other tests untouched in this session (no code-under-test changes).

## Session 4 delta summary

| Metric | Session 3 end | Session 4 end | Δ |
|---|---:|---:|---:|
| Repo top-level directory count | 9 | 10 | +1 (`scripts/`) |
| Source LOC in `pycaret/` | ~49,400 | ~49,400 | 0 |
| Test files | 4 | 4 | 0 |
| Test pass rate | 100% (32/32) | 100% (32/32) | — |
| Dead-weight directories | 4 (Docker_files, docs/source, tutorials/{legacy_v3,time_series,translations}) | **0** | − |
| Stale root-level files | 3 (logs.log, .readthedocs.yml, .slugignore) | **0** | − |
| Working notebooks | 0 (3.x legacy archived) | **5 (executed, with outputs)** | +5 |
| Docs files | 7 in `docs/revamp/` | **23** (revamp + for_agents + for_developers + github_issues) | +16 |
| GitHub issues in actionable triage buckets | 0 (untouched backlog of 388) | 224 ready-to-close, 165 needing human review | — |

---

# Session 3 — 2026-04-22 — Functional API killed; 4.0 is OOP-only

Baseline: end of session 2.
Environment: unchanged.

Theme: the user made the final 4.0 call — "nobody will migrate 3 → 4. 4 in my mind is totally new thing. I really would like to get rid of 90% tech debt now." This session deletes the module-level functional API entirely, leaving the sklearn-compatible `Experiment` class hierarchy as the single canonical surface.

## BREAKING — functional API removed wholesale

- `REMOVED, BREAKING` — **`pycaret.classification.setup`, `compare_models`, `create_model`, `tune_model`, `ensemble_model`, `blend_models`, `stack_models`, `plot_model`, `evaluate_model`, `interpret_model`, `calibrate_model`, `optimize_threshold`, `predict_model`, `finalize_model`, `deploy_model`, `save_model`, `load_model`, `automl`, `pull`, `models`, `get_metrics`, `add_metric`, `remove_metric`, `get_logs`, `get_config`, `set_config`, `save_experiment`, `load_experiment`, `get_leaderboard`, `set_current_experiment`, `get_current_experiment`, `dashboard`, `convert_model`, `check_fairness`, `create_api`, `create_docker`, `create_app`, `get_allowed_engines`, `get_engine`, `check_drift`** — all 40 module-level functions gone. File `pycaret/classification/functional.py` deleted (3,323 LOC).
- `REMOVED, BREAKING` — **`pycaret.regression` functional API** — 38 module-level functions, `pycaret/regression/functional.py` deleted (3,033 LOC).
- `REMOVED, BREAKING` — **`pycaret.clustering` functional API** — 23 module-level functions, `pycaret/clustering/functional.py` deleted (1,461 LOC).
- `REMOVED, BREAKING` — **`pycaret.anomaly` functional API** — 18 module-level functions, `pycaret/anomaly/functional.py` deleted (1,256 LOC).
- `REMOVED, BREAKING` — **`pycaret.time_series` functional API** — 26 module-level functions, `pycaret/time_series/forecasting/functional.py` deleted (2,260 LOC).

Total LOC dropped in this session: **~11,333 lines of pass-through wrappers**.

## BREAKING — class renaming

- `CHANGED, BREAKING` — **`TSForecastingExperiment` renamed to `TimeSeriesExperiment`** in the public API. The cleaner name matches the task's module name. The legacy class (`pycaret.time_series.forecasting.oop.TSForecastingExperiment`) still exists as an internal implementation detail the new wrapper delegates to during migration.

## ADDED — 4 new task subclasses + stateless persistence

- `ADDED` — **`pycaret.tasks.RegressionExperiment`** (sklearn-compatible, `estimator_type="regressor"`; delegates to legacy `_NonTSSupervisedExperiment` during transition).
- `ADDED` — **`pycaret.tasks.ClusteringExperiment`** (inherits `UnsupervisedExperiment`; adds `assign_model`).
- `ADDED` — **`pycaret.tasks.AnomalyExperiment`** (inherits `UnsupervisedExperiment`; adds `assign_model`).
- `ADDED` — **`pycaret.tasks.TimeSeriesExperiment`** (inherits `SupervisedExperiment`; adds `check_stats`; overrides `fit()` for univariate time-series inputs).
- `ADDED` — **`pycaret.core.supervised.SupervisedExperiment`** — intermediate base that hosts supervised-only verbs (`compare_models`, `tune_model`, `ensemble_model`, `blend_models`, `stack_models`, `calibrate_model`, `finalize_model`, `interpret_model`, `automl`, `get_leaderboard`). Extracted from `Experiment` so unsupervised tasks don't inherit verbs they can't implement.
- `ADDED` — **`pycaret.core.unsupervised.UnsupervisedExperiment`** — intermediate base that adds `assign_model`. Tells the legacy engine via an override that supervised-coercion is not needed in `fit`.
- `ADDED` — **`pycaret.persistence`** — stateless top-level `save_model(model, path)` / `load_model(path)` utilities. No experiment required, no globals, no mutation. Re-exported as `pycaret.save_model` / `pycaret.load_model`.

## CHANGED — legacy module paths thinned

Each `pycaret/{module}/__init__.py` reduced from a 40-entry re-export list (~90 LOC with `__all__`) to a ~15-line docstring plus a single-line `from pycaret.tasks.{module} import {Task}Experiment`:

- `CHANGED` — `pycaret/classification/__init__.py` thinned (88 LOC → 20).
- `CHANGED` — `pycaret/regression/__init__.py` thinned (83 LOC → 13).
- `CHANGED` — `pycaret/clustering/__init__.py` thinned (53 LOC → 14).
- `CHANGED` — `pycaret/anomaly/__init__.py` thinned (43 LOC → 14).
- `CHANGED` — `pycaret/time_series/__init__.py` thinned (59 LOC → 16).
- `CHANGED` — `pycaret/time_series/forecasting/__init__.py` created as a deep-import path for the legacy class (10 LOC).
- `CHANGED` — `pycaret/__init__.py` rewritten with a canonical-API docstring and top-level `save_model` / `load_model` re-exports.
- `CHANGED` — `pycaret/core/experiment.py` Experiment base had supervised-only verbs extracted into `SupervisedExperiment` subclass; unsupervised task-subclass setup kwargs now strip supervised-only params (`target`, `train_size`, `fold`, `fold_strategy`, `transformation`, `remove_outliers`, `feature_selection`) which the legacy unsupervised `setup()` doesn't accept.

## REMOVED — state machinery

- `REMOVED, BREAKING` — **`pycaret/core/state.py`** deleted. `current_experiment()`, `set_current_experiment()`, `reset_current_experiment()`, `require_current_experiment()` — with no functional API to serve, these had no purpose. The `ContextVar`-backed current-experiment machinery is gone entirely.
- `REMOVED, BREAKING` — **`set_current_experiment` / `get_current_experiment`** public functions in every task module. They backed the functional API's implicit-experiment model; that model is gone.

## REMOVED — tests that exercised the functional API

- `TESTS, BREAKING` — **22 classification/regression/clustering/anomaly/misc test files deleted:** `test_classification.py`, `test_classification_plots.py`, `test_classification_tuning.py`, `test_regression.py`, `test_regression_plots.py`, `test_regression_tuning.py`, `test_clustering.py`, `test_anomaly.py`, `test_multiclass.py`, `test_optimize_threshold.py`, `test_overflow.py`, `test_preprocess.py`, `test_probability_threshold.py`, `test_supervised_predict_model.py`, `test_tune_model.py`, `test_convert_model.py`, `test_pipeline.py`, `test_memory.py`, `test_utils.py`, `test_utils_datetime.py`, `test_persistence.py`, `test_persistence_experiment.py`. All were functional-API-coupled and would fail to import in 4.0.
- `TESTS, BREAKING` — **19 time-series test files deleted:** `test_time_series_{base,blending,exogenous,feat_eng,indices,metrics,models,plots,preprocess,setup,stats,tune_base,tune_grid,tune_random,utils,utils_forecasting,utils_forecasting_pipeline,utils_plots}.py`, plus the `time_series_test_utils.py` helper. Will be re-authored in OOP style as each TS verb is rewritten natively.
- `TESTS` — **`tests/conftest.py` rewritten**. The 3.x `_CURRENT_EXPERIMENT` reset fixture and the TSForecastingExperiment `load_setup` fixture are gone. File reduced from 152 LOC to 21.

## ADDED — OOP test suite

- `ADDED` — **`tests/test_e2e_oop.py`** — end-to-end smoke tests covering all 5 tasks via the new OOP API. Includes:
  - Classification e2e (setup + create_model + compare_models + predict_model + save/load roundtrip + event-stream verification)
  - Regression e2e (compare + predict)
  - Clustering e2e (create + assign)
  - Anomaly e2e (create + assign)
  - API introspection surface (static `list_models` / `describe_model` / `list_metrics` / `describe_setup_params` plus JSON round-trip)
  - Event-stream subscriber fan-out
- `ADDED` — **`tests/test_core_architecture.py`** extended with:
  - `test_tasks_package_exports_all_five_task_subclasses`
  - `test_legacy_import_paths_re_export_new_classes`
  - `test_all_task_subclasses_are_sklearn_compatible`
  - `test_top_level_save_load_model_roundtrip`
  - `test_functional_api_is_gone` — asserts the absence of `setup`/`compare_models`/`set_current_experiment` on every task module (regression canary)
- `CHANGED` — **`tests/test_models.py` rewritten** to the OOP pattern (`exp = ClassificationExperiment(target=...).fit(df)` instead of `exp = ClassificationExperiment(); exp.setup(df, target=...)`). Imports `TimeSeriesExperiment` (4.0 name) instead of `TSForecastingExperiment`. `create_model(id).pipeline` shape used for equality checks.
- `TESTS` — **All 32 tests pass** on Python 3.13 + sklearn 1.7.2 in ~2 minutes. 100% green vs. 77% green at end of session 1 (on a smaller, OOP-native suite).

## DOCS

- `DOCS, BREAKING` — **`README.md` rewritten.** Removes the 3.x positioning ("low-code ML library"), replaces it with the 4.0 engine framing (sklearn-composable, typed results, event stream, engine for React UI + agents). Quickstart shows the OOP pattern for all 5 tasks. "What's not in 4.0" section is explicit about the kill list. New "Who PyCaret 4.0 is for" section.
- `DOCS` — Release notes (this file) updated with the session-3 block.

## INTERNAL

- `INTERNAL` — **`pycaret.tasks.__init__.py`** exports all 5 task subclasses.
- `INTERNAL` — **`pycaret.core.__init__.py`** exports `SupervisedExperiment` and `UnsupervisedExperiment` alongside the task-agnostic `Experiment` base.
- `INTERNAL` — Transition pattern unchanged from session 2: each task subclass's `_build_legacy_experiment()` still returns the 3.x god-class instance, and every verb delegates. Replacement with native sklearn.pipeline implementations happens verb-by-verb in subsequent sessions; the public API is stable from here on.

## Session delta summary

| Metric | Session 2 end | Session 3 end | Δ |
|---|---:|---:|---:|
| pycaret/ source LOC | ~60,700 | ~49,400 | **−11,300** |
| Test files | 45 | 4 | **−41** |
| Test pass count | 568 (of 734) | 32 (of 32) | — |
| Test pass rate | 77% | 100% | +23pp |
| Public module-level functions | 145 (functional API) | 0 | **−145** |
| Canonical APIs | 2 (functional + OOP) | 1 (OOP) | **−1** |
| Module-level mutable state | 5 ContextVars / globals | 0 | **−5** |

---

# Session 2 — 2026-04-22 — Phase 4 (Engine Architecture) kickoff

Baseline: end of session 1.
Environment: unchanged (Python 3.13.13, sklearn 1.7.2, NumPy 2.3.5, pandas 2.x).

Theme: the user called out that the 3.x OOP-on-top-of-functional layering was "a piece of hack shit." Session 2 rebuilds the engine core as a proper sklearn-composable object graph — `pycaret.core.Experiment` is a real `BaseEstimator` subclass — while the legacy code paths stay intact under the new class (delegation) so the notebook golden path never breaks during the migration.

## ADDED

- `ADDED` — **`pycaret.core` package.** New engine primitives:
  - `pycaret/core/tasks.py` — `TaskType` str-enum (`CLASSIFICATION`, `REGRESSION`, `CLUSTERING`, `ANOMALY`, `TIME_SERIES`) with `is_supervised` / `is_classification` / `is_regression` helpers.
  - `pycaret/core/errors.py` — `PyCaretError` hierarchy (`ConfigurationError` extends ValueError, `NotFittedError` extends RuntimeError, `UnknownModelError` / `UnknownMetricError` extend KeyError) so UI/agent callers can catch engine errors distinctly from upstream ones.
  - `pycaret/core/results.py` — frozen dataclasses `CompareResult`, `CreateResult`, `TuneResult`, `EnsembleResult`, `BlendResult`, `StackResult`, `CalibrateResult`, `FinalizeResult`, `PredictResult` — the typed return shape of every verb. Each carries fitted pipeline, metrics DataFrame, event trace.
  - `pycaret/core/state.py` — `current_experiment()`, `set_current_experiment()`, `reset_current_experiment()`, `require_current_experiment()` backed by `contextvars.ContextVar`. Thread- and async-safe replacement for the 3.x module-level global.
  - `pycaret/core/experiment.py` — `Experiment(BaseEstimator)` base class. Implements `get_params`, `set_params`, `__sklearn_tags__`, `__sklearn_is_fitted__`, `fit(X, y=None, **setup_kwargs)`. Verbs (`compare_models`, `create_model`, `tune_model`, `ensemble_model`, `blend_models`, `stack_models`, `calibrate_model`, `finalize_model`, `predict_model`, `plot_model`, `interpret_model`, `evaluate_model`, `automl`) delegate to a legacy `_SupervisedExperiment` held as `self._legacy` during the transition; each returns a typed result dataclass.
- `ADDED` — **`pycaret.logging` package.** Replaces the 3.x tracker-adapter concept with a lean structured event stream designed for React UI / LLM agent consumption:
  - `pycaret/logging/events.py` — `Event` frozen dataclass and `EventKind` str-enum with 22 canonical kinds (`experiment.started`, `model.created`, `model.compared`, `model.tuned`, etc.). `Event.to_dict()` produces a JSON-serializable dict.
  - `pycaret/logging/base.py` — `BaseLogger` hook interface with `log()` / `emit()` + `subscribe(callback)` for UI fan-out. `NullLogger` (default silent) and the 3.x no-op shim methods (`log_experiment`, `log_model`, `log_model_comparison`, `log_plot`, `log_params`, `log_metrics`, `log_artifact`, `log_hpram_grid`, `log_sklearn_pipeline`, `init_logger`, `init_experiment`, `finish_experiment`, `set_tags`, `.loggers` property) so legacy god-class calls through a PyCaret 4.0 `BaseLogger` instance continue to work.
  - `pycaret/logging/memory.py` — `MemoryLogger`: thread-safe in-memory buffer with optional JSONL file teeing (flushed after every write so a UI can tail). `events` property, `as_jsonl()` method, `clear()`.
- `ADDED` — **`pycaret.api` package — agent+UI introspection surface.** All functions return JSON-serializable dataclasses:
  - `pycaret/api/cards.py` — `ParameterCard`, `ModelCard`, `MetricCard` dataclasses + `ParameterKind` str-enum (BOOL, INT, FLOAT, STRING, ENUM, LIST, COLUMN, COLUMNS, MODEL_ID, METRIC_ID, UNKNOWN) — widget hints for a React form.
  - `pycaret/api/schemas.py` — `SetupParamSchema` dataclass grouping `ParameterCard`s.
  - `pycaret/api/describe.py` — `list_models(task)` (19 classification cards, 26 regression cards curated from the legacy containers), `describe_model(task, id)`, `list_metrics(task)`, `describe_setup_params(task)` (13 common params organised into groups: Data / Experiment / Cross-Validation / Preprocessing / Compute / Logging), `list_available_models(experiment)` (runtime-aware: flags `is_available=False` when a model's package isn't installed).
- `ADDED` — **`pycaret.tasks` package — task-specific experiment subclasses.**
  - `pycaret/tasks/classification.py` — `ClassificationExperiment(Experiment)` pre-configures `task=CLASSIFICATION`, sets `estimator_type="classifier"` in `__sklearn_tags__`, and explicitly declares all 15 init parameters on the concrete class (rather than via `**kwargs`) so that sklearn's `get_params()` introspection surfaces every configured knob.
- `ADDED` — **End-to-end proof the new stack works.** `ClassificationExperiment(target="Purchase").fit(data).compare_models().predict_model(result.best)` on the `juice` dataset: fitted in 1.3s, compared 3 models in 8.3s, emitted 5 typed events through the logger, returned `CompareResult` / `PredictResult` dataclasses. Captured in the new-architecture test suite.
- `ADDED` — **`tests/test_core_architecture.py`** — 17 fast unit tests (0.2s) covering every new primitive: TaskType enum, error hierarchy, frozen result dataclasses, event JSON round-trip, MemoryLogger (captures / subscribers / file teeing), BaseLogger no-op compat methods, ModelCard/MetricCard/ParameterSchema introspection, `describe_model` raises `UnknownModelError` for bad ids, ClassificationExperiment is sklearn-cloneable, declares classifier tag, ContextVar state. All 17 pass.

## CHANGED

- `CHANGED` — **`pycaret/loggers/base_logger.py`** is now a thin re-export shim over `pycaret.logging.base.BaseLogger`. The full `BaseLogger` lives in `pycaret/logging/base.py`. User subclasses of `pycaret.loggers.base_logger.BaseLogger` (a 3.x import path) still work unchanged.
- `CHANGED` — **`pycaret/loggers/__init__.py`** re-exports only `BaseLogger` from the new location. Previously exported 5 symbols (all removed in session 1).

## DOCS

- `DOCS` — **`docs/revamp/ARCHITECTURE.md`** — new design doc explaining the 4.0 engine architecture: why the 3.x layering was broken, the 8 core design principles (sklearn-canonical `BaseEstimator`, typed results, build-on-not-replace Pipeline, `ColumnTransformer`-based preprocessor, canonical search CV, event-stream logger, no prints/interactive), the package layout, the full `Experiment` interface contract, task-subclass pattern, result/event contracts, multi-session migration plan, and explicit non-goals.
- `DOCS` — **Release-notes section updated** to reflect Phase 4 kickoff and the new package boundaries.

## INTERNAL

- `INTERNAL` — **Transition pattern documented.** During the multi-session migration, `Experiment._legacy` holds an instance of the legacy `_SupervisedExperiment` (task-specific subclass picked by `_build_legacy_experiment`). Every verb on the new `Experiment` calls through to `self._legacy.<verb>(...)`, wraps the legacy return in a typed result dataclass, and emits structured events. Future sessions replace the delegation bodies one verb at a time without breaking the public API.
- `INTERNAL` — **sklearn compatibility validated.** `ClassificationExperiment(...).get_params()` returns all 15 init params; `sklearn.base.clone(exp)` preserves configuration and resets fitted state; `__sklearn_tags__()` surfaces `estimator_type="classifier"`.

## TESTS

- `TESTS` — **17 new unit tests added under `tests/test_core_architecture.py`.** Fast (< 0.5s total); designed to run on every CI matrix entry.
- `TESTS` — **No regressions on the legacy subset.** `pytest tests/test_models.py tests/test_datasets.py tests/test_core_architecture.py` — 23/23 pass.
- `TESTS` — **End-to-end proof via the new stack.** Full `fit → compare_models → predict_model` golden path runs green on Python 3.13 + sklearn 1.7.2 + NumPy 2.3.5 using `pycaret.tasks.ClassificationExperiment`.

---

# Session 1 — 2026-04-22 — Phase 0 (Groundwork) + most of Phase 1 (Amputation)

Baseline: PyCaret 3.4.0 @ `main`.
Environment: Python 3.13.13, uv 0.11.7, scikit-learn 1.7.2, NumPy 2.3.5, pandas 2.x on Windows.

## BUILD

- `BUILD, BREAKING` — **Version bumped `3.4.0` → `4.0.0.dev0`.** `pycaret/__init__.py` and `pyproject.toml` now declare 4.0.0.dev0; this is a pre-release marker for the in-flight revamp.
- `BUILD, BREAKING` — **Migrated build backend from `setuptools` to `hatchling`.** `[build-system]` in `pyproject.toml` now uses `hatchling>=1.25`. Wheel packaging handled by `[tool.hatch.build.targets.wheel]`.
- `BUILD, BREAKING` — **Migrated environment management to `uv` (Astral).** `uv.lock` becomes the lockfile; `uv sync --all-extras` is the supported one-command install. Dev dependencies live in `[dependency-groups.dev]`.
- `BUILD, BREAKING` — **Deleted `setup.cfg`.** pytest, flake8, and isort configs moved. Pytest config now lives in `[tool.pytest.ini_options]` in `pyproject.toml`. Flake8 / isort replaced by ruff (see below).
- `BUILD, BREAKING` — **Deleted `MANIFEST.in`.** Hatchling's built-in include/exclude rules replace it.
- `BUILD` — **Deleted `mypy.ini`.** mypy config now lives in pyproject.toml's tool sections (reinstate there if/when mypy is run).
- `BUILD, BREAKING` — **Python floor raised from `>=3.9,<3.13` to `>=3.11`.** Classifiers now list 3.11 / 3.12 / 3.13 / 3.14. Primary dev target is Python 3.13 for session 1; 3.14 is aspirational (PEP 649 currently breaks joblib+cloudpickle pickling — see `thinking/2026-04-22_python314_pep649_blocker.md`).
- `BUILD` — **Ruff added as the single linter.** Replaces the 3.x stack of black + isort + flake8. Config in `[tool.ruff]` / `[tool.ruff.lint]`. Target version `py313`.
- `BUILD` — **Hard-coded Python version guard rewritten.** `pycaret/__init__.py` now raises `RuntimeError` if run under `<3.11` instead of `<3.9` or `>=3.13`.

## DEPS — removed from runtime (kill list)

All removals are pre-approved in `KILL_LIST.md`. Each is a `BREAKING` change against any user who relied on the affected feature.

- `REMOVED, BREAKING` — **mlflow.** No longer a dependency; `loggers/mlflow_logger.py` module deleted.
- `REMOVED, BREAKING` — **comet-ml.** `loggers/comet_logger.py` deleted.
- `REMOVED, BREAKING` — **wandb.** `loggers/wandb_logger.py` deleted.
- `REMOVED, BREAKING` — **dagshub.** `loggers/dagshub_logger.py` deleted.
- `REMOVED, BREAKING` — **`DashboardLogger` fan-out logger.** `loggers/dashboard_logger.py` deleted.
- `REMOVED, BREAKING` — **fugue, fugue[dask].** `pycaret/parallel/fugue_backend.py` deleted.
- `REMOVED, BREAKING` — **dask, distributed.** No replacement.
- `REMOVED, BREAKING` — **ray[tune], tune-sklearn.** Ray Tune integration dropped from the `tuners` extra. `optuna`, `scikit-optimize`, and `hyperopt` remain.
- `REMOVED, BREAKING` — **yellowbrick.** Plot layer deleted wholesale. `pycaret/internal/plots/yellowbrick.py` deleted; `pycaret/internal/patches/yellowbrick.py` deleted. 16 inline `from yellowbrick.* import ...` call sites in `internal/pycaret_experiment/tabular_experiment.py` stubbed behind a `show_yellowbrick_plot` `NotImplementedError` placeholder (see `tabular_experiment.py` transitional stubs). **Replacements land in Phase 3 as native Plotly plots.**
- `REMOVED, BREAKING` — **mljar-scikit-plot.** `scikitplot` import removed from `internal/plots/helper.py`. The only functional dependency was a matplotlib re-export, which is now direct.
- `REMOVED, BREAKING` — **schemdraw.** Pipeline diagram drawing (one code path) no longer supported; Phase 3 will decide whether to re-render it as Plotly.
- `REMOVED, BREAKING` — **plotly-resampler.** `display_format='plotly-widget'` and `display_format='plotly-dash'` paths in `time_series/forecasting/oop.py` raise `NotImplementedError`. Plain Plotly rendering still works.
- `REMOVED, BREAKING` — **evidently.** `check_drift` method (and its test) removed from all experiment classes.
- `REMOVED, BREAKING` — **fairlearn.** `check_fairness` method (and its test) removed.
- `REMOVED, BREAKING` — **ydata-profiling.** `eda` method scheduled for removal; extras entry deleted.
- `REMOVED, BREAKING` — **explainerdashboard.** `dashboard` method scheduled for removal; extras entry deleted.
- `REMOVED, BREAKING` — **gradio.** `create_app` method scheduled for removal; extras entry deleted.
- `REMOVED, BREAKING` — **fastapi, uvicorn.** `create_api` method scheduled for removal; extras entry deleted. If a web server layer comes back it will be a separate `pycaret-server` package.
- `REMOVED, BREAKING` — **boto3.** `deploy_model`'s S3 path no longer supported. `test_persistence.py` S3 fixtures/tests removed.
- `REMOVED, BREAKING` — **m2cgen.** `convert_model` method scheduled for removal; extras entry deleted.
- `REMOVED, BREAKING` — **moto.** Test-only dep for boto3 S3 mocking; removed.
- `REMOVED, BREAKING` — **flask, Werkzeug.** `parallel` extras artifacts; removed with dask removal.
- `REMOVED, BREAKING` — **dash[testing].** `explainerdashboard` test dep; removed.
- `REMOVED, BREAKING` — **scikit-learn-intelex.** Intel oneAPI sklearn-plus-daal4py extension dropped. `engine="sklearnex"` paths in the model containers will fail to match; corresponding tests were deleted.
- `REMOVED, BREAKING` — **trio<0.25.** Legacy httpcore workaround removed.
- `REMOVED, BREAKING` — **statsforecast.** Demoted from the `timeseries` extra. (Needed a C toolchain on Python 3.14 with no wheel; no PyCaret code imports it directly. Users can install manually.)
- `REMOVED, BREAKING` — **tbats.** Demoted from the `timeseries` extra. (Declares `numpy<2`, incompatible with NumPy 2.x modernization. `BATSContainer` / `TBATSContainer` now try-import and silently mark themselves inactive if the package is missing; users can still `pip install tbats` manually.)

## DEPS — upper-bound pins removed (modernization)

All of these caps blocked PyCaret from running on modern Python or modern scientific-stack versions.

- `CHANGED, BREAKING` — **`scikit-learn` cap lifted from `<1.5` to `>=1.7,<1.8`.** The `<1.8` upper bound is *transitional*: `sktime` (the `timeseries` extra) caps sklearn at `<1.8`. When sktime releases with sklearn 1.8 support, we bump.
- `CHANGED, BREAKING` — **`numpy` cap lifted from `<1.27` to no upper bound.** Floor is `>=1.26` (for NumPy 2.x API). Codebase updated to NumPy 2 compatibility; see fixes below.
- `CHANGED, BREAKING` — **`pandas` cap lifted from `<2.2` to no upper bound.** Floor is `>=2.2`.
- `CHANGED, BREAKING` — **`scipy` cap lifted from `<=1.11.4` to no upper bound.** Floor is `>=1.11`.
- `CHANGED` — **`joblib` cap lifted from `<1.5` to no upper bound.** Floor is `>=1.4`.
- `REMOVED` — **`matplotlib` upper bound `<3.8.0` removed.** matplotlib now declared `>=3.9` as a transitional core dep (only used by residual non-Plotly plot paths — Phase 3 will remove it entirely).
- `REMOVED` — **`sktime` pinned release `>=0.31.0,<0.31.1` unpinned to `>=0.36`.**
- `REMOVED` — **`shap` upper bound `<0.47.0` removed.** Now `>=0.46`.
- `REMOVED` — **`fairlearn==0.7.0` pin removed** (fairlearn itself dropped).
- `REMOVED` — **`evidently<0.4.30` pin removed** (evidently dropped).
- `REMOVED` — **`pmdarima` exact version constraints relaxed.** Floor `>=2.0.4`.
- `REMOVED` — **`dask<2024.6.3`, `distributed<2024.6.3`, `fugue==0.9.1` pins removed** (packages dropped).
- `REMOVED` — **`Werkzeug>=2.2,<3.0` pin removed** (dropped).
- `REMOVED` — **`moto<5.0.0` pin removed** (dropped).

## DEPS — optional-extras restructuring

Extras reorganized from the 3.x layout (`full`, `analysis`, `models`, `tuners`, `mlops`, `parallel`, `prophet`, `dev`, `test`) to a cleaner 4.0 layout:

- `CHANGED, BREAKING` — **`mlops` extra deleted.** Its contents (mlflow, gradio, fastapi, uvicorn, m2cgen, boto3, evidently) are on the kill list.
- `CHANGED, BREAKING` — **`parallel` extra deleted** (dask/distributed/fugue gone).
- `ADDED` — **`anomaly` extra.** Isolates pyod + numba so classification/regression users don't pay the install cost.
- `ADDED` — **`timeseries` extra.** Contains statsmodels, sktime, pmdarima; isolated from core because sktime's dep closure is heavy.
- `CHANGED` — **`full` extra** now means `pycaret[models,tuners,analysis,anomaly,timeseries]` (plain alias — no duplicated list).
- `CHANGED` — **`test` extra** gains `pytest-xdist`, `pytest-cov`, `nbval`; loses `fugue[dask]`, `dash[testing]`, `moto`.
- `CHANGED` — **`dev` dependency group** replaces `dev` extra. Contents: `ruff`, `mypy`, `pre-commit`. (black/isort/flake8 dropped in favour of ruff.)
- `BUILD` — **kaleido pinned to `>=0.2`** (core dep for Plotly static image export).
- `BUILD` — **Core deps trimmed from 30 to 19.** Removed from core: `deprecation`, `markupsafe` (Colab workaround), `wurlitzer`, `importlib_metadata` (stdlib now), `setuptools` (runtime), `mljar-scikit-plot`, `schemdraw`, `plotly-resampler`, `yellowbrick`, `trio`, plus the timeseries packages which moved to their extra.

## REMOVED — modules and tests

- `REMOVED, BREAKING` — **`pycaret/parallel/`** directory deleted.
- `REMOVED, BREAKING` — **`pycaret/internal/parallel/`** directory deleted.
- `REMOVED, BREAKING` — **`pycaret/loggers/{mlflow,comet,wandb,dagshub,dashboard}_logger.py`** — five logger modules deleted.
- `REMOVED, BREAKING` — **`pycaret/internal/patches/yellowbrick.py`** deleted.
- `REMOVED, BREAKING` — **`pycaret/internal/plots/yellowbrick.py`** deleted.
- `TESTS` — **14 test files deleted:**
  - `test_classification_parallel.py`, `test_regression_parallel.py`, `test_time_series_parallel.py` (parallel gone)
  - `test_mlflow_artifacts.py` (was empty)
  - `test_time_series_mlflow.py`
  - `test_create_api.py`, `test_create_app.py`, `test_create_docker.py`
  - `test_dashboard.py`
  - `test_check_drift.py`, `test_check_fairness.py`
  - `test_clustering_engines.py` (daal4py-only)
  - `test_classification_engines.py`, `test_regression_engines.py`, `test_time_series_engines.py` (sklearnex-only)
- `TESTS` — **`test_persistence.py` reduced to a single-line stub comment** (all its tests were boto3/moto/S3 specific).

## CHANGED — API / signature changes

- `CHANGED, BREAKING` — **`compare_models(parallel=...)` argument removed** from 7 files: `classification/{functional,oop}.py`, `regression/{functional,oop}.py`, `time_series/forecasting/{functional,oop}.py`, `internal/pycaret_experiment/supervised_experiment.py`. Passing `parallel=` to `compare_models` will now raise `TypeError: unexpected keyword argument`.
- `REMOVED, BREAKING` — **`_parallel_compare_models` method deleted** from `internal/pycaret_experiment/supervised_experiment.py`.
- `CHANGED, BREAKING` — **`setup(log_experiment=...)` no longer accepts string shortcuts** (`"mlflow"`, `"comet_ml"`, `"wandb"`, `"dagshub"`). Only `bool` or `BaseLogger` instances are valid now. `_validate_log_experiment` / `_convert_log_experiment` rewritten in `internal/pycaret_experiment/tabular_experiment.py`. The `list[...]` form is preserved but validates each element to the new rule.
- `CHANGED` — **`_convert_log_experiment` now always returns a `BaseLogger` instance, never a bool.** Downstream hooks (`log_experiment`, `log_model`, `log_model_comparison`, `log_plot`, `.loggers`) can be called unconditionally — the default `BaseLogger()` is a silent no-op for every method. This eliminates the `if self.logging_param:` truthiness checks that permeated the experiment classes.
- `CHANGED` — **`BaseLogger` (in `pycaret/loggers/base_logger.py`) rewritten to be the null/identity logger.** Was an `ABC`; is now a concrete class with every hook method implemented as a silent no-op. Adds `.loggers` property returning `[self]` to replace the removed `DashboardLogger` fan-out. Adds `log_experiment`, `log_model`, `log_model_comparison`, `log_plot`, `log_hpram_grid`, `log_artifact`, `log_sklearn_pipeline` as no-op overridable hooks. Users who had a `BaseLogger` subclass should still work if they only override hooks (they will inherit no-op defaults for the new methods).
- `CHANGED` — **`pycaret/loggers/__init__.py` now exports only `BaseLogger`.** Previously exported `MlflowLogger`, `CometLogger`, `WandbLogger`, `DagshubLogger`, `DashboardLogger` — all deleted.

## CHANGED — transitional stubs (Phase 3 will replace)

These raise a clear `NotImplementedError` at call time rather than at import time, so the package remains importable while killed features are progressively re-implemented.

- `CHANGED, BREAKING` — **`show_yellowbrick_plot` stubbed** in `internal/pycaret_experiment/tabular_experiment.py`. All `plot_model` calls that historically routed through yellowbrick now raise `NotImplementedError` with a pointer to the kill-list doc and Phase 3.
- `CHANGED, BREAKING` — **`MlflowLogger` / `CometLogger` / `WandbLogger` / `DagshubLogger` stubs** installed in `tabular_experiment.py`. Any code constructing one raises `NotImplementedError` mentioning "pass a custom `BaseLogger` subclass or `log_experiment=False`".
- `CHANGED, BREAKING` — **`scikitplot.metrics.plot_lift_curve / plot_cumulative_gain / plot_ks_statistic` stubbed** in `tabular_experiment.py`. `plot_model('lift' | 'gain' | 'ks')` will raise `NotImplementedError` until Phase 3 ships the Plotly replacements.
- `CHANGED, BREAKING` — **`plotly_resampler.FigureResampler / FigureWidgetResampler` stubbed** in `time_series/forecasting/oop.py`. `display_format='plotly-widget'` and `'plotly-dash'` raise `NotImplementedError`.
- `CHANGED` — **`with patch(...)` yellowbrick mock-patches replaced with `contextlib.nullcontext()`** in `tabular_experiment.py`. Preserves body indentation until Phase 3 removes the wrapper blocks entirely.

## FIXED — modernization bugs

- `FIXED` — **`distutils.version.LooseVersion` import removed.** Python 3.12 removed the `distutils` module. `pycaret/utils/_dependencies.py` rewritten to use `packaging.version.Version` + stdlib `importlib.metadata` (replaces the `importlib_metadata` backport).
- `FIXED` — **`joblib.Memory` `bytes_limit` kwarg handling updated.** `joblib>=1.4` moved `bytes_limit` off the constructor and onto `reduce_size(bytes_limit=...)`. `FastMemory.__init__` in `pycaret/internal/memory.py` now strips the old kwarg and forwards it into `reduce_size` on each reduction.
- `FIXED` — **`np.NaN` → `np.nan`.** `pycaret/internal/preprocess/preprocessor.py` line 682. NumPy 2.0 removed the capitalised alias.
- `FIXED` — **`sklearn.metrics._regression._check_reg_targets` new signature.** sklearn 1.7 inserted `sample_weight` as a positional parameter and expanded the return from 4 to 5 values. Custom MAPE metric in `pycaret/containers/metrics/regression.py` updated to pass the new arg and unpack the new return.
- `FIXED` — **BATS / TBATS containers guard against missing `tbats`.** `pycaret/containers/models/time_series.py`: `BATSContainer.__init__` and `TBATSContainer.__init__` now try-import the sktime wrapper inside a `try`/`except` and set `self.active = False` on failure, rather than crashing the entire container registry.
- `FIXED` — **`MatplotlibDefaultDPI` rewritten** in `pycaret/internal/plots/helper.py`. Previously went through `scikitplot.metrics.plt`; now talks to matplotlib directly. Fixes an `ImportError` at module load.

## DOCS

- `DOCS` — **Created `docs/revamp/` directory** — the authoritative narrative of the revamp. Index in `README.md`.
- `DOCS` — **Created `docs/revamp/AUDIT.md`** — baseline inventory of 3.4.0: 62,164 LOC, monster-file hotspots, dep upper-bound audit, kill-list evidence with file paths, test landscape, headline risks.
- `DOCS` — **Created `docs/revamp/KILL_LIST.md`** — every dep and subsystem being removed, with replacements and rationale. Pre-approved removals.
- `DOCS` — **Created `docs/revamp/ROADMAP.md`** — phased plan (Phase 0 groundwork / Phase 1 amputation / Phase 2 modernization / Phase 3 Plotly plot rewrite / Phase 4 agent+UI API / Phase 5 docs+release). Exit criteria per phase.
- `DOCS` — **Created `docs/revamp/DECISIONS.md`** — ADR-style decision log.
- `DOCS` — **Created `docs/revamp/STATUS.md`** — current session status, headline metrics, next steps.
- `DOCS` — **Created `docs/revamp/thinking/2026-04-22_session1_framing.md`** — scoping conversation and rejected approaches.
- `DOCS` — **Created `docs/revamp/thinking/2026-04-22_python314_pep649_blocker.md`** — why Python 3.13 not 3.14 for primary dev (upstream joblib+cloudpickle blocked on PEP 649).
- `DOCS` — **Created `docs/revamp/thinking/2026-04-22_session1_outcomes.md`** — quantitative/qualitative session notes intended to feed the research paper.
- `DOCS` — **Created `docs/revamp/thinking/phase0_failure_landscape.md`** — 158 test failures clustered into 5 root causes with ROI-ordered fix list.
- `DOCS` — **Created `docs/revamp/release_notes_pycaret4.md`** (this file) — engineering change log; user-facing notes will be generated from it.
- `DOCS` — **Created user-memory files under `.claude/memory/`** — workspace layout, kill list, version targets, revamp style, user profile, research-paper framing. Persist across sessions.

## TESTS

- `TESTS` — **Stripped `from mlflow.tracking import MlflowClient` imports** from `test_classification.py`, `test_regression.py`, `test_clustering.py`, `test_anomaly.py`.
- `TESTS` — **Deleted `TestClassificationExperimentCustomTags` class** (3 mlflow-custom-tag tests) from `test_classification.py`.
- `TESTS` — **Deleted `TestRegressionExperimentCustomTags` class** (3 mlflow-custom-tag tests) from `test_regression.py`.
- `TESTS` — **Deleted mlflow-specific `test_clustering` function** from `test_clustering.py`. File now has a `data` fixture but no tests — will be populated in a later session.
- `TESTS` — **Deleted mlflow-specific `test_anomaly` function** from `test_anomaly.py`. Same state as `test_clustering.py`.
- `TESTS` — **First post-amputation full run captured** in `docs/revamp/thinking/phase0_pytest_run1.log`: 568 passed / 158 failed / 8 skipped in 34:26 (77.4% pass rate). Three further engine-only test files deleted after the run.

## INTERNAL

- `INTERNAL` — **`_show_versions` reference to `tbats` left in place.** Even though tbats is no longer a declared extra, `pycaret/utils/_show_versions.py` still lists it in its introspection table; the BATS/TBATS containers gracefully no-op when the package is missing.
- `INTERNAL` — **uv dev-dependencies migrated from `[tool.uv.dev-dependencies]` to `[dependency-groups.dev]`.** The former is deprecated in current uv versions.

---

<!--
To add a new session below, follow this template:

# Session N — YYYY-MM-DD — <phase or task focus>

Baseline: <commit or state>.
Environment: <changes from previous session, if any>.

## <CATEGORY TAG>

- `TYPE[, BREAKING]` — **Short title.** Detail. Files. Rationale.
-->
