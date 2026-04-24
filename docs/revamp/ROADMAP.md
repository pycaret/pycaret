# PyCaret 4.0 — Phased Roadmap

A phase is not "done" until its exit criteria are met and a `DECISIONS.md` entry is written. Items checked off are complete; items unchecked are either in-flight or not-yet-started.

---

## Part 1: Engine revamp (sessions 1-6+)

### Phase 0 — Groundwork — ✅ COMPLETE (session 1)

- [x] Clone upstream, install `uv`
- [x] Verify sklearn support matrix ⇒ Python 3.13 as primary dev target (see DECISIONS.md for the 3.14/PEP-649 finding)
- [x] Scaffold `docs/revamp/`
- [x] Complete baseline audit (`AUDIT.md`) and kill list (`KILL_LIST.md`)
- [x] Write new `pyproject.toml` v4 (lean deps, uv-first, hatchling)
- [x] Create `.venv` via `uv sync`, verify `import pycaret` + all 6 public submodules import
- [x] End-to-end smoke (`setup → compare_models → predict_model`) green on `juice` dataset
- [x] Full test-suite run captured in `thinking/phase0_pytest_run1.log`

### Phase 1 — Amputation (remove the kill list) — ✅ COMPLETE (sessions 1 + 6)

- [x] Removed `pycaret/parallel/` and `pycaret/internal/parallel/`; dropped `parallel` args from 7 files.
- [x] Removed `loggers/{mlflow,comet,wandb,dagshub,dashboard}_logger.py`.
- [x] Removed `internal/patches/yellowbrick.py`; plot branches stubbed.
- [x] Deleted 11 initial kill-list test files + mlflow-custom-tag blocks.
- [x] *(session 6)* Removed `pycaret.loggers` shim; 7 import sites re-pointed to `pycaret.logging.base`.
- [x] *(session 6)* Removed `pycaret/distributions.py` (0 callers).
- [x] *(session 6)* Removed `pycaret/internal/cloudpickle_compat.py` (0 callers).
- [x] *(session 6)* Removed `pycaret/internal/cuml_wrappers.py` (143 LOC) + replaced GPU-fallback call sites with `NotImplementedError`.
- [x] *(session 6)* Removed 9 killed verb methods from the god-class + task oop wrappers: `check_fairness`, `check_drift`, `dashboard`, `create_api`, `create_docker`, `create_app`, `convert_model`, `deploy_model`, `eda`. **~1,156 LOC dropped.**

### Phase 2 — Modernization (sklearn 1.7 / NumPy 2 / pandas 2 / Python 3.13) — ✅ MOSTLY COMPLETE (sessions 1 + 6)

- [x] Replaced `distutils.LooseVersion` with `packaging.version.Version` (Python 3.12+).
- [x] Replaced `np.NaN` with `np.nan` (NumPy 2.0).
- [x] Replaced `np.product` with `np.prod` (NumPy 2.0).
- [x] Fixed `joblib.Memory(bytes_limit=...)` → `Memory.reduce_size(bytes_limit=...)` (joblib 1.4+).
- [x] Fixed `sklearn.metrics._regression._check_reg_targets` signature change (sklearn 1.7: added `sample_weight`, returns 5-tuple).
- [x] BATS / TBATS containers guard missing `tbats` (its `numpy<2` pin conflicts with ours).
- [x] scikitplot removed from `internal/plots/helper.py` (replaced with direct matplotlib).
- [x] Unpinned `sktime` (0.31 → 0.36+); absorbed API drift into `time_series/` container guards.
- [ ] *(Phase 3 sweep)* Any remaining `FutureWarning`s from the sklearn 1.7 / pandas 2 transition.
- [ ] Full sklearn `__sklearn_tags__` rollout across every custom transformer in `internal/preprocess/*`.

**Exit criteria progress:** `pytest tests/` 32/32 green on Python 3.13 + sklearn 1.7.2 + NumPy 2.3.5 in ~2 min. ✅

### Phase 3 — Plotly plot rewrite — 🟡 PLANNED

Goal: replace the remaining yellowbrick / scikit-plot path with a flat Plotly module.

1. [ ] New `pycaret/plots/` (flat, no `internal/plots/`). One file per plot family:
   - `classification_curves.py` (ROC, PR, threshold)
   - `classification_matrix.py` (confusion, class prediction error, classification report)
   - `regression_diagnostics.py` (residuals, prediction error, Cook's distance)
   - `clustering.py` (elbow, silhouette, intercluster distance)
   - `feature.py` (RadViz, manifold)
   - `model_selection.py` (learning curve, validation curve, RFECV)
2. [ ] `plot_model` dispatches via a registry `dict[str, Callable]`; no giant if/elif chain.
3. [ ] Unified Plotly theme, dark-mode-friendly.
4. [ ] Retire `matplotlib`, `schemdraw`, `kaleido` from core deps after this lands.

**Exit criteria:** `plot_model(kind=...)` returns a `plotly.graph_objects.Figure` for every `kind`; `matplotlib` removed from `pyproject.toml` core deps.

### Phase 3.5 — Functional API killed, OOP-only — ✅ COMPLETE (session 3)

- [x] All 5 `functional.py` files deleted (~11,300 LOC).
- [x] `pycaret.tasks` exports all 5 task subclasses.
- [x] `pycaret.core.SupervisedExperiment` / `UnsupervisedExperiment` intermediate bases.
- [x] `pycaret.save_model` / `pycaret.load_model` stateless top-level utilities.
- [x] `pycaret/core/state.py` deleted (no ContextVar, no implicit state).
- [x] All 6 task-module `__init__.py`s collapsed to thin re-exports.
- [x] 41 functional-API-coupled tests deleted; 4 OOP-native test files remain (32/32 pass).
- [x] README rewritten for 4.0 positioning; tutorials doc updated; 3.x notebooks archived.

### Phase 4 — API for agents / React UI — ✅ ARCHITECTURE LANDED (session 2)

- [x] Public `pycaret.api` submodule: `list_models(task)`, `describe_model(task, id)`, `list_metrics(task)`, `describe_setup_params(task)` returning JSON-serializable dataclasses.
- [x] Typed return objects for every verb in `pycaret.core.results`.
- [x] Streaming events through `pycaret.logging.MemoryLogger`; `BaseLogger.subscribe(callback)` fans out.
- [x] `Experiment(BaseEstimator)` in `pycaret.core.experiment` — sklearn-compatible.
- [x] 5 task subclasses in `pycaret.tasks.*`.
- [ ] Progress-bar prints inside legacy god-class still go to stdout — audit remains for Phase 5.

### Phase 5 — God-class drain + release — 🟡 IN FLIGHT

Goal: empty `pycaret/internal/pycaret_experiment/` verb by verb and cut `4.0.0alpha0`.

Recommended verb-migration order (see `docs/for_developers/DRAINING_THE_GODCLASS.md`):
1. [ ] `save_model` / `load_model` — thinnest, reference implementation.
2. [ ] `predict_model` — no CV, no training.
3. [ ] `create_model` — single-model CV via `sklearn.model_selection.cross_validate`.
4. [ ] `tune_model` — `GridSearchCV` / `HalvingRandomSearchCV` / `optuna.integration.OptunaSearchCV` wrap.
5. [ ] `ensemble_model` — `BaggingClassifier` / `AdaBoostClassifier` wrap.
6. [ ] `blend_models` — `VotingClassifier` / `VotingRegressor` wrap.
7. [ ] `stack_models` — `StackingClassifier` / `StackingRegressor` wrap.
8. [ ] `calibrate_model` — `CalibratedClassifierCV` wrap.
9. [ ] `compare_models` — the heaviest; loop over registry + rank.
10. [ ] `finalize_model` — refit on full data.

Between verbs:
- [ ] `pytest tests/test_e2e_oop.py` must stay green.
- [ ] Legacy method deleted from `internal/pycaret_experiment/*`.
- [ ] Release-notes `CHANGED, INTERNAL` entry appended.

After all verbs drained:
- [ ] Delete `pycaret/internal/pycaret_experiment/` entirely.
- [ ] Delete `Experiment._legacy` and `_build_legacy_experiment()`.
- [ ] Core dep count drops to ~15.
- [ ] **Release `pycaret==4.0.0alpha0` to PyPI.**

### Phase 6 — Engine docs + notebooks + CI matrix — 🟡 PARTIAL (docs done; notebooks+CI in CI section)

- [x] `docs/revamp/` complete.
- [x] `docs/for_agents/` + `docs/for_developers/`.
- [x] `AGENTS.md` + `CONTRIBUTING.md` rewritten for 4.0.
- [x] README rewritten with 4.0 positioning + WIP banner + master-branch announcement.
- [x] 5 executed end-to-end notebooks (`notebooks/01-05`).
- [x] CI green across Python 3.11 / 3.12 / 3.13 × Ubuntu + Windows (`v4` branch, `ci-status` job).
- [ ] CI notebook-execution nightly job stays green for 7 consecutive days.
- [ ] Open-issue triage executed (224 bulk-closable per `docs/revamp/github_issues/PLAYBOOK.md`).

---

## Part 2: Application Platform (new scope — starts after Phase 5 released)

Full design in [`docs/revamp/PLATFORM_PLAN.md`](PLATFORM_PLAN.md).

**Gate:** Part 2 does not start until Phase 5 is done — that is, `pycaret==4.0.0alpha0` is on PyPI, the god-class is drained, and the library is demonstrably lightweight with extremely few deps.

### Phase 7 — CLI utility (`pycaret-cli`) — 🟡 PARTIAL (session 9)

- [x] `pycaret-server` CLI with `serve` / `version` subcommands shipped in `pycaret-server`.
- [ ] Separate `pycaret-cli/` package (project-export, YAML-driven runs, admin) — session 10+.

### Phase 8 — Database layer — ✅ COMPLETE (session 9)

- [x] `pycaret-server/pycaret_server/db/` with full SQLAlchemy 2.x models + session factory + FastAPI `get_db` dependency.
- [x] SQLite default (`sqlite:///./pycaret.db`); Postgres / MySQL driver selection via `PYCARET_DATABASE_URL`.
- [x] 14 tables (matches `PLATFORM_PLAN.md § 3` exactly): `users`, `workspaces`, `workspace_members`, `projects`, `data_sources`, `experiments`, `runs`, `events`, `artifacts`, `fold_metrics`, `pipelines`, `pipeline_project_links`, `deployments`, `api_keys`, `sessions`.
- [x] First-run bootstrap flow implemented end-to-end (`POST /api/v1/setup/bootstrap` creates admin + workspace + workspace_member + session, returns token pair).
- [ ] Alembic baseline migration — session 10 (currently boot-time `Base.metadata.create_all` on SQLite).

### Phase 9 — Backend API (`pycaret-server`) — 🟡 MOSTLY DONE (session 9)

- [x] FastAPI app factory with CORS + lifespan (creates tables on first boot).
- [x] Auth: bcrypt password hashing + JWT access-token + rotating refresh-token with session-row storage. `/api/v1/auth/{login,refresh,logout,me}`.
- [x] `POST /api/v1/setup/{status,bootstrap}` first-run flow.
- [x] `GET /api/v1/describe/{models,models/{id},metrics,setup-params}` engine-introspection proxy over `pycaret.api`.
- [x] CRUD on `/api/v1/workspaces`, `/api/v1/workspaces/{id}/projects`, `/api/v1/projects/{id}/experiments`.
- [x] OpenAPI at `/docs` + `/openapi.json`; health at `/healthz`.
- [x] **14 integration tests (pytest + httpx TestClient)** — green in ~8 s.
- [ ] `POST /api/v1/experiments/{id}/runs` → background-worker dispatch to `pycaret.tasks.*Experiment` — session 10.
- [ ] WebSocket `/ws/runs/{id}/events` — session 10.
- [ ] `/api/v1/deployments/*` + in-house serving (`DeploymentRegistry`, catch-all `/predict` route) — session 10.
- [ ] Data-source connectors (CSV upload, S3, Postgres) — session 10.

### Phase 10 — Frontend (`pycaret-ui`) — 🔴 NOT STARTED

- [ ] Vite + React 18 + TypeScript + Tailwind + TanStack Query + Zustand + Plotly.js.
- [ ] Typed API client auto-generated from `/openapi.json`.
- [ ] 8 screens: setup / login / workspaces / project / experiment / run / admin-users / admin-workspace.
- [ ] Setup form 100% driven by `describe_setup_params` (zero UI code hard-codes param names).
- [ ] Live event-stream rendering via WebSocket.
- [ ] Dark-mode first.

### Phase 11 — Docker / deploy — 🟡 PARTIAL (session 9)

- [x] `docker/Dockerfile.api` (multi-stage Python 3.13-slim + uv + non-root runtime user + healthcheck).
- [x] `docker/docker-compose.yml` (dev compose; SQLite + artifact volume at `./data/`).
- [ ] `docker/Dockerfile.ui` — after frontend phase.
- [ ] `docker/docker-compose.prod.yml` with reverse-proxy + TLS — after frontend.
- [ ] `deploy/k8s/` manifests as stretch goal.

### Phase 12 — Platform release — 🔴 NOT STARTED

- [ ] Per-package READMEs.
- [ ] 5-minute quickstart (clone → compose up → first experiment).
- [ ] Deployment guide (local / docker / k8s / cloud).
- [ ] Video walkthrough.
- [ ] Tag `pycaret-server==0.1.0` + `pycaret-cli==0.1.0` + `@pycaret/ui@0.1.0`.

---

## Out of scope (explicit non-goals for the whole programme)

- Multi-GPU / distributed training — no parallel.
- Hosted experiment-tracking SaaS (mlflow, comet, wandb) — out of core.
- Model serving — MLServer / Seldon / BentoML already do this; we link, don't replace.
- Multi-tenant hosted SaaS with billing — someone else builds this on top of the self-hostable platform.
- Backward compatibility with PyCaret 3.x internal APIs — only the OOP golden path is stable.
- Kubernetes operator — Compose is the default; K8s is thin manifests.
- GraphQL — REST + OpenAPI is simpler for this surface.
