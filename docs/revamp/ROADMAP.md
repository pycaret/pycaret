# PyCaret 4.0 — Phased Roadmap

*Last revised: session 13 (2026-04-24). Restructured around Control Plane spec.*

This roadmap maps the product vision in [`VISION.md`](VISION.md) and the
full technical specification in [`CONTROL_PLANE_SPEC.md`](CONTROL_PLANE_SPEC.md)
onto concrete engineering phases. A phase is "done" when its exit criteria are
met and an entry has been appended to [`DECISIONS.md`](DECISIONS.md).

Checkbox legend: [x] complete · [ ] not started · 🟡 in flight · 🟢 mostly done · ✅ fully done · 🔴 not started.

---

## MVP 1 — Engine (pip-installable PyCaret 4)

Status: 🟢 **MOSTLY DONE** (sessions 1–8). 4.0.0a1 on PyPI.

- [x] Python 3.11+ / sklearn 1.7+ / NumPy 2 / pandas 2 modernization (Phases 0–2).
- [x] Kill-list removal: 9 killed verbs (`check_fairness`, `check_drift`, `dashboard`, `create_api`, `create_docker`, `create_app`, `convert_model`, `deploy_model`, `eda`), mlflow/comet/wandb/dagshub/yellowbrick/fugue/dask/ray/etc.
- [x] Functional API killed; OOP-only (`Experiment` / `ClassificationExperiment` / ...).
- [x] Typed introspection (`pycaret.api.list_models`, `describe_model`, `list_metrics`, `describe_setup_params`).
- [x] Structured event logger (`pycaret.logging.BaseLogger` + `MemoryLogger` + subscriber pattern).
- [x] 32 engine tests green across Ubuntu + Windows × Python 3.11 / 3.12 / 3.13.
- [x] Dep floor dropped to ~13 packages (from ~65 pre-revamp).
- [x] Published to PyPI as `4.0.0a1`.
- [ ] **Stateless `engine.run(config)` entry point** on top of the existing OOP surface — session 14 kickoff.
- [ ] **RunConfig Pydantic model** (§ 6.1 spec) as the single contract driving notebook / API / UI / LLM-generated runs. Ships in `packages/shared-schemas` once built.
- [ ] **God-class drain (Phase 5)** — the 10 OOP verbs still delegate to `self._legacy`. Migrate verb-by-verb to native sklearn. ~10 sessions worth of work. Order: `save_model → predict_model → create_model → tune_model → ensemble_model → blend_models → stack_models → calibrate_model → compare_models → finalize_model`.
- [ ] **Plotly plot rewrite (Phase 3)** — new `pycaret/plots/` flat module. Enables dropping matplotlib/schemdraw/kaleido from core.
- [ ] **Release 4.0.0 (non-alpha)** to PyPI once god-class is drained.

### Exit criteria
- `pip install pycaret` gives you a stateless engine driven by `RunConfig`.
- God-class drained; `pycaret/internal/pycaret_experiment/` deleted.
- 4.0.0 on PyPI.

---

## MVP 2 — Backend (`services/api`)

Status: ✅ **FULLY DONE** (sessions 9–11). 30 integration tests green.

- [x] FastAPI app factory + CORS + lifespan.
- [x] SQLAlchemy 2.x + Alembic baseline migration. 14 tables: `users`, `sessions`, `api_keys`, `workspaces`, `workspace_members`, `data_sources`, `projects`, `experiments`, `runs`, `events`, `artifacts`, `fold_metrics`, `pipelines`, `pipeline_project_links`, `deployments`.
- [x] Auth: bcrypt + JWT access + rotating refresh tokens + session revocation.
- [x] First-run bootstrap (`/setup/status`, `/setup/bootstrap`).
- [x] Engine introspection proxy (`/describe/models`, `/describe/metrics`, `/describe/setup-params`).
- [x] Workspace / Project / Experiment CRUD.
- [x] **Run execution** — `RunOrchestrator` with `ThreadPoolExecutor`, `DBEventLogger(BaseLogger)` bridging engine events to DB + WebSocket. Plans: `setup` / `create` / `compare`.
- [x] **WebSocket event stream** — `/runs/{id}/events/ws?token=…` with `EventBroker` bridging worker-thread emission to asyncio via `call_soon_threadsafe`.
- [x] **Data sources** — CSV upload (64 MB cap, SHA-256, column sample) + S3/Postgres connector registration.
- [x] **Deployments + in-house serving** — `/runs/{id}/promote` → `Pipeline` row → `/pipelines/{id}/deployments` → `/deployments/{slug}/predict`. `DeploymentRegistry` in-process LRU + p50/p95 rolling window.
- [x] **Run cancellation** — cooperative `threading.Event` polled at stage boundaries.
- [x] Alembic migrations (auto-migrate SQLite dev; explicit `pycaret-server migrate` for prod).
- [x] 30 integration tests green (server) + 62/62 combined with engine.
- [ ] **Trial entity** — expand `Run.leaderboard` JSON into first-class `trials` table rows (one per AutoML candidate). Needed for the Trials tab in the UI.
- [ ] **Prediction Log + Drift Report tables + routes** — § 4.11 / § 4.12 of spec.
- [ ] **Model Library DB entity** — move engine's hardcoded registry into editable `model_library` rows synced from engine metadata.
- [ ] **Job queue** — upgrade from `ThreadPoolExecutor` to a `Job` table + `services/worker` runner (Celery / RQ / Arq pluggable).
- [ ] **LLM gateway** (see MVP 3 below — provider router + 6 advisory endpoints).
- [x] **API keys** — DB table + CRUD routes + `X-PyCaret-Key` auth middleware (sessions 19 + 20).
- [x] **Workspace member management** — invite / list / PATCH role / remove + last-admin guard (session 20).
- [x] **Audit logs** — append-only table + middleware that records every mutating API call + `/admin/audit` viewer (session 21, SPEC § 17.4).
- [x] **Drift reports + drift analyst** — DriftReport table + 3 CRUD routes + 6th LLM copilot + deployment UI surface (session 21, SPEC § 4.12 / § 11.2 / § 12.2).
- [ ] **Secrets encryption** for LLM keys, cloud credentials.

### Exit criteria
- Every endpoint listed in [`CONTROL_PLANE_SPEC.md § 14`](CONTROL_PLANE_SPEC.md#14-api-surface) implemented (current: ~40 of ~300 planned).
- Trials, Jobs, LLM, Drift, Model-Library tables all exist and are exercised by tests.

---

## MVP 3 — Web UI (`apps/web`)

Status: 🟢 **CORE COMPLETE** (sessions 12 + 14 + 15 + 16). 12/14 screens. 33 tests green. Full product loop shipped.

- [x] Vite 5 + React 18 + TypeScript 5 + Tailwind 3 + TanStack Query + Zustand + React Router 6 scaffold.
- [x] Typed API client (hand-written; `npm run gen:api` wired for growth).
- [x] Auth: Zustand store + localStorage refresh token + axios single-flight refresh interceptor + `<AuthGate>` session restore.
- [x] Dark-mode-first Tailwind palette + component primitives (`.btn-*`, `.input`, `.card`, etc.).
- [x] Screens shipped (session 12): `/setup`, `/login`, `/` (workspaces), `/workspaces/:id` (projects).
- [x] Production bundle: 83 kB gzipped. 6 tests green. Docker image + CI job live.
- [x] *(session 14)* **`<DynamicForm>` + `<ParamInput>`** — 100%-data-driven form infrastructure dispatching on `ParamKind` (bool / int / float / enum / column / string) and grouped by engine-declared `group`. `applyDefaults` / `stripDefaults` helpers so API payloads carry user intent only. 13 tests locking the contract.
- [x] *(session 14)* **`/workspaces/:wsId/projects/:projectId`** — project detail with experiments list + "New experiment" link.
- [x] *(session 14)* **`/workspaces/:wsId/projects/:projectId/experiments/new`** — experiment setup wizard **100% driven by `describe_setup_params(task)`**. Zero UI code hard-codes a parameter name.
- [x] *(session 14)* **`/workspaces/:wsId/projects/:projectId/experiments/:experimentId`** — experiment detail with config overview, runs table (auto-polls while pending), and a minimal new-run sidebar (plan / model / sklearn sample dataset).
- [x] *(session 15)* **`/runs/:runId`** — dedicated run detail. Live WebSocket event stream (`<EventStream>` with connection + replay + sentinel handling + single-retry reconnect). Sortable leaderboard (`<Leaderboard>` — zero hard-coded metric names). Cancel button while pending. Promote-to-pipeline form on success. Polls run row every 2 s until terminal. Full request snapshot for reproducibility.
- [x] *(session 15)* **Experiment sidebar upgrade** — model picker driven by `describeApi.models(task)`; data-source picker mixing workspace CSV uploads + sklearn sample fallbacks; runs-table rows now clickable through to `/runs/:id`.
- [x] *(session 16)* **`/workspaces/:wsId/pipelines`** + **`/workspaces/:wsId/pipelines/:id`** — workspace-scoped fitted-pipeline registry + deploy-form sidebar (slug regex validation, auth-mode selector) + list of existing deployments per pipeline.
- [x] *(session 16)* **`/workspaces/:wsId/deployments`** + **`/deployments/:id`** — workspace deployments list + single-deployment view with 4 stat cards (predictions / errors / p50 / p95) + `<PredictTester>` (JSON textarea → predictions table + latency). Polls every 3–5 s to keep metrics fresh.
- [x] *(session 16)* **`<DataSourcesCard>`** in `WorkspaceDetail` — CSV upload + list + delete. Multipart upload via `dataSourcesApi.uploadCsv`. Row count / file size / column count rendered inline.
- [ ] **`/datasets/:id`** — dataset overview / schema / profile / quality / versions.
- [ ] **`/deployments/:id`** — endpoint details + test form + logs + metrics + drift tab.
- [ ] **`/monitoring`** — deployment health + drift alerts.
- [ ] **`/admin/users`** + **`/admin/workspace`** + **`/admin/integrations`** — workspace admin surface.
- [ ] **AI Assistant widget** — surfaces LLM suggestions inline (design generator on `/experiments/new`, run explainer on `/runs/:id`, drift analyst on `/monitoring`).
- [ ] Light-mode opt-in (dark-mode-first remains default).

### Exit criteria
- All 14 sidebar entries in [`CONTROL_PLANE_SPEC.md § 13.1`](CONTROL_PLANE_SPEC.md#131-sidebar) wired.
- Every backend endpoint has a UI touchpoint.
- Live event stream renders during an AutoML run.

---

## MVP 4 — Self-hosted distribution (`infra/docker`)

Status: 🟢 **MOSTLY DONE** (sessions 9, 12).

- [x] `infra/docker/Dockerfile.api` (multi-stage, non-root, healthchecked).
- [x] `infra/docker/Dockerfile.ui` (multi-stage, nginx runtime, SPA fallback + `/api` + WebSocket reverse proxy).
- [x] `infra/docker/docker-compose.yml` (api + web services, SQLite + artifact volume, one-command startup).
- [x] `infra/docker/nginx.ui.conf` with 1h WS upgrade timeouts for long AutoML runs.
- [ ] **`infra/docker/docker-compose.prod.yml`** with Postgres + MinIO + Redis (optional) + reverse-proxy (Caddy / Traefik) + TLS.
- [ ] **`services/worker` container** wired into compose once the Job queue lands.
- [ ] **`services/deployment-runtime` container** for prod serving (separate from the API process).

### Exit criteria
- `docker compose up` = full stack with managed Postgres + object storage.
- Prod compose variant with TLS + reverse proxy in under 10 minutes from clone.

---

## V2 — Enterprise readiness

Status: 🔴 **NOT STARTED**. Each bullet is roughly one session.

- [ ] **User roles** — expand from admin/member to owner / admin / project_admin / ml_engineer / data_scientist / viewer / service_account (§ 17.2).
- [ ] **Audit logs** — append-only table, UI viewer, retention policy.
- [ ] **API keys** — programmatic access with scoped permissions.
- [ ] **SSO / SAML / OAuth / LDAP** — one provider per session.
- [ ] **Secrets encryption** — KMS / Vault integration for LLM keys + cloud credentials.
- [ ] **Backup / restore** — DB snapshot + artifact archive workflow.
- [ ] **Model Library UI** — admin enable/disable, edit search spaces.
- [ ] **LLM Assistant UI** — all 6 advisory features wired into their screens (§ 12.2).
- [ ] **AutoML full pipeline search** — preprocessing + model + hyperparameters in one search (§ 7).
- [ ] **Drift monitoring** — § 11.2. PSI / KS / Jensen-Shannon feature drift, prediction drift, periodic cron.
- [ ] **Deployment rollback + scaling** — version history + replica control.
- [ ] **Scheduled retraining** — cron jobs that re-run an experiment on a fresh dataset snapshot.
- [ ] **Cloud deployment templates** — `infra/terraform/{aws,gcp,azure}` each fully implemented.
- [ ] **Kubernetes** — `infra/helm/pycaret/` chart with prod-grade values.
- [ ] **Electron desktop** — `apps/desktop/` fully built; signed installers per OS; auto-update.
- [ ] **Python SDK** — `packages/sdk-python/` published to PyPI as `pycaret-client`.

### Exit criteria
- SSO-authed, audit-logged Control Plane runs on EKS / GKE / AKS from Terraform.
- Electron desktop installer works on macOS / Windows / Linux.
- `pip install pycaret-client` + Control-Plane instance reproduces the UI's full workflow in code.

---

## V3 — Scale + governance

Status: 🔴 **NOT STARTED**. Long-term.

- [ ] Kubernetes-native execution (runs as K8s Jobs, not in-process workers).
- [ ] Distributed AutoML (Ray / Dask opt-in backend).
- [ ] Approval workflows (reviewer required before `promote` / `deploy`).
- [ ] Model cards + governance reports.
- [ ] Multi-environment deployments (dev / staging / prod promotion).
- [ ] Feature store integrations (Feast / Tecton / Snowflake Feature Store).
- [ ] Advanced monitoring (Prometheus metrics export, OpenTelemetry traces).
- [ ] Plugin system (custom preprocessors + models loaded at runtime).
- [ ] Marketplace for community models + preprocessors.

---

## Current session ledger

| Session | Theme | Ships |
|---|---|---|
| 1–6 | Engine revamp Phases 0–3.5 | 4.0.0 OOP engine scaffold |
| 7–8 | Dep cut + `4.0.0a1` to PyPI | Lean install |
| 9 | Backend scaffold (Part 2 kickoff) | 14 tables + auth + CRUD |
| 10 | Run execution + WebSocket | Event-streamed AutoML |
| 11 | Phase 9 finish | Data sources + deployments + cancel + Alembic |
| 12 | Frontend scaffold (Phase 10 start) | 4 screens live |
| 13 | Monorepo restructure + Control Plane spec | Canonical structure + docs |
| 14 | Project detail + Experiment wizard (dynamic form) | Experiment creation UI + runs sidebar + 13 new tests |
| 15 | `/runs/:id` with live WebSocket + leaderboard + data-source + model pickers | Run view + promote button + 8 new tests |
| 16 | Pipelines + Deployments + CSV upload UI — full serving loop | 4 screens + 2 components + 6 tests + E2E verified |
| 17 | LLM router (Claude + OpenAI) + dataset consultant + settings screen | 2 DB tables + 6 API routes + 1 UI screen + 1 modal + 12 new tests + live E2E |
| 18 | Experiment designer + Run explainer advisories | 2 consultations + 2 routes + 2 UI components + 11 new tests (118/118 combined) |
| 19 | Failure debugger + Deployment reviewer + API keys | 2 copilots + 3 API-key routes + 3 UI components + 1 screen + 16 new tests (134/134) |
| 20 | Workspace members + X-PyCaret-Key auth middleware | 4 member CRUD routes + API-key auth fallback + 1 UI screen + 18 new tests (148/148) |
| 21 | Drift analyst + audit logs | DriftReport table + 3 drift routes + 6th copilot + AuditLog + middleware + 2 viewer routes + 2 UI components + 1 screen + 22 new tests (174/174) |
| 22 | God-class drain #1 — persistence verbs | save_model/load_model/save_experiment/load_experiment off self._legacy. 7 new tests (181/181). 4 of 10 god-class verbs drained. |
| 23 | God-class drain #2 — `predict_model` | Native task-aware dispatch. 12 new tests (197/197). 5 of 10 verbs drained. |
| 24 | God-class drain #3 — `create_model` (supervised) | CreateResult.pipeline now a real sklearn Pipeline. 10 new tests (207/207). 6 of 10 verbs drained (classification + regression only; clustering/anomaly/TS still delegate). |
| 25 | God-class drain #4 — `tune_model` (supervised) | RandomizedSearchCV with registry tune_grid. TuneResult.search populated. 9 new tests (216/216). 7 of 10 verbs drained. |
| 26 | God-class drain #5 — `compare_models` (supervised) | Native iteration over registry, reuses drained create_model. 10 new tests (226/226). 8 of 10 verbs drained. |
| 27 | God-class drain #6 — ensemble + blend + stack + calibrate + finalize | All 5 remaining supervised verbs drained in one batch. 13 new tests (239/239). ALL 13 supervised verbs done. 0 of supervised verbs still on self._legacy. |
| 28 | God-class drain #7 — unsupervised (clustering + anomaly) | create_model + assign_model native for clustering + anomaly. CreateResult.pipeline is now a real sklearn Pipeline for ALL non-TS tasks. 11 new tests (250/250). |
| 29 | Property drain — user-facing data accessors | X / X_train / X_test / y / y_train / y_test / preprocess_pipeline read from self._fit_state, not self._legacy. Public API surface fully drained. 4 new tests (254/254). |
| 30 | Internal-state drain — transformed splits + fold generator + model registry | X_transformed / X_train_transformed / y_transformed / y_train_transformed / fold_generator / model_registry promoted to self._fit_state. 13 internal `_legacy` reads in drained verbs eliminated. 5 new tests (259/259). |
| 31 | Secondary-verb drain — pull / models / get_metrics | pull() reads from _fit_state["last_metrics"], updated by every native modeling verb. models() + get_metrics() build DataFrames natively from the snapshot + metric registry. 8 new tests (267/267). |
| 32 | Per-Experiment metric registry + add_metric / remove_metric drain | Metric registry promoted to _fit_state["metric_registry"]; add_metric mutates it; CV / leaderboard / predict all read from it. Custom metrics actually show up in CV now (real bugfix). 10 new tests (277/277). |
| 33 | get_config / set_config drain | Last drainable secondary verbs done. get_config reads from _fit_state + ctor params; set_config has tight allowlist (session_id/n_jobs/verbose/fold/log_experiment). 10 new tests (287/287). DRAIN COMPLETE on the public surface. |
| 34 | Fix sklearn 1.6+ `squared=` deprecation in regression metrics | RMSE / RMSLE containers use root_mean_squared_(log_)error directly. -69 warnings per test run. 4 new tests (291/291). |
| 35 | Native `setup()` (phase 1, simple supervised) | Skip self._legacy.setup() entirely for clf+reg with no complex preprocessing flags. Native train/test split + impute + encode + fold generator + registry-via-proxy. 10 new tests (301/301). The biggest remaining drain target lands incrementally. |
| 36 | Native `setup()` phase 2: normalize + transformation | StandardScaler + PowerTransformer(yeo-johnson) chain into the native numeric branch. normalize=True / transformation=True no longer force legacy. 10 new tests (311/311). |
| 37 | Native `setup()` phase 3: remove_outliers + feature_selection | IsolationForest drops 5% most anomalous train rows; SelectFromModel(median) with ExtraTrees keeps above-median features + appends to preprocess pipeline. EVERY supervised constructor flag now native. 8 new tests (319/319). |
| 38 | Native `setup()` phase 4: unsupervised tabular (clustering + anomaly) | `_native_setup_unsupervised` mirrors the supervised chain (impute + ordinal + optional StandardScaler / PowerTransformer) on the full frame — no train/test split, no fold generator. Predicate accepts CLUSTERING + ANOMALY tasks. Time-series is now the only legacy.setup() path. 9 new tests (328/328 engine). |
| 39 | Native `setup()` phase 5a: time-series soft drain | `_native_setup_timeseries` populates `_fit_state` (y / y_train / y_test / fh / seasonal_period / fold_generator / model_registry / preprocess_pipeline) so user-facing accessors work for TS. Predicate accepts TIME_SERIES. predict_model + models() defer to legacy for TS until verb drain. The TS native path still calls legacy.setup() under the hood — verbs aren't drained yet. 9 new tests (191/191 engine). |
| 40 | Phase 5b: drain `TimeSeriesExperiment.create_model` | First TS verb fully native. Resolves estimator from sktime registry, wires into `ForecastingPipeline` via `_add_model_to_pipeline`, runs CV through the existing `cross_validate` helper, refits on full y_train, returns CreateResult with a real `ForecastingPipeline`. Adds `_build_ts_metric_registry` + `_primary_sp_to_use` helpers. 10 new tests (201/201 engine). 1 of 6 TS verbs drained. |
| 41 | Phase 5c: drain `TimeSeriesExperiment.predict_model` | Second TS verb native. Calls `get_predictions_with_intervals` for forecasts and `calculate_metrics` for ground-truth scoring against `_fit_state["y_test"]`. Supports `return_pred_int`, custom `fh`, exogenous X, and bare forecasters (auto-wires into preprocess). 8 new tests (209/209 engine). 2 of 6 TS verbs drained. |
| 42 | Phase 5c (cont.): drain `TimeSeriesExperiment.compare_models` | Third TS verb native. Iterates the sktime registry → calls native `create_model` per candidate → builds leaderboard ranked by MASE (ascending). Supports `include` / `exclude` / `turbo` / `errors` / `n_select`. Filters `ensemble_forecaster` by default. 9 new tests (218/218 engine). 3 of 6 TS verbs drained. |
| **43** | **Phase 5c (cont.): drain `TimeSeriesExperiment.tune_model`** | **Fourth TS verb native. Wraps sktime `ForecastingGridSearchCV` / `ForecastingRandomizedSearchCV` around the experiment's preprocess pipeline using container `tune_grid` / `tune_distributions` (or `custom_grid=`). Auto-converts pycaret `Distribution` objects via `get_base_distributions`. Strips pipeline prefixes off `best_params_`, refits via native `create_model`. `choose_better=True` keeps whichever wins on `optimize`. 10 new tests (228/228 engine). 4 of 6 TS verbs drained.** |
| 44+ | Phase 5c (cont.) / 5d / 6 → 4.0.0 release | Drain remaining TS verbs (finalize / assign); strip legacy.setup() from `_native_setup_timeseries`; delete pycaret/internal/pycaret_experiment; ship 4.0.0 |

Roughly **7–8 sessions to MVP 3 completion** (all 8 UI screens wired + LLM assist + full AutoML flow). Then a handful of V2 items. Then god-class drain for MVP 1 release.

---

## Out of scope (forever)

- Multi-GPU / distributed training in the engine (V3 opt-in via Ray).
- Hosted experiment-tracking SaaS (mlflow / comet / wandb) — out of core.
- Notebook-as-a-service / data warehouse / job scheduler — not us.
- Backward compatibility with PyCaret 3.x internal APIs — only the OOP golden path is stable.
- GraphQL — REST + OpenAPI is simpler for this surface.
