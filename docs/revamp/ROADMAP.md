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
- [ ] **Audit logs** + **API keys** tables + routes.
- [ ] **Secrets encryption** for LLM keys, cloud credentials.

### Exit criteria
- Every endpoint listed in [`CONTROL_PLANE_SPEC.md § 14`](CONTROL_PLANE_SPEC.md#14-api-surface) implemented (current: ~40 of ~300 planned).
- Trials, Jobs, LLM, Drift, Model-Library tables all exist and are exercised by tests.

---

## MVP 3 — Web UI (`apps/web`)

Status: 🟡 **IN FLIGHT** (session 12). 4/8 screens. 6 tests green.

- [x] Vite 5 + React 18 + TypeScript 5 + Tailwind 3 + TanStack Query + Zustand + React Router 6 scaffold.
- [x] Typed API client (hand-written; `npm run gen:api` wired for growth).
- [x] Auth: Zustand store + localStorage refresh token + axios single-flight refresh interceptor + `<AuthGate>` session restore.
- [x] Dark-mode-first Tailwind palette + component primitives (`.btn-*`, `.input`, `.card`, etc.).
- [x] Screens shipped: `/setup`, `/login`, `/` (workspaces), `/workspaces/:id` (projects).
- [x] Production bundle: 83 kB gzipped. 6 tests green. Docker image + CI job live.
- [ ] **`/projects/:id`** — project detail: experiments list + New Experiment button.
- [ ] **`/experiments/:id`** — experiment setup form **100% driven by `describe_setup_params`** (zero UI code hard-codes a parameter name). RunConfig modes: manual / assisted / auto / expert.
- [ ] **`/runs/:id`** — live event stream via WebSocket + leaderboard table + artifact download + promote-to-pipeline.
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
| **13** | **Monorepo restructure + Control Plane spec** | **Canonical structure + docs** |
| 14 | `/projects/:id` + `/experiments/:id` setup wizard (dynamic form) | Experiment creation UI |
| 15 | `/runs/:id` with live WebSocket + leaderboard | Run view |
| 16 | Trial entity + Model Library table sync | DB expansion |
| 17 | LLM router + first 2 advisory endpoints | AI assist MVP |
| 18 | Dataset upload UI + profile screen | Data-source UI |
| 19 | Admin screens + API keys + audit logs | V2 foundation |
| 20+ | God-class drain → 4.0.0 release | Engine finish |

Roughly **7–8 sessions to MVP 3 completion** (all 8 UI screens wired + LLM assist + full AutoML flow). Then a handful of V2 items. Then god-class drain for MVP 1 release.

---

## Out of scope (forever)

- Multi-GPU / distributed training in the engine (V3 opt-in via Ray).
- Hosted experiment-tracking SaaS (mlflow / comet / wandb) — out of core.
- Notebook-as-a-service / data warehouse / job scheduler — not us.
- Backward compatibility with PyCaret 3.x internal APIs — only the OOP golden path is stable.
- GraphQL — REST + OpenAPI is simpler for this surface.
