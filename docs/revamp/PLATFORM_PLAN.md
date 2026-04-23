# PyCaret 4.0 — Application Platform Plan

*Authored: 2026-04-23. Status: **design, not yet implemented.***

This document captures the scope and phased execution plan for turning the PyCaret 4.0 engine into a full open-source AutoML platform — a credible self-hostable alternative to DataRobot and H2O.ai.

The **engine** work (sessions 1-6) built the lean, sklearn-composable Python library. The **platform** work below builds everything on top of that engine: CLI, FastAPI backend, database, React frontend, and Docker deployment.

> **Execution starts only after the engine cleanup is fully done** (ROADMAP Phase 5 god-class drained, dependency set stable, notebooks green). Until then, this document is authored in advance so the architecture is already agreed-upon when we get there.

---

## 1. Vision

**Problem statement.** Teams who want AutoML today have two choices:

1. **Pay** for DataRobot / H2O.ai / enterprise Azure ML — six- to seven-figure licenses, vendor lock-in.
2. **Roll their own** on top of sklearn + notebooks + glue — works for data scientists, doesn't scale to org-wide governance.

There's no serious open-source middle ground. PyCaret today is a library; enterprise tools are applications. The gap is: "I want a button that says 'Run AutoML on this CSV', I want to manage experiments in a project, I want to share results with my team, I don't want to buy anything."

**What we build.** An open-source application platform where a user clones a repo, runs two commands (`docker compose up` or `pycaret serve`), and ends up with:

- A **web UI** at `http://localhost:3000` that looks like a modern enterprise ML tool.
- A **workspace → project → experiment → pipeline** hierarchy for governance.
- **First-run setup wizard** — self-service admin creation, no external config.
- **The full PyCaret engine** driving every experiment — same notebook API, same sklearn outputs.
- **Docker Compose** for single-command production deploy.

**Non-goals** (explicit):
- Not a hosted SaaS — self-hosted only. Someone else can build a hosted layer on top.
- Not a multi-tenant SaaS with billing / org-level quotas.
- Not a model-serving platform (MLServer / Seldon / BentoML already exist). We link to those; we don't replace them.

---

## 2. The entire stack, at a glance

```
   ┌───────────────────────────────────────────────────────────────────┐
   │                         React UI (TypeScript)                    │
   │    • Workspace / Project / Experiment views                       │
   │    • Dynamic forms driven by pycaret.api.describe_setup_params    │
   │    • Real-time event stream (WebSocket subscriber)                │
   │    • Charts rendered from engine's Plotly figures (Phase 3)       │
   └────────────────────────────┬──────────────────────────────────────┘
                                │ REST + WebSocket
   ┌────────────────────────────▼──────────────────────────────────────┐
   │                       FastAPI backend                             │
   │    • Typed endpoints — request/response = pycaret.api dataclasses │
   │    • Auth: JWT with local user store (OAuth providers as plugin)  │
   │    • Background jobs: run_experiment dispatches to a worker       │
   │    • WebSocket: fan-out of BaseLogger events to subscribed UIs    │
   └────────┬─────────────────────────────────┬────────────────────────┘
            │ SQLAlchemy                      │ In-process
   ┌────────▼─────────────┐         ┌─────────▼────────────────────────┐
   │     Database         │         │    PyCaret 4.0 engine (library)  │
   │  default: SQLite     │         │    pycaret.tasks.*Experiment     │
   │  optional: Postgres  │         │    pycaret.api / pycaret.logging │
   │  option: MySQL       │         │    pycaret.core — typed results  │
   └──────────────────────┘         └──────────────────────────────────┘

   ┌───────────────────────────────────────────────────────────────────┐
   │                           CLI utility                             │
   │   `pycaret serve`  → spin up backend + frontend                   │
   │   `pycaret run experiment.yaml` → scripted runs without UI        │
   │   `pycaret export project` → dump to YAML for git-based workflow  │
   │   `pycaret admin create-user …` → initial admin bootstrap         │
   └───────────────────────────────────────────────────────────────────┘
```

---

## 3. Data model — workspace → project → experiment → run → pipeline → deployment

Single authoritative hierarchy. Every domain object below is an SQLAlchemy model.

```
Workspace
├── members (User × role)
├── config (theme, default compute profile, data-source allowlist)
├── pipelines                     ← workspace-scoped; shareable across projects (§ decision 3)
│   └── Pipeline
│       ├── name, description, tags
│       ├── origin_run_id (the Run that created it, if any)
│       ├── model_id (pycaret id, e.g. "lr")
│       ├── stored_path (fitted sklearn Pipeline pickle)
│       ├── sha256
│       └── linked_projects[]    ← many-to-many; a Pipeline can be used by multiple Projects
├── deployments                  ← in-house serving (§ decision 4)
│   └── Deployment
│       ├── pipeline_id (FK → Pipeline)
│       ├── endpoint_slug         ← stable path: /api/v1/deployments/{slug}/predict
│       ├── status (active / paused / archived)
│       ├── inference_count, last_inference_at
│       └── auth_mode (workspace / api-key / public)
└── projects
    └── Project
        ├── metadata (name, description, tags, owner)
        ├── data_sources (CSV upload, S3, Postgres — § decision 2)
        ├── pipeline_refs[]       ← references to workspace-scoped Pipelines
        └── experiments
            └── Experiment
                ├── config (task, target, setup params — serialized SetupParamSchema)
                └── runs                       ← many-to-one with Experiment
                    └── Run
                        ├── started_at, finished_at, status
                        ├── events[]           ← engine Event stream captured (append-only)
                        ├── leaderboard       ← CompareResult.leaderboard serialized
                        ├── artifacts[]       ← fitted pipeline pickle + notebook (§ decision 1)
                        ├── fold_metrics[]    ← per-fold × per-model metrics (§ decision 6)
                        ├── metrics_summary   ← leaderboard-shaped aggregates
                        └── produced_pipelines[] ← references to Pipelines registered from this Run
```

**Core tables (v1, 14 total):** `users`, `workspaces`, `workspace_members`, `projects`, `data_sources`, `experiments`, `runs`, `events`, `artifacts`, `fold_metrics`, `pipelines`, `pipeline_project_links`, `deployments`, `api_keys`, `sessions`.

**Granularity (§ decision 6 — "both, very comprehensive"):** `runs.metrics_summary` stores the leaderboard shape (one row per model, `mean_*` / `std_*` columns). `fold_metrics` stores every per-fold × per-model × per-metric value (one row per `(run, model, fold, metric)` — roughly `n_models × n_folds × n_metrics` rows per Run). Both are queryable from the UI: summary drives the leaderboard screen; per-fold drives detailed model-inspection and any future time-to-train / variance-across-folds dashboards.

**First-run setup flow** (self-service, no external config):
1. User runs `docker compose up` (or `pycaret serve`).
2. Frontend loads, detects `users` table is empty → redirects to `/setup`.
3. Setup wizard collects: admin email + password, workspace name, default compute (local/docker/cloud).
4. Backend POST `/api/v1/setup/bootstrap` → creates admin user, workspace, emits setup-complete event.
5. Subsequent loads go to normal login.

**Roles (v1):** `admin`, `member`. Admin manages users + workspace settings. Member creates projects/experiments. Future: `viewer` (read-only), per-project ACLs.

---

## 4. The platform as separate packages

To keep the engine lean (the 4.0 promise) we **do not** fold the platform into `pycaret/`. Instead:

```
pycaret/                              the library (what we have today)
├── pyproject.toml                    name: pycaret
└── pycaret/ …

pycaret-server/                       NEW — the FastAPI backend
├── pyproject.toml                    name: pycaret-server; depends on pycaret>=4.0
└── pycaret_server/
    ├── app.py                        FastAPI app factory
    ├── config.py                     env-driven config
    ├── auth/                         JWT + user store
    ├── db/                           SQLAlchemy models + Alembic migrations
    ├── api/                          route modules: workspaces.py, projects.py, …
    ├── engine/                       thin adapter over pycaret.tasks.*Experiment
    ├── jobs/                         background-worker dispatch (RQ or Celery-lite)
    └── websocket.py                  event-stream fan-out

pycaret-ui/                           NEW — the React frontend
├── package.json                      name: @pycaret/ui; framework: Vite + React
└── src/
    ├── api/                          typed OpenAPI client generated from pycaret-server
    ├── pages/                        Setup, Login, Workspaces, Project, Experiment, Run
    ├── components/                   FormFromSchema, Leaderboard, EventStream, …
    ├── charts/                       Plotly wrappers (figures come from engine)
    └── store/                        Zustand or Redux for UI state

pycaret-cli/                          NEW — the `pycaret` command
├── pyproject.toml                    name: pycaret-cli; depends on pycaret, pycaret-server
└── pycaret_cli/
    ├── main.py                       Typer / Click entrypoint
    ├── serve.py                      `pycaret serve` (wraps uvicorn + npm start in dev)
    ├── run.py                        `pycaret run experiment.yaml`
    ├── admin.py                      `pycaret admin create-user …`
    └── docker.py                     `pycaret docker build/up/down`

deploy/                               NEW — docker compose, k8s manifests
├── docker-compose.yml                full-stack local deploy
├── docker-compose.prod.yml           hardened prod variant
├── Dockerfile.api                    server container
├── Dockerfile.ui                     frontend container
└── k8s/                              optional kubernetes manifests
```

**Why separate packages:** a user who wants just the library (notebooks, scripts, programmatic use) does `pip install pycaret` and gets ~20 deps. A user who wants the full application does `pip install pycaret-server pycaret-cli` or `docker compose up` and accepts FastAPI + uvicorn + SQLAlchemy. The library stays lean; the application is opt-in.

**Monorepo vs. multi-repo:** single monorepo at `github.com/pycaret/pycaret` with the four top-level packages as siblings. CI builds them independently, publishes to PyPI / npm separately.

---

## 5. Phased execution plan

### Phase 6 — Engine cleanup completion (prerequisite)

**Must finish before platform work starts.** See `ROADMAP.md` Phase 5 (god-class draining) and Phase 3 (Plotly plot rewrite).

Exit criteria:
- God-class drained to ≤1K LOC residual.
- All plots are Plotly-native.
- Core deps down from 19 to ≤15.
- 4.0.0alpha0 released on PyPI.

### Phase 7 — CLI utility

Scope: `pycaret-cli` package. Single binary entry point.

Features:
- `pycaret serve [--port 8000] [--ui-port 3000]` — spins up backend + optional frontend dev server.
- `pycaret run <experiment.yaml>` — YAML-driven scripted runs; no UI; writes results to a folder.
- `pycaret admin create-user <email>` / `pycaret admin list-workspaces`.
- `pycaret export <project-id> --out path/` — dump project + experiments to git-committable YAML.
- `pycaret import path/` — round-trip the export back in.

Stack: Typer (built on Click), Rich for terminal UX.

Exit: `pycaret serve` successfully starts a minimal API + UI locally; `pycaret run` trains a classifier from a YAML.

### Phase 8 — Database layer (`pycaret-server/db`)

Scope: SQLAlchemy models + Alembic migrations.

- Default driver: SQLite (`sqlite:///./pycaret.db`) — zero setup.
- Optional: Postgres, MySQL — switch via `DATABASE_URL` env var.
- All models inherit a `Base` with `id` (UUID), `created_at`, `updated_at`, `created_by`.
- Alembic migrations checked into `pycaret-server/db/migrations/`.

**Tables (v1, 14):**

| Table | Purpose |
|---|---|
| `users` | Local user store (email + bcrypt hash). |
| `workspaces` | Top-level container. |
| `workspace_members` | User × Workspace × role (`admin` / `member`). |
| `projects` | Project inside a workspace. |
| `data_sources` | CSV upload / S3 / Postgres connection (§ decision 2). |
| `experiments` | Configured `Experiment` (task, target, setup params). |
| `runs` | One invocation of an experiment; captures status + timings. |
| `events` | Append-only engine event stream per run. |
| `artifacts` | Run outputs: pickle, notebook, plots (§ decision 1). |
| `fold_metrics` | Per-fold × per-model × per-metric values (§ decision 6). |
| `pipelines` | Workspace-scoped fitted sklearn Pipeline registry (§ decision 3). |
| `pipeline_project_links` | Many-to-many: a Pipeline can be used by multiple Projects. |
| `deployments` | In-house serving record (§ decision 4). |
| `api_keys` | Per-user / per-workspace programmatic access tokens (SaaS standard). |
| `sessions` | Active login sessions (for refresh-token rotation and forced logout). |

Exit: `alembic upgrade head` on a fresh SQLite file produces a valid schema; smoke-insert + query on every table.

### Phase 9 — Backend API (`pycaret-server`)

Scope: FastAPI app with typed endpoints.

- Auth: local user store, bcrypt password, JWT access + refresh tokens. OAuth (Google / GitHub) as a pluggable layer in a follow-up.
- Endpoints (v1):
  - `POST /api/v1/setup/bootstrap` — first-run admin creation.
  - `POST /api/v1/auth/login` / `POST /api/v1/auth/refresh` / `POST /api/v1/auth/logout`.
  - `CRUD /api/v1/workspaces` (admin).
  - `CRUD /api/v1/workspaces/{id}/projects`.
  - `CRUD /api/v1/projects/{id}/experiments`.
  - `POST /api/v1/experiments/{id}/runs` — enqueue a run; returns `run_id`.
  - `GET /api/v1/runs/{id}` — status + leaderboard.
  - `GET /api/v1/describe/setup-params?task=classification` — proxies `pycaret.api.describe_setup_params`.
  - `GET /api/v1/describe/models?task=classification` — proxies `pycaret.api.list_models`.
  - `GET /api/v1/describe/metrics?task=classification`.
- WebSocket: `GET /ws/runs/{run_id}/events` — subscribe to the engine's event stream; server fans out via `pycaret.logging.BaseLogger.subscribe(...)`.
- Background jobs: simple threading-based worker for v1 (no Redis). Queued runs execute serially per workspace. V2 considers RQ or Celery.

OpenAPI spec auto-generated (FastAPI native). Served at `/docs` (Swagger UI) and `/openapi.json`.

Exit:
- Full CRUD on every table, exercised by `pytest tests/`.
- End-to-end: POST run → stream events via WS → GET run returns leaderboard.

### Phase 10 — Frontend (`pycaret-ui`)

Stack:
- **Vite + React 18** + TypeScript.
- **Tailwind CSS** for styling (utility-first, modern, minimalistic).
- **TanStack Query** for server state.
- **Zustand** for UI state.
- **Plotly.js** for charts (engine emits Plotly JSON; UI just renders).
- **Orval** or similar to auto-generate the typed API client from `/openapi.json`.

Screens (v1):
- `/setup` — first-run wizard.
- `/login`.
- `/` — workspace home (shows member's workspaces).
- `/workspaces/:id` — project list.
- `/projects/:id` — experiment list + "New experiment" button.
- `/experiments/:id` — setup form (rendered dynamically from `describe_setup_params`).
- `/experiments/:id/runs/:runId` — live event stream + leaderboard + artifacts.
- `/admin/users` — admin-only user management.
- `/admin/workspace` — workspace settings.

Design principles:
- **Minimalistic**: no chrome, no marketing, no noise. Single-column forms, generous whitespace, keyboard-first.
- **Dark-mode first**, light-mode opt-in.
- **Responsive**, but desktop-first (this is an analyst tool).
- **No icons without labels**. No mystery meat.

Exit:
- All 8 screens functional against a live `pycaret-server`.
- Event stream renders in real time during a run.
- Setup form is 100% driven by `describe_setup_params` — zero UI code knows what a "normalize" parameter is.

### Phase 11 — In-house serving + Docker/deploy

Two deliverables in this phase — the **serving subsystem** and the **Docker deployment story**.

**Serving subsystem (§ decision 4):**
- New module `pycaret-server/engine/serving.py` implementing `DeploymentRegistry` (in-memory map of `slug → loaded pipeline`).
- New API routes:
  - `POST /api/v1/pipelines/{pipeline_id}/deploy` — create deployment, register route.
  - `POST /api/v1/deployments/{slug}/predict` — single catch-all inference endpoint.
  - `POST /api/v1/deployments/{slug}/pause` / `archive`.
  - `GET  /api/v1/deployments` — list for a workspace, with rolled-up inference counts.
- On FastAPI startup: `DeploymentRegistry.bootstrap()` loads every `status='active'` deployment's pipeline from disk.
- Per-deployment auth: `workspace` (JWT) / `api-key` (header `X-PyCaret-Key`) / `public` (rate-limited).
- Per-deployment metrics table: inference count, last-used timestamp, p50/p95 latency rollups.
- Stretch (v1.1): per-request payload logging to a drift-monitoring store (default: same DB; S3-backed option).

**Docker / deploy:**
- `Dockerfile.api` — multi-stage (python:3.13-slim + uv).
- `Dockerfile.ui` — Node build + nginx serve of the dist.
- `docker-compose.yml` — full stack: `api`, `ui`, `db` (Postgres optional for prod; SQLite volume-mounted for default).
- `docker-compose.prod.yml` — traefik or caddy reverse proxy, TLS termination, healthchecks, restart policies.
- K8s manifests (`deploy/k8s/`) as a stretch goal.

**Exit criteria:**
- `docker compose up` from a fresh clone produces a running app at http://localhost:3000 with a valid first-run setup page, no additional config.
- A Pipeline can be deployed from the UI and a `curl POST /api/v1/deployments/<slug>/predict` returns predictions.
- Deployment survives backend restart (registry rehydrates from DB).

### Phase 12 — Docs + release

Scope:
- Per-package READMEs.
- Architecture doc for the platform (adapted from this file).
- First-run guide (clone → compose up → create first experiment, in < 5 minutes).
- Deployment guide (local / docker / k8s / cloud variants).
- Video walkthrough.
- `pycaret==4.0.0` + `pycaret-server==0.1.0` + `pycaret-cli==0.1.0` + `@pycaret/ui@0.1.0` released.

---

## 6. Dependency discipline (platform-side)

To mirror the engine's "lean" ethos, the platform side also has a kill list.

**Forbidden until proven necessary:**
- **Celery / RabbitMQ / Redis** for job queueing — v1 uses threading; swap later only if load demands it.
- **Multiple auth providers** bundled — local-user + JWT in v1; plug-in interface for OAuth; no Auth0/Keycloak/Okta in the core.
- **Kubernetes operator** — Docker Compose is the default; K8s is a thin manifest layer.
- **GraphQL** — REST + OpenAPI is simpler for this surface.

**Allowed:**
- FastAPI + uvicorn + Starlette.
- SQLAlchemy 2.x + Alembic.
- Pydantic 2 (FastAPI already pulls it; reuse for DTO shapes).
- React 18+ + Vite + Tailwind + TanStack Query + Zustand.
- Plotly.js.
- bcrypt + pyjwt.
- `nbconvert` (render the generated notebooks to HTML for in-app preview — § decision 1).
- `boto3` (S3 data-source connector — § decision 2; gated behind an `s3` extra).
- `psycopg[binary]` or `asyncpg` (Postgres data-source connector — § decision 2; gated behind a `postgres` extra).
- `python-multipart` (CSV upload handling).
- `joblib` (deployment pipeline loading — already in the engine; re-used in the server).

---

## 7. Resolved design decisions

Project owner has answered the six parked questions. Each answer is now binding and has been propagated into §3 Data model, §5 Phased plan, and §6 Dep discipline above.

### § Decision 1 — Notebook artifacts: do what a modern SaaS would do

Every Run persists a first-class artifact bundle:

- `run.ipynb` — the executed notebook (programmatically generated from `pycaret.api.describe_setup_params` config + the engine's event stream). Stored in object storage (local disk v1; S3 when deployed).
- `fitted_pipeline.pkl` — the sklearn Pipeline joblib-pickled.
- `leaderboard.json` — serialized `CompareResult.leaderboard`.
- `events.jsonl` — the full `MemoryLogger` event stream.
- `preview.html` — pre-rendered HTML of the notebook (via nbconvert) for fast UI preview.

Modern-SaaS expectations also covered: versioned (each Run is immutable), shareable via signed URL, downloadable, previewable in-app without download.

### § Decision 2 — Data-source connectors: build a small set locally first

v1 ships three connectors; AWS-first since the owner will deploy to AWS for initial testing:

| Connector | Purpose | v1 scope |
|---|---|---|
| `csv-upload` | Direct file upload through the UI | Full support |
| `s3` | Read CSV / Parquet from an S3 bucket | Read-only v1; list + sample + load |
| `postgres` | Read a table / view from Postgres | Read-only v1; list tables + load |

Plugin interface (`DataSourceConnector` ABC) is in place from v1 so community/maintainer can add Snowflake / Google Sheets / MySQL later without touching core.

### § Decision 3 — Pipelines are workspace-scoped and shareable across projects

`Pipeline` is promoted out of `Project` into `Workspace` (see updated §3). Projects reference pipelines via a many-to-many link table. Model-registry-style: one fitted pipeline, many consumers.

UI affordances:
- Project experiment view shows "Use an existing Pipeline from the workspace" selector.
- Workspace has a "Pipelines" top-level screen listing every registered pipeline with a search/filter.
- Deploying a Pipeline is a workspace-level action; scoping discussion deferred to v2.

### § Decision 4 — In-house serving, not MLServer / BentoML

Own the serving layer. Design for v1 (single-process, self-hosted):

**Storage model.** Each deployed Pipeline has a `deployments` row with an `endpoint_slug` (url-safe id) and `status`. On backend startup, the FastAPI app reads all `deployments WHERE status='active'` and loads each `fitted_pipeline.pkl` from storage into memory.

**Routing.** A single catch-all route:

```python
@app.post("/api/v1/deployments/{slug}/predict")
def predict(slug: str, req: PredictRequest, auth=Depends(resolve_deployment_auth)):
    deployment = DeploymentRegistry.get_or_404(slug)
    df = pd.DataFrame(req.records)
    preds = deployment.pipeline.predict(df)
    proba = deployment.pipeline.predict_proba(df) if hasattr(deployment.pipeline, "predict_proba") else None
    DeploymentMetrics.record(slug, n=len(df), latency_ms=...)
    return PredictResponse(predictions=preds.tolist(), probabilities=proba.tolist() if proba is not None else None)
```

**Auth modes (per deployment):**
- `workspace` — requires a workspace-member JWT. Default.
- `api-key` — requires one of the `api_keys` rows linked to the deployment. For scripts / CI.
- `public` — no auth, rate-limited. Opt-in only; UI shows a red warning.

**Operational surface:**
- "Deploy" button on a Pipeline → creates `deployments` row → `DeploymentRegistry.register(...)`.
- "Pause" / "Archive" actions change status; the registry drops the pipeline from memory.
- Per-deployment metrics: inference count, p50/p95 latency, error rate. Stored in a time-series rollup.

**What we deliberately skip in v1:**
- Per-deployment Docker isolation (add in v2 if memory/security requires it).
- Auto-scaling / load balancing (single-process + Docker Compose is fine for the target teams-under-20 use case).
- A/B model routing, shadow deployments, canary traffic (v2+).
- Model drift monitoring as a first-class product (the event stream captures request data; a post-v1 module surfaces drift dashboards).

Phase 11 becomes **Phase 11 — Docker + in-house serving** (see updated §5).

### § Decision 5 — Dual-license for the platform packages

- `pycaret` (engine library): stays **MIT**. No change.
- `pycaret-server` / `pycaret-cli` / `pycaret-ui`: **dual-licensed**.
  - **MIT** for self-hosted and internal-enterprise use. Clone, deploy, modify, ship to your team — no restrictions.
  - **Business Source License (BSL 1.1)** for any deployment that offers the platform as a multi-tenant hosted service to third parties. BSL converts to MIT/Apache-2.0 after 3 years, so it's a commercial-use gate, not a freeze.
- A `CONTRIBUTING.md`-level Contributor License Agreement (CLA) is required on PRs so the project owner can relicense if needed.

Concrete effect: if Moez (or an acquirer) wants to run a hosted SaaS on top someday, the license permits it while still letting everyone else self-host freely.

### § Decision 6 — Metrics: store both summary AND every per-fold value

Two tables, as already shown in §3:
- `runs.metrics_summary` — leaderboard-shaped aggregates (one row per model per run, `mean_*` / `std_*` columns).
- `fold_metrics` — per-fold × per-model × per-metric. One row per `(run_id, model_id, fold_idx, metric_name) → value`.

Rationale: the summary drives the leaderboard; the fold table unlocks variance-across-folds plots, time-to-train analysis, stability checks, and any future "is this model actually better than the runner-up within CV noise?" screens. Storage cost is trivial relative to the fitted-pipeline pickles.

---

---

## 8. Licensing posture (§ decision 5)

| Component | License |
|---|---|
| `pycaret` (engine) | **MIT** — unchanged from 3.x. |
| `pycaret-server` | **Dual: MIT for self-hosted + internal-enterprise; BSL 1.1 for hosted multi-tenant SaaS.** BSL auto-converts to MIT after 3 years. |
| `pycaret-cli` | Same dual-license as `pycaret-server`. |
| `pycaret-ui` | Same dual-license. |

A Contributor License Agreement (CLA) is added to `CONTRIBUTING.md` so the project owner retains the right to relicense a future hosted variant commercially. Self-hosters, internal deployments, and anyone cloning the repo to run it themselves are covered by the MIT side of the dual license and are not affected.

This mirrors the posture of Sentry / Cal.com / Plausible / Supabase — credible OSS core + commercial freedom for a future hosted layer.

---

## 9. Success criteria for the platform

The platform is "done enough to talk about" when:

1. A new user can go from `git clone` to "I just trained my first classifier in the UI" in **under 10 minutes** on a fresh laptop.
2. A team of 3 can share a workspace, each person's experiments are visible to the others.
3. The full stack deploys to a single VM via `docker compose up -d` with a single env file.
4. The engine is completely replaceable — i.e., the platform doesn't reach *around* `pycaret.tasks.*Experiment` to do anything the library can't also do from a notebook.
5. The platform is a believable alternative to a $100K/year DataRobot license for teams under ~20 people.

---

*End of plan. Execution is gated on the engine cleanup being truly done (all 32+ tests green, deps ≤15, god-class drained). Tracking in `ROADMAP.md`.*
