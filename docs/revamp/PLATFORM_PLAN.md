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

## 3. Data model — workspace → project → experiment → run

Single authoritative hierarchy. Every domain object below is an SQLAlchemy model.

```
Workspace
├── members (User × role)
├── config (theme, default compute profile, data-source allowlist)
└── projects
    └── Project
        ├── metadata (name, description, tags, owner)
        ├── data_sources (registered CSVs / DB connections / S3 paths)
        └── experiments
            └── Experiment
                ├── config (task, target, setup params — serialized SetupParamSchema)
                ├── runs                      ← many-to-one with Experiment
                │   └── Run
                │       ├── started_at, finished_at, status
                │       ├── events[]          ← engine Event stream captured here
                │       ├── leaderboard        ← CompareResult.leaderboard serialized
                │       ├── artifacts[]       ← fitted pipeline .pkl paths
                │       └── metrics           ← per-model CV metrics
                └── pipelines                 ← many-to-one with Experiment
                    └── Pipeline               ← a named fitted sklearn pipeline
                        ├── run_id (origin)
                        ├── model_id (pycaret id, e.g. "lr")
                        ├── stored_path
                        └── sha256
```

**Core tables:** `users`, `workspaces`, `workspace_members`, `projects`, `data_sources`, `experiments`, `runs`, `events`, `artifacts`, `pipelines`, `sessions` (auth).

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

Tables (v1): `users`, `workspaces`, `workspace_members`, `projects`, `data_sources`, `experiments`, `runs`, `events`, `artifacts`, `pipelines`, `sessions`.

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

### Phase 11 — Docker / deploy

Scope:
- `Dockerfile.api` — multi-stage (python:3.13-slim + uv).
- `Dockerfile.ui` — Node build + nginx serve of the dist.
- `docker-compose.yml` — full stack: `api`, `ui`, `db` (Postgres optional for prod; SQLite volume-mounted for default).
- `docker-compose.prod.yml` — traefik or caddy reverse proxy, TLS termination, healthchecks, restart policies.
- K8s manifests (`deploy/k8s/`) as a stretch goal.

**Target: `docker compose up` from a fresh clone produces a running app at http://localhost:3000 with a valid first-run setup page, no additional config.**

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
- SQLAlchemy + Alembic.
- Pydantic (FastAPI already pulls it; reuse for DTO shapes).
- React + Vite + Tailwind + TanStack Query + Zustand.
- Plotly.js.
- bcrypt + pyjwt.

---

## 7. Open questions (parked for future decision)

1. **Notebook persistence** — should runs store their produced Jupyter notebook as an artifact, or is the event-stream replay enough?
2. **Data-source connectors** — v1 supports local CSV upload. v1.1: Postgres / Snowflake / S3 as plugins. v2: live-data refresh semantics.
3. **Model registry** — do we expose Pipelines as a first-class shareable object across projects? Or keep them scoped to their project?
4. **Serving** — do we add a "deploy this pipeline" button that pushes to MLServer / BentoML? Or keep serving out of scope?
5. **Hosted SaaS** — someone will eventually build this on top. Do we keep the core MIT-only, or dual-license?
6. **Metrics warehouse** — do we store every per-fold metric in the DB, or only the leaderboard summary?

---

## 8. Success criteria for the platform

The platform is "done enough to talk about" when:

1. A new user can go from `git clone` to "I just trained my first classifier in the UI" in **under 10 minutes** on a fresh laptop.
2. A team of 3 can share a workspace, each person's experiments are visible to the others.
3. The full stack deploys to a single VM via `docker compose up -d` with a single env file.
4. The engine is completely replaceable — i.e., the platform doesn't reach *around* `pycaret.tasks.*Experiment` to do anything the library can't also do from a notebook.
5. The platform is a believable alternative to a $100K/year DataRobot license for teams under ~20 people.

---

*End of plan. Execution is gated on the engine cleanup being truly done (all 32+ tests green, deps ≤15, god-class drained). Tracking in `ROADMAP.md`.*
