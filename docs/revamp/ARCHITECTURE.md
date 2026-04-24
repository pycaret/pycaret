# PyCaret — System Architecture

*Last revised: session 13 (2026-04-24). Covers the full Control Plane (engine + backend + UI + infra). For the engine-internal architecture (class hierarchy, event system, legacy god-class drain plan) see [`ARCHITECTURE_ENGINE.md`](ARCHITECTURE_ENGINE.md).*

---

## 1. Monorepo layout

One git repo, uv workspace + npm workspace, four top-level homes:

```
pycaret/                              repo root
├── pyproject.toml                    workspace manifest only (no package)
├── uv.lock                           Python lockfile
├── README.md  AGENTS.md  CONTRIBUTING.md
│
├── packages/                         SHIPPABLE LIBRARIES (pip / npm install)
│   ├── engine/                       → published as `pycaret` on PyPI
│   │   ├── pyproject.toml            hatchling; 4.0.0a1
│   │   ├── pycaret/                  the importable package
│   │   └── tests/                    32 engine tests
│   ├── sdk-python/                   (V2) python client → `pycaret-client` on PyPI
│   └── shared-schemas/               (V2) JSON schemas shared between Python + TS
│
├── services/                         LONG-RUNNING DEPLOYABLES
│   ├── api/                          FastAPI backend → `pycaret-server` on PyPI
│   │   ├── pyproject.toml
│   │   ├── pycaret_server/           importable package
│   │   ├── alembic.ini + migrations/
│   │   └── tests/                    30 server tests
│   ├── worker/                       (V2) background job runner
│   └── deployment-runtime/           (V2) standalone inference server
│
├── apps/                             USER-FACING APPLICATIONS
│   ├── web/                          React SPA → `@pycaret/ui` (internal package)
│   │   ├── package.json
│   │   ├── src/                      6 vitest tests
│   │   └── dist/                     (built)
│   └── desktop/                      (V2) Electron wrapper
│
├── infra/                            DEPLOYMENT & OPS
│   ├── docker/                       Dockerfile.api, Dockerfile.ui, compose
│   ├── helm/                         (V2) Kubernetes chart
│   └── terraform/                    (V2) AWS / GCP / Azure modules
│
├── docs/
│   └── revamp/                       VISION, SPEC, ROADMAP, STATUS, DECISIONS
│
└── .github/workflows/                CI: lint, pytest matrix, UI pipeline
```

Three rules:

1. **`packages/` publishes, `services/` runs, `apps/` is UI, `infra/` is ops.** Every directory at the root has exactly one reason to exist.
2. **Python package names are independent of source-tree paths.** `pip install pycaret` and `import pycaret` continue to work exactly as they did before the restructure; only the source location changed. Same for `pycaret-server`.
3. **No cross-contamination.** `packages/engine` has zero knowledge of `services/api`. `services/api` imports `pycaret` as a normal dependency. `apps/web` talks only to `services/api` over HTTP + WebSocket.

---

## 2. Service topology

```
         ┌─────────────────┐      ┌────────────────────────┐
         │   apps/web      │      │   apps/desktop (V2)    │
         │   React SPA     │      │   Electron shell       │
         └────────┬────────┘      └───────────┬────────────┘
                  │ HTTP + WS                 │ (hosts both)
                  ▼                           ▼
         ┌──────────────────────────────────────────────────┐
         │              services/api                       │
         │     FastAPI + SQLAlchemy + JWT auth             │
         │                                                  │
         │  /api/v1/workspaces  …projects  …experiments    │
         │  /api/v1/runs        …artifacts …deployments    │
         │  /api/v1/describe    …llm       …monitoring     │
         │  /api/v1/deployments/{slug}/predict  ← serving  │
         │  /ws runs/{id}/events                ← stream   │
         └────┬───────────────┬────────────┬────────────┬──┘
              │               │            │            │
              ▼               ▼            ▼            ▼
        ┌─────────┐     ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ engine  │     │    DB    │ │ artifact │ │   LLM    │
        │ in-proc │     │ Postgres │ │  store   │ │ provider │
        │ (thread │     │ / SQLite │ │ fs/S3/.. │ │ router   │
        │  pool)  │     │          │ │          │ │          │
        └─────────┘     └──────────┘ └──────────┘ └──────────┘
              ▲
              │ (V2 promotion)
              ▼
        ┌─────────────────────────┐
        │ services/worker (V2)    │
        │   Job queue consumer    │
        └─────────────────────────┘
```

Current MVP: in-process `ThreadPoolExecutor` inside the API process runs engine work. V2 moves that to a separate `services/worker` pulling `Job` rows. Same interface on both sides (the `RunOrchestrator` abstraction hides it).

---

## 3. Engine layer (`packages/engine`)

See [`ARCHITECTURE_ENGINE.md`](ARCHITECTURE_ENGINE.md) for depth. Summary:

- **`pycaret.tasks`** — 5 task subclasses (`ClassificationExperiment`, `RegressionExperiment`, `ClusteringExperiment`, `AnomalyExperiment`, `TimeSeriesExperiment`), all sklearn-composable (`BaseEstimator` subclasses).
- **`pycaret.api`** — typed introspection: `list_models`, `describe_model`, `list_metrics`, `describe_setup_params`. Drives the dynamic form in `/experiments/:id`.
- **`pycaret.logging`** — `BaseLogger` + `Event` + `EventKind`. The backend's `DBEventLogger` subclasses this to persist + broadcast.
- **`pycaret.core.results`** — typed dataclasses for every verb (`CompareResult`, `TuneResult`, `CreateResult`, ...).

**Future direction** (MVP 1 exit): `pycaret.engine.run(config: RunConfig) → RunResult` as a single stateless entry. Wraps the task subclasses. Same contract as notebook / API / UI / LLM-generated config use.

---

## 4. Backend (`services/api`)

### 4.1 Routers (current surface, ~40 endpoints under `/api/v1/`)

| Router | File | Responsibility |
|---|---|---|
| setup | `api/setup.py` | First-run bootstrap + status |
| auth | `api/auth.py` | login / refresh / logout / me |
| describe | `api/describe.py` | Engine introspection proxy |
| workspaces | `api/workspaces.py` | Workspace CRUD + members |
| projects | `api/projects.py` | Project CRUD |
| experiments | `api/experiments.py` | Experiment CRUD |
| runs | `api/runs.py` | Run submit / list / get / events / wait / cancel + WebSocket |
| data_sources | `api/data_sources.py` | CSV upload + S3/Postgres register |
| deployments | `api/deployments.py` | Pipeline promote + deployment CRUD + `/predict` |

### 4.2 Data model (14 tables, [`CONTROL_PLANE_SPEC.md § 4`](CONTROL_PLANE_SPEC.md#4-main-domain-model))

```
User ─┬─ Session (refresh tokens)
      └─ ApiKey

Workspace ─┬─ WorkspaceMember (user × role)
           ├─ DataSource       (CSV / S3 / Postgres)
           ├─ Pipeline ─────┐  (workspace-scoped model registry)
           ├─ Deployment ←──┘
           └─ Project ─┬─ Experiment ─┬─ Run ─┬─ Event   (event stream)
                       │               │       ├─ Artifact
                       │               │       └─ FoldMetric
                       └─ PipelineProjectLink  (m2m to Pipeline)
```

Additions planned for MVP 2 completion (see ROADMAP):

- **Trial** — one row per AutoML candidate inside a Run.
- **PredictionLog** — per-request log for deployed endpoints.
- **DriftReport** — periodic drift scores.
- **ModelLibrary** — admin-editable model catalogue (today: hardcoded in engine).
- **Job** — background work queue ([`CONTROL_PLANE_SPEC.md § 16`](CONTROL_PLANE_SPEC.md#16-background-jobs)).
- **LLMProviderSetting** + **LLMConsultation** — AI assistant ([§ 12](CONTROL_PLANE_SPEC.md#12-llm--ai-assistant-system)).
- **AuditLog** — every admin-relevant action.

### 4.3 Run execution

```
POST /experiments/{id}/runs
   │
   ▼
Run row (status=queued) + Run.snapshot (full reproducibility)
   │
   ▼
RunOrchestrator.submit(RunSpec)
   │  threading.Event for cancellation
   ▼
Worker thread
   ├─ _load_data()              sklearn_dataset | data_inline | data_source_path
   ├─ _build_experiment()       dispatches to pycaret.tasks.*
   ├─ exp.logger = DBEventLogger(run_id=…, event_broker)
   ├─ exp.fit(df)
   ├─ execute_plan(setup | create | compare)
   │  _checkpoint()             — cancellation poll at stage boundaries
   ├─ _save_pipeline()          cloudpickle → artifact row
   └─ transition(status=succeeded, leaderboard, duration_ms, …)
   │
   ▼
event_broker.close_run(run_id)  → WS subscribers receive {kind: "run.closed"}
```

Every engine `Event` flows through `DBEventLogger.emit()`:

1. Write an `events` row (synchronous, scoped session).
2. `event_broker.publish(run_id, event.to_dict())` — fans out to any WebSocket subscribers via `loop.call_soon_threadsafe(queue.put_nowait, event)`.

### 4.4 Deployment + serving

```
Run (succeeded + pipeline_pickle artifact)
   │  POST /runs/{id}/promote
   ▼
Pipeline row        (workspace-scoped, shareable across projects)
   │  POST /pipelines/{id}/deployments
   ▼
Deployment row      (slug + auth_mode + metrics counters)
   │
   ▼
DeploymentRegistry  (in-process LRU + p50/p95 rolling window)
   │
   ▼
POST /api/v1/deployments/{slug}/predict
```

Auth modes per deployment: `workspace` (JWT) / `api-key` (V2) / `public` (V2, rate-limited).

---

## 5. Frontend (`apps/web`)

### 5.1 Stack

Vite 5 + React 18 + TypeScript 5 (strict, `verbatimModuleSyntax`) + Tailwind 3 (dark-mode-first) + TanStack Query + Zustand + React Router 6 + axios. Production bundle: 83 kB gzipped.

### 5.2 Directory

```
apps/web/src/
├── main.tsx                  React root + QueryClient
├── App.tsx                   route table
├── index.css                 Tailwind + component primitives
├── api/
│   ├── client.ts             axios instance + single-flight 401 refresh
│   ├── endpoints.ts          one function per backend route
│   └── types.ts              hand-written mirrors of Pydantic schemas
├── state/
│   └── auth.ts               Zustand store; refresh token in localStorage
├── components/
│   ├── AuthGate.tsx          guards authenticated routes
│   └── Layout.tsx            top-nav shell for authed screens
└── pages/
    ├── Setup.tsx             /setup
    ├── Login.tsx             /login
    ├── Workspaces.tsx        /
    └── WorkspaceDetail.tsx   /workspaces/:id
```

### 5.3 Design

Per [`CONTROL_PLANE_SPEC.md § 13`](CONTROL_PLANE_SPEC.md#13-ui--navigation-specification):

- **Minimalistic.** No chrome, no noise. Single-column forms, keyboard-first.
- **Dark-mode first.** Tailwind `darkMode: 'class'` with `<html class="dark">`. Light mode opt-in (V2).
- **Desktop-first.** Analyst tool, not a mobile app.
- **No icons without labels.** No mystery meat navigation.

---

## 6. Infra (`infra/`)

Currently: `infra/docker/` runs the full local stack.

```bash
docker compose -f infra/docker/docker-compose.yml up --build
# → http://localhost:3000
```

UI container's nginx reverse-proxies `/api` + `/ws` to the API container so the browser sees a single origin (no CORS headaches). WebSocket upgrade on `/api/v1/runs/*` with 1h timeouts for long AutoML runs.

`infra/helm/` + `infra/terraform/{aws,gcp,azure}/` are V2 stubs.

---

## 7. LLM router (new in MVP 2 final stretch)

Per [decision 3 of session 13](DECISIONS.md#2026-04-24-session-13--restructure-decision-3--llm-router-not-a-single-provider):

```
services/api/pycaret_server/llm/
├── router.py               LLMRouter (provider selection + retries + usage)
├── providers/
│   ├── base.py             LLMProvider Protocol (chat_completion, tool_use)
│   ├── anthropic.py        Claude via anthropic SDK
│   ├── openai.py           GPT-4 / o-series via openai SDK
│   └── __init__.py         registry
├── consultations/
│   ├── dataset_analysis.py Per-type prompt templates + output schemas
│   ├── experiment_design.py
│   ├── run_explainer.py
│   ├── failure_debugger.py
│   ├── deployment_review.py
│   └── drift_analyst.py
└── schemas.py              Pydantic: LLMConsultation, LLMProviderSetting
```

**Crucial constraint**: LLM output is *advisory*. Every consultation returns `suggested_config_json` + `suggested_action` + `reasoning_summary` + `risk_flags`. The deterministic engine executes what the user approves. See [`CONTROL_PLANE_SPEC.md § 12.3`](CONTROL_PLANE_SPEC.md#123-important-constraint).

---

## 8. RunConfig — the single contract

One JSON schema drives four interfaces:

```
               ┌─────────────────────────────────────────┐
               │              RunConfig (JSON)           │
               │  dataset + task + preprocessing +       │
               │  model_selection + evaluation +         │
               │  automl + tuning + explainability       │
               └───────┬─────────┬───────────┬───────────┘
                       │         │           │
                notebook    API/CLI       UI wizard
                `engine.    POST          dynamic
                 run(cfg)`  /runs         form from
                                          describe_setup_params
                       │         │           │
                       └─────────┼───────────┘
                                 ▼
                       LLM-generated config
                        (reviewed + approved)
```

Schema lives in `packages/shared-schemas/` (V2 implementation); today the API accepts it as a loose dict on `Run.snapshot.setup_params`. MVP 1 exit requires migrating to a strict Pydantic `RunConfig`.

See [`CONTROL_PLANE_SPEC.md § 6`](CONTROL_PLANE_SPEC.md#6-run-configuration-system) for the full schema.

---

## 9. CI

`.github/workflows/test.yml` runs on every push to `v4`:

| Job | Matrix | Gate |
|---|---|---|
| Lint (ruff) | ubuntu-latest | blocking |
| Tests | Ubuntu + Windows × Python 3.11/3.12/3.13 | blocking |
| Web (tsc + eslint + vitest + vite build) | ubuntu-latest, Node 22 | blocking |
| Notebooks (re-execute canonical) | ubuntu-latest, nightly only | advisory |
| ci-status | aggregate gate | blocking |

Test counts after session 13: **32 engine + 30 server + 6 web = 68 total**.

---

## 10. What the architecture deliberately is *not*

- **No microservice hell.** One API process. One worker process (eventually). One UI bundle. That's it.
- **No GraphQL.** REST + OpenAPI is simpler for this surface.
- **No home-grown ORM.** SQLAlchemy 2.x.
- **No home-grown auth.** JWT + bcrypt, both standard.
- **No lock-in to one LLM provider.** Router abstraction covers Claude + OpenAI from day one.
- **No dependency on MLflow, Comet, Weights & Biases.** All mechanisms we needed from them (event stream, artifact registry) we own. Third-party trackers can be added as optional adapters in V3.
- **No implicit global state in the engine.** 3.x's `ContextVar`-based session state is gone. `Experiment` instances are independent. `RunConfig` is the only input.

---

## 11. Links

- [`VISION.md`](VISION.md) — 1-page product statement.
- [`CONTROL_PLANE_SPEC.md`](CONTROL_PLANE_SPEC.md) — full spec (24 sections).
- [`ROADMAP.md`](ROADMAP.md) — MVP 1–4 / V2 / V3 phase breakdown.
- [`STATUS.md`](STATUS.md) — where we are right now.
- [`DECISIONS.md`](DECISIONS.md) — ADR log.
- [`ARCHITECTURE_ENGINE.md`](ARCHITECTURE_ENGINE.md) — engine-internal detail (god-class, events, class hierarchy).
- [`PLATFORM_QUICKSTART.md`](PLATFORM_QUICKSTART.md) — clone-to-running in 5 minutes.
