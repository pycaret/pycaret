# PyCaret 4.0 Revamp — Status

*Updated: 2026-04-24, end of session 17*

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
