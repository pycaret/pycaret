# AGENTS.md — PyCaret agent instructions

> Read by AI coding agents (Claude, Cursor, Copilot, etc.) before touching the repo. Single-source briefing for any agent contributing to PyCaret. Humans should read it too.

## TL;DR — the 60-second briefing

- **We are building PyCaret — an open-source, self-hosted ML platform** (engine + backend + web UI). Product name: **PyCaret**. UI branding: **PyCaret Control Plane**. See `docs/revamp/VISION.md` for the one-pager.
- **Monorepo: `apps/`, `services/`, `packages/`, `infra/`.** Every top-level dir has one reason to exist. See `docs/revamp/ARCHITECTURE.md § 1` for the layout rules.
- **Engine (`packages/engine/`) is stateless.** Built on sklearn 1.7+. Shipped on PyPI as `pycaret`. OOP-only; the 3.x functional API is gone.
- **Backend (`services/api/`) is FastAPI + SQLAlchemy.** Shipped on PyPI as `pycaret-server`. Hosts the Control Plane.
- **Web (`apps/web/`) is Vite + React 18 + TypeScript.** Dark-mode-first, single-column forms, no mystery meat.
- **The single contract is `RunConfig`.** A strict JSON schema that drives notebook / API / UI / LLM-generated runs. Spec: `docs/revamp/CONTROL_PLANE_SPEC.md § 6`.
- **Every non-trivial change gets logged** in `docs/revamp/release_notes_pycaret4.md` under the current session block (tags: `BREAKING`, `REMOVED`, `ADDED`, `CHANGED`, `FIXED`, `DOCS`, `BUILD`, `TESTS`, `DEPS`, `INTERNAL`).

## Start here

Read these in order before writing code:

1. **`docs/revamp/VISION.md`** — 1-page product statement.
2. **`docs/revamp/CONTROL_PLANE_SPEC.md`** — full technical spec (24 sections). The canonical scope.
3. **`docs/revamp/ARCHITECTURE.md`** — live system architecture. Maps spec onto current code.
4. **`docs/revamp/ROADMAP.md`** — MVP 1–4 / V2 / V3 phase breakdown. Find the phase you're contributing to.
5. **`docs/revamp/STATUS.md`** — what's landed and what's in play. Newest session first.
6. **`docs/revamp/DECISIONS.md`** — ADRs. If an option "feels wrong," check here; it's probably already been litigated.
7. **`docs/revamp/KILL_LIST.md`** — everything deliberately removed from the engine. Never reintroduce any of it.
8. **`docs/revamp/release_notes_pycaret4.md`** — engineering change log. You'll append to this.

## Non-negotiables

### Universal rules

1. **Engine is stateless.** `result = engine.run(config)`, not `setup() + compare_models()`. No module-level `_CURRENT_EXPERIMENT`. No `ContextVar` implicit-state.
2. **Config is the contract.** The same `RunConfig` JSON must work from a notebook, the REST API, the UI wizard, and an LLM-generated payload. Don't invent a parallel shape for any single surface.
3. **Artifacts are immutable.** Every promotion / retrain creates a new `pipeline_pickle`. Never mutate a completed artifact.
4. **Deployments are versioned.** Every `Deployment` row points at one specific `Pipeline` row. No "moving target" endpoints.
5. **LLM is advisory.** LLM calls return `suggested_config_json` + `reasoning_summary` + `risk_flags`. The user approves. The deterministic engine executes. **Never let the LLM directly trigger a destructive action.** (See CONTROL_PLANE_SPEC § 12.3.)
6. **Every public verb returns a typed result dataclass** — `CompareResult`, `TuneResult`, `PredictResult`. Never a bare DataFrame.
7. **Every long-running operation emits a structured event** through `self.logger.log(EventKind.X, ...)`. No `print()` inside the engine.
8. **No upper-bound version pins** on NumPy, pandas, scipy, sklearn, joblib. The whole point of 4.0 was removing those.
9. **No reintroducing kill-listed dependencies.** See `docs/revamp/KILL_LIST.md`.

### Tooling conventions

- **Python target:** 3.13 primary; 3.11 floor.
- **Node target:** 22 primary; 20 floor.
- **Python env:** `uv` for env + lockfile, `hatchling` build backend, `ruff` for lint + format, `pytest` for tests, Alembic for migrations.
- **Node env:** `npm` (workspace) with `package-lock.json` checked in, Vite for dev/build, Vitest for tests, ESLint flat config, TypeScript 5.6+ with `verbatimModuleSyntax`.
- **Imports (Python):** absolute only inside `pycaret/` and `pycaret_server/`. No star imports. Lazy-import heavy optional deps inside the function that needs them.
- **Imports (TS):** use `@/` alias to `src/`. Prefer named exports. Use `import type` for types (enforced by `verbatimModuleSyntax`).
- **Type hints:** everywhere on new Python code. `from __future__ import annotations` at the top of every module. TS strict mode is on.
- **Docstrings:** numpydoc style, as short as truthful. Describe *why*, not *what*.

## Repo map

```
pycaret/                              repo root
├── pyproject.toml                    workspace manifest only (no package)
├── uv.lock
├── AGENTS.md  CONTRIBUTING.md  README.md  LICENSE
│
├── packages/                         SHIPPABLE LIBRARIES
│   ├── engine/                       → `pycaret` on PyPI (4.0.0a1)
│   │   ├── pyproject.toml            hatchling build config
│   │   ├── pycaret/                  the importable package
│   │   │   ├── api/                  typed introspection (for UI + agents)
│   │   │   ├── core/                 Experiment, results, errors, tasks
│   │   │   ├── tasks/                5 task subclasses (public API)
│   │   │   ├── logging/              event-stream logger
│   │   │   ├── containers/           model-registry containers (being drained)
│   │   │   └── internal/             LEGACY god-class (drain in Phase 5)
│   │   └── tests/                    32 engine tests
│   ├── sdk-python/                   (V2) Python client (README stub)
│   └── shared-schemas/               (V2) JSON schemas shared Python ↔ TS
│
├── services/                         LONG-RUNNING DEPLOYABLES
│   ├── api/                          → `pycaret-server` on PyPI (0.1.0a0)
│   │   ├── pyproject.toml
│   │   ├── alembic.ini
│   │   ├── pycaret_server/
│   │   │   ├── api/                  HTTP routers (setup, auth, describe,
│   │   │   │                           workspaces, projects, experiments,
│   │   │   │                           runs, data_sources, deployments)
│   │   │   ├── auth/                 bcrypt + JWT helpers
│   │   │   ├── db/                   SQLAlchemy models + session + bootstrap
│   │   │   ├── migrations/           Alembic env + versions
│   │   │   ├── runs/                 RunOrchestrator + broker + logger_bridge
│   │   │   ├── serving.py            DeploymentRegistry (in-proc inference)
│   │   │   ├── config.py             pydantic-settings
│   │   │   ├── app.py                FastAPI factory
│   │   │   └── cli.py                `pycaret-server serve | migrate`
│   │   └── tests/                    30 server tests
│   ├── worker/                       (V2) background job runner (README stub)
│   └── deployment-runtime/           (V2) standalone serving (README stub)
│
├── apps/                             USER-FACING APPLICATIONS
│   ├── web/                          → `@pycaret/ui` (internal)
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── api/                  typed client (client + endpoints + types)
│   │   │   ├── state/                Zustand stores (auth)
│   │   │   ├── components/           AuthGate, Layout
│   │   │   └── pages/                Setup, Login, Workspaces, WorkspaceDetail
│   │   └── (6 vitest tests)
│   └── desktop/                      (V2) Electron wrapper (README stub)
│
├── infra/                            OPS & DEPLOYMENT
│   ├── docker/                       Dockerfile.api, Dockerfile.ui, compose, nginx
│   ├── helm/                         (V2) Kubernetes chart (README stub)
│   └── terraform/                    (V2) AWS / GCP / Azure modules (stubs)
│
├── docs/revamp/                      VISION + SPEC + ROADMAP + STATUS + DECISIONS
│                                     + release_notes + PLATFORM_QUICKSTART
│                                     + ARCHITECTURE + ARCHITECTURE_ENGINE
│                                     + AUDIT + KILL_LIST
├── notebooks/                        5 working end-to-end notebooks (01–05)
├── scripts/                          maintenance scripts
└── .github/workflows/                CI: lint + test matrix + web + notebooks
```

## Which phase am I in?

Quick decision tree:

- **Are you changing Python code inside `packages/engine/pycaret/`?** You're working on the engine (MVP 1). Follow `docs/for_developers/DRAINING_THE_GODCLASS.md` if you're migrating a verb off `_legacy`.
- **Are you adding a route / table / service under `services/api/`?** MVP 2. Add the SQLAlchemy model, write an Alembic migration (autogenerate works well here), add the router, write the integration test.
- **Are you adding a screen / component to `apps/web/`?** MVP 3. Match the existing dark-mode palette + component primitives. 100% TypeScript strict. Tests in `vitest`.
- **Are you editing Docker / Helm / Terraform?** MVP 4 (docker) or V2 (helm / terraform). Stay within `infra/`.
- **Are you wiring LLM functionality?** Uses the `services/api/pycaret_server/llm/` router (Claude + OpenAI). Every call returns an advisory `LLMConsultation` row; the user approves before execution.

## Workflow

1. **Plan.** For any non-trivial change, sketch what you'll edit + why in the response to the user before editing.
2. **Write small, cohesive diffs.** One concern per commit.
3. **Run the relevant test subset locally.** For the engine: `uv run pytest packages/engine/tests/ -q`. For the API: `uv run --package pycaret-server pytest services/api/tests/ -q`. For the web: `cd apps/web && npm run typecheck && npm run lint && npm test && npm run build`.
4. **Append a release-notes entry** in `docs/revamp/release_notes_pycaret4.md` under the current session block.
5. **Update `docs/revamp/STATUS.md`** if you finished a roadmap item.
6. **Update `docs/revamp/ROADMAP.md`** if you closed a phase or added scope.
7. **Record non-obvious design choices** in `docs/revamp/DECISIONS.md` as a new ADR entry (newest first).

## Common tasks

### Add a new backend route

1. Define the SQLAlchemy model(s) in `services/api/pycaret_server/db/models.py` if needed.
2. Generate the migration: `cd services/api && uv run alembic revision --autogenerate -m "<slug>"`. Review + format the generated file.
3. Add the Pydantic schemas in `services/api/pycaret_server/api/schemas.py`.
4. Create / extend the router in `services/api/pycaret_server/api/<module>.py`.
5. Mount it in `services/api/pycaret_server/app.py`.
6. Write the integration test in `services/api/tests/test_<module>.py` using the TestClient fixture pattern.
7. Run the server suite: `uv run --package pycaret-server pytest services/api/tests/ -q`.

### Add a new frontend screen

1. Add the typed endpoint(s) to `apps/web/src/api/endpoints.ts` and the response types to `apps/web/src/api/types.ts`.
2. Create the page under `apps/web/src/pages/<Name>.tsx`.
3. Route it in `apps/web/src/App.tsx` (inside the `<Layout>` for authed, outside for public).
4. Write at least one Vitest component test.
5. Check everything: `cd apps/web && npm run typecheck && npm run lint && npm test && npm run build`.

### Drain a god-class verb (engine, Phase 5)

1. Current state: the verb calls `self._legacy.<verb>(*args, **kwargs)` and wraps the return in a typed dataclass.
2. Reimplement natively using `sklearn.pipeline.Pipeline`, `sklearn.model_selection`, etc.
3. Keep the signature + return type identical.
4. Emit the same structured events.
5. Add a test in `packages/engine/tests/test_e2e_oop.py`.
6. Release-notes entry under `CHANGED` + `INTERNAL`.

### Add an LLM advisory feature

1. Add a new file under `services/api/pycaret_server/llm/consultations/<type>.py` with the prompt template + output schema.
2. Route through the existing `LLMRouter` — don't import `anthropic` or `openai` directly outside `services/api/pycaret_server/llm/providers/`.
3. Persist results as an `LLMConsultation` row.
4. Output must include `suggested_config_json`, `suggested_action`, `reasoning_summary`, `risk_flags`. The user sees all four before anything runs.
5. Never let the LLM cause a side effect directly.

## Deep dives

- `docs/revamp/ARCHITECTURE_ENGINE.md` — engine-internal architecture (god-class, class hierarchy, event system).
- `docs/for_agents/ENGINE_WALKTHROUGH.md` — what happens at every step of `fit` → `compare_models` → `predict_model`.
- `docs/for_agents/TYPED_RESULTS.md` — every result dataclass, its fields, when it's produced.
- `docs/for_agents/EVENT_STREAM.md` — the canonical `EventKind`s, what they carry, how to subscribe.
- `docs/for_agents/INTROSPECTION_API.md` — `list_models` / `describe_model` / `describe_setup_params` contract.
- `docs/for_developers/SETUP.md` — dev environment, linting, test matrix.
- `docs/for_developers/TESTING.md` — how to run / add tests.
- `docs/for_developers/DRAINING_THE_GODCLASS.md` — the playbook for migrating a verb off `_legacy`.
