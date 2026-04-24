# Contributing to PyCaret

Thanks for considering a contribution.

> **⚠ If you are an AI coding agent, read [`AGENTS.md`](AGENTS.md) first.** It is your 60-second briefing and lists the non-negotiables.

## The 30-second overview

- **PyCaret is an open-source ML platform.** Engine (`packages/engine`) + backend (`services/api`) + web UI (`apps/web`). See [`docs/revamp/VISION.md`](docs/revamp/VISION.md).
- **The engine is OOP-only, config-driven, stateless.** The 3.x functional API (`setup(...)`, `compare_models(...)`) is gone. One `Experiment` subclass per task.
- **The repo is a monorepo:** `apps/` (UI), `services/` (API + workers), `packages/` (libraries), `infra/` (deployment). See [`docs/revamp/ARCHITECTURE.md`](docs/revamp/ARCHITECTURE.md) for the canonical layout.
- **The contract is `RunConfig`.** Same JSON drives notebook / API / UI / LLM-generated runs.

## How to get set up

Zero-to-first-green-test in under 5 minutes:

```bash
git clone https://github.com/pycaret/pycaret.git
cd pycaret

# Python side — engine + backend share a uv workspace
uv python install 3.13
uv sync --all-packages --all-extras

# engine tests (32, ~90s)
uv run pytest packages/engine/tests/ -q

# backend tests (30, ~30s)
uv run --package pycaret-server pytest services/api/tests/ -q

# frontend
cd apps/web
npm install
npm run typecheck && npm run lint && npm test && npm run build
```

See [`docs/revamp/PLATFORM_QUICKSTART.md`](docs/revamp/PLATFORM_QUICKSTART.md) for the full-stack local-dev flow.

## How to contribute a change

### 1. Open an issue first (for non-trivial work)

For anything beyond a typo or a one-line bugfix, open an issue describing what you want to do. The roadmap is active; we might already be working on it or have ruled it out.

### 2. Check the current state

Before you start writing code:

- [`docs/revamp/VISION.md`](docs/revamp/VISION.md) — is your idea aligned with the product direction?
- [`docs/revamp/ROADMAP.md`](docs/revamp/ROADMAP.md) — is this part of a current MVP / V2 / V3 phase?
- [`docs/revamp/DECISIONS.md`](docs/revamp/DECISIONS.md) — has this design call already been litigated?
- [`docs/revamp/KILL_LIST.md`](docs/revamp/KILL_LIST.md) — is what you want to add already deliberately removed from the engine?

### 3. Follow the architecture

Universal rules (from [`AGENTS.md`](AGENTS.md)):

- **Engine is stateless.** `engine.run(config)`, not `setup() + compare_models()`.
- **Config is the contract.** Don't invent a parallel shape for one surface.
- **Artifacts are immutable. Deployments are versioned.**
- **LLM is advisory.** LLM proposes; user approves; deterministic engine executes.
- **Every verb returns a typed result.** No bare DataFrames.
- **Every long-running operation emits a structured event.** No `print()`.

### 4. Code style

- **Python:** `ruff` enforces formatting + import order. Run `uv run --with ruff ruff check packages/engine services/api --fix` before committing.
- **TypeScript:** ESLint flat config, `--max-warnings 0`. Run `cd apps/web && npm run lint` before committing.
- **No upper-bound version pins** on NumPy, pandas, scipy, sklearn, joblib.

### 5. Tests

Every non-trivial PR needs test coverage:

- **Engine bug fix** → regression test in `packages/engine/tests/`.
- **New backend route** → integration test in `services/api/tests/` using the TestClient fixture.
- **New frontend screen** → at least one Vitest component test in `apps/web/src/`.
- **New feature crossing all three layers** → tests at each layer.

### 6. Release-notes entry

Append to [`docs/revamp/release_notes_pycaret4.md`](docs/revamp/release_notes_pycaret4.md) under the current session block. One bullet per change, tagged: `BREAKING` / `REMOVED` / `ADDED` / `CHANGED` / `FIXED` / `DEPRECATED` / `SECURITY` / `DOCS` / `BUILD` / `TESTS` / `DEPS` / `INTERNAL`. The user-facing `CHANGELOG.md` is generated from this file at release time.

### 7. PR checklist

- [ ] Tests added / updated
- [ ] `ruff check` + `eslint` pass locally
- [ ] Relevant test suite passes locally
- [ ] Release-notes entry appended
- [ ] For user-visible changes: README / notebook / doc updated if relevant
- [ ] For new deps: ADR added in `DECISIONS.md`
- [ ] For new scope: `ROADMAP.md` / `STATUS.md` updated

### 8. What makes a good PR

- **Small and focused.** One concern per PR.
- **Linked to an issue.** "Fixes #123".
- **Passes CI.** The merge queue won't take red PRs.
- **Explains the why.** A paragraph in the PR description. The diff already shows the what.

## Where to go deeper

- [`AGENTS.md`](AGENTS.md) — agent briefing; also useful for humans.
- [`docs/revamp/VISION.md`](docs/revamp/VISION.md) — 1-page product statement.
- [`docs/revamp/CONTROL_PLANE_SPEC.md`](docs/revamp/CONTROL_PLANE_SPEC.md) — full technical spec.
- [`docs/revamp/ARCHITECTURE.md`](docs/revamp/ARCHITECTURE.md) — system architecture.
- [`docs/revamp/ROADMAP.md`](docs/revamp/ROADMAP.md) — phase breakdown.
- [`docs/for_developers/`](docs/for_developers/) — dev onboarding, testing, release process, god-class-draining playbook.
- [`docs/for_agents/`](docs/for_agents/) — deep dives for tooling integrators.

## Licensing

- Engine (`packages/engine/`) is **MIT**.
- Platform packages (`services/*`, `apps/*`) are **dual-licensed MIT OR BUSL-1.1**. See [`docs/revamp/DECISIONS.md § 2026-04-23 · decision 5`](docs/revamp/DECISIONS.md) for the rationale.
- Contributions to platform packages imply acceptance of the BSL posture (self-host freely; if you run it as a multi-tenant hosted SaaS to third parties, the BSL grant becomes relevant until it converts to MIT/Apache-2.0 after 3 years).

## Code of conduct

See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
