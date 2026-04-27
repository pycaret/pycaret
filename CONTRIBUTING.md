# Contributing to PyCaret

Thanks for considering a contribution.

> **⚠ If you are an AI coding agent, read [`AGENTS.md`](AGENTS.md) first** (cross-vendor briefing) **and [`CLAUDE.md`](CLAUDE.md) if you are Claude Code** (slash commands, sub-agents, permission allowlist).

## The Claude-Code-first contributor flow

This project is **Claude-Code-first**. Instead of a CI bot fixing
issues automatically, contributions look like this:

1. **You open an issue** describing the bug or feature.
2. **The maintainer ([@moezali1](https://github.com/moezali1))** reviews and either closes it or adds the **`Approved`** label.
3. **You** (or anyone) clone the repo, fire up Claude Code in your own checkout (using **your own** Claude subscription / API key), and run:
   ```bash
   claude
   > /work-on-approved-issue
   ```
   The slash command lists Approved issues, you pick one, and the agent fixes it end-to-end — branch, failing test, fix, lint, PR — locally, on your laptop, with your credits.
4. The agent opens a PR against `main`. Maintainer reviews and merges.

There is **no `ANTHROPIC_API_KEY` in this repo's secrets**. There is **no GitHub Action that runs Claude on issues automatically**. Compute is community-funded by whoever runs the agent. Contributors who don't use Claude Code can still contribute the traditional way (clone, edit, PR) — both flows are first-class.

The Claude Code config lives in [`CLAUDE.md`](CLAUDE.md) (entry point), [`.claude/`](.claude/) (commands + sub-agents + settings), and the per-directory `CLAUDE.md` files under `packages/engine/`, `apps/web/`, and `apps/site/`.

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

- Engine (`packages/engine/`) and the public site (`apps/site/`) are **FSL-1.1-MIT** (Functional Source License, MIT Future Variant). Free for any non-Competing use; auto-converts to MIT two years after each release. See [`LICENSE`](LICENSE) for the full text.
- Platform packages (`services/api/`, `apps/web/`) are **dual-licensed FSL-1.1-MIT OR BUSL-1.1**. See [`docs/revamp/DECISIONS.md`](docs/revamp/DECISIONS.md) for rationale.
- Contributions imply you grant the project the right to distribute your contribution under these licenses, including the future-MIT grant.

