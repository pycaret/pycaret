# Contributing to PyCaret 4.0

Thanks for considering a contribution! PyCaret 4.0 is under active architectural revamp. This file is the short version of the contributor guide; deeper references are linked below.

> **⚠ If you are an AI coding agent, read [`AGENTS.md`](AGENTS.md) first.** It is your 60-second briefing and lists the non-negotiables.

## The 30-second overview

- **PyCaret 4.0 is OOP-only.** The 3.x module-level functional API (`setup(...)`, `compare_models(...)`) is gone.
- **One sklearn-compatible `Experiment` subclass per task.** `ClassificationExperiment`, `RegressionExperiment`, etc.
- **The repo is in the middle of a multi-session revamp.** See [`docs/revamp/STATUS.md`](docs/revamp/STATUS.md) for what's landed and what's still in play.
- **The notebook golden path must always work.** `fit → compare_models → tune_model → predict_model → save_model`.

## How to get set up

See [`docs/for_developers/SETUP.md`](docs/for_developers/SETUP.md). Zero-to-first-green-test in < 5 minutes.

```bash
git clone https://github.com/pycaret/pycaret.git
cd pycaret
uv python install 3.13
uv sync --all-extras
uv run pytest tests/test_core_architecture.py tests/test_datasets.py -q   # ~5s green
```

## How to run tests

See [`docs/for_developers/TESTING.md`](docs/for_developers/TESTING.md).

## How to contribute a change

### 1. Open an issue first (for non-trivial work)

For anything beyond a typo or a one-line bugfix, open an issue describing what you want to do. This avoids duplicate effort and lets us flag whether your idea conflicts with the ongoing revamp.

### 2. Check the current state

Before you start writing code:

- [`docs/revamp/ROADMAP.md`](docs/revamp/ROADMAP.md) — are we already working on this in a current phase?
- [`docs/revamp/KILL_LIST.md`](docs/revamp/KILL_LIST.md) — is what you want to add already deliberately removed?
- [`docs/revamp/DECISIONS.md`](docs/revamp/DECISIONS.md) — has this design call already been litigated?

### 3. Follow the architecture

- **No new module-level public functions.** The functional API is dead. All user-facing operations are methods on an `Experiment` subclass.
- **No new module-level mutable state.** No globals, no ContextVars.
- **Every verb returns a typed result dataclass.** See [`docs/for_agents/TYPED_RESULTS.md`](docs/for_agents/TYPED_RESULTS.md).
- **Every long-running operation emits structured events.** See [`docs/for_agents/EVENT_STREAM.md`](docs/for_agents/EVENT_STREAM.md).

### 4. Code style

See [`docs/for_developers/CODING_STYLE.md`](docs/for_developers/CODING_STYLE.md). Enforced by `ruff`. Run `uv run ruff check pycaret/ tests/ --fix` before committing.

### 5. Tests

Every non-trivial PR needs test coverage:

- **Bug fix:** add a regression test that fails without the fix.
- **New feature:** add a unit test in `tests/test_core_architecture.py` (for primitive shapes) and/or an e2e test in `tests/test_e2e_oop.py` (for verb behaviour).

### 6. Release-notes entry

Append to [`docs/revamp/release_notes_pycaret4.md`](docs/revamp/release_notes_pycaret4.md) under the current session block. One bullet per change, tagged with the appropriate category (`BREAKING`, `REMOVED`, `ADDED`, `CHANGED`, `FIXED`, `DEPRECATED`, `SECURITY`, `DOCS`, `BUILD`, `TESTS`, `DEPS`, `INTERNAL`). The user-facing `CHANGELOG.md` is generated from this file at release time.

### 7. PR checklist

- [ ] Tests added / updated
- [ ] `uv run ruff check pycaret/ tests/` passes
- [ ] `uv run pytest tests/test_core_architecture.py tests/test_datasets.py -q` passes locally
- [ ] Release-notes entry appended
- [ ] For user-visible changes: README / notebook / doc updated if relevant
- [ ] For new deps: ADR added in `DECISIONS.md`

### 8. What makes a good PR

- **Small and focused.** One concern per PR.
- **Linked to an issue.** "Fixes #123".
- **Passes CI.** The merge queue won't take red PRs.
- **Explains the why.** A paragraph in the PR description. The diff already shows the what.

## Where to go deeper

- [`AGENTS.md`](AGENTS.md) — briefing for AI agents; also useful for humans
- [`docs/revamp/ARCHITECTURE.md`](docs/revamp/ARCHITECTURE.md) — the 4.0 design
- [`docs/for_developers/`](docs/for_developers/) — dev onboarding, testing, release process, god-class-draining playbook
- [`docs/for_agents/`](docs/for_agents/) — deep dives for tooling integrators (engine walkthrough, typed results, event stream, introspection API, verb × task cheatsheet)

## Code of conduct

See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the MIT license. See [`LICENSE`](LICENSE).
