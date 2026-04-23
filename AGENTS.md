# AGENTS.md — PyCaret 4.0 agent instructions

> This file is read by AI coding agents (Claude, Cursor, Copilot, etc.) before they touch the repo. It is the single-source briefing for any agent contributing to PyCaret 4.0. If you are a human: read it too; it's short and useful.

## TL;DR — the 60-second briefing

- **PyCaret 4.0 is a ground-up revamp** of a ~62K-LOC ML library that was unmaintained for 3 years. It's a clean break from 3.x — the functional API, mlflow, comet, wandb, dagshub, fugue, dask, yellowbrick, m2cgen, gradio, fastapi, boto3, evidently, fairlearn, and ~20 other dependencies are gone.
- **The public API is OOP-only.** One sklearn-compatible `Experiment` subclass per task. `Experiment.fit(data)` replaces the 3.x `setup(data)`.
- **`pycaret/internal/pycaret_experiment/` is a 16,000-LOC god-class** that's still alive during the migration. The new `Experiment` wraps it via `self._legacy`. **Do not rewrite it in one pass.** Drain it verb-by-verb.
- **The notebook golden path must always work.** `setup` → `compare_models` → `tune_model` → `predict_model` → `save_model`. If your change breaks that on the `juice` / `boston` / `jewellery` / `anomaly` / `airline` datasets, revert.
- **Everything non-trivial gets logged** in `docs/revamp/release_notes_pycaret4.md` under the appropriate category tag (`BREAKING`, `REMOVED`, `ADDED`, `CHANGED`, `FIXED`, `DOCS`, `BUILD`, `TESTS`, `DEPS`, `INTERNAL`). Newest session at the top; append-only.

## Start here

Read these in order before writing code:

1. **`README.md`** — what PyCaret 4.0 is and the canonical usage pattern.
2. **`docs/revamp/ARCHITECTURE.md`** — 8 core design principles, package layout, `Experiment` interface contract, migration plan.
3. **`docs/revamp/STATUS.md`** — what's landed and what's still in play. Newest session first.
4. **`docs/revamp/ROADMAP.md`** — phased plan. Find the phase you're contributing to.
5. **`docs/revamp/DECISIONS.md`** — ADRs. If an option "feels wrong," check here; it's probably already been litigated.
6. **`docs/revamp/KILL_LIST.md`** — everything deliberately removed from 4.0. Don't reintroduce any of it.
7. **`docs/revamp/release_notes_pycaret4.md`** — engineering change log. You'll append to this.

## Non-negotiables

### Rules

1. **OOP-only API.** No module-level `setup` / `compare_models` / etc. If you catch yourself writing one, stop.
2. **No new module-level mutable state.** No `_CURRENT_EXPERIMENT`. No `ContextVar` implicit-state. If state needs to flow, put it on the `Experiment` instance.
3. **Every public verb returns a typed result dataclass** — `CompareResult`, `TuneResult`, `PredictResult`, etc. Never return a bare DataFrame or bare estimator.
4. **Every long-running operation emits a structured event** through `self.logger.log(EventKind.X, ...)`. No `print()` inside the engine. (Top-level `save_model(..., verbose=True)` is an explicit opt-in exception.)
5. **No upper-bound version pins** on NumPy, pandas, scipy, sklearn, joblib. The whole point of 4.0 was removing those.
6. **No reintroducing kill-listed dependencies.** See `docs/revamp/KILL_LIST.md`.
7. **Don't delete `internal/pycaret_experiment/`** wholesale. Delegation is the escape hatch keeping the public API stable while verbs are rewritten.
8. **Don't add backward-compat shims for the 3.x functional API.** 4.0 is a clean break; "nobody will migrate 3 → 4" (project owner).

### Conventions

- **Python target:** 3.13 primary; 3.11 floor. 3.14 is tracked but currently blocked on upstream joblib/cloudpickle PEP 649 support — do not write 3.14-only code.
- **Tooling:** `uv` for env + lockfile, `hatchling` build backend, `ruff` for lint + format, `pytest` for tests.
- **Imports:** absolute only inside `pycaret/`. No star imports. Lazy-import heavy optional deps inside the function that needs them.
- **Type hints:** everywhere on new code. `from __future__ import annotations` at the top of every module.
- **Docstrings:** numpydoc style, as short as truthful. Describe *why*, not *what* (the code already says what).
- **Filenames:** `snake_case.py`. Task subclasses named `{Task}Experiment`.
- **Tests live in `tests/`.** The canonical OOP smoke set is in `tests/test_e2e_oop.py`; the architecture unit tests are in `tests/test_core_architecture.py`.

### Workflow

1. **Make a plan.** For any non-trivial change, sketch what you'll edit + why in the response to the user before editing.
2. **Write the code.** Small diffs, one concern per commit.
3. **Run the relevant test subset** (`uv run pytest tests/test_core_architecture.py tests/test_e2e_oop.py`) — it's fast.
4. **Append a release-notes entry** in `docs/revamp/release_notes_pycaret4.md` under the current session block. Every non-trivial change.
5. **Update `docs/revamp/STATUS.md` / `ROADMAP.md`** if you finished a roadmap item.
6. **If you made a non-obvious design choice**, record it in `docs/revamp/DECISIONS.md` as a new ADR entry (newest first).

## Repo map

```
pycaret/
├── README.md                       <- user-facing entry point
├── AGENTS.md                       <- this file
├── CONTRIBUTING.md                 <- human contributor guide
├── LICENSE                         <- MIT
├── pyproject.toml                  <- deps, tool config, build
├── uv.lock                         <- locked resolution
│
├── pycaret/                        <- the engine (~49K LOC, shrinking)
│   ├── __init__.py                 <- version + save_model / load_model re-exports
│   ├── persistence.py              <- stateless save/load utilities
│   ├── core/                       <- Experiment base + typed results + errors
│   │   ├── experiment.py           <- Experiment(BaseEstimator), task-agnostic verbs
│   │   ├── supervised.py           <- SupervisedExperiment (classification/regression/TS)
│   │   ├── unsupervised.py         <- UnsupervisedExperiment (clustering/anomaly)
│   │   ├── results.py              <- CompareResult, TuneResult, PredictResult, ...
│   │   ├── tasks.py                <- TaskType enum
│   │   └── errors.py               <- PyCaretError hierarchy
│   ├── tasks/                      <- 5 task subclasses (the public API)
│   │   ├── classification.py
│   │   ├── regression.py
│   │   ├── clustering.py
│   │   ├── anomaly.py
│   │   └── time_series.py
│   ├── api/                        <- typed introspection (for UI / agents)
│   │   ├── cards.py                <- ModelCard, MetricCard, ParameterCard
│   │   ├── schemas.py              <- SetupParamSchema
│   │   └── describe.py             <- list_models, describe_model, ...
│   ├── logging/                    <- event-stream logger
│   │   ├── events.py               <- Event dataclass + EventKind enum
│   │   ├── base.py                 <- BaseLogger + NullLogger
│   │   └── memory.py               <- MemoryLogger (in-mem + optional JSONL file)
│   ├── classification/             <- thin re-export shim (BC for import paths)
│   ├── regression/                 <- thin re-export shim
│   ├── clustering/                 <- thin re-export shim
│   ├── anomaly/                    <- thin re-export shim
│   ├── time_series/                <- thin re-export shim
│   ├── datasets.py                 <- dataset loaders (kept)
│   ├── loggers/                    <- legacy import path; re-exports pycaret.logging
│   ├── distributions.py            <- re-export of pycaret.internal.distributions
│   ├── containers/                 <- legacy model/metric registries (still used)
│   ├── internal/                   <- LEGACY god-class (Experiment._legacy)
│   │   ├── pycaret_experiment/     <- ~10K LOC supervised/tabular god-class — to drain
│   │   ├── preprocess/             <- transformers, imputers (migrating in Phase 4)
│   │   ├── plots/                  <- legacy plot helpers (Phase 3 replacement: Plotly)
│   │   └── patches/                <- sklearn monkey-patches (remove as sklearn catches up)
│   └── utils/                      <- version checks, soft-dep introspection
│
├── tests/                          <- pytest suite
│   ├── conftest.py                 <- minimal; no implicit-state fixtures
│   ├── test_core_architecture.py   <- fast unit tests for core primitives
│   ├── test_e2e_oop.py             <- end-to-end smoke per task
│   ├── test_models.py              <- model-registry equality tests (OOP)
│   └── test_datasets.py
│
├── notebooks/                      <- working end-to-end examples
│   ├── 01_classification.ipynb
│   ├── 02_regression.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_time_series.ipynb
│
├── datasets/                       <- bundled sample CSVs
├── scripts/                        <- maintenance scripts (notebook build, etc.)
└── docs/
    ├── images/                     <- logo, diagrams for README
    ├── revamp/                     <- THE revamp narrative (read first)
    ├── for_agents/                 <- agent-specific deep dives
    └── for_developers/             <- dev onboarding
```

## Common tasks

### Add a new verb / rewrite a legacy verb natively

1. Current state: the verb calls `self._legacy.<verb>(*args, **kwargs)` and wraps the return in a typed dataclass.
2. Replacement: reimplement the verb natively using `sklearn.pipeline.Pipeline`, `sklearn.model_selection`, and whatever upstream helpers fit.
3. Keep the signature identical. Keep the return type identical.
4. Emit the same structured events (`MODEL_TUNE_STARTED` / `MODEL_TUNED`, etc.).
5. Add a test in `tests/test_e2e_oop.py`; do not bring back the old test.
6. Release-notes entry under `CHANGED` + `INTERNAL` (not `BREAKING` — external API is identical).

### Add a new task

1. Add `TaskType.NEW_TASK` to `pycaret/core/tasks.py`.
2. Decide: `SupervisedExperiment` or `UnsupervisedExperiment`?
3. Create `pycaret/tasks/<new_task>.py` with the subclass; pre-configure `task`, override `__sklearn_tags__` if useful, override `_build_legacy_experiment()` if a legacy target exists.
4. Extend `pycaret.api.describe` with model/metric cards.
5. Release-notes entry under `ADDED`.

### Fix an issue from the open-issue list

1. Find it in `docs/revamp/github_issues/triage.md` and its bucket.
2. If it's in the **fixed-in-4.0** bucket, close with a link to the relevant commit / release note.
3. If it's in the **still-relevant** bucket, fix it; reference the issue number in the release-notes entry.
4. If it's in the **out-of-scope** bucket, reply pointing at `KILL_LIST.md` and close.

## Deep dives

- `docs/for_agents/ENGINE_WALKTHROUGH.md` — what happens at every step of `fit` → `compare_models` → `predict_model`.
- `docs/for_agents/TYPED_RESULTS.md` — every result dataclass, its fields, and when it's produced.
- `docs/for_agents/EVENT_STREAM.md` — the 22 canonical `EventKind`s, what they carry, how to subscribe.
- `docs/for_agents/INTROSPECTION_API.md` — `list_models` / `describe_model` / `describe_setup_params` contract for UI / form-building.
- `docs/for_developers/SETUP.md` — dev environment, linting, test matrix.
- `docs/for_developers/TESTING.md` — how to run / add tests.
- `docs/for_developers/DRAINING_THE_GODCLASS.md` — the playbook for migrating a verb off `_legacy`.
