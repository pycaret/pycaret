# Phase 0 — Data Model Reconciliation (spec)

*Status: drafted, awaiting review. No code lands until this is signed off.*

This is the foundational refactor. Every later phase assumes the model
described below. We do it first, while the dev DB is small and there's
no production data.

---

## Background

The spec in [`PHASES.md`](PHASES.md) (architectural decision A3) defines:

- **Trial** = a *logical pipeline candidate*. Stable identity. `Logistic Regression`, `Tuned XGBoost`, `Bagged Decision Tree`, `Stacked Ensemble` are each a Trial.
- **Run** = one *execution instance* of a Trial. `Run 1` is the initial training; `Run 2` is a retrain on fresher data; `Run 3` is a drift-triggered retrain.

The **current** code does the opposite:

- A Run owns the compare/search execution and the leaderboard JSON.
- A Trial is a row born from a Run's compare loop — they currently share a 1:N parent-child where Run is the parent.
- Tuning a Trial today *inserts a new Trial row in the same Run* (the model we agreed on a few sessions ago, which we now know is wrong long-term).

The new model inverts this:

- An Experiment owns Trials directly.
- A Trial owns Runs (1:N).
- A Run owns the metrics + artifact.
- Compare produces N Trials in one click, each with Run 1.
- Tune produces a new Trial with Run 1.
- Retraining the same Trial produces Run 2.

---

## Target entity model

```
Workspace
└── Project
    └── Experiment        (problem definition: data + target + preprocessing config)
        └── Trial         (logical pipeline candidate)
            └── Run       (execution instance — metrics, artifact, status, started_at)
                ↑
            Deployment ───┘ (references trial_id + run_id together)
```

### `trials` (revised)

| column | type | notes |
|---|---|---|
| `id` | uuid PK | |
| `experiment_id` | uuid FK → `experiments` ON DELETE CASCADE | **NEW** — Trial belongs to Experiment, not Run. |
| `workspace_id` | uuid FK → `workspaces` | denorm for scoping queries |
| `kind` | varchar(16) NOT NULL DEFAULT `'compare'` | `compare \| tuned \| ensembled \| blended \| stacked \| manual` |
| `name` | varchar(128) | engine model id, e.g. `xgboost`, or a generated name like `tuned_xgboost` |
| `model_id` | varchar(64) | engine registry id when applicable |
| `parent_trial_ids` | jsonb (list of uuids) | lineage — sources for tune/blend/stack |
| `created_by_action_id` | uuid | groups Trials spawned by one user action (e.g. one `compare_models` click). NULL for standalone Trials. |
| `created_at` / `updated_at` | timestamps | |
| `created_by` | uuid FK → `users` | |

Removed: `run_id` FK (Trial is no longer scoped to one Run), `rank`, `metrics`, `is_best`, `stored_path`, `sha256`, `size_bytes`, `params`, `notes`. These move to Run (artifact / metrics) or stay on Trial as derived (best-run metrics) computed on read.

### `runs` (revised)

| column | type | notes |
|---|---|---|
| `id` | uuid PK | |
| `trial_id` | uuid FK → `trials` ON DELETE CASCADE | **NEW** — Run belongs to Trial. |
| `experiment_id` | uuid FK → `experiments` | denorm to dodge a join when fetching by experiment |
| `workspace_id` | uuid FK → `workspaces` | denorm |
| `sequence` | int NOT NULL | per-trial run number: 1, 2, 3… Unique on (trial_id, sequence). |
| `status` | varchar(16) | `queued \| running \| succeeded \| failed \| cancelled` |
| `started_at` / `finished_at` | timestamps | |
| `duration_ms` | float | |
| `metrics` | jsonb | per-fold + aggregated metrics |
| `stored_path` | varchar(1024) | object-store URI for the fitted pipeline pickle |
| `sha256` | varchar(64) | |
| `size_bytes` | int | |
| `params` | jsonb | estimator hyperparams + (for tuned) `_best_params`, `_cv_history` |
| `error` | text | failure reason |
| `snapshot` | jsonb | request payload — for reproducibility |
| `triggered_by` | varchar(32) | `user \| schedule \| drift \| api` |
| `triggered_by_id` | uuid | nullable FK to the trigger source |
| `created_at` / `updated_at` | timestamps | |

### `deployments` (revised)

| column | change |
|---|---|
| `pipeline_id` | **renamed** to `registered_model_id` (placeholder — wired up in Phase 7; for Phase 0 we keep `pipeline_id` and add a NOT NULL `trial_id` + `run_id` pair). |
| `trial_id` | **NEW** — FK → `trials`, NOT NULL. |
| `run_id` | **NEW** — FK → `runs`, NOT NULL. (trial_id + run_id together = exact reproducibility) |

We don't kill `pipelines` in Phase 0 — that's Phase 7. We just add `(trial_id, run_id)` so every deployment can be traced to its exact training event.

### `experiments` (unchanged in Phase 0)

No schema change. But behaviour changes — an Experiment's "leaderboard" view becomes *"all Trials, each scored by its best Run's primary metric."*

---

## API contract changes

Every trial/run endpoint mutates. Summary:

| current | becomes |
|---|---|
| `POST /experiments/:id/runs` (creates a Run with N trials embedded) | `POST /experiments/:id/runs` (creates a `compare-batch`: N Trials, each with Run 1, sharing `created_by_action_id`. Returns the Trial list.) |
| `GET /runs/:run_id` | `GET /runs/:run_id` (returns one Run + its parent Trial + Experiment) |
| `GET /runs/:run_id/trials` | `GET /experiments/:id/trials` (Trials live under Experiments now) |
| `GET /runs/:run_id/trials/:trial_id` | `GET /trials/:trial_id` (or `/trials/:trial_id/runs/:run_id` for run-level detail) |
| `POST /runs/:run_id/trials/:trial_id/tune` | `POST /trials/:trial_id/tune` — creates a new Trial + Run 1; returns new trial id |
| `POST /runs/:run_id/blend` | `POST /experiments/:id/blend` — body lists source `trial_ids`; creates new Trial + Run 1 |
| `POST /runs/:run_id/trials/:trial_id/promote` | `POST /trials/:trial_id/runs/:run_id/promote` (Run-specific; reproducibility) |
| `GET /runs/:run_id/events/ws` | unchanged — WebSocket still scoped by Run |
| `POST /experiments/:id/retrain` | **NEW** — same Experiment, same Trial(s), produces new Run(s). The Trial is unchanged. Body: `trial_ids?: [...] // default: all trials in the experiment` |

The frontend trial detail URL goes from `/runs/:run_id/trials/:trial_id` to `/trials/:trial_id` (with run-specific child route `/trials/:trial_id/runs/:run_id`).

---

## Engine impact

The engine itself doesn't change. Compare / Tune / Ensemble / Blend / Stack already return their results as objects — we just persist them into the new Trial/Run shape on the platform side.

The orchestrator's `submit(spec)` becomes `submit_action(action_spec)`:

- `action: compare` → emits N Trials + N Run-1s.
- `action: tune | ensemble` → emits 1 Trial + 1 Run-1.
- `action: blend | stack` → emits 1 Trial + 1 Run-1.
- `action: retrain` → emits 0 new Trials + N new Runs (one per source trial).

Each Trial gets a stable `name` (computed from action + sources, e.g. `tuned_xgboost`, `stacked(rf+xgb+lr)`).

---

## Migration plan

This is **destructive** for the dev DB. We're OK with that — no real data lives there.

1. New Alembic revision `f0a1b2c3d4e5_session28_phase0_trial_run_pivot.py`:
   - drop columns `trials.{run_id, rank, metrics, is_best, stored_path, sha256, size_bytes, params, notes}`
   - add columns `trials.{experiment_id, name, created_by_action_id}` (kind + parent_trial_ids already exist)
   - add columns `runs.{trial_id, sequence, snapshot, triggered_by, triggered_by_id}` (most fields exist, just need to fix FKs and add seq)
   - add columns `deployments.{trial_id, run_id}`
   - migrate data: for every existing Trial, create an experiment-scoped row and a Run row populated from the legacy Run that owned it; assign sequence=1.
2. Bootstrap detector updated: a `trials` table with an `experiment_id` column ⇒ Phase 0 fingerprint.
3. `pycaret-server migrate --reset-dev` flag added for clean wipes during development.

### Rollback

The migration is reversible per-step (Alembic `downgrade`), but the data lineage between legacy Run.leaderboard JSON and the new shape is lossy. So in practice: roll back by wiping the dev DB and reverting to a prior Alembic revision. We commit to *never* shipping Phase 0 to a prod DB without an upgrade script. (None exists yet — none needed yet.)

---

## Code surface that has to change

| layer | files | scope |
|---|---|---|
| **DB** | `services/api/pycaret_server/db/models.py` | Trial + Run + Deployment ORM updates |
| **Migration** | `services/api/pycaret_server/migrations/versions/` | one new revision (above) |
| **Bootstrap** | `services/api/pycaret_server/db/bootstrap.py` | fingerprint for Phase 0 |
| **Schemas** | `services/api/pycaret_server/api/schemas.py` | `TrialResponse`, `RunResponse`, request bodies — all updated |
| **Routes** | `services/api/pycaret_server/api/runs.py` | most endpoints move or change shape |
| **Orchestrator** | `services/api/pycaret_server/runs/orchestrator.py` | `submit_action` instead of `submit`; emits Trial+Run rows in the new shape |
| **Dispatch** | `services/api/pycaret_server/runs/dispatch.py` | `dispatch_action(experiment, payload)` returns the list of new Trial ids |
| **SDK** | `packages/sdk-python/pycaret_client/client.py` | mirror the API surface |
| **Frontend types** | `apps/web/src/api/types.ts` | Trial + Run + Experiment shapes |
| **Frontend API** | `apps/web/src/api/endpoints.ts` | every trial/run helper |
| **Frontend pages** | `RunDetail`, `TrialDetail`, `TrialsCard`, `ExperimentDetail`, `TrialCompare`, `ModelCard` | URL routes + data fetching + chip rendering all touched |
| **Frontend routes** | `apps/web/src/App.tsx` | `/trials/:trial_id`, `/trials/:trial_id/runs/:run_id` |

---

## Test plan

### Backend
- New test file `tests/test_phase0_model.py`:
  - Compare on iris → 1 experiment, N trials (one per algorithm), N×1 runs.
  - Trial spawned by compare has `created_by_action_id` set; trials from same call share it.
  - Tune trial X → 1 new trial with `kind=tuned`, `parent_trial_ids=[X]`, 1 new run with `sequence=1`.
  - Retrain Experiment → existing trials get a new Run with `sequence=2`.
  - Blend [X, Y, Z] → 1 new trial `kind=blended`, parents `[X, Y, Z]`, 1 new run.
  - Stack same.
  - Deployment creation requires both `trial_id` and `run_id`; either missing → 400.
  - Deployment lookup returns the exact Run snapshot.
- Existing tests that use legacy run/trial shapes get rewritten or marked obsolete.

### Frontend
- Vitest tests for the new TrialDetail / RunDetail components.
- Smoke test: render the Experiment page, the Trial page, the Run page; no console errors.

### End-to-end manual
1. Fresh DB (`pycaret-server migrate --reset-dev`).
2. Upload boston.csv.
3. Create an Experiment.
4. Click "Compare models" → see Trials populate one-by-one.
5. Click any Trial → see its Run 1 details.
6. Click Tune → see new Trial created → Run 1 streaming.
7. Click "Retrain experiment" → all existing Trials get Run 2.
8. Promote → pick (trial, run) pair → deploy → predict.

---

## Breaking changes — call list

- **Dev databases wipe** required. No production data to worry about today.
- **Trial detail URL changes** from `/runs/:run_id/trials/:trial_id` to `/trials/:trial_id`. Old bookmarks 404.
- **SDK clients** that call `runsApi.trials(run_id)` need to migrate to `experimentsApi.trials(experiment_id)`.
- **WebSocket subscriptions** still keyed by `run_id` — no change there, but `run_id` now identifies one execution of one trial, not one compare batch.

---

## Success criteria (recap)

A user can:
1. Upload a CSV → create an Experiment → click Compare → see N Trials appear, each with Run 1.
2. Click any Trial → see its Run history (just Run 1 initially).
3. Click Tune → see a new Trial created with `kind=tuned`, its Run 1 streaming live in the event log.
4. Click "Retrain experiment" → see every existing Trial get a new Run.
5. Open a Trial that has multiple Runs → see them in a sub-list, pick one, promote it.
6. Deploy a specific `(trial, run)` pair. Roll back by changing the deployment's `(trial, run)` reference (Phase 7 wires this fully, Phase 0 just sets the foundation).

All four backend session-test files pass against the new shape. tsc clean. 59+ UI tests still pass (some new, some updated).

---

## Open questions before coding

1. **Trial.name** — should the platform assign it (`tuned_xgboost`, `stacked_3`) or let the user override on creation? Suggest: auto-generated, user-editable post-hoc.
2. **Deployment FK Pair** — Phase 0 keeps `pipeline_id` alongside the new `(trial_id, run_id)` for backward compat. Phase 7 collapses to `(registered_model_id, version)` and drops the Trial/Run FKs. OK?
3. **Retrain endpoint shape** — body is `{trial_ids?: [...]}` (default: all trials of the experiment, only succeeded ones). Confirm?
4. **Created-by-action grouping** — a `created_by_action_id` UUID per action (compare/retrain) is enough; we don't need a separate `actions` table yet. Confirm?

Once these four are answered I start migration coding.
