# PyCaret Platform — Forward Roadmap (Phases 0–14)

*Drafted after the architecture-decisions conversation captured in
[`thinking/`](thinking/). Reads forward; the legacy MVP 1/2/3 timeline
lives in [`ROADMAP.md`](ROADMAP.md).*

This document is the **single source of truth for what we're building
next**. Each phase has its own detailed spec at `phase-N-spec.md`,
written *before* code starts so the design is reviewable.

---

## Architectural decisions (locked)

These shape every phase below.

| # | Decision | Rationale |
|---|---|---|
| **A1** | Solo engineering (maintainer + AI). No calendar dates, no team scaling assumptions. | Plan for what actually ships. |
| **A2** | Primary user: **open-source community** self-hosting on their own infra. | DX, docs, easy install, Git integration move *up*. SSO/SAML/multi-tenant/billing move *down*. |
| **A3** | **Trial = logical pipeline candidate; Run = execution instance inside a Trial.** Reverses the current model. | Cleaner semantics; reproducibility (Deployment references a specific Run); maps to user mental model. |
| **A4** | `compare_models` produces N Trials in one click, each with Run 1. Grouping = Experiment (no separate batch entity). Tag spawned-together trials with `created_by_action_id` for traceability. | Cheap, matches how users think. |
| **A5** | Queue + worker process separation, object-storage abstraction, and Postgres-first prod path are **foundational and ship before any new product surface**. | Every feature stacked on the current in-process executor would need re-engineering. |
| **A6** | Datasource entity model lands in Phase 4 but with **only the CSV driver** plus one DB driver to prove the shape. New sources are 1 driver + UI tweaks each, not a refactor. | Avoids over-building. |
| **A7** | Notebook runtime is **one Jupyter container per session**. Cold-start cost accepted in exchange for isolation. | Reproducibility > efficiency for v1. |
| **A8** | Every phase ships **docs, tests, and a `phase-N-spec.md` reviewed before code**. Docs/DX is a *discipline*, not a phase. | Quality bar that holds across phases. |

---

## Phase sequencing — at a glance

| # | Name | Why this slot | Rough size |
|---|---|---|---|
| **0** | Data model reconciliation (Trial-as-logical / Run-as-execution) | Blocks everything else. Cheap now, expensive later. | S |
| **1** | Queue + worker separation (Redis + RQ, worker container, Job table) | Unblocks scale, scheduling, GPU pool, drift jobs. | M |
| **2** | Object storage abstraction (MinIO/S3/local adapter) | Required before notebooks, Git, multi-worker. | S |
| **3** | Postgres-first production path | Required before multi-user, multi-tenant, real prod. | S |
| **4** | Datasource entity model (CSV driver + one DB driver, lineage table from day one) | Biggest "real platform" lift; unblocks everything data-side. | L |
| **5** | Git integration v1 (project ↔ repo, YAML export, config-as-code) | OSS-first: devs expect this. Moved up. | M |
| **6** | Realtime callbacks v2 (per-fold / per-iteration events, pub/sub fan-out, modern training UI) | Delivers the "modern real-time UI" promise; unblocks tune/ensemble visualization roadmap. | M |
| **7** | Model registry v2 (Pipeline → `RegisteredModel.version`; Deployment references run+version) | Unblocks governance, reproducibility, rollback. | M |
| **8** | Notebook runtime (JupyterLab containers per session) | Major analyst surface. | L |
| **9** | Scheduling v2 + drift-triggered retrain + batch inference jobs | Closes the lifecycle loop. | M |
| **10** | Monitoring v2 (latency / throughput / drift / alert rules → Slack/email) | Production-grade ML gate. | M |
| **11** | Statistical computing v1 (Analysis entity as peer to Experiment; t-tests, ANOVA, χ², regression diagnostics, survival, forecasting) | Begins SAS-parity story; meaningful breadth in one phase. | L |
| **12** | Governance basics (RBAC v2 scoped to org/workspace/project; approval workflow; audit log UI; lineage graph viewer) | Enterprise-readiness without the enterprise feature creep. | M |
| **13** | Enterprise deployment polish (Helm chart, Compose for self-host, airgapped install docs) | Sells the platform to OSS enterprises. | L |
| **14** | Distributed / GPU workers (worker pool segmentation by class, GPU queue) | Performance scale-out. | L |

**Long tail** (post-14, each a phase of its own when scoped):
Statistical computing v2 (SAS-parity deep cuts — MANOVA, Bayesian, multilevel, causal inference, A/B platform) ·
AI assistant v2 (conversational threads on runs, suggested next experiments, AI cohort discovery) ·
Feature store (Feast adapter or native) ·
Explainability v2 (SHAP, PDP, ICE, counterfactuals as first-class UI) ·
AutoML preset (setup → compare → tune → ensemble → deploy in one click) ·
SSO/SAML/OIDC (only when an enterprise prospect needs it) ·
Multi-tenancy / Org tier (only when SaaS path opens) ·
Marketplace for community plugins.

---

## Phase 0 — Data model reconciliation

**Goal.** Adopt the spec's Trial/Run model verbatim. Trial = logical pipeline candidate (`Logistic Regression`, `Tuned XGBoost`, `Stacked Ensemble`). Run = one execution of a Trial (`Run 1`, `Run 2 — retrained`).

**Scope.**
- New `trials` schema: `kind`, `parent_trial_ids`, `created_by_action_id` (group spawn), `experiment_id`, removes `run_id` FK.
- New `runs` schema: `trial_id` FK (mandatory), `status`, `started_at`, `finished_at`, `metrics`, `stored_path`, `sha256`, `size_bytes`, `params`. Existing run-level fields (`leaderboard`, `metrics_summary`, `error`) attach to Run.
- Compare semantics: one engine call → N Trials, each with Run 1, all sharing `created_by_action_id`.
- Tune/Ensemble/Blend/Stack: new Trial + Run 1.
- Retraining: same Trial, Run N+1.
- Deployment: references `(trial_id, run_id)` pair (both required for reproducibility).
- All API responses + UI views updated accordingly.
- New seed/migrate scripts to wipe dev DB and re-bootstrap.

**Prerequisites.** None. Should ship first.

**Breaking changes.**
- DB wipe required (dev only — no production data yet).
- API contract change for every trial/run endpoint.
- Frontend types regenerated; pages reworked.

**Success criteria.** Compare on iris produces an Experiment with N Trials (one per algorithm), each with Run 1; Tune produces a new Trial with Run 1; Retrain produces a new Run on the same Trial. Deployment points at `(trial_id, run_id)`. Engine tests + UI tests green; manual end-to-end smoke pass.

**Detailed spec.** [`phase-0-spec.md`](phase-0-spec.md) (drafted; review before coding).

---

## Phase 1 — Queue + worker separation

**Goal.** Replace the in-process `ThreadPoolExecutor` with Redis + RQ workers running in a separate process. New `Job` entity owns the lifecycle of every async unit of work.

**Scope.**
- Redis service in `docker-compose.yml`.
- `Job` table: `id, kind, status, payload, run_id?, created_at, started_at, finished_at, attempts, error`.
- Worker entrypoint (`pycaret-worker serve`) — long-lived process polling RQ.
- Backend produces `Job` rows + enqueues; never executes ML.
- All current orchestrator paths (compare/create/search/tune/ensemble/blend/stack) move to RQ jobs.
- Heartbeat + retry/back-off semantics.
- `RUNS_BACKEND=inprocess|redis` env toggle so devs can still run in-process during early iteration.

**Prerequisites.** Phase 0 (Run is the unit of execution → maps cleanly to one Job per Run).

**Breaking changes.** Test fixtures that depend on `orchestrator.wait_for(...)` migrate to `job.wait(...)`.

**Success criteria.** `compare_models` runs via a worker container, backend stays sub-100ms p99 during a tune, kill the worker mid-job and it resumes on restart.

---

## Phase 2 — Object storage abstraction

**Goal.** Stop writing pickles to local disk. Adapter pattern: local-fs / S3 / MinIO / GCS / Azure Blob — one driver per backend.

**Scope.**
- `pycaret_server.storage.ObjectStore` protocol with `put`, `get`, `presigned_url`, `delete`, `exists`.
- LocalFs driver (default for dev, current behaviour preserved).
- S3 driver (boto3) for prod self-host on AWS.
- MinIO driver for Docker Compose self-host.
- All `stored_path` columns become object-store URIs (`s3://bucket/key`, `file:///abs/path`).
- Migration of existing artifacts.
- Download/upload presigned URLs for the trial pickle download.

**Prerequisites.** Phase 1 (workers need write access to shared storage when running on a different host).

**Breaking changes.** Local dev still works without S3; URIs in DB swap from absolute path to `file://` form.

**Success criteria.** Same trial pkl downloadable from UI with backend reading from S3.

---

## Phase 3 — Postgres-first production path

**Goal.** Postgres becomes the production DB. SQLite stays the zero-config dev default.

**Scope.**
- Postgres service in `docker-compose.yml`.
- Bootstrap honours both backends; pool config; sane connection limits.
- Alembic migrations validated against both.
- Test matrix runs against both.
- Document the migration path for SQLite-dev → Postgres-prod (export + reimport script).

**Prerequisites.** Phase 0 (schema is now stable).

**Success criteria.** Same backend image runs against both DBs depending on `PYCARET_DATABASE_URL`. CI runs the test suite against both.

---

## Phase 4 — Datasource entity model

**Goal.** Lay the catalog primitives Databricks/Snowflake users expect — Datasource, Connection, Secret, Dataset (versioned), Lineage. Ship CSV + one DB driver (Postgres) end-to-end. Adding the next driver becomes a 1-file PR.

**Scope.**
- `data_sources` (already exists) — extend with `kind ∈ {csv_upload, postgres, snowflake, bigquery, s3, ...}`, `config`, `secret_id`, `created_at`, `updated_at`.
- `connections` (new) — `id, workspace_id, kind, config, secret_id, last_tested_at`.
- `secrets` (new) — encrypted at-rest, scoped to workspace.
- `datasets` (new) — `id, datasource_id, name, version, schema_json, row_count, byte_count, snapshot_uri, created_at`.
- `lineage` (new) — `source_kind, source_id, target_kind, target_id, relation` — captures *dataset → experiment*, *run → registered_model*, *deployment → run*, etc.
- Driver layer: `DatasourceDriver` protocol with `test_connection`, `list_tables`, `read_sample`, `read_full`. CSV + Postgres drivers ship.
- UI: "New datasource" wizard (kind picker → connection form → test → save); dataset profile view; lineage graph viewer.
- Backend: introspect on register (schema, stats, sample); refresh on demand.

**Prerequisites.** Phase 1 (introspection happens in a worker), Phase 2 (snapshots land in object storage), Phase 3 (catalog scales).

**Success criteria.** Register a Postgres table as a Datasource → schema + sample rendered in the UI → can be picked as an Experiment's training data → run completes against live Postgres rows → lineage shows dataset→trial→deployment graph.

---

## Phase 5 — Git integration v1

**Goal.** Project ↔ Git repo mapping. Experiments / Trials / Runs export as YAML + manifest files on every state change. Git stores configs + lineage; object storage holds artifacts.

**Scope.**
- `git_repositories` entity: workspace-scoped, GitHub/GitLab/Gitea/Bitbucket; access via PAT or app installation.
- One-way **export sync** on commit: `/experiments/<name>/experiment.yaml`, `/trials/<name>/trial.yaml`, `/trials/<name>/runs/<id>/{metadata,metrics,params,lineage}.json`.
- Manifest carries `artifact_uri: s3://...` — never raw model bytes.
- "Publish to Git" button on Experiment, Trial, and Run detail pages.
- "Open in GitHub" link from every entity.
- UI: connect-repo wizard, push status indicator, repo browser pane.

**Prerequisites.** Phase 0 (stable entity shapes), Phase 2 (`artifact_uri` is the manifest's pointer).

**Out of scope (v2):** bidirectional sync, Git-as-source-of-truth, PR-driven model changes. Read-only export first.

**Success criteria.** Run finishes → commit lands in repo within 5s with metadata + lineage; pull the repo, every run is reproducible from its manifest + artifact URI.

---

## Phase 6 — Realtime callbacks v2

**Goal.** The "watch a model train" experience users get from W&B / Comet. Per-fold and per-iteration events emitted by the engine, fanned out via Redis pub/sub, rendered as live charts in the UI.

**Scope.**
- Engine emits richer events: `FOLD_STARTED`, `FOLD_FINISHED` (with per-fold metrics), `TUNE_ITERATION` (with score), `STACK_BASE_FITTED`, etc.
- Pub/sub backbone: workers publish to Redis, backend subscribes, WebSocket fans out to browsers.
- UI: live optimization-history charts on the running card (extends the tune chart we already have); per-fold metric ribbons; "trained models so far" mini-leaderboard with diff vs current best.
- Drawer event log gains real-time filtering by event kind.

**Prerequisites.** Phase 1 (workers publish to Redis), Phase 0 (Trial/Run granularity is right).

**Success criteria.** Watching a 30s tune feels alive — chart fills iteration by iteration, fold metrics stream as they land.

---

## Phase 7 — Model registry v2

**Goal.** Separate the *artifact a Run produces* from the *named, versioned operational thing a Deployment references*. Eliminates the current ambiguity where promoting a trial creates a Pipeline AND mutates the trial.

**Scope.**
- `registered_models` (new): `id, workspace_id, name, project_id, description, current_version_id, owner, created_at`.
- `registered_model_versions` (new): `id, registered_model_id, version, run_id, trial_id, stored_path, sha256, params, status ∈ {staging, production, archived}, promoted_at, promoted_by`.
- Deployment now references `(registered_model_id, version)` — not run directly.
- "Promote" wizard: pick existing model name (or create new) → version bumps automatically → assigns to dev/staging/prod.
- Rollback: change Deployment.version to a prior one; no model retraining needed.
- UI: model registry page (list / detail / version history / promotion graph).

**Prerequisites.** Phase 0 (Run is stable), Phase 5 (export reflects the lineage).

**Breaking changes.** Existing `pipelines` rows migrate to `registered_models` + one version each. Deployment FK points at version.

**Success criteria.** Train v1 → promote to staging → deploy → train v2 → promote to staging → flip prod from v1 to v2 → rollback to v1 — all without retraining or unpickling.

---

## Phase 8 — Notebook runtime

**Goal.** First-class JupyterLab notebooks inside the platform, with data/secret injection and Git-backed checkpointing.

**Scope.**
- `notebooks` entity: `id, project_id, path, kernel, last_modified, last_executed`.
- `notebook_sessions` entity: `id, notebook_id, container_id, status, port, started_at, last_active_at, idle_timeout`.
- Notebook Manager service: spawns isolated Jupyter containers per session (memory/CPU cap, mount workspace-scoped object storage as `/data`, inject workspace secrets as env vars).
- Frontend: notebook list per project; "Open" launches a session and iframes the JupyterLab UI; idle sessions auto-shutdown.
- Git integration: notebooks live in `/notebooks/` of the project repo; "Save" commits.
- A `pycaret` Python client preinstalled — read/write platform entities from within a notebook (`pycaret_client.runs.create(...)`, etc.).

**Prerequisites.** Phase 2 (object storage), Phase 5 (Git repo), Phase 4 (Datasource — notebook can read a Datasource).

**Success criteria.** Open a notebook in 10s, read a registered Datasource, run a quick model, commit to Git. Restart the platform and re-open the same notebook intact.

---

## Phase 9 — Scheduling v2 + drift-triggered retrain + batch inference

**Goal.** Production lifecycle automation. Schedules become first-class jobs that can produce new Runs, trigger drift checks, or run batch predictions.

**Scope.**
- Extend `scheduled_jobs.kind` to `retrain`, `drift_check`, `batch_predict`, `dataset_refresh`.
- Schedule editor in UI (cron expression + human readable).
- Drift-triggered retrain: drift score over threshold → enqueue a retrain Job → new Run on the same Trial → optional auto-promote.
- Batch predict: pick a deployment + an input dataset → produce an output dataset with predictions; results land in object storage and a new Dataset row.
- Logs + run history per schedule.

**Prerequisites.** Phase 1 (jobs), Phase 4 (Datasource for batch IO), Phase 7 (registry — drift retrains a specific model).

**Success criteria.** Schedule a weekly retrain → new Run appears every week → if metric regresses, no auto-promote; if improves, promote to staging.

---

## Phase 10 — Monitoring v2

**Goal.** A deployment dashboard that answers "is my model OK *right now*?" in five seconds. Alerts that route to Slack/email when it's not.

**Scope.**
- Per-deployment metric panels: requests, latency (p50/p95/p99), error rate, prediction distribution, drift score over time.
- Time-series store (Postgres TimescaleDB extension or a simple roll-up table for v1).
- `alert_rules` entity: `metric, comparator, threshold, window, destination`.
- Destinations: webhook, Slack, email.
- Alert delivery via worker, dedup'd per `(rule, deployment, window)`.

**Prerequisites.** Phase 1 (alerts run on workers), Phase 4 (drift uses Datasource baseline).

**Success criteria.** Deploy a model → simulate load → see latency curve fill → trip a threshold → Slack message arrives within 60s.

---

## Phase 11 — Statistical computing v1

**Goal.** The first chunk of the SAS-parity story. "Analysis" becomes a peer to "Experiment" — workflows that don't train a predictive model but answer statistical questions on data.

**Scope.**
- New entity: `analyses` (id, project_id, kind, params, status, runs).
- Analyses ship as a small library of typed procedures with a uniform UI surface:
  - **Compare two groups**: t-test, Welch's t, Mann–Whitney U, paired-t.
  - **Compare many groups**: one-way + two-way ANOVA, Kruskal–Wallis.
  - **Categorical association**: χ², Fisher's exact, Cramér's V.
  - **Regression diagnostics**: OLS with full diagnostic suite (residual plots, Q-Q, leverage, Durbin–Watson, VIF, Cook's distance).
  - **Survival**: Kaplan–Meier, log-rank, Cox PH.
  - **Forecasting**: ARIMA + Prophet wrappers (alongside the existing time-series experiment kind).
- Result UI: hypothesis statement, test statistic, p-value, effect size, confidence interval, plain-English interpretation, plot.
- Each analysis produces a Run (consistent with Phase 0 model) and can be exported to Git (Phase 5) and rendered in a notebook (Phase 8).

**Prerequisites.** Phase 0 (Run model), Phase 4 (Datasource as analysis input).

**Success criteria.** Pick a Datasource → choose "One-way ANOVA" → pick grouping column + measure column → run → see the table, plot, and interpretation. Save to Git as YAML.

---

## Phase 12 — Governance basics

**Goal.** Enough RBAC, audit, and lineage UI for an OSS enterprise self-hoster to feel safe.

**Scope.**
- RBAC v2: roles scoped to org → workspace → project → resource. Default roles: `owner`, `admin`, `member`, `viewer`.
- Approval workflows: configurable on Deployment.create / RegisteredModelVersion.promote_to_production.
- Audit log UI (already have the table — needs a viewer + filters + export).
- Lineage graph viewer: a clickable DAG of Datasource → Experiment → Trial → Run → RegisteredModel → Deployment.

**Prerequisites.** Phase 4 (lineage exists), Phase 7 (registry).

**Success criteria.** A non-admin user can browse but not deploy. An admin sees who promoted what + when in the audit log.

---

## Phase 13 — Enterprise deployment polish

**Goal.** A self-hoster on their laptop or a corporate VM should be running in ≤15 minutes.

**Scope.**
- One-command Docker Compose stack: `frontend, backend, worker, postgres, redis, minio` wired and ready.
- Helm chart for Kubernetes self-host.
- `pycaret-server` CLI: `init`, `migrate`, `serve`, `worker`, `doctor`.
- Airgapped install bundle (tarball with all images + a one-liner install script).
- Comprehensive `INSTALL.md` + `OPERATIONS.md`.

**Prerequisites.** Everything above; this is the polish phase.

**Success criteria.** `docker compose up` on a fresh laptop → working platform at http://localhost:3000 in <15 min. Same on a fresh EC2 box.

---

## Phase 14 — Distributed / GPU workers

**Goal.** Worker pool segmentation by hardware class so heavy training doesn't starve light inference.

**Scope.**
- Queue names by class: `default`, `cpu-heavy`, `gpu`, `inference`.
- Worker startup flag selects the queues it listens on.
- Job routing logic: tune/compare/search → `cpu-heavy`; predict → `inference`; anything tagged GPU-capable → `gpu`.
- UI: per-class queue depth + worker count on an admin page.

**Prerequisites.** Phase 1.

**Success criteria.** Run a tune job on `cpu-heavy` and an inference job on `inference` simultaneously; latency on the inference path stays sub-second.

---

## Cross-cutting concerns (every phase)

- **Specs first.** `phase-N-spec.md` written and reviewed before code.
- **Tests at every layer.** Engine unit tests, backend API tests, UI tests; coverage doesn't drop.
- **Migrations.** Every schema change ships an Alembic migration + a bootstrap-detector fingerprint update.
- **Docs.** Every new entity / endpoint / page gets a short paragraph in the platform docs.
- **Status journal.** `STATUS.md` gets an entry per phase exit; `DECISIONS.md` gets an entry for every architectural choice.
- **Breaking changes.** Called out explicitly in the phase spec; migration script + announcement note.
- **Performance gates.** Backend p99 stays sub-100ms under nominal load. Worker throughput measured.

---

## How we use this doc

1. We work one phase at a time, in order.
2. Before any code lands for a phase, `phase-N-spec.md` is written and you sign off.
3. Phase exits when its success criteria are met **and** an entry is appended to `DECISIONS.md`.
4. Anything that doesn't fit in the current phase goes to the long-tail list at the top of this doc.
5. Mid-phase pivots are fine but get appended to the spec doc with a dated note.
