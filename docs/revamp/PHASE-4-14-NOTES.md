# Phases 4 / 5 / 6 / 7 / 9 / 10 / 12 / 14 — implementation notes

All landed in one consolidated migration (`b2c3d4e5f6a8`) + a batch of
new modules. Strategy: ship the schema, the API surface, the worker
handlers, and the frontend client for each phase. Heavy UI work (per-
phase pages, dashboards) is the next session's job — types and client
methods are in place so each surface is one component away.

The three phases that didn't ship (8, 11, 13) are documented at the
bottom; each genuinely needs the elbow room of its own session.

---

## Phase 4 — Datasource entity model

**Tables.** `secrets`, `connections`, `datasets`, `lineage`.

**Backend.**

- `pycaret_server.datasources` package: `DatasourceDriver` protocol,
  factory, two end-to-end drivers (`csv_upload`, `postgres`). Adding
  Snowflake or BigQuery is one new module + one `register()` call.
- `api/connections.py`:
  - `GET/POST /workspaces/{ws}/secrets` (encrypted at-rest via Fernet —
    same key as LLM provider settings).
  - `GET/POST/DELETE /workspaces/{ws}/connections` + `POST .../test`.
  - `GET /datasource-kinds` (drives the UI's New-Connection wizard).
  - `GET /data-sources/{id}/datasets` and `POST .../refresh` — version
    bumps + schema/sample capture via the driver.
  - `GET /workspaces/{ws}/lineage` — full edge list, with optional
    `node_kind` + `node_id` + `depth` for a focused subgraph (BFS in
    both directions up to 6 hops).
  - `record_lineage(...)` helper used by the registry's promote flow
    to emit `Run → RegisteredModelVersion (relation=produced)` edges.

**Frontend.** `secretsApi`, `connectionsApi`, `datasetsApi`, `lineageApi`
+ matching types in `api/types.ts`.

---

## Phase 5 — Git integration v1

**Tables.** `git_repositories`.

**Backend.**

- `pycaret_server.git.exporter` — pure serialisers turning Experiment /
  Trial / Run rows into wire-shaped dicts. Used by the live publisher
  and the future CLI export.
- `pycaret_server.git.service.publish_project` — clones the repo (or
  inits empty on first push), writes the directory layout under
  `path_prefix/`, commits, pushes. PAT auth via the linked Secret.
- `api/git_repos.py`:
  - `GET/POST/DELETE /workspaces/{ws}/git-repositories`
  - `POST /git-repositories/{id}/publish` — synchronous v1; an async
    worker handler swap-in is a one-line change when needed.

**On-disk layout the publisher writes:**

```
<repo>/<path_prefix>/
  experiments/
    <experiment_name>/
      experiment.yaml
      trials/
        <trial_name>/
          trial.yaml
          runs/
            <run_id>/
              metadata.yaml
              metrics.json
              params.json
              artifact.pointer.yaml      # carries artifact_uri (s3://...)
```

Manifests carry URIs, never raw model bytes. Pull the repo + walk the
URIs → reproducible.

**Frontend.** `gitApi` + `GitRepository` / `PublishResult` types.

---

## Phase 6 — Realtime callbacks v2

**No new tables.** Phase 6 ships the *bridge* between the worker
process and the API process so WebSocket fan-out keeps working in
`RUNS_BACKEND=redis` mode.

**Backend.**

- `pycaret_server.runs.pubsub.publish_to_redis(run_id, event)` — the
  worker calls this right after the in-process publish; events land on
  `pycaret:events:run:<id>` Redis channels.
- `pycaret_server.runs.pubsub.ensure_subscribed(run_id)` — the API's
  WebSocket handler calls this on connect; a background task reads
  from Redis and re-publishes into the local broker so existing
  WebSocket clients see the events transparently. Idempotent (only
  one subscriber per run).

**Deferred to a follow-up cut.** The engine-side event widening
(`FOLD_STARTED`, `FOLD_FINISHED`, `TUNE_ITERATION`, `STACK_BASE_FITTED`
events with per-fold metrics) — the bus is in place; emitting richer
payloads is a per-plan edit on the engine side.

---

## Phase 7 — Model registry v2

**Tables.** `registered_models`, `registered_model_versions`. Plus
`deployments` gains `registered_model_id` + `registered_model_version_id`
FKs; the legacy `pipeline_id` relaxes to nullable so new Deployments
can skip it cleanly.

**Backend.** `api/registry.py`:

- `GET/POST/DELETE /workspaces/{ws}/registered-models`
- `GET /registered-models/{id}`
- `GET/POST /registered-models/{id}/versions` — promote a (trial_id, run_id)
  pair into a new immutable version; also drops a lineage edge
  (`run → registered_model_version`).
- `POST /registered-models/{id}/versions/{v}/promote` — flip status
  (`staging` / `production` / `archived`). Only one production version
  per model at a time — older prod versions auto-archive.
- `POST /registered-models/{id}/versions/{v}/rollback` — same flip,
  but reads as "set current pointer back".

Existing pipelines + deployments keep working unchanged.

**Frontend.** `registryApi` (`list`, `create`, `versions`, `promote`,
`setStatus`, `rollback`, `delete`) + types.

---

## Phase 9 — Schedule v2 / drift retrain / batch predict

**No new tables.** Phase 9 builds on the Phase 1 `jobs` table.

**Backend — new worker handlers in `pycaret_server.worker`:**

- `drift_check` — runs the monitoring evaluator for a workspace.
- `retrain` — kicks off a new Run on an existing Trial. Same code
  path the user-clicked retrain endpoint uses, with `triggered_by`
  attribution defaulting to `drift`/`schedule`.
- `batch_predict` — stub handler that records a lineage edge
  `data_source → deployment (batch_predict_input)`. The
  predictions-on-disk pipeline is the next bit of substantive work.
- `dataset_refresh` — re-runs the DataSource's driver `introspect`
  and bumps the `Dataset` version row.

Existing `scheduled_jobs` infrastructure remains; the new job kinds
queue via the Phase 1 dispatch path.

---

## Phase 10 — Monitoring v2

**Tables.** `alert_rules`, `metric_points`.

**Backend.** `api/monitoring.py`:

- `GET/POST /workspaces/{ws}/alert-rules`
- `PATCH /alert-rules/{id}`, `DELETE /alert-rules/{id}`
- `GET /deployments/{id}/metrics` — time-series read-out, defaults to
  last hour, optional `metric` + `since_seconds` + `limit`.
- `POST /deployments/{id}/metrics` — ingest. Workers, recorders, and
  the future serving-side latency instrument all hit this.
- `evaluate_rules_for_workspace(db, ws_id)` — pure function, walked by
  the `drift_check` worker handler. Aggregates points over each rule's
  window; returns rules that fired so the caller delivers to Slack /
  email / webhook destinations.

**Destinations.** `slack` (webhook URL), `email` (`{"to": [...]}`),
`webhook` (generic JSON POST). The shapes are in place; SMTP / Slack
HTTPS delivery is a one-function add per destination.

**Frontend.** `monitoringApi` (rules CRUD + metric read/ingest) +
types.

---

## Phase 12 — Governance basics

**Tables.** `approval_workflows`.

**Backend.** `api/governance.py`:

- `GET/POST /workspaces/{ws}/approvals`
- `POST /approvals/{id}/approve`, `/reject`, `/execute`

Lifecycle: a user opens an Approval with target+action+payload;
approvers `POST .../approve` until `len(approvals) >= required`;
the requester calls `execute` and a tiny dispatch table runs the
gated action. v1 wires
`(registered_model_version, promote_to_production)` so the registry
endpoint stays unguarded for solo dev but the workflow exists for
multi-user installs.

**Frontend.** `governanceApi` (`list`, `open`, `approve`, `reject`,
`execute`) + types.

---

## Phase 14 — Distributed / GPU workers

**No new tables.**

**Backend.**

- `dispatch_run` picks the queue from `experiment.setup_params.queue` /
  `plan_params.queue` (falls back to `default`). Validates against the
  known set `default | cpu-heavy | gpu | inference` so a typo never
  strands a Run.
- The worker entrypoint (`pycaret-server worker --queues gpu`) already
  accepts a comma-separated list — only listens on its declared
  queues. Phase 1 work already laid this groundwork.
- `api/queue_admin.py`:
  - `GET /admin/queues` — per-queue counts across statuses +
    `recent_throughput_1h`.
  - `GET /admin/workers` — workers currently holding a Job lock.

**Frontend.** `queueAdminApi` + `QueueRow` / `WorkerRow` types.

---

## What's wired into `app.py`

New routers registered: `connections`, `git_repos`, `registry`,
`monitoring`, `governance`, `queue_admin`. All under `/api/v1/`.

## Migration

One head: `b2c3d4e5f6a8` (chains `f0a1b2c3d4e5 → a1b2c3d4e5f7 → b2c3d4e5f6a8`).
On a fresh dev DB:

```powershell
pycaret-server migrate --reset-dev
```

Bootstrap detector recognises the new tables and stamps head when an
existing DB already has them.

---

## Why phases 8, 11, 13 didn't ship

Each one is genuinely a separate session's worth of work, and trying
to land them in this one would mean either a broken build or
load-bearing stubs that can't actually run.

### Phase 8 — Notebook runtime

Needs:
- A "Notebook Manager" service that spawns a JupyterLab Docker
  container per session (resource limits, mount data dir as `/data`,
  inject workspace secrets as env vars).
- `notebooks` + `notebook_sessions` tables.
- An idle-shutdown reaper.
- A `pycaret` Python client preinstalled in the kernel image
  (so `pycaret_client.runs.create(...)` works inside notebooks).
- An iframe-friendly UI surface.

Container lifecycle is non-trivial — single-host-only, then K8s pod
spec, then resource quotas. Tackling it inside this session means
writing 1500+ lines that nobody can test until the container infra
lands. Carved out for its own session.

### Phase 11 — Statistical computing v1

Needs:
- `analyses` table with a kind discriminator.
- A *library* of typed procedures: t-test variants, ANOVA (1-way,
  2-way), Kruskal–Wallis, χ², Fisher's exact, Cramér's V, OLS with
  full diagnostics (residual plots, Q-Q, leverage, Durbin–Watson,
  VIF, Cook's distance), Kaplan–Meier + log-rank + Cox PH, ARIMA +
  Prophet wrappers.
- A uniform result envelope (test statistic, p-value, effect size,
  CI, plain-English interpretation, plot).
- New "New analysis" UI page per kind.

This is the SAS-parity story — meaningful breadth is the whole point.
Five hours of fast typing produces five half-baked procedures; a
session's worth of focus produces the full breadth done well.

### Phase 13 — Enterprise deployment polish

Needs:
- Helm chart with values for every service (api, worker, postgres,
  redis, minio, web), TLS via cert-manager, ingress, secrets via
  external-secrets-operator.
- Airgapped install bundle: `docker save` every image, `pip download`
  every wheel, package, write the install script, test it.
- `pycaret-server init` interactive bootstrap.
- INSTALL.md / OPERATIONS.md covering: TLS, backup/restore (Postgres
  dump + object-storage rsync), upgrade runbook, observability
  recommendations, perf tuning, scaling guidance.

Polish phase by definition — depends on everything above being stable,
which is what makes it the LAST phase. Lots of writing, lots of
testing, very little code.

---

## Smoke test order in the morning

```powershell
# 1. Apply the new schema cleanly.
pycaret-server migrate --reset-dev

# 2. Sanity check.
pycaret-server doctor

# 3. Existing Phase 0 tests stay green.
pytest services/api/tests/test_session28_phase0.py -v

# 4. New phases haven't broken anything else.
pytest services/api/tests/ -v

# 5. Frontend type check.
cd apps/web; npm run typecheck

# 6. (Optional) Full prod-shaped stack.
docker compose -f infra/docker/docker-compose.prod.yml up --build
```

All new endpoints follow the existing auth + workspace-scoping
patterns, so a token from the existing login flow will hit every
new surface without ceremony.
