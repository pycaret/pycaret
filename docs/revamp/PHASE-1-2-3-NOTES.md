# Phases 1, 2, 3 — implementation notes (session 29)

Scaffolding for the next three platform phases. Each phase ships its
foundation so the rest of the work (the heavyweight changes — worker
runtime, S3 migration of existing artifacts, dual-engine CI) can land
incrementally without another schema flip.

---

## Phase 1 — Queue + worker separation

**What landed**

- `jobs` table (migration `a1b2c3d4e5f7`). Generic shape so future Job
  kinds (`drift_check`, `batch_predict`, `dataset_refresh`) don't need
  a schema change.
- `PYCARET_RUNS_BACKEND=inprocess|redis` env toggle.
  - `inprocess` (default): existing `ThreadPoolExecutor` runs Jobs in the
    API process. Every dispatch still writes a Job row as a passive
    audit trail.
  - `redis`: backend `LPUSH`es to `pycaret:queue:<name>`; the
    `pycaret-server worker` process `BRPOP`s, atomically claims the Job
    via `locked_by`, executes, marks terminal.
- `pycaret-server worker` CLI subcommand + `pycaret_server.worker`
  module with a registry-based handler dispatcher (`register_handler`).
  Phase 1 ships the `run` handler; Phase 9 adds more.
- `pycaret-server doctor` health check (DB / Redis / storage).
- Bootstrap detector updated; head revision is now `a1b2c3d4e5f7`.

**What's deferred to a follow-up cut**

- Visible retry/back-off semantics on the dashboard. The `attempts`
  column ticks up but there's no UI for it yet.
- `RUNS_BACKEND=redis` end-to-end smoke. Code path exists, falls back
  to in-process on enqueue failure so it's safe to leave behind.

**How to try it**

```bash
# In-process (default — no change vs before).
pycaret-server serve

# Redis mode (needs redis-py + a running Redis):
pip install redis
docker run -d -p 6379:6379 redis:7-alpine
PYCARET_RUNS_BACKEND=redis pycaret-server serve &
PYCARET_RUNS_BACKEND=redis pycaret-server worker
```

---

## Phase 2 — Object storage abstraction

**What landed**

- `pycaret_server.storage` package: `ObjectStore` protocol + factory.
- `LocalFsObjectStore` driver (default). Writes pickles under
  `settings.artifact_dir`; returns `file://` URIs. Refuses key traversal
  (`../../etc/passwd` → `ObjectStoreError`). Accepts bare absolute
  paths for back-compat with pre-Phase-2 DB rows.
- `S3ObjectStore` driver. Same protocol, talks to AWS S3 or MinIO via
  `endpoint_url`. `boto3` is an optional extra (`pycaret-server[s3]`).
- Orchestrator's `_save_pipeline` + `_save_trial_artifact` write through
  the factory. `Run.stored_path` is now a URI, not an absolute path.
- `api/runs.py` download endpoint resolves URIs:
  - `file://` → `FileResponse` (auth + range-request behavior preserved).
  - `s3://` → 302 to a presigned URL.
- `api/plots.py` `_load_pipeline_artifact` accepts URIs and downloads
  cloud objects to a tempfile before handing to `joblib.load`.
- Unit tests (`tests/test_storage.py`) cover put/get round-trip,
  traversal refusal, bare-path back-compat, factory selection.

**What's deferred**

- One-time migration script that rewrites existing bare absolute paths
  to `file://` URIs in the DB. The driver tolerates both shapes
  indefinitely, so this can wait.
- S3 driver tested against a mocked boto3 — needs `moto` in test extras.
- Pre-signed upload URLs for client-side dataset uploads.

**How to try it**

```bash
# Local (default).
pycaret-server serve  # writes to ./artifacts, URIs are file://

# MinIO.
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address :9001
PYCARET_STORAGE_BACKEND=minio \
PYCARET_STORAGE_ENDPOINT_URL=http://localhost:9000 \
PYCARET_STORAGE_BUCKET=pycaret-artifacts \
PYCARET_STORAGE_ACCESS_KEY=minioadmin \
PYCARET_STORAGE_SECRET_KEY=minioadmin \
pycaret-server serve
```

---

## Phase 3 — Postgres-first production path

**What landed**

- `db/session.py` is now dual-backend-aware.
  - SQLite: `check_same_thread=False`, default pool. Zero-config dev.
  - Postgres: `pool_size=5`, `max_overflow=10`, `pool_recycle=1800`,
    `pool_pre_ping=True`.
- `infra/docker/docker-compose.prod.yml` — production-shaped stack:
  `postgres`, `redis`, `minio`, `minio-bootstrap`, `api`, `worker`, `web`.
- All env vars wired (`PYCARET_DATABASE_URL`, `PYCARET_RUNS_BACKEND=redis`,
  `PYCARET_STORAGE_BACKEND=minio`).

**What's deferred**

- Dual-engine CI matrix (SQLite + Postgres). Backend code is now
  driver-agnostic; the GitHub Actions matrix can land independently.
- SQLite → Postgres migration script (export + reimport with engine-
  agnostic dumps). Phase 13's enterprise polish slice will own this.

**How to try it**

```bash
# Whole prod stack with one command.
docker compose -f infra/docker/docker-compose.prod.yml up --build
# Open http://localhost:3020. MinIO at http://localhost:9001.
```

---

## Why phases 4–14 didn't ship tonight

Each remaining phase is realistically multi-week professional engineering
work, and rushing them through one session means broken builds and
half-wired features. The roadmap below scopes the next session's slice.

- **Phase 4 (Datasource entity model)** — 4 new tables, driver protocol,
  Postgres + CSV drivers, lineage capture, "New datasource" wizard +
  dataset profile UI. ~2–3 weeks.
- **Phase 5 (Git integration v1)** — PAT/app credentials, repo-mapping
  entity, YAML export on every state change, push-status indicator,
  repo browser UI. ~1–2 weeks.
- **Phase 6 (Realtime callbacks v2)** — engine event surface widening,
  Redis pub/sub fanout, live optimization charts on the running card.
  Depends on Phase 1 finishing (Redis). ~1–2 weeks.
- **Phase 7 (Model registry v2)** — `RegisteredModel` +
  `RegisteredModelVersion` tables, Deployment FK swap, promotion +
  rollback flows, registry page. ~2–3 weeks.
- **Phase 8 (Notebook runtime)** — Jupyter container manager, per-session
  isolation, idle shutdown, iframe UI, `pycaret_client` SDK. ~3–4 weeks.
- **Phase 9 (Schedule v2 + drift retrain + batch predict)** — three new
  Job kinds, schedule editor, drift threshold rules, batch-predict
  pipeline. ~2 weeks.
- **Phase 10 (Monitoring v2)** — TimescaleDB or roll-up tables, alert
  rules, Slack/email destinations. ~2 weeks.
- **Phase 11 (Statistical computing v1)** — Analysis entity, library of
  procedures (t-test, ANOVA, χ², regression diagnostics, Kaplan-Meier,
  ARIMA/Prophet), result UI with interpretation. ~3–4 weeks.
- **Phase 12 (Governance basics)** — RBAC v2, approval workflows,
  lineage graph viewer, audit-log UI. ~2 weeks.
- **Phase 13 (Enterprise deployment polish)** — Helm chart, airgap bundle,
  `pycaret-server init`, INSTALL/OPERATIONS docs. ~1–2 weeks.
- **Phase 14 (Distributed / GPU workers)** — queue class routing,
  worker class flag, per-class admin UI. ~1 week (builds on Phase 1).

Each one has its own success criteria in `PHASES.md`. The next-session
order is: Phase 4 first (data-side foundation), then Phase 5 (Git is
high-value for the OSS audience and small enough to ship cleanly), then
Phase 7 (model registry v2 — the unblocking of governance + rollback).
