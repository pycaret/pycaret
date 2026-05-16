# PyCaret Platform — operations guide

How to run, observe, back up, upgrade, and tune a real deploy.

---

## Backups

The platform is two pieces of stateful data: the **DB** (Postgres) and
the **object store** (S3 / MinIO). Backing up one without the other
leaves you with metadata pointing at missing artifacts (or vice
versa) — always back up both.

### Postgres

```powershell
# Compose stack:
docker exec pycaret-postgres pg_dump -U pycaret pycaret \
  | gzip > pycaret-$(Get-Date -Format yyyyMMdd-HHmmss).sql.gz

# Kubernetes:
kubectl -n pycaret exec sts/pycaret-pycaret-postgres -- pg_dump -U pycaret pycaret \
  | gzip > pycaret-$(Get-Date -Format yyyyMMdd-HHmmss).sql.gz
```

Restore is `pg_restore` against a fresh DB. Test restores quarterly.

### Object store (MinIO / S3)

`mc mirror` is the easiest path:

```powershell
mc alias set source http://minio:9000 $env:S3_ACCESS_KEY $env:S3_SECRET_KEY
mc mirror source/pycaret-artifacts ./backup/artifacts
```

For real S3, `aws s3 sync` is equivalent. Schedule via cron / a
Kubernetes `CronJob`; retain at least the last 14 daily + 6 monthly.

### Coordinating the two

Take the DB dump **first**, then the object-store mirror. The DB row
might reference an artifact that hasn't synced yet — re-running the
sync the next day backfills. The reverse (mirror first → dump second)
would point the dump at artifacts that weren't captured.

---

## Upgrade runbook

1. **Read the release notes** — every PyCaret release calls out
   migrations + breaking changes in `docs/revamp/PHASE-*-NOTES.md`.
2. **Snapshot the DB + object store** before touching anything.
3. **Pull the new image** (`docker pull` / Helm `--set global.imageTag=…`).
4. **Apply migrations** before bringing the API back up:
   ```powershell
   docker exec pycaret-api pycaret-server migrate
   ```
   (Or `kubectl exec` against an api pod.) Alembic is forward-compatible
   — applying migrations against a still-running old API is safe
   *for additive changes*; for breaking schema flips (the Phase 0
   pivot is the example), the API must be down.
5. **Roll the API**. Drain the worker first (let in-flight Runs finish)
   then roll the worker.
6. **Smoke test** — `pycaret-server doctor` from one of the pods.

A bad release: roll the image back, then `migrate --revision <previous>`
to undo the schema changes (every migration has a `downgrade`).

---

## Observability

### Health endpoints

- `GET /healthz` — liveness. Returns `200` if the process is up.
- `GET /readyz` — readiness (planned). Returns `200` only once the DB
  + Redis + storage are reachable.
- `pycaret-server doctor` — CLI variant of the above; useful in
  scripts and Kubernetes `initContainers`.

### Logs

The API and worker emit structured logs to stderr. The default level
is `INFO`; set `PYCARET_DEBUG=true` to bump it to `DEBUG` for one
session. Pipe to Loki / CloudWatch / Datadog via the container
runtime's logging driver.

### Metrics

- `GET /admin/queues` — per-queue depth + 1-hour throughput.
- `GET /admin/workers` — workers currently holding a Job lock.
- `GET /deployments/{id}/metrics?metric=p95_latency_ms` — per-
  deployment time-series.

A future cut exposes the lot via Prometheus `/metrics`; for now, scrape
the admin endpoints from a custom exporter or use the platform's
built-in alert rules + Slack/email destinations.

---

## Scaling

| Resource              | When to scale                                          | How |
|-----------------------|--------------------------------------------------------|-----|
| API replicas          | p95 latency > 200ms                                    | `helm upgrade … --set api.replicaCount=N` |
| Worker (default queue)| `queued` jobs accumulating                             | bump `worker.replicaCount` |
| Worker (gpu queue)    | tune jobs taking too long                              | dedicated Helm release with `worker.queues=gpu` |
| Postgres              | CPU > 70% sustained                                    | bump instance class on managed Postgres; vertical first, replicas later |
| Redis                 | high pub/sub volume + queue depth                      | rare — single-instance Redis handles >10k events/s |
| MinIO / S3            | storage > 80% of allocated PVC                         | bump PVC; S3 scales itself |

### Queue separation (Phase 14)

Run dedicated worker deployments per class:

- **`default`** — the catch-all. Compare / create / search jobs.
- **`cpu-heavy`** — long tuning runs that would starve the default
  pool. Set `setup_params.queue=cpu-heavy` on the experiment.
- **`gpu`** — nodes with `nvidia.com/gpu` advertised. Set
  `setup_params.queue=gpu` plus an `nvidia.com/gpu=1` resource limit
  on the worker deployment.
- **`inference`** — light, latency-sensitive predict-only jobs. Keep
  the replica count low and the CPU request high so latency stays
  predictable.

---

## Security

- **JWT secret** — 48+ bytes of randomness. Rotate annually; bump the
  refresh-token TTL to "1d" temporarily during rotation so existing
  sessions don't break.
- **Secrets-key (Fernet)** — encrypts LLM API keys + Phase 4 secrets
  + Phase 5 PATs at rest. Rotation re-encrypts every Secret row on
  startup; budget a few minutes of downtime per 100k secrets.
- **JupyterLab tokens** — generated per-session, never reused. The
  iframe URL is the only place the token surfaces; the platform never
  stores it cleartext beyond the session row.
- **Approval workflows (Phase 12)** — wire `promote_to_production`
  through approvals for any deploy that matters. Default workflows
  ship empty (single-signature self-approve) so they don't block
  solo-dev installs.

---

## Troubleshooting

| Symptom                                                | Likely cause | Fix |
|--------------------------------------------------------|-----|-----|
| `pycaret-server doctor` reports DB FAIL                | wrong `PYCARET_DATABASE_URL` or DB down | check the URL in `.env` / values.yaml |
| Worker stays at 0 throughput                           | wrong queue list                  | confirm `--queues` matches the queue the dispatcher set on the Job |
| WebSocket fan-out silent in Redis mode                 | API can't reach Redis             | check `PYCARET_REDIS_URL` from inside the api pod |
| Promote endpoint 409s                                  | trial already has a Pipeline      | un-promote first, or use the registry v2 versions endpoint |
| Notebook session iframe is blank                       | `notebook_backend=local`          | switch to `docker` and restart the API; check `docker ps` for the spawned container |
| Stats procedure errors "no numeric values"             | column was a string, e.g. "1,234" | pre-clean in a Connection query or DataSource transform |

---

## What's next

- [`INSTALL.md`](INSTALL.md) — first-time setup.
- [`docs/revamp/PHASES.md`](docs/revamp/PHASES.md) — roadmap.
