# PyCaret Platform — install guide

Three supported install paths, in increasing order of operational
complexity:

1. **Single-process dev** — one-command, SQLite + local-fs, no Redis.
2. **Docker Compose** — full stack on one host (Postgres + Redis + MinIO).
3. **Kubernetes via Helm** — production self-host.

All three serve the same UI and expose the same API. Pick by audience.

---

## 1. Single-process dev (10 seconds)

```powershell
pip install pycaret-server
pycaret-server init             # writes ./data/.env + applies migrations
pycaret-server serve            # API at http://localhost:8020
```

Open <http://localhost:3020> for the UI (run `npm run dev` from
`apps/web` for live reload, or use the bundled web image).

- **DB**: SQLite at `./data/pycaret.db`.
- **Artifacts**: `./data/artifacts/`.
- **Queue**: in-process; no Redis required.
- **Object store**: local-fs; URIs stored as `file:///…`.

Reset to a clean state with:

```powershell
pycaret-server migrate --reset-dev
```

---

## 2. Docker Compose (production-shaped, one host)

```powershell
docker compose -f infra/docker/docker-compose.prod.yml up --build
```

Brings up:

| service     | port  | role |
|-------------|-------|------|
| `web`       | 3020  | React UI |
| `api`       | 8020  | FastAPI backend |
| `worker`    | —     | Phase 1 job worker |
| `postgres`  | —     | Phase 3 production DB |
| `redis`     | —     | Phase 1 + 6 queue + pub/sub |
| `minio`     | 9000 + 9001 | Phase 2 object store + admin UI |

Defaults to dev credentials (`minioadmin` / `minioadmin`,
`postgres:pycaret-dev-only`). For anything real, set:

```powershell
$env:PYCARET_PG_PASSWORD = "…"
$env:PYCARET_JWT_SECRET  = "…"
$env:PYCARET_SECRETS_KEY = "…"           # Fernet key
$env:PYCARET_S3_ACCESS_KEY = "…"
$env:PYCARET_S3_SECRET_KEY = "…"
docker compose -f infra/docker/docker-compose.prod.yml up
```

Volumes (`postgres-data`, `redis-data`, `minio-data`) persist across
`docker compose down`; `docker compose down --volumes` wipes them.

---

## 3. Kubernetes via Helm

Prereqs: Kubernetes 1.27+, Helm 3.12+, an ingress controller, a
cert-manager `ClusterIssuer` (default config assumes `letsencrypt-prod`),
and a default StorageClass.

```powershell
# Create the required secrets up front.
kubectl create namespace pycaret
kubectl -n pycaret create secret generic pycaret-jwt --from-literal=secret="$(openssl rand -base64 48)"
kubectl -n pycaret create secret generic pycaret-encryption --from-literal=key="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
kubectl -n pycaret create secret generic pycaret-postgres --from-literal=password="$(openssl rand -base64 24)"
kubectl -n pycaret create secret generic pycaret-minio --from-literal=password="$(openssl rand -base64 24)"

# Install the chart.
helm install pycaret ./infra/helm/pycaret \
    --namespace pycaret \
    --set global.domain=pycaret.example.com
```

After install:

- API at `https://pycaret.example.com/api/v1/healthz`.
- UI at `https://pycaret.example.com/`.
- MinIO console is **not** exposed by default; port-forward for admin:
  `kubectl -n pycaret port-forward svc/pycaret-pycaret-minio 9001:9001`.

### Bring-your-own dependencies

Point at a managed Postgres / Redis / S3:

```powershell
helm install pycaret ./infra/helm/pycaret -n pycaret \
    --set postgres.enabled=false \
    --set externalPostgres.url=postgresql+psycopg://user:pass@host:5432/db \
    --set redis.enabled=false \
    --set externalRedis.url=redis://host:6379/0 \
    --set minio.enabled=false \
    --set externalS3.endpoint=https://s3.amazonaws.com \
    --set externalS3.bucket=my-pycaret-bucket
```

### GPU worker pool (Phase 14)

Spin up a second `worker` Deployment listening on the `gpu` queue:

```powershell
helm install pycaret-gpu ./infra/helm/pycaret -n pycaret \
    --set api.replicaCount=0 \
    --set web.replicaCount=0 \
    --set postgres.enabled=false \
    --set redis.enabled=false \
    --set minio.enabled=false \
    --set worker.replicaCount=2 \
    --set worker.queues=gpu \
    --set worker.resources.limits."nvidia\.com/gpu"=1
```

Submit a Run with `setup_params.queue=gpu` and the GPU workers pick it
up while the default-queue workers stay idle.

---

## Airgapped install

A future cut ships a tarball with every image preloaded + a wheelhouse
of every Python wheel. Until then, use the standard
`docker save <image> | gzip > image.tar.gz` flow + `pip download` of
the `pycaret-server` requirements.

---

## What's next

- [`OPERATIONS.md`](OPERATIONS.md) — backup, upgrade, observability.
- [`docs/revamp/PHASES.md`](docs/revamp/PHASES.md) — the phase roadmap.
