# PyCaret Control Plane — 5-minute quickstart

Get a working full-stack Control Plane running locally in under 5 minutes.

> **Status:** session 13. Backend is feature-complete (auth, workspaces, projects, experiments, runs, data sources, deployments, in-house serving, run cancellation, Alembic migrations). Frontend covers the bootstrap → workspace/project flow; experiment / run / deployment / admin screens land in session 14+.

## Option A — Local dev (preferred during development)

```bash
git clone -b v4 https://github.com/pycaret/pycaret.git
cd pycaret

# 1. Install Python 3.13 + all Python workspace packages (engine + api)
uv python install 3.13
uv sync --all-packages --all-extras

# 2. Terminal 1 — backend
uv run --package pycaret-server pycaret-server serve --reload
#   → http://127.0.0.1:8000

# 3. Terminal 2 — frontend
cd apps/web
npm install
npm run dev
#   → http://127.0.0.1:3000  (proxies /api → :8000)
```

Open **http://127.0.0.1:3000/setup** to run the first-run wizard.

You can also skip the UI and hit the API directly:
- `http://127.0.0.1:8000/docs` — interactive Swagger UI.
- `http://127.0.0.1:8000/openapi.json` — machine-readable OpenAPI schema.
- `http://127.0.0.1:8000/healthz` — liveness probe.

## Option B — Docker compose (full stack, one command)

```bash
git clone -b v4 https://github.com/pycaret/pycaret.git
cd pycaret
docker compose -f infra/docker/docker-compose.yml up --build
```

Open **http://localhost:3000**. The web container fronts the API (same origin), so `/api/v1/*` and `/ws/*` are reverse-proxied. SQLite DB + artifacts persist to `./data/` on the host.

## First-run flow

In the UI:

1. Navigate to http://127.0.0.1:3000/setup.
2. Fill the bootstrap form: admin email, password (min 8), display name, workspace name.
3. Click **Create workspace** → redirected to `/` with a live session.
4. Click **New workspace** on the right panel to add another.
5. Click any workspace → land on `/workspaces/:id` → **New project**.

Or via the API (`http://localhost:8000/docs`):

1. `GET /api/v1/setup/status` → `{"is_bootstrapped": false, ...}`.
2. `POST /api/v1/setup/bootstrap`:

   ```json
   {
     "email": "admin@example.com",
     "password": "supersecret",
     "display_name": "Admin",
     "workspace_name": "My Workspace"
   }
   ```

3. Returns an `access_token` + `refresh_token` pair. Copy the access token.
4. Use Swagger "Authorize" → paste the token. Every protected route now works.

## End-to-end demo: CSV upload → AutoML run → deploy → predict

```bash
export TOKEN=$(curl -sX POST http://localhost:8000/api/v1/setup/bootstrap \
  -H 'content-type: application/json' \
  -d '{"email":"me@x","password":"supersecret","workspace_name":"Demo"}' | jq -r .access_token)

export WS=$(curl -sH "authorization: bearer $TOKEN" http://localhost:8000/api/v1/workspaces | jq -r '.[0].id')

export PROJ=$(curl -sX POST "http://localhost:8000/api/v1/workspaces/$WS/projects" \
  -H "authorization: bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"name":"Iris"}' | jq -r .id)

export EXP=$(curl -sX POST "http://localhost:8000/api/v1/projects/$PROJ/experiments" \
  -H "authorization: bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"name":"baseline","task":"classification","target":"target",
       "setup_params":{"session_id":42,"fold":2,"verbose":false}}' | jq -r .id)

# Fire a run using the built-in sklearn iris dataset (no CSV needed)
export RUN=$(curl -sX POST "http://localhost:8000/api/v1/experiments/$EXP/runs" \
  -H "authorization: bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"plan":"create","model_id":"lr","sklearn_dataset":"iris"}' | jq -r .id)

# Block until done
curl -sX POST "http://localhost:8000/api/v1/runs/$RUN/wait?timeout_s=120" \
  -H "authorization: bearer $TOKEN" | jq '.status'

# Promote the fitted pipeline to the workspace registry
export PIPE=$(curl -sX POST "http://localhost:8000/api/v1/runs/$RUN/promote" \
  -H "authorization: bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"name":"iris-v1"}' | jq -r .id)

# Deploy it behind a slug
curl -sX POST "http://localhost:8000/api/v1/pipelines/$PIPE/deployments" \
  -H "authorization: bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"endpoint_slug":"iris-v1"}'

# Serve a prediction
curl -sX POST "http://localhost:8000/api/v1/deployments/iris-v1/predict" \
  -H "authorization: bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"rows":[{"sepal length (cm)":5.1,"sepal width (cm)":3.5,
                 "petal length (cm)":1.4,"petal width (cm)":0.2}]}'
```

## Run the tests

```bash
# engine (32)
uv run pytest packages/engine/tests/ -q

# backend (30)
uv run --package pycaret-server pytest services/api/tests/ -q

# web (6)
cd apps/web && npm test
```

Total: **68/68** green.

## Config (env vars with `PYCARET_` prefix)

Override any setting via env or a `.env` file at the repo root:

| Env | Default | Purpose |
|---|---|---|
| `PYCARET_DATABASE_URL` | `sqlite:///./pycaret.db` | SQLAlchemy URL. Postgres, MySQL supported. |
| `PYCARET_JWT_SECRET` | **dev-fallback, override in prod** | JWT HMAC secret. |
| `PYCARET_ACCESS_TOKEN_TTL_MINUTES` | 60 | Access-token lifetime. |
| `PYCARET_REFRESH_TOKEN_TTL_DAYS` | 30 | Refresh-token lifetime. |
| `PYCARET_ARTIFACT_DIR` | `./artifacts` | Where run artifacts + CSV uploads land. |
| `PYCARET_CORS_ORIGINS` | `["http://localhost:3000"]` | CORS allowlist for the web UI. |
| `PYCARET_DEBUG` | `false` | Verbose logging + SQL echo. |

## Migrations

Fresh DB on local dev is auto-migrated when `PYCARET_DATABASE_URL` starts with `sqlite:`. For Postgres or any prod deploy, run Alembic explicitly before starting the server:

```bash
uv run --package pycaret-server pycaret-server migrate
# or pin a revision
uv run --package pycaret-server pycaret-server migrate --revision head
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 invalid access token` after 1h | Access token expired | UI refreshes automatically; for raw API use `POST /api/v1/auth/refresh` |
| `409 instance already bootstrapped` | Trying to bootstrap twice | Delete `./pycaret.db` (or `./data/pycaret.db`) to reset |
| `500` on `/describe/models` | Engine can't find a task | Task must be one of `classification`, `regression`, `clustering`, `anomaly`, `time_series` |
| ImportError on `uvicorn` launch | Workspace not synced | `uv sync --all-packages --all-extras` |
| Alembic `CommandError: Path doesn't exist` | CWD mismatch in a custom runner | Fixed in session 11 — `_run_alembic` resolves the script path absolutely |

## References

- [`docs/revamp/VISION.md`](VISION.md) — 1-page product statement.
- [`docs/revamp/CONTROL_PLANE_SPEC.md`](CONTROL_PLANE_SPEC.md) — full technical spec.
- [`docs/revamp/ARCHITECTURE.md`](ARCHITECTURE.md) — system architecture.
- [`docs/revamp/ROADMAP.md`](ROADMAP.md) — MVP 1–4 / V2 / V3 phase breakdown.
- [`docs/revamp/DECISIONS.md`](DECISIONS.md) — ADR log.
