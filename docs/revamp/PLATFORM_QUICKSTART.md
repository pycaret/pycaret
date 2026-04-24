# PyCaret 4.0 platform — 5-minute quickstart

Get a working backend running locally against SQLite in under 5 minutes.

> **Status:** session 9 scaffolding. Backend is API-complete for auth + workspaces / projects / experiments + engine introspection; run-dispatch + WebSocket event fan-out + deployments land next session.

## Option A — Local dev (preferred during development)

```bash
git clone -b v4 https://github.com/pycaret/pycaret.git
cd pycaret

# Install Python 3.13 + all workspace packages (engine + server)
uv python install 3.13
uv sync --all-packages --all-extras

# Start the backend
uv run --package pycaret-server pycaret-server serve --reload
# or equivalently:
#   uv run --package pycaret-server uvicorn pycaret_server.app:create_app --factory --reload
```

Server listens on `http://127.0.0.1:8000`.

Browse:
- `http://127.0.0.1:8000/docs` — interactive Swagger UI.
- `http://127.0.0.1:8000/openapi.json` — machine-readable OpenAPI schema (the React UI will generate a typed client from this).
- `http://127.0.0.1:8000/healthz` — liveness probe.

## Option B — Docker compose (closer to prod)

```bash
git clone -b v4 https://github.com/pycaret/pycaret.git
cd pycaret
docker compose -f docker/docker-compose.yml up --build
```

Same URL (`http://localhost:8000/docs`). SQLite DB + artifacts persist to `./data/` on the host.

## First-run flow

1. Open `http://localhost:8000/docs`.
2. `GET /api/v1/setup/status` → `{"is_bootstrapped": false, "user_count": 0, ...}`.
3. `POST /api/v1/setup/bootstrap` with:

   ```json
   {
     "email": "admin@example.com",
     "password": "supersecret",
     "display_name": "Admin",
     "workspace_name": "My Workspace"
   }
   ```

   Returns an `access_token` + `refresh_token` pair. Copy the access token.

4. Use the Swagger UI "Authorize" button → paste `<access_token>`. Every protected route now works.

5. Try the flow:
   - `POST /api/v1/workspaces` → create a second workspace.
   - `POST /api/v1/workspaces/{id}/projects` → create a project.
   - `POST /api/v1/projects/{id}/experiments` with e.g. `{"name": "baseline", "task": "classification", "target": "churn"}`.
   - `GET /api/v1/describe/setup-params?task=classification` → live JSON schema a React form renders from.

## Run the tests

```bash
uv run --package pycaret-server pytest pycaret-server/tests -v
```

Should be 14/14 green. Uses an in-memory SQLite per test — no state leakage.

## Config (env vars with `PYCARET_` prefix)

Override any setting via env or a `.env` file at the repo root:

| Env | Default | Purpose |
|---|---|---|
| `PYCARET_DATABASE_URL` | `sqlite:///./pycaret.db` | SQLAlchemy URL. Postgres, MySQL supported |
| `PYCARET_JWT_SECRET` | **dev-fallback, override in prod** | JWT HMAC secret |
| `PYCARET_ACCESS_TOKEN_TTL_MINUTES` | 60 | Access-token lifetime |
| `PYCARET_REFRESH_TOKEN_TTL_DAYS` | 30 | Refresh-token lifetime |
| `PYCARET_ARTIFACT_DIR` | `./artifacts` | Where run artifacts land |
| `PYCARET_CORS_ORIGINS` | `["http://localhost:3000"]` | CORS allowlist for the React UI |
| `PYCARET_DEBUG` | `false` | Verbose logging + SQL echo |

## What's in this scaffolding (session 9)

Implemented:

- **Config** (`pycaret_server/config.py`) — pydantic-settings, env-driven.
- **Database** (`pycaret_server/db/`) — 14 SQLAlchemy models, session factory, FastAPI `get_db` dependency.
- **Auth** (`pycaret_server/auth/`) — bcrypt password hashing, JWT access + rotating refresh tokens, `CurrentUser` dependency.
- **Routes** (`pycaret_server/api/`) — setup (bootstrap + status), auth (login / refresh / logout / me), describe (engine introspection proxy), workspaces CRUD, projects CRUD, experiments CRUD.
- **App factory** (`pycaret_server/app.py`) — FastAPI application with CORS + lifespan that auto-creates SQLite tables on first boot.
- **CLI** (`pycaret-server serve` command).
- **Docker** — multi-stage `Dockerfile.api` + dev `docker-compose.yml`.
- **Tests** — 14 integration tests exercising every route.

Coming next session:

- **Runs** (`POST /api/v1/experiments/{id}/runs`) — dispatches to a background worker that loads the configured data source, constructs the engine's `Experiment` class, runs `compare_models`, captures events + metrics + artifacts into the database.
- **WebSocket** (`GET /ws/runs/{id}/events`) — subscribes to the engine's `BaseLogger` event stream and fans out to connected UIs.
- **Deployments** (in-house serving per `PLATFORM_PLAN.md` § decision 4) — `POST /api/v1/pipelines/{id}/deploy`, catch-all `POST /api/v1/deployments/{slug}/predict`.
- **Data-source connectors** — CSV upload + S3 + Postgres readers per `PLATFORM_PLAN.md` § decision 2.
- **Alembic migrations** — replacing the boot-time `create_all` fallback.

## Hitting the API with `curl`

```bash
# 1. Bootstrap
curl -s -X POST http://localhost:8000/api/v1/setup/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"supersecret","workspace_name":"Demo"}' | tee tokens.json

# 2. Stash the access token
export TOKEN=$(python -c "import json,sys; print(json.load(open('tokens.json'))['access_token'])")

# 3. List workspaces
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/workspaces

# 4. Introspect the classification setup schema (no auth needed)
curl -s "http://localhost:8000/api/v1/describe/setup-params?task=classification" | python -m json.tool
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 invalid access token` after 1h | Access token expired | `POST /api/v1/auth/refresh` with the refresh token |
| `409 instance already bootstrapped` | Trying to bootstrap twice | Delete `./pycaret.db` (or `./data/pycaret.db`) to reset |
| `500` on `/describe/models` | Engine can't find a task | Task parameter must be one of `classification`, `regression`, `clustering`, `anomaly`, `time_series` |
| ImportError on `uvicorn` launch | Workspace not synced | `uv sync --all-packages --all-extras` |

## References

- Design: `docs/revamp/PLATFORM_PLAN.md`
- Decisions: `docs/revamp/DECISIONS.md` § session 6 entries for the 6 platform-design calls
- Roadmap: `docs/revamp/ROADMAP.md` Part 2 phases 7-12
