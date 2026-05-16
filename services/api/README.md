# pycaret-server

FastAPI backend for the PyCaret 4.0 application platform. Serves a typed REST + WebSocket API in front of the `pycaret` engine.

> **Status:** Alpha. This package is part of the [PyCaret 4.0 Part-2 plan](../docs/revamp/PLATFORM_PLAN.md). It depends on `pycaret >= 4.0.0a1`.

## Install (dev)

From the repo root:

```bash
uv sync --package pycaret-server --all-extras
uv run --package pycaret-server pycaret-server serve --reload
# or
uv run --package pycaret-server uvicorn pycaret_server.app:create_app --factory --reload
```

Default database: SQLite at `./pycaret.db`.
Default artifact dir: `./artifacts/`.
API docs: http://localhost:8020/docs (port 8020 keeps us out of
resumly's 8000 when both servers run on the same laptop).

## Config (env vars, prefix `PYCARET_`)

| Env | Default | Purpose |
|---|---|---|
| `PYCARET_DATABASE_URL` | `sqlite:///./pycaret.db` | SQLAlchemy URL. Also supports Postgres / MySQL |
| `PYCARET_JWT_SECRET` | *(dev fallback)* | JWT signing secret. **Override in prod.** |
| `PYCARET_ACCESS_TOKEN_TTL_MINUTES` | 60 | Access-token lifetime |
| `PYCARET_REFRESH_TOKEN_TTL_DAYS` | 30 | Refresh-token lifetime |
| `PYCARET_ARTIFACT_DIR` | `./artifacts` | Where run artifacts land |
| `PYCARET_CORS_ORIGINS` | `["http://localhost:3020"]` | Origins allowed by CORS |
| `PYCARET_DEBUG` | `false` | SQLAlchemy echo + FastAPI debug mode |

See `.env.example` in the repo root.

## First-run flow

1. Start the server — it creates SQLite tables on first boot.
2. Open http://localhost:8020/docs in the browser.
3. Hit `GET /api/v1/setup/status` — returns `{"is_bootstrapped": false}`.
4. Hit `POST /api/v1/setup/bootstrap` with an admin email / password / workspace name.
5. Use the returned access token for every subsequent call.

## Tests

```bash
uv run --package pycaret-server pytest pycaret-server/tests -v
```

## Routes

| Group | Prefix | Purpose |
|---|---|---|
| **setup** | `/api/v1/setup/*` | First-run bootstrap + status |
| **auth** | `/api/v1/auth/*` | login / refresh / logout / me |
| **describe** | `/api/v1/describe/*` | Engine introspection (proxies `pycaret.api`) |
| **workspaces** | `/api/v1/workspaces[/{id}]` | Workspace CRUD |
| **projects** | `/api/v1/workspaces/{id}/projects[/...]` | Project CRUD |
| **experiments** | `/api/v1/projects/{id}/experiments[/...]` | Experiment CRUD |
| *runs + deployments* | coming next phase | |

## Data model

14 SQLAlchemy tables — see `pycaret_server/db/models.py` and [`docs/revamp/PLATFORM_PLAN.md § 3`](../docs/revamp/PLATFORM_PLAN.md#3-data-model--workspace--project--experiment--run--pipeline--deployment).

## License

Dual-licensed: **MIT** for self-hosted / internal-enterprise use; **BSL 1.1** for hosted multi-tenant SaaS. See `../docs/revamp/DECISIONS.md` § decision 5.
