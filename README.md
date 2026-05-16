<div align="center">

<img src="docs/images/logo.png" alt="PyCaret" width="200"/>

# PyCaret — open-source self-hosted ML platform

### The engine, the control plane, and the UI — all in one box.

[![CI](https://github.com/pycaret/pycaret/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/pycaret/pycaret/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/pycaret/pycaret)
[![License](https://img.shields.io/pypi/l/pycaret.svg)](https://github.com/pycaret/pycaret/blob/main/LICENSE)

[Vision](docs/revamp/VISION.md) ·
[Architecture](docs/revamp/PLATFORM_ARCHITECTURE.md) ·
[Roadmap](docs/revamp/ROADMAP.md) ·
[Spec](docs/revamp/CONTROL_PLANE_SPEC.md) ·
[Agent guide](AGENTS.md)

</div>

---

> ## ⚠ 4.0 is **work in progress** — you're on `main` (the 4.0 line)
>
> PyCaret 4.0 is a ground-up architectural revamp. The 3.x line is frozen on PyPI as `pycaret 3.4.0` — no further commits.
>
> Track progress in [`docs/revamp/STATUS.md`](docs/revamp/STATUS.md) and [`docs/revamp/ROADMAP.md`](docs/revamp/ROADMAP.md).

---

## What you get

A self-hosted ML platform you can run on your laptop in 5 minutes:

- **Engine** — sklearn-1.7-based AutoML library (`pycaret` on PyPI).
- **Control plane** — FastAPI backend with workspaces, projects, experiments, runs, model registry, deployments, approvals, monitoring, drift, lineage, webhooks, schedules, LLM-advisory copilots.
- **Web UI** — React + Vite, dark-mode-first. Configure everything via point-and-click; nothing requires editing YAML.
- **One-command local install** — `docker compose up` and the whole thing is running locally on `http://localhost:3020`.

### How it's built today

Today's `docker compose` ships a **deliberately compact** two-container shape: one for the nginx-served React bundle, one for a FastAPI process that ALSO runs the in-process scheduler, the in-process compute, and a local SQLite file (persisted to a Docker volume) — the same single-binary pattern Plausible / Vaultwarden / n8n use for self-hosters. Great for laptops, fine for a small-team production deploy.

The architecture is designed to split apart cleanly: every external dependency (storage, DB, secrets, auth, queue, compute) is meant to sit behind a `Protocol` with multiple implementations, so the same codebase can target separate api / worker / runtime containers backed by RDS / S3 / SecretsManager / SQS / Fargate when you outgrow the single-binary shape. Most of those `Protocol` extractions are still ahead of us — see [PLATFORM_ARCHITECTURE.md](docs/revamp/PLATFORM_ARCHITECTURE.md) for the gap matrix and the phasing plan. We're publishing the foundation now and growing it in the open.

---

## Quickstart — local install in 5 minutes

**Prerequisites:** Docker Desktop 4.27+ (or Docker Engine 25.0+ with Compose v2.24+). That's it.

```bash
git clone https://github.com/pycaret/pycaret.git
cd pycaret
docker compose up --build
```

First build takes ~5 min (Python deps + npm install + Vite build). Subsequent starts are < 30 sec.

Once you see `pycaret-api  | INFO: Application startup complete.` and `pycaret-web  | … starting nginx`, open:

**http://localhost:3020**

The setup screen will prompt you to create your first admin account + workspace. You're in.

### What's running

| Service | URL | What it is |
|---|---|---|
| `pycaret-web` | http://localhost:3020 | React UI (nginx-served bundle, proxies `/api` + `/ws` to backend) |
| `pycaret-api` | http://localhost:8020 | FastAPI + SQLAlchemy + APScheduler (in-process worker for V1) |
| `pycaret-data` (volume) | (named volume) | SQLite DB + uploaded CSVs + fitted `.pkl` files + Fernet key |

### Common ops

```bash
docker compose logs -f api      # tail backend logs
docker compose logs -f web      # tail UI logs
docker compose restart api      # restart backend only
docker compose down             # stop everything (volume PRESERVED — data + key safe)
docker compose down -v          # NUKE everything including the data volume (wipes DB + key)
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop / Docker Engine not running | Start Docker Desktop, wait for the whale icon, retry |
| `bind: address already in use` on `:3020` or `:8020` | Another process holds the port | Override: `PYCARET_WEB_PORT=3030 PYCARET_API_PORT=8030 docker compose up` |
| Build hangs at `pip install` for >10 min | Slow network on first build, or Docker Desktop is RAM-starved | Open Docker Desktop → Settings → Resources → bump RAM to ≥6 GB, retry |
| `no space left on device` during build | Old Docker images filling the disk | `docker system prune -a` then retry |
| `/bin/sh^M: bad interpreter` in api logs | Shell script got CRLF endings on a Windows checkout | This shouldn't happen — `.gitattributes` forces LF. If it does, re-clone with `git clone --config core.autocrlf=false` |
| UI loads but every `/api` call returns 404 | `pycaret-api` container hasn't finished bootstrapping (Alembic migrations) — happens on the very first `up` | Wait ~30 sec for the health check to pass; `docker compose logs -f api` to watch |
| Forgot the admin password | DB is encrypted; you can't recover it | `docker compose down -v` to wipe and re-bootstrap |

### Config

All overrideable via a `.env` file at the repo root (gitignored — see [`.env.example`](.env.example) for the template):

```bash
cp .env.example .env
# edit .env: set PYCARET_JWT_SECRET to something real, etc.
docker compose up
```

The biggest knobs:
- **`PYCARET_SECRETS_KEY`** — Fernet key for encrypting LLM API keys + connection passwords at rest. Auto-generated on first run + persisted to the data volume; only set this explicitly if you want the same key across multiple instances.
- **`PYCARET_JWT_SECRET`** — Used to sign auth tokens. **Must** be a strong random value in any prod deploy.
- **`PYCARET_DATABASE_URL`** — Swap from SQLite to Postgres in prod (`postgresql+psycopg://…`).
- **`PYCARET_STORAGE_BACKEND=s3`** + bucket creds — swap artifacts from local FS to S3.

---

## Take it for a spin (golden path)

1. **Setup** — create the first admin account + your workspace.
2. **Configure LLM (optional)** — sidebar → **Settings → LLM** → paste your Anthropic or OpenAI API key. Enables the AI dataset consultant + experiment designer copilots.
3. **Upload a dataset** OR pick one — sidebar → **Build → Datasets** → click **Browse samples** to grab a bundled CSV (juice, bank, iris, …) with one click.
4. **Create a project** — sidebar → **Build → Projects** → New project.
5. **Run an experiment** — inside the project → New experiment → pick your dataset, pick the task type (classification / regression / clustering / anomaly / time series), pick the target column. The run starts automatically and trains ~12 algorithms.
6. **Promote the winner** — on the Run detail page, the leaderboard ranks all 12 trials. Pick one → **Promote** → it lands as v1 on the **Model registry**.
7. **Deploy** — Model registry → click your model → on the version row, **Deploy** → give it an endpoint slug (e.g. `juice-prod`).
8. **Predict** — Deployment detail page → use the **Test a prediction** panel, or `curl POST http://localhost:8020/api/v1/deployments/juice-prod/predict` with a JSON row.

That's the full Train → Register → Deploy → Predict loop in ~10 minutes.

---

## Local development (without docker compose)

If you want to hack on the code with hot-reload, the docker compose setup is overkill — run the services directly:

```bash
# Prereqs: Python 3.13 + Node 22 + uv 0.11+ + npm 10+

# Install everything (engine + control plane + UI deps)
uv sync --all-packages --all-extras
cd apps/web && npm install && cd ..

# Two terminals:
uv run --package pycaret-server pycaret-server serve --reload   # backend :8020
cd apps/web && npm run dev                                       # frontend :3020
```

`apps/web/vite.config.ts` proxies `/api` and `/ws` to `:8020` — same browser experience as the docker setup.

---

## Security — what's safe to push, what's not

A `scripts/check-secrets.sh` scanner blocks accidental secret commits. Run it manually any time:

```bash
bash scripts/check-secrets.sh
```

Or install as a git pre-push hook (recommended):

```bash
cp scripts/check-secrets.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

It scans for Anthropic / OpenAI / Stripe / Slack / GitHub / AWS / Google API key shapes, Fernet ciphertext blobs, and PEM private keys. Whole-file allow-list: [`scripts/.secrets-allowlist`](scripts/.secrets-allowlist). Single-line allow: append `# pragma: allow-secret` to the line.

Things that NEVER get committed by design:
- `*.db` / `*.sqlite` (SQLite files — may contain encrypted secret blobs)
- `.env` (your local config)
- `*.pem` / `*.key` / `credentials.json` / `aws-credentials*`
- `/data/` (docker-compose volume mount point if you ever bind-mount it)

See [`.gitignore`](.gitignore) for the full list.

---

## Architecture, honestly

**Today** (what `docker compose up` actually runs): two containers. The api one runs a single FastAPI / uvicorn process that ALSO holds the in-process scheduler (APScheduler), the in-process compute (ThreadPoolExecutor), the inference runtime, and a SQLAlchemy session pointing at a SQLite file in a Docker volume. The web one is an nginx serving the built React bundle + reverse-proxying `/api` to the api container. That's the entire production deployment today.

**Tomorrow** (where we're going): every external dependency (storage, DB, secrets, auth, queue, compute, notifications) sits behind a `Protocol` with a local impl AND a cloud impl. Storage + DB are already abstracted that way; the remaining five `Protocol` extractions are queued as Phase 1 work. Once that's done, splitting api / worker / runtime into separate containers and pointing them at RDS / S3 / SQS / Secrets Manager becomes a config swap, not a rewrite.

We're publishing the foundation now and growing it in the open. Full design + gap matrix: [PLATFORM_ARCHITECTURE.md](docs/revamp/PLATFORM_ARCHITECTURE.md).

---

## Contributing

This project is **Claude Code-first**: contributors clone, run `claude` in the repo, pick an approved issue, let the agent do the work, open a PR. See [AGENTS.md](AGENTS.md) for the full agent brief and [CONTRIBUTING.md](CONTRIBUTING.md) for the human workflow.

---

## License

PyCaret 4.0 is dual-licensed:
- The **engine** (`packages/engine/`) — **MIT** (free for any use, including commercial).
- The **control plane** (`services/api/`, `apps/web/`, `infra/`) — **BUSL-1.1** with a 4-year change date to Apache-2.0. Production-self-host is fine for any organisation under the [Additional Use Grant](LICENSE); managed-SaaS-of-PyCaret-itself is the restricted case.

See [LICENSE](LICENSE) for the legal text, [LICENSE.engine.txt](LICENSE.engine.txt) for the engine MIT terms.
