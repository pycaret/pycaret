<div align="center">

<img src="docs/images/logo.png" alt="PyCaret" width="200"/>

# PyCaret — open-source ML platform

### The engine, the control plane, and the UI — all in one self-hosted box.

[![CI](https://github.com/pycaret/pycaret/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/pycaret/pycaret/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/pycaret/pycaret)
[![License](https://img.shields.io/pypi/l/pycaret.svg)](https://github.com/pycaret/pycaret/blob/main/LICENSE)

[Vision](docs/revamp/VISION.md) ·
[Architecture](docs/revamp/ARCHITECTURE.md) ·
[Roadmap](docs/revamp/ROADMAP.md) ·
[Spec](docs/revamp/CONTROL_PLANE_SPEC.md) ·
[Quickstart](docs/revamp/PLATFORM_QUICKSTART.md) ·
[Agent guide](AGENTS.md)

</div>

---

> ## ⚠ 4.0 is **work in progress** — you're looking at `main` (the 4.0 line)
>
> PyCaret 4.0 is a ground-up architectural revamp. It now lives on `main`. The 3.x release line is frozen on PyPI as `pycaret 3.4.0` (no further commits).
>
> Track progress in [`docs/revamp/STATUS.md`](docs/revamp/STATUS.md) and [`docs/revamp/ROADMAP.md`](docs/revamp/ROADMAP.md).

---

## What you get

**PyCaret** is two layers of one product:

**The engine** (`packages/engine/`) — `pip install pycaret`. Config-driven, stateless, sklearn-composable. Use it in a notebook:

```python
from pycaret.tasks import ClassificationExperiment
from pycaret.datasets import get_data

df = get_data("juice")
exp = ClassificationExperiment(target="Purchase", session_id=42).fit(df)
best = exp.compare_models().best
tuned = exp.tune_model(best).pipeline
exp.save_model(tuned, "baseline")
```

**PyCaret Control Plane** (`services/api/` + `apps/web/` + `infra/`) — the full self-hosted web platform that wraps the engine. Workspaces, projects, datasets, experiments, runs, artifacts, deployments, monitoring, LLM-assisted experiment design. Run it on a laptop, a Docker host, or Kubernetes.

Three deployment modes (current + roadmapped):

| Mode | For | Status |
|---|---|---|
| Notebook (`pip install pycaret`) | Data scientist workflow | ✅ `4.0.0a1` on PyPI |
| Local dev (uv + npm) | Building against the Control Plane | ✅ shipped |
| Single-server Docker compose | Small-team self-hosted | ✅ shipped |
| Kubernetes + Helm + Terraform | Enterprise cloud | 🔴 V2 (stubs scaffolded) |
| Electron desktop | Analyst, no Docker | 🔴 V2 (stub scaffolded) |

## Repo layout

```
pycaret/                  ← monorepo
├── packages/
│   └── engine/           → `pycaret` on PyPI
├── services/
│   ├── api/              → `pycaret-server` on PyPI (FastAPI)
│   ├── worker/           (V2) background job runner
│   └── deployment-runtime/ (V2) standalone serving
├── apps/
│   ├── web/              React + Vite (Control Plane UI)
│   └── desktop/          (V2) Electron
├── infra/
│   ├── docker/           Dockerfile.api, Dockerfile.ui, compose
│   ├── helm/             (V2) Kubernetes chart
│   └── terraform/        (V2) AWS / GCP / Azure modules
└── docs/revamp/          VISION, SPEC, ARCHITECTURE, ROADMAP, STATUS, DECISIONS
```

See [`docs/revamp/ARCHITECTURE.md`](docs/revamp/ARCHITECTURE.md) for the full system architecture.

## Try it locally — 3 minutes

**Just the engine, in a notebook:**

```bash
pip install pycaret
# or with every optional extra:
pip install "pycaret[full]"
```

Supported: Python 3.11 / 3.12 / 3.13.

**The full Control Plane, from source:**

```bash
git clone https://github.com/pycaret/pycaret.git
cd pycaret

# Backend (terminal 1)
uv python install 3.13
uv sync --all-packages --all-extras
uv run --package pycaret-server pycaret-server serve --reload

# Frontend (terminal 2)
cd apps/web
npm install
npm run dev
# → http://localhost:3000/setup
```

**Or with Docker** (full stack, one command):

```bash
docker compose -f infra/docker/docker-compose.yml up --build
# → http://localhost:3000
```

See [`docs/revamp/PLATFORM_QUICKSTART.md`](docs/revamp/PLATFORM_QUICKSTART.md) for the full quickstart.

## Engine quickstart

```python
from pycaret.datasets import get_data
from pycaret.tasks import ClassificationExperiment
from pycaret import save_model, load_model

df = get_data("juice")
exp = ClassificationExperiment(target="Purchase", session_id=42).fit(df)

# Compare models — returns a typed CompareResult
result = exp.compare_models()
best = result.best
print(result.leaderboard)

# Tune — returns a TuneResult
tuned = exp.tune_model(best).pipeline

# Predict — returns a PredictResult
preds = exp.predict_model(tuned).predictions

# Save + load
save_model(tuned, "artifacts/best")
restored = load_model("artifacts/best")
```

Same shape for the other task types:

```python
from pycaret.tasks import (
    RegressionExperiment,
    ClusteringExperiment,
    AnomalyExperiment,
    TimeSeriesExperiment,
)
```

## Introspection — for UIs and LLM agents

```python
from pycaret.api import (
    list_models, describe_model, list_metrics, describe_setup_params,
)

list_models("classification")           # -> list[ModelCard]
describe_model("classification", "lr")  # -> ModelCard
list_metrics("classification")          # -> list[MetricCard]

# UI-form schema — JSON-serializable, renders directly as a dynamic form
schema = describe_setup_params("classification")
```

The Control Plane UI renders its entire experiment-setup form from `describe_setup_params`. Zero UI code hard-codes a parameter name.

## Event stream

```python
from pycaret.logging import MemoryLogger

log = MemoryLogger()
log.subscribe(lambda event: print(event.kind.value, event.message))

exp = ClassificationExperiment(target="y", logger=log).fit(df)
exp.compare_models()   # emits experiment.started → model.compare.finished → ...
```

The Control Plane backend subclasses `BaseLogger` with `DBEventLogger` — every engine event becomes a DB row **and** streams live to any connected WebSocket clients.

## What's deliberately not here

- Module-level functional API (`setup`, `compare_models`) — use OOP `Experiment` classes.
- External experiment trackers: mlflow, comet-ml, wandb, dagshub — the Control Plane owns this story now.
- Distributed backends: fugue, dask, ray (V3 opt-in).
- Visualization: yellowbrick, mljar-scikit-plot, schemdraw — Plotly-only rewrite in progress.
- In-engine deployment helpers: `create_api`, `create_app`, `create_docker`, `dashboard`, `convert_model`, `deploy_model` — the Control Plane owns serving + deployment.
- Drift / fairness in the engine: `check_drift`, `check_fairness` — moved to the monitoring layer.

See [`docs/revamp/KILL_LIST.md`](docs/revamp/KILL_LIST.md) for the exhaustive list.

## Who this is for

- **Data scientists** who want AutoML in a notebook without vendor lock-in.
- **ML engineers** who want an open-source control plane they can self-host — train, deploy, monitor, improve.
- **Small teams** (≤20 people) who need the whole loop without Databricks licenses.
- **Enterprises** who need SSO + audit logs + multi-cloud deployment in the same repo they started prototyping with.
- **LLM agents** that introspect and drive ML experiments — every model, metric, and parameter is a serializable dataclass.

See [`docs/revamp/VISION.md`](docs/revamp/VISION.md) for the product statement.

## Licensing

PyCaret 4.0 ships under the **Functional Source License (FSL-1.1-MIT)** — the same license Sentry, Convex, and Keep use. The short version:

- **Free** for individual use, internal corporate use, non-commercial education / research, and consulting work delivered on top of PyCaret.
- **Not free** to use as the basis of a competing AutoML product or hosted service.
- **Auto-converts to MIT** two years after each release. The 4.0.0 release becomes plain MIT in 2028, the next minor in two years from its release date, and so on.

See [`LICENSE`](LICENSE) for the full text.

**Per-package detail:**

- `packages/engine/` (the `pycaret` library) and `apps/site/` are FSL-1.1-MIT.
- `services/api/` and `apps/web/` (the Control Plane backend + frontend) are **dual-licensed FSL-1.1-MIT OR BUSL-1.1**. Self-host freely; the BSL grant kicks in for multi-tenant hosted commercialisation and auto-converts after three years.

Rationale and the chain of decisions: [`docs/revamp/DECISIONS.md`](docs/revamp/DECISIONS.md).

The 3.x line on PyPI (`pycaret <= 3.4.0`) remains MIT — license changes only apply to 4.0+.

## Contributing

PyCaret is under active revamp and is **Claude-Code-first**: anyone can clone the repo, run Claude Code in their own checkout (using their own Claude credentials), pick a maintainer-`Approved` issue, and let the agent open a PR.

```bash
gh repo clone pycaret/pycaret && cd pycaret
claude
> /work-on-approved-issue
```

Compute is community-funded — there's **no** Claude API key in this repo's secrets and **no** CI bot that auto-fixes issues. The Claude Code setup lives in [`CLAUDE.md`](CLAUDE.md) (entry point), [`.claude/`](.claude/) (slash commands + sub-agents + permissions), and per-directory `CLAUDE.md` files. Cross-vendor instructions for non-Claude agents live in [`AGENTS.md`](AGENTS.md).

Traditional contributions — clone, edit, PR — are also first-class. Read [`CONTRIBUTING.md`](CONTRIBUTING.md). Bug reports welcome; large feature PRs should discuss in an issue first.
