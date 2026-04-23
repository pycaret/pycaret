<div align="center">

<img src="docs/images/logo.png" alt="PyCaret" width="200"/>

# PyCaret 4.0 — lean, modern, agent-native AutoML

### A sklearn-composable AutoML engine built to power notebooks, LLM agents, and a modern React UI.

[![CI](https://github.com/pycaret/pycaret/actions/workflows/test.yml/badge.svg?branch=v4)](https://github.com/pycaret/pycaret/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/pycaret/pycaret)
[![License](https://img.shields.io/pypi/l/pycaret.svg)](https://github.com/pycaret/pycaret/blob/v4/LICENSE)

[Changelog](docs/revamp/release_notes_pycaret4.md) ·
[Architecture](docs/revamp/ARCHITECTURE.md) ·
[Roadmap](docs/revamp/ROADMAP.md) ·
[Agent guide](AGENTS.md) ·
[Dev docs](docs/for_developers/)

</div>

---

> ## ⚠ 4.0 is **work in progress** — you're looking at the `v4` branch
>
> PyCaret 4.0 is a ground-up architectural revamp of PyCaret. It lives on the **`v4` branch**. The `master` branch is still 3.4.0.
>
> **Status (as of the latest session):**
> - **Public API**: OOP-only `Experiment` classes. Functional API is gone. 145 module-level functions + ~11,300 LOC of pass-through wrappers removed.
> - **Source**: ~49K LOC (down from ~62K at 3.4.0), still shrinking. The 3.x god-class in `pycaret/internal/pycaret_experiment/` is still wrapped by `Experiment._legacy` and being drained verb-by-verb (Phase 5 of the roadmap).
> - **Tests**: 32/32 green on Python 3.11 / 3.12 / 3.13 + scikit-learn 1.7 + NumPy 2. Runs in ~2 min.
> - **Dependencies**: 19 core deps (down from 30). mlflow / comet / wandb / dagshub / fugue / dask / ray / yellowbrick / gradio / fastapi / boto3 / m2cgen / evidently / fairlearn all removed. See [`docs/revamp/KILL_LIST.md`](docs/revamp/KILL_LIST.md).
> - **Notebooks**: 5 working executed end-to-end examples under [`notebooks/`](notebooks/).
>
> **Don't use 4.0 in production yet.** It's not on PyPI and the internal delegation layer is still being drained. Track progress in [`docs/revamp/STATUS.md`](docs/revamp/STATUS.md) and [`docs/revamp/ROADMAP.md`](docs/revamp/ROADMAP.md). The first installable release will be `4.0.0alpha0`.
>
> **What works today:** `git clone -b v4`, `uv sync --all-extras`, `uv run pytest` — full green. The 5 notebooks in [`notebooks/`](notebooks/) run end-to-end. The OOP API is stable; internal refactors will not break it.

---

## Why 4.0 is different

PyCaret 3.x shipped two overlapping APIs (functional + OOP) and ~62K LOC of code. In 2026, most of that was tech debt: unmaintained tracker integrations (mlflow/comet/wandb), broken-on-modern-sklearn plot wrappers (yellowbrick), never-used distributed backends (fugue/dask/ray), and signature-drift bugs between the two APIs.

**PyCaret 4.0 is a ~35% tech-debt cut.** Result:

- **OOP-only**, sklearn-composable `Experiment` classes (`ClassificationExperiment`, `RegressionExperiment`, …) that inherit from `sklearn.base.BaseEstimator` — they respond to `get_params`, `set_params`, `clone`, `__sklearn_tags__`, `__sklearn_is_fitted__` like any other sklearn object.
- **Typed result dataclasses** — every operation returns a `CompareResult` / `TuneResult` / `PredictResult` / … with the fitted pipeline, metrics, and an event trace.
- **Structured event stream** (`pycaret.logging`) designed to be consumed by a React UI or an LLM agent over `BaseLogger.subscribe(callback)`.
- **First-class introspection** (`pycaret.api`) — every model, metric, and setup parameter is a serializable dataclass a UI can render directly.
- **Engine-oriented:** the forthcoming PyCaret React UI runs on this engine; so do LLM-agent workflows that drive PyCaret programmatically.

See [`docs/revamp/ARCHITECTURE.md`](docs/revamp/ARCHITECTURE.md) for the full design rationale and [`docs/revamp/release_notes_pycaret4.md`](docs/revamp/release_notes_pycaret4.md) for the engineering change log.

## Installation

PyCaret 4.0 uses `uv` for environment management.

```bash
# Install the `uv` tool if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# Core installation
uv pip install pycaret

# With every optional extra (models, tuners, analysis, anomaly, timeseries)
uv pip install "pycaret[full]"
```

Plain `pip` works too:

```bash
pip install pycaret
pip install "pycaret[full]"
```

**Supported:** Python 3.11 / 3.12 / 3.13. (Python 3.14 tracked, pending upstream `joblib` + `cloudpickle` support for PEP 649.)

## Quickstart — Classification

```python
from pycaret.datasets import get_data
from pycaret.tasks import ClassificationExperiment
from pycaret import save_model, load_model

df = get_data("juice")

exp = ClassificationExperiment(target="Purchase", session_id=42).fit(df)

# Compare top models; returns a CompareResult dataclass
result = exp.compare_models()
best = result.best
print(result.leaderboard)         # notebook-friendly DataFrame

# Tune the best model; returns a TuneResult
tuned = exp.tune_model(best).pipeline

# Predict on new data; returns a PredictResult
preds = exp.predict_model(tuned).predictions

# Persist the fitted pipeline
save_model(tuned, "artifacts/best")
restored = load_model("artifacts/best")
```

## Quickstart — Regression / Clustering / Anomaly / Time Series

Same shape, task-specific subclass:

```python
from pycaret.tasks import (
    RegressionExperiment,
    ClusteringExperiment,
    AnomalyExperiment,
    TimeSeriesExperiment,
)

reg = RegressionExperiment(target="medv").fit(boston_df)
best = reg.compare_models().best

cluster = ClusteringExperiment().fit(jewellery_df)
km = cluster.create_model("kmeans", num_clusters=4).pipeline
labelled = cluster.assign_model(km)

anom = AnomalyExperiment().fit(anomaly_df)
iforest = anom.create_model("iforest").pipeline
anom.assign_model(iforest)

ts = TimeSeriesExperiment(fh=12).fit(airline_series)
forecast = ts.predict_model(ts.compare_models().best).predictions
```

## Introspection — build a UI or drive it from an agent

```python
from pycaret.api import (
    list_models, describe_model, list_metrics, describe_setup_params,
)

# Static listings — no Experiment required, works for docs generation
list_models("classification")           # -> list[ModelCard]
describe_model("classification", "lr")  # -> ModelCard
list_metrics("classification")          # -> list[MetricCard]

# UI-form schema — JSON-serializable, renders directly as a form
schema = describe_setup_params("classification")
import json
json.dumps(schema.to_dict())
```

## Event stream — for UIs and LLM agents

```python
from pycaret.logging import MemoryLogger

log = MemoryLogger()
log.subscribe(lambda event: print(event.kind.value, event.message))

exp = ClassificationExperiment(target="y", logger=log).fit(df)
exp.compare_models()   # emits experiment.started → model.compare.finished → ...
```

## What's not in 4.0 (deliberate)

Dropped in the 4.0 revamp because they were either unused, duplicated, or duplicated what the upcoming React UI will provide:

- Module-level functional API (`setup`, `compare_models`, etc.) — **use the OOP `Experiment` classes**.
- External experiment trackers: mlflow, comet-ml, wandb, dagshub — **replaced by the built-in event stream**.
- Distributed / parallel backends: fugue, dask, ray, distributed — **removed**.
- Visualization: yellowbrick, mljar-scikit-plot, schemdraw, plotly-resampler — **Plotly-only plots coming in the next release**.
- Deployment helpers: `create_api`, `create_app`, `create_docker`, `dashboard`, `deploy_model` (S3), `convert_model` (m2cgen) — **out of scope for the engine**.
- Drift / fairness: `check_drift`, `check_fairness` — **the React UI owns these**.

See [`docs/revamp/KILL_LIST.md`](docs/revamp/KILL_LIST.md) for the exhaustive list.

## Who PyCaret 4.0 is for

- **Data scientists in notebooks** who want the fastest path from a DataFrame to a fitted pipeline.
- **Engineering teams** building ML workflows on top of a sklearn-compatible engine with a clean, typed API.
- **LLM agents** that introspect and drive ML experiments — every model, metric, and parameter is a serializable dataclass.
- **UI builders** — the forthcoming open-source React UI runs on this engine; you can build your own.

## License

MIT — see [`LICENSE`](LICENSE).

## Contributing

PyCaret 4.0 is under active architectural revamp. See [`docs/revamp/ROADMAP.md`](docs/revamp/ROADMAP.md) for the phased plan and [`docs/revamp/STATUS.md`](docs/revamp/STATUS.md) for current progress. Bug reports welcome; large feature PRs should discuss in an issue first.
