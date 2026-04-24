# PyCaret 4.0 — engine

The PyCaret 4.0 engine. Config-driven, stateless, notebook-friendly, built on
scikit-learn 1.7+ / NumPy 2 / pandas 2 / Python 3.11+.

Paired with [`services/api`](../../services/api) (the FastAPI backend) and
[`apps/web`](../../apps/web) (the React UI) to form **PyCaret Control Plane** —
a self-hosted ML platform. But the engine works standalone in a notebook too:

```python
from pycaret.datasets import get_data
from pycaret.tasks import ClassificationExperiment

df = get_data("juice")
exp = ClassificationExperiment(target="Purchase", session_id=42).fit(df)
best = exp.compare_models().best
exp.save_model(best, "baseline")
```

## Status

4.0.0a1 on PyPI. Published wheel name: `pycaret`. See the repo root
[`docs/revamp/STATUS.md`](../../docs/revamp/STATUS.md) for the programme-wide
status and [`docs/revamp/ROADMAP.md`](../../docs/revamp/ROADMAP.md) for what's
next.

## Layout

```
packages/engine/
├── pyproject.toml       # builds the `pycaret` wheel
├── pycaret/             # the package
│   ├── api/             # list_models, describe_model, describe_setup_params
│   ├── core/            # Experiment base class + results + errors
│   ├── tasks/           # 5 task subclasses (classification/regression/...)
│   ├── logging/         # BaseLogger + events + MemoryLogger
│   ├── containers/      # model-registry containers (legacy, being drained)
│   └── internal/        # god-class (being drained verb-by-verb — Phase 5)
├── tests/               # pytest suite (32 tests)
└── README.md            # this file
```

## Dev

From the repo root:

```bash
uv sync --all-packages --all-extras          # all workspace members + extras
uv run pytest packages/engine/tests/ -q      # 32 engine tests
```
