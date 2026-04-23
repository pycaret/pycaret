# PyCaret 4.0 — Kill List

Every removal below is pre-approved by the project owner (2026-04-22). No re-litigation; this file is the record.

## Dependencies removed from `pyproject.toml`

| Package | Where used | Replacement | Rationale |
|---|---|---|---|
| `mlflow` | `loggers/mlflow_logger.py`, inline imports in `pycaret_experiment.py`, `time_series/forecasting/oop.py` | Lean built-in logger (`pycaret/logging/`) designed for the React UI | Out of scope for 4.0; React UI owns tracking. |
| `comet-ml` | `loggers/comet_logger.py` | Same as above | Out of scope. |
| `wandb` | `loggers/wandb_logger.py` | Same as above | Out of scope. |
| `dagshub` | `loggers/dagshub_logger.py` | Same as above | Out of scope. |
| `fugue`, `fugue[dask]` | `parallel/fugue_backend.py`, parallel tests | *No replacement.* | "Nobody uses that anyways." |
| `dask`, `distributed` | same as above | *No replacement.* | Same. |
| `ray[tune]`, `tune-sklearn` | `tuners` extra | Keep `optuna` + sklearn native search | Ray is heavy; optuna covers search needs. |
| `yellowbrick` | `internal/patches/yellowbrick.py`, `internal/plots/yellowbrick.py`, 16 inline imports in `tabular_experiment.py` | New internal Plotly plots in `pycaret/plots/` | Yellowbrick is stuck on old sklearn API; plots can look much better in Plotly. |
| `mljar-scikit-plot` | 2 import sites | Folded into new Plotly plots | Redundant. |
| `schemdraw` | One diagram | Drop the diagram, or render as Plotly | Pinned to 0.15; maintenance risk. |
| `plotly-resampler` | Time-series plots | Evaluate; drop if the built-in Plotly toggle suffices | Marginal value. |
| `evidently` | `check_drift` | Drop `check_drift` | Not part of notebook golden path; UI will own drift monitoring. |
| `fairlearn` | `check_fairness` | Drop `check_fairness` | Same reasoning. |
| `ydata-profiling` | EDA helper | Drop | Heavy dep; UI will own EDA. |
| `explainerdashboard` | Dashboard helper | Drop (React UI replaces this) | Duplicates UI. |
| `gradio` | `create_app` | Drop | Duplicates React UI. |
| `uvicorn`, `fastapi` (in `mlops`) | `create_api` | Drop from core; if we need a server it goes in a separate `pycaret-server` package | Engine shouldn't ship a web server. |
| `boto3` | `deploy_model` AWS | Drop — deploy is out of scope for core | Can be a separate extra later. |
| `m2cgen` | `convert_model` | Drop | Niche; re-add on demand. |
| `moto` | Test dep for boto3 | Drops with boto3 | – |
| `flask`, `Werkzeug` (in `parallel` extra) | dask UI | Drops with parallel | – |
| `dash[testing]` | test dep | Drop with dashboard | – |
| `schemdraw==0.15` | one diagram | Drop | – |
| `trio<0.25` | "fixes httpcore" workaround | Remove; revisit with modern httpcore | Legacy pin. |
| `setuptools` runtime dep | Python 3.12 workaround | Shouldn't be runtime dep | – |
| `wurlitzer` | fd redirection | Audit — likely droppable with modern display backend | – |

## Subsystems removed wholesale

| Path | Action |
|---|---|
| `pycaret/parallel/` | `git rm -r` |
| `pycaret/internal/parallel/` | `git rm -r` |
| `pycaret/loggers/comet_logger.py` | remove |
| `pycaret/loggers/wandb_logger.py` | remove |
| `pycaret/loggers/dagshub_logger.py` | remove |
| `pycaret/loggers/mlflow_logger.py` | remove |
| `pycaret/loggers/dashboard_logger.py` | remove (dispatcher of above) |
| `pycaret/internal/patches/yellowbrick.py` | remove |
| `pycaret/internal/plots/yellowbrick.py` | remove; rewrite |
| `tests/test_*_parallel.py` | remove |
| `tests/test_mlflow_artifacts.py` | remove (empty already) |
| `tests/test_time_series_mlflow.py` | remove |
| `tests/test_create_api.py` | remove |
| `tests/test_create_app.py` | remove |
| `tests/test_create_docker.py` | remove |
| `tests/test_dashboard.py` | remove |
| `tests/test_check_drift.py` | remove |
| `tests/test_check_fairness.py` | remove |

## Features removed from public API

- `pycaret.*.create_api()`
- `pycaret.*.create_app()`
- `pycaret.*.create_docker()`
- `pycaret.*.dashboard()`
- `pycaret.*.check_drift()`
- `pycaret.*.check_fairness()`
- `pycaret.*.deploy_model()` (AWS/S3 specifically; local save/load stays)
- `pycaret.*.eda()` (ydata-profiling wrapper)
- `pycaret.*.convert_model()` (m2cgen wrapper)
- The `parallel_backend` argument on `setup()` / `compare_models()` → dropped from signatures.

## Features preserved (explicit)

Nothing in the notebook-user golden path is touched. Signatures may tighten but the call shape stays:

```python
from pycaret.classification import *
s = setup(data, target='y')
best = compare_models()
tuned = tune_model(best)
plot_model(tuned, 'auc')
preds = predict_model(tuned, data=new_data)
save_model(tuned, 'model')
```
Same for `regression`, `clustering`, `anomaly`, `time_series`.
