# Task cheatsheet — verbs × tasks

Which verbs exist on which task. Everything else is a method lookup error.

|                      | Classification | Regression | Clustering | Anomaly | TimeSeries |
|---|:-:|:-:|:-:|:-:|:-:|
| `fit(X, y=None)`     | ✓ | ✓ | ✓ (no y) | ✓ (no y) | ✓ |
| `create_model(id)`   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `compare_models()`   | ✓ | ✓ | — | — | ✓ |
| `tune_model(m)`      | ✓ | ✓ | — | — | ✓ |
| `ensemble_model(m)`  | ✓ | ✓ | — | — | — |
| `blend_models(ms)`   | ✓ | ✓ | — | — | ✓ |
| `stack_models(ms)`   | ✓ | ✓ | — | — | — |
| `calibrate_model(m)` | ✓ | — | — | — | — |
| `finalize_model(m)`  | ✓ | ✓ | — | — | ✓ |
| `predict_model(m)`   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `assign_model(m)`    | — | — | ✓ | ✓ | — |
| `plot_model(m, k)`   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `evaluate_model(m)`  | ✓ | ✓ | ✓ | ✓ | ✓ |
| `interpret_model(m)` | ✓ | ✓ | — | — | — |
| `save_model(m, p)`   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `load_model(p)`      | ✓ | ✓ | ✓ | ✓ | ✓ |
| `save_experiment(p)` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `load_experiment(p)` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pull()`             | ✓ | ✓ | ✓ | ✓ | ✓ |
| `models()`           | ✓ | ✓ | ✓ | ✓ | ✓ |
| `get_metrics()`      | ✓ | ✓ | ✓ | ✓ | ✓ |
| `add_metric(...)`    | ✓ | ✓ | ✓ | ✓ | ✓ |
| `remove_metric(id)`  | ✓ | ✓ | ✓ | ✓ | ✓ |
| `get_config(k)`      | ✓ | ✓ | ✓ | ✓ | ✓ |
| `set_config(k, v)`   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `automl()`           | ✓ | ✓ | — | — | ✓ |
| `get_leaderboard()`  | ✓ | ✓ | — | — | ✓ |
| `check_stats()`      | — | — | — | — | ✓ |

Properties (accessible after `fit`):

|                  | Classification | Regression | Clustering | Anomaly | TimeSeries |
|---|:-:|:-:|:-:|:-:|:-:|
| `X`, `X_train`, `X_test`    | ✓ | ✓ | ✓ | ✓ | ✓ |
| `y`, `y_train`, `y_test`    | ✓ | ✓ | — | — | ✓ |
| `preprocess_pipeline`       | ✓ | ✓ | ✓ | ✓ | ✓ |
| `events` (replay of logger) | ✓ | ✓ | ✓ | ✓ | ✓ |

Construction signature (init kwargs you can pass):

| Parameter         | C | R | Cl | A | T | Notes |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `target`          | ✓ | ✓ | — | — | opt | column name of the label |
| `session_id`      | ✓ | ✓ | ✓ | ✓ | ✓ | RNG seed |
| `train_size`      | ✓ | ✓ | — | — | — | default 0.7 |
| `fold`            | ✓ | ✓ | — | — | ✓ | CV folds; TS default=3 |
| `fold_strategy`   | ✓ | ✓ | — | — | ✓ | string or splitter |
| `preprocess`      | ✓ | ✓ | ✓ | ✓ | ✓ | |
| `normalize`       | ✓ | ✓ | ✓ | ✓ | — | |
| `transformation`  | ✓ | ✓ | ✓ | ✓ | — | power transform |
| `remove_outliers` | ✓ | ✓ | — | — | — | |
| `feature_selection` | ✓ | ✓ | ✓ | ✓ | — | |
| `n_jobs`          | ✓ | ✓ | ✓ | ✓ | ✓ | -1 = all cores |
| `use_gpu`         | ✓ | ✓ | ✓ | ✓ | ✓ | |
| `logger`          | ✓ | ✓ | ✓ | ✓ | ✓ | `BaseLogger` instance |
| `log_experiment`  | ✓ | ✓ | ✓ | ✓ | ✓ | auto-install `MemoryLogger` if True |
| `verbose`         | ✓ | ✓ | ✓ | ✓ | ✓ | legacy progress bar toggle |
| `fh`              | — | — | — | — | ✓ | forecast horizon (TS only) |
| `seasonal_period` | — | — | — | — | ✓ | TS only |

## Import paths

```python
# Canonical (use this in new code):
from pycaret.tasks import ClassificationExperiment
from pycaret.tasks import RegressionExperiment
from pycaret.tasks import ClusteringExperiment
from pycaret.tasks import AnomalyExperiment
from pycaret.tasks import TimeSeriesExperiment

# Legacy-friendly (re-exports the same classes):
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.anomaly import AnomalyExperiment
from pycaret.time_series import TimeSeriesExperiment

# Stateless top-level utilities:
from pycaret import save_model, load_model, show_versions

# Agent / UI introspection:
from pycaret.api import list_models, describe_model, list_metrics, describe_setup_params

# Event stream:
from pycaret.logging import MemoryLogger, BaseLogger, EventKind, Event
```
