# PyCaret 3.4.0 → 4.0 Baseline Audit

*Date:* 2026-04-22
*Method:* Static inventory of the 3.4.0 source tree at HEAD, plus a support-matrix check against the current scikit-learn release.

## 1. Size

| Scope | LOC |
|---|---:|
| Total (`pycaret/*.py`) | **62,164** |
| `pycaret/internal/` | 23,775 |
| `pycaret/containers/` | 9,201 |
| `pycaret/time_series/` | 8,115 |
| `pycaret/classification/` | 6,876 |
| `pycaret/regression/` | 5,926 |
| `pycaret/utils/` | 3,659 |
| `pycaret/clustering/` | 1,866 |
| `pycaret/anomaly/` | 1,464 |
| `pycaret/loggers/` | 849 |
| `pycaret/parallel/` | 233 |

### Monster files (candidates for surgery)

| File | LOC | Notes |
|---|---:|---|
| `internal/pycaret_experiment/supervised_experiment.py` | 5,886 | God-class; `_BaseSupervisedExperiment`. Everything tune/compare/predict flows through here. |
| `internal/pycaret_experiment/tabular_experiment.py` | 2,862 | Contains ALL yellowbrick imports (19 sites) inline in `plot_model` branches. |
| `internal/pycaret_experiment/unsupervised_experiment.py` | 1,392 | Unsupervised counterpart. |
| `internal/plots/time_series.py` | 1,367 | Plotly-native, likely salvageable. |
| `internal/preprocess/preprocessor.py` | 1,035 | Core preprocessor — must preserve. |
| `internal/plots/residual_plots.py` | 753 | Candidate for Plotly rewrite along with yellowbrick replacements. |
| `internal/preprocess/transformers.py` | 630 | Core — preserve. |
| `internal/preprocess/iterative_imputer.py` | 562 | Check whether sklearn's own IterativeImputer now covers this; likely deletable. |

## 2. Current dependency surface (from `pyproject.toml`)

### Upper-bound pins — all block modern Python / sklearn

| Package | Current pin | Problem |
|---|---|---|
| `scikit-learn` | `<1.5` | Blocks sklearn 1.6+ (current is **1.8.0**). |
| `numpy` | `<1.27` | Blocks NumPy 2.x entirely. |
| `pandas` | `<2.2` | Blocks pandas 2.2+ |
| `scipy` | `<=1.11.4` | Badly stale. |
| `joblib` | `<1.5` | Minor but still a pin. |
| `matplotlib` | `<3.8.0` | Blocks a lot; also matplotlib-only usage is small. |
| `sktime` | `>=0.31.0,<0.31.1` | Pinned to one point release — fragile. |
| `requires-python` | `>=3.9,<3.13` | Blocks 3.13, 3.14. |

### Core dependencies — count

- **Hard (`dependencies`):** 30 packages
- **`full` extras:** 30+ more
- **Redundant plotting stacks:** matplotlib + plotly + yellowbrick + mljar-scikit-plot + schemdraw + plotly-resampler — at least 3 are redundant.

## 3. Kill-list evidence (user pre-approved removals)

### MLflow / Comet / Wandb / Dagshub
```
pycaret/loggers/mlflow_logger.py           ← full module
pycaret/loggers/comet_logger.py            ← full module
pycaret/loggers/wandb_logger.py            ← full module
pycaret/loggers/dagshub_logger.py          ← full module
pycaret/loggers/dashboard_logger.py        ← full module (dispatcher)
pycaret/internal/pycaret_experiment/pycaret_experiment.py:249  ← inline mlflow import
pycaret/time_series/forecasting/oop.py:1432                    ← inline mlflow import
```
→ Replace with a single lean built-in logger designed for the React UI.

### Parallel (Fugue / Dask / Distributed)
```
pycaret/parallel/                          ← full module (233 LOC, incl. fugue_backend.py)
pycaret/internal/parallel/                 ← internal parallel helpers
tests/test_classification_parallel.py
tests/test_regression_parallel.py
tests/test_time_series_parallel.py
```
→ Delete wholesale. No replacement.

### Yellowbrick
```
pycaret/internal/patches/yellowbrick.py    ← monkey patches over yellowbrick
pycaret/internal/plots/yellowbrick.py      ← wrapper layer
pycaret/internal/pycaret_experiment/tabular_experiment.py   ← 16 yellowbrick imports, one per plot kind:
  KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance,
  ResidualsPlot, ROCAUC, DiscriminationThreshold, PrecisionRecallCurve,
  ConfusionMatrix, ClassPredictionError, PredictionError, CooksDistance,
  ClassificationReport, DecisionViz, RFECV, LearningCurve, Manifold,
  ValidationCurve, RadViz
```
→ Rewrite each as a Plotly function in a new `pycaret/plots/` flat module.

### mljar-scikit-plot / schemdraw
- 2 import sites for scikit-plot-style calls → fold into Plotly rewrite.
- `schemdraw==0.15` pin is a ticking time bomb; only used for one diagram → drop or replace.

## 4. Test landscape

- **59 test files** under `tests/` (flat layout, no `unit/`/`integration/` split).
- Tests with kill-list in the name (delete):
  - `test_mlflow_artifacts.py` (already empty file)
  - `test_time_series_mlflow.py`
  - `test_classification_parallel.py`
  - `test_regression_parallel.py`
  - `test_time_series_parallel.py`
- Tests for features likely cut:
  - `test_create_api.py`, `test_create_app.py`, `test_create_docker.py`, `test_dashboard.py` — Gradio/FastAPI/Docker deploy helpers; redundant given React UI.
  - `test_check_drift.py` (evidently), `test_check_fairness.py` (fairlearn).
- **pytest config** in `setup.cfg`; uses `python_files = *.py` which is loose.

## 5. Python / sklearn support matrix (target)

- **scikit-learn 1.8.0** (current latest): `requires_python >= 3.11`, classifiers list 3.11 / 3.12 / 3.13 / 3.14.
- **PyCaret 4.0 target:**
  - Floor: **Python 3.11** (aligned with sklearn 1.8 floor)
  - CI matrix: 3.11, 3.12, 3.13, 3.14
  - Primary dev target: **3.14**
  - scikit-learn: latest stable (1.8.x), no upper cap.

## 6. Build system

- Current: `setuptools` backend declared in `pyproject.toml`, but with legacy artifacts (`setup.cfg`, `MANIFEST.in`).
- 4.0: `hatchling` or `uv_build`; `uv.lock` committed; one-command dev bootstrap (`uv sync --all-extras`).

## 7. Notebooks / examples

- `tutorials/` has 6 notebooks (classification binary + multiclass, regression, clustering, anomaly, time series).
- Must be re-run end-to-end on 4.0 and checked into the repo with fresh outputs.

## 8. Headline risks

1. **`supervised_experiment.py` (5,886 LOC)** is a god-class — refactor risk is concentrated here.
2. **`tabular_experiment.py` plot dispatcher** is the single largest behaviour change area (16 yellowbrick rewrites).
3. **sktime 0.31.x pin** — unpinning will churn time-series code; time-series tests are the bulk of the suite.
4. **Preprocessor** (`internal/preprocess/*`, ~2,200 LOC) is where sklearn 1.5+ API changes will bite (transformer protocol, `set_output`, `__sklearn_tags__`).
