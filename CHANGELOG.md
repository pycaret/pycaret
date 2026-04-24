# Changelog

User-facing release notes for PyCaret. Every release entry is a summarised view of the engineering log in [`docs/revamp/release_notes_pycaret4.md`](docs/revamp/release_notes_pycaret4.md) — go there for full commit-by-commit detail.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [PEP 440](https://peps.python.org/pep-0440/).

---

## [4.0.0a1] — 2026-04-23

**Second test release of PyCaret 4.0** — focused on aggressive dependency discipline and unpinning scikit-learn.

### Install

```bash
pip install https://github.com/pycaret/pycaret/releases/download/v4.0.0a1/pycaret-4.0.0a1-py3-none-any.whl
```

### Changed

- **scikit-learn is now unpinned** — your install gets whatever the latest scikit-learn is (tested against 1.8). Previously pinned to `>=1.7,<1.8` because `sktime` capped it; `sktime` moved to the `timeseries` extra so the core install is no longer constrained.
- **Core install pulls ~41 packages** (vs. ~90+ in 3.x, ~65 in 4.0.0a0). sklearn 1.8, NumPy 2.4, pandas 3.0 on the default install.

### Removed from the default install

PyCaret 4.0 is opinionated: the library is a minimal ML control pane on top of scikit-learn. Everything below is now **user-installed** if wanted — pycaret's model / metric / plot registries auto-detect them via soft-dependency checks.

- `lightgbm`, `xgboost`, `catboost` — pick your own; pycaret lights the container up when it finds one installed.
- `kmodes`, `mlxtend` — same pattern.
- `optuna`, `optuna-integration`, `scikit-optimize`, `hyperopt` — the whole `tuners` extra is gone. Use sklearn's built-in `GridSearchCV` / `RandomizedSearchCV` / `HalvingRandomSearchCV`, or install a backend yourself and pass a fitted search-cv object to `tune_model`.
- `shap`, `interpret`, `umap-learn` — the whole `analysis` extra is gone. Interpretability is out of scope for 4.0 alpha; will return targeted in a later release.
- `prophet` — the entire `prophet` extra is gone.
- `matplotlib` — no longer a core dep. Lazy-imported with graceful fallbacks wherever a residual non-Plotly plot path exists. Plotly is the single chosen plot library for 4.0.
- `kaleido` — moved from core to the new `export` extra (only needed for static Plotly image export).
- `xxhash` — no longer core; `pycaret.internal.memory.FastMemory` now falls back to `hashlib.blake2b` if xxhash isn't available.
- `cloudpickle` — no longer core; `load_experiment()` lazy-imports it and raises a clean error if missing.
- `psutil` — no longer core; system-info logging falls back to stdlib `os.cpu_count()`.
- `nbformat` — moved to the `notebook` extra (only needed for run-notebook artifact generation in the forthcoming platform layer).
- `ipywidgets` — moved to the `notebook` extra.
- `ipython` — **kept** in core (widely used, small footprint, and the 4.0 Jupyter integration expects it).

### New extras structure

- `pycaret[notebook]` — `ipywidgets`, `nbformat` (for progress widgets + run-notebook artifact generation).
- `pycaret[export]` — `kaleido` (static Plotly image export).
- `pycaret[anomaly]` — `pyod`, `numba` (AnomalyExperiment backend).
- `pycaret[timeseries]` — `sktime`, `statsmodels`, `pmdarima` (TimeSeriesExperiment backend).
- `pycaret[full]` — all of the above in one go.

### Fixed

- `lightgbm` container in `pycaret.containers.models.*` now gracefully disables itself when lightgbm is missing (was ImportError-at-load before).

### Known transitional deps (still in core; targeted for removal)

- `imbalanced-learn` — required because `pycaret.internal.pipeline.Pipeline` inherits from `imblearn.pipeline.Pipeline`. Phase 4 preprocessor rewrite removes this.
- `category-encoders` — used by the legacy preprocessor. sklearn 1.3+ has native `OrdinalEncoder` / `OneHotEncoder` / `TargetEncoder` replacements; Phase 4 swaps them in.

These two are listed as transitional in `pyproject.toml` and will be cut in a future alpha.

---

## [4.0.0a0] — 2026-04-23

**First test release of PyCaret 4.0.** This is an alpha — the OOP public API is stable, but the legacy internal implementation is still being progressively replaced. Feel free to install, try, and file feedback; do **not** rely on it for production workloads yet.

### Install

```bash
pip install --pre pycaret==4.0.0a0
# or with every optional extra:
pip install --pre "pycaret[full]==4.0.0a0"
```

Python 3.11 / 3.12 / 3.13 supported.

### Quickstart

```python
from pycaret.datasets import get_data
from pycaret.tasks import ClassificationExperiment
from pycaret import save_model, load_model

df = get_data("juice")
exp = ClassificationExperiment(target="Purchase", session_id=42).fit(df)
result = exp.compare_models()
best = result.best
preds = exp.predict_model(best).predictions
save_model(best, "juice_classifier")
```

Five task classes available:

```python
from pycaret.tasks import (
    ClassificationExperiment,
    RegressionExperiment,
    ClusteringExperiment,
    AnomalyExperiment,
    TimeSeriesExperiment,
)
```

### Changed

- **The public API is now OOP-only.** The 3.x module-level functional API (`setup()`, `compare_models()`, ...) has been removed. Every task has one `Experiment` subclass with a fluent method interface.
- **`Experiment` subclasses are `sklearn.base.BaseEstimator`.** `get_params()`, `set_params()`, `sklearn.base.clone()`, and `__sklearn_tags__()` all work. You can pickle, clone, and introspect experiments like any other sklearn object.
- **Every verb returns a typed dataclass.** `compare_models()` returns `CompareResult`; `tune_model()` returns `TuneResult`; `predict_model()` returns `PredictResult`; and so on. All dataclasses live in `pycaret.core.results`. The old notebook-friendly DataFrame is still available as `result.leaderboard` / `result.predictions` / etc.
- **Structured event stream.** Every long-running operation emits typed `Event` dataclasses through `pycaret.logging.BaseLogger`. Subscribe via `logger.subscribe(callback)` for real-time progress in UIs and agents. `MemoryLogger` is the default when `log_experiment=True`.
- **Typed introspection API for UIs and LLM agents.** `pycaret.api.list_models(task)`, `describe_model(task, id)`, `list_metrics(task)`, `describe_setup_params(task)` return JSON-serializable dataclasses you can feed directly to a React form or an LLM prompt.
- **Class rename:** `TSForecastingExperiment` → `TimeSeriesExperiment`. Matches the task module name.

### Removed

Clean break from 3.x. These dependencies and features were cut to keep the library lean and modern:

- **Trackers:** `mlflow`, `comet-ml`, `wandb`, `dagshub`. Replaced by the built-in event stream. If you need external-tracker integration, subclass `BaseLogger`.
- **Distributed / parallel:** `fugue`, `dask`, `ray[tune]`, `tune-sklearn`. No replacement — PyCaret is a single-node library in 4.0.
- **Visualisation adapters:** `yellowbrick`, `mljar-scikit-plot`, `schemdraw`, `plotly-resampler`. Plotly-native plot rewrite is on the roadmap.
- **Deploy/serving helpers:** `create_api`, `create_app`, `create_docker`, `deploy_model` (S3), `dashboard`, `convert_model`, `eda`, `check_drift`, `check_fairness`. These will live in the forthcoming `pycaret-server` package instead.
- **GPU via `scikit-learn-intelex` / `cuml`:** out of scope for 4.0 core. GPU-accelerated gradient-boosted models (xgboost / lightgbm / catboost) still work via their own flags.
- **Module-level state:** `set_current_experiment()`, `get_current_experiment()`, and the `_CURRENT_EXPERIMENT` global are gone. Hold onto your `Experiment` instance directly.

See [`docs/revamp/KILL_LIST.md`](docs/revamp/KILL_LIST.md) for the exhaustive list with rationale.

### Fixed

- Runs on modern Python (3.11 / 3.12 / 3.13), modern scikit-learn (1.7), modern NumPy (2.x), modern pandas (2.x). The 3.x upper-bound version pins have been removed.
- `distutils.LooseVersion` usage replaced with `packaging.version.Version` (Python 3.12 dropped `distutils`).
- `np.NaN`, `np.product` usage updated to NumPy 2.0 equivalents.
- `joblib.Memory(bytes_limit=...)` — updated to joblib 1.4+ API.
- `sklearn._check_reg_targets` — updated to sklearn 1.7 signature.

### Dependencies

Core runtime dependencies cut from 30 → 19. Optional extras reorganised into focused tiers: `models`, `tuners`, `analysis`, `anomaly`, `timeseries`, `prophet`, `full`.

### Known limitations in this alpha

- **Legacy internals are still in place.** The new `Experiment` classes delegate to the 3.x god-class under the hood via `self._legacy`. Each verb is being rewritten natively on top of `sklearn.pipeline.Pipeline` in subsequent releases. Public API is stable; internals will churn.
- **Plot rewrite not complete.** Some plot types still error if Plotly hasn't replaced them yet — tracking in ROADMAP Phase 3.
- **Python 3.14 not yet supported.** Upstream `joblib` / `cloudpickle` need to ship PEP 649 support first; see `docs/revamp/thinking/2026-04-22_python314_pep649_blocker.md`.

### Documentation

- [`README.md`](README.md) — quickstart and positioning.
- [`AGENTS.md`](AGENTS.md) — briefing for AI coding agents contributing to PyCaret.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — human contributor guide.
- [`docs/for_agents/`](docs/for_agents/) — deep dives (engine walkthrough, typed results, event stream, introspection API, task cheatsheet).
- [`docs/for_developers/`](docs/for_developers/) — setup, testing, god-class-draining playbook, coding style, release process.
- [`docs/revamp/`](docs/revamp/) — the full engineering narrative (audit, architecture, roadmap, decisions, release notes, platform plan).
- [`notebooks/`](notebooks/) — five executed end-to-end examples, one per task.

### Migrating from 3.x

**Short answer: don't, unless you want to.** 3.x keeps working; PyPI serves it as the default `pip install pycaret`. 4.0 is opt-in via the pre-release flag.

If you do want to migrate, the shape is:

| 3.x (functional) | 4.0 (OOP) |
|---|---|
| `from pycaret.classification import setup, compare_models` | `from pycaret.tasks import ClassificationExperiment` |
| `setup(data, target='y')` | `exp = ClassificationExperiment(target='y').fit(data)` |
| `compare_models()` | `exp.compare_models()` → `CompareResult` |
| `pull()` | `result.leaderboard` |
| `tune_model(model)` | `exp.tune_model(model).pipeline` |
| `predict_model(m, data=new_df)` | `exp.predict_model(m, data=new_df).predictions` |
| `save_model(m, 'f')` | `from pycaret import save_model; save_model(m, 'f')` |

More migration patterns in [`notebooks/README.md`](notebooks/README.md).

### License

MIT. Unchanged from 3.x.

---

*For every non-trivial change in this release (~180 entries across 6 sessions), see [`docs/revamp/release_notes_pycaret4.md`](docs/revamp/release_notes_pycaret4.md).*
