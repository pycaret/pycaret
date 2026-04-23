# Phase 0 Failure Landscape

*Source: `phase0_pytest_run1.log`, 2026-04-22, Python 3.13.13 + sklearn 1.7.2 + NumPy 2.3.5 + pandas 2.x*

## Headline numbers

| Bucket | Count | % of run |
|---|---:|---:|
| **Passed** | **568** | **77.4%** |
| Failed | 158 | 21.5% |
| Skipped | 8 | 1.1% |
| Collected | 734 | — |
| Runtime | 34:26 | — |
| Warnings | 9,443 | — |

(Test count further dropped to ~706 after deleting the sklearnex-coupled engine test files — `test_regression_engines.py`, `test_classification_engines.py`, `test_time_series_engines.py` — during the post-run cleanup.)

## Failure root-cause clusters

Sorted by blast radius. Numbers are *our estimate* of how many of the 158 failures each cluster explains; they sum to slightly more than 158 because some tests fail for compound reasons.

### A. Time-series — pandas `PeriodIndex` `name` drift · **~90 failures**

Symptom:
```
assert PeriodIndex([...], name='period[M]') == PeriodIndex([...], name='Period')
```
Clusters: `test_time_series_base.py` (30), `test_time_series_tune_grid.py` (28), `test_time_series_tune_random.py` (28), `test_time_series_tune_base.py` (6), `test_time_series_blending.py` (5), `test_time_series_exogenous.py` (2), `test_time_series_setup.py` (2).

Likely cause: pandas 2.x changed the default `name` on `PeriodIndex.from_timestamps` to `'period[M]'` (the frequency alias) instead of `'Period'`. pycaret's test assertions hard-coded `'Period'`. Fix is almost certainly in the test harness, not production code. Single-point repair.

### B. Preprocess — `IterativeImputer._validate_data` removed · **13 failures**

Symptom:
```
AttributeError: 'IterativeImputer' object has no attribute '_validate_data'.
Did you mean: '_validate_params'?
```
Cluster: `test_preprocess.py::test_iterative_imputer[*]`.

Cause: sklearn 1.7 removed the private `_validate_data` method in favour of a different validation pathway (`validate_data` free function / `_validate_params`). pycaret vendors its own `IterativeImputer` in `pycaret/internal/preprocess/iterative_imputer.py` which still calls `self._validate_data(...)`. Fix: update the vendored imputer to use the new sklearn validation helpers. Consider dropping it entirely if sklearn's stock `IterativeImputer` now covers the feature set.

### C. Tune/tunable estimators — attribute / API drift · **~6 failures**

Symptoms include:
```
AttributeError: 'EnsembleForecaster' object has no attribute '__sklearn_tags__'
AttributeError (tunable_mlp, tunable_voting_estimator)
```
Cluster: `test_tune_model.py::test_tunable_{voting_estimator,mlp}[*]`.

Cause: sklearn 1.6+ introduced `__sklearn_tags__` as the replacement for `_get_tags()`. pycaret's `internal/tunable.py` + sktime's `EnsembleForecaster` both pre-date this and don't implement it. Fix: implement `__sklearn_tags__` on the tunable subclasses; file an upstream issue / workaround for sktime's forecasters that pycaret wraps.

### D. sklearnex engine tests · **17 failures (now deleted)**

Symptoms:
```
AssertionError: assert (False or False)  # parent_library.startswith('sklearnex')
```
Clusters: `test_regression_engines.py` (10), `test_classification_engines.py` (7), `test_time_series_engines.py` (4 of its failures).

Cause: tests assume the Intel oneAPI sklearn extension is installed and active. We cut `scikit-learn-intelex` from the kill list. **Post-run cleanup deleted these three test files** — the feature is gone, the tests are gone. Will not appear in the Phase 2 run.

### E. Logger refactor regression · **fixed this session, count not included above**

Symptom:
```
AttributeError: 'bool' object has no attribute 'log_experiment'
```

Cause: my Phase-0 rewrite of `_convert_log_experiment` returned `False` when tracking was off; downstream code assumed a logger object. **Fixed** by having `_convert_log_experiment` always return a `BaseLogger` instance (the default no-op), and by expanding `BaseLogger` in `pycaret/loggers/base_logger.py` to implement the full hook surface (`log_experiment`, `log_model`, `log_model_comparison`, `log_plot`, `.loggers`, etc.) as no-ops.

### F. sklearn `_check_reg_targets` signature change · **fixed this session**

Symptom:
```
TypeError: _check_reg_targets() missing 1 required positional argument: 'multioutput'
ValueError: too many values to unpack (expected 4)
```

Cause: sklearn 1.7 added `sample_weight` as a required positional to `_check_reg_targets` and expanded the return from 4 to 5 values. pycaret's custom MAPE container was calling the old shape. **Fixed** in `pycaret/containers/metrics/regression.py`.

### G. Miscellaneous · **~20 failures**

- `test_utils.py::test_utils` — categorical levels mismatch (likely pandas 2.x `CategoricalDtype` drift).
- `test_multiclass.py` (3 failures) — probably AD-hoc sklearn API drift.
- `test_overflow.py` (2) — numeric overflow handling; could be NumPy 2.
- `test_convert_model.py` (2) — m2cgen feature was killed but tests remain; delete.
- `test_probability_threshold.py` (1) — probably metric-driven.
- `test_regression_plots.py` (1) — plot-model path; Phase 3 will rewrite plots anyway.
- `test_regression.py` (4) / `test_classification.py` (4) — post-mlflow cleanup leftovers; likely the same logger regression (E) that bit other places.

## What this tells us about Phase 2

Phase 2 ("Modernization — compat with current sklearn / NumPy / pandas / Python") now has a concrete punch list ordered by ROI:

1. **One-file fix in `internal/preprocess/iterative_imputer.py`** — unblocks 13 tests.
2. **Hunt-and-replace in the TS test fixtures** for `PeriodIndex(name='Period')` → `name='period[M]'`. Unblocks ~90 tests if the pattern is consistent.
3. **Add `__sklearn_tags__` to `internal/tunable.py`** — unblocks ~6 tests and any downstream user whose model wraps a pycaret tunable.
4. **Delete `test_convert_model.py`** (m2cgen feature is killed).
5. Fold remaining ~10 scattered failures into a clean second run after (1)–(4).

If (1)–(4) land, projected pass rate jumps from **77% → ~92%**, which would take the suite to credibly green enough to call Phase 2 complete for supervised modules.

## What this does NOT tell us

- **Nothing about Phase 3 (Plotly plot rewrite).** Plot-model paths mostly hit the `_v4_removed` stubs at runtime which would show up as `NotImplementedError` in plot-specific tests. Only 1 such test (`test_regression_plots.py`) failed — because most plot tests are in the skipped / parameterised-only-on-certain-plots bracket. Phase 3 needs its own failure inventory once the stubs are replaced.
- **Nothing about parallelised / distributed behaviour** — cut.
- **Nothing about tracking correctness** — we stubbed loggers as no-ops; the *real* 4.0 logger lands in Phase 4 with its own test set.
