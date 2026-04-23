# PyCaret 4.0 — Open-issue triage

*Auto-generated from `open_issues_raw.json` by `scripts/triage_issues.py`.*

Total open issues: **388**. Triage buckets:

| Bucket | Count | % | Suggested action |
|---|---:|---:|---|
| `fixed_in_4_0` | 8 | 2% | Close with pointer to `release_notes_pycaret4.md` and the 4.0 release announcement. |
| `out_of_scope` | 92 | 24% | Close with pointer to `KILL_LIST.md`. Optionally tag `wontfix-in-core` and leave for a community-maintained `pycaret-extras` repo. |
| `stale` | 123 | 32% | Reopener ping: "Does this still reproduce on PyCaret 4.0?" Auto-close after 30 days of silence. |
| `still_relevant_bug` | 58 | 15% | Label `4.0-candidate`. Triage into Phase 5 repair queue. |
| `still_relevant_enhancement` | 107 | 28% | Label `4.0-candidate`. Decide per-issue whether to accept, defer to 4.1+, or close. |

---

## `fixed_in_4_0` — 8 issue(s)

**Action:** Close with pointer to `release_notes_pycaret4.md` and the 4.0 release announcement.

| # | Title | Labels | Updated | Reason |
|---:|---|---|---|---|
| [#4173](https://github.com/pycaret/pycaret/issues/4173) | [BUG]: Unable to use in Python version 3.12 | bug | 2025-12-07 | Matches 4.0 revamp fixes: python\s*3\.1[2-9] |
| [#4123](https://github.com/pycaret/pycaret/issues/4123) | [ENH]: Support for pandas 2.2 | enhancement | 2025-02-19 | Matches 4.0 revamp fixes: pandas\s*2\.[2-9] |
| [#4121](https://github.com/pycaret/pycaret/issues/4121) | [ENH]: Support for Python 3.13 | enhancement | 2026-01-23 | Matches 4.0 revamp fixes: python\s*3\.1[2-9] |
| [#4054](https://github.com/pycaret/pycaret/issues/4054) | [ENH]: python 3.12 support failed | enhancement | 2025-04-04 | Matches 4.0 revamp fixes: python\s*3\.1[2-9] |
| [#3994](https://github.com/pycaret/pycaret/issues/3994) | [BUG]: inconsistent date columns transformations between training and infernece | bug | 2024-05-29 | Matches 4.0 revamp fixes: np\.NaN |
| [#3908](https://github.com/pycaret/pycaret/issues/3908) | [BUG]: Incompatible with Pandas 2.0 | bug | 2024-06-28 | Matches 4.0 revamp fixes: pandas\s*2\.[2-9] |
| [#3717](https://github.com/pycaret/pycaret/issues/3717) | [BUG]:  Pycaret applying normalize to categorical variables after encoding | bug | 2023-08-28 | Matches 4.0 revamp fixes: np\.NaN |
| [#2079](https://github.com/pycaret/pycaret/issues/2079) | AttributeError: 'numpy.ndarray' object has no attribute 'columns' | bug | 2022-04-14 | Matches 4.0 revamp fixes: np\.NaN |

## `out_of_scope` — 92 issue(s)

**Action:** Close with pointer to `KILL_LIST.md`. Optionally tag `wontfix-in-core` and leave for a community-maintained `pycaret-extras` repo.

| # | Title | Labels | Updated | Reason |
|---:|---|---|---|---|
| [#4170](https://github.com/pycaret/pycaret/issues/4170) | Issue finding out how to use results when using spark as a back end | documentation | 2025-12-28 | Body mentions killed feature(s): \bfugue\b |
| [#4161](https://github.com/pycaret/pycaret/issues/4161) | [BUG]: Longer reported times for initial execution of compare_models in regression experiment | bug | 2025-04-16 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4158](https://github.com/pycaret/pycaret/issues/4158) | [BUG]: compare_models with 'lightgbm' is 50 times slower than it should be | bug | 2025-03-25 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4148](https://github.com/pycaret/pycaret/issues/4148) | [BUG]: Cannot cast object dtype to float64 | bug | 2025-12-02 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4117](https://github.com/pycaret/pycaret/issues/4117) | [BUG]: AttributeError: 'SimpleImputer' object has no attribute 'keep_empty_features' | bug | 2025-05-21 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4112](https://github.com/pycaret/pycaret/issues/4112) | [BUG]: Timeseries forecasting ignores passed timerange and uses test set timerange (cutoff date) | bug | 2024-12-23 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4107](https://github.com/pycaret/pycaret/issues/4107) | [BUG]: ValueError: "Seasonal periodicity must be greater than 1" when tuning ARIMA model in Time Series | bug | 2025-08-19 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4100](https://github.com/pycaret/pycaret/issues/4100) | [BUG]: PyCaret MLflow logger incompatibility with thread safe release of MLflow | bug | 2024-12-13 | Title mentions killed feature(s): \bmlflow\b |
| [#4084](https://github.com/pycaret/pycaret/issues/4084) | [BUG]: error occured after execute second plot_model | bug | 2024-10-30 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4083](https://github.com/pycaret/pycaret/issues/4083) | [BUG]: test_data in classification set up does not work | bug | 2024-10-28 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4081](https://github.com/pycaret/pycaret/issues/4081) | [BUG]: with "fix_imbalance=True", finalise_model throws error | bug | 2025-05-21 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4080](https://github.com/pycaret/pycaret/issues/4080) | [BUG]: tune_model does not work with BaggingRegressor | bug | 2024-10-15 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4079](https://github.com/pycaret/pycaret/issues/4079) | [BUG]: When applying the smote algorithm, an error is reported | bug | 2024-10-13 | Body mentions killed feature(s): \bboto3\b, \bdaal4py\b, \bdask\b, \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \bscikit[_-]plot\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b, pandas[_-]profiling, scikit[_-]learn[_-]intelex\b |
| [#4071](https://github.com/pycaret/pycaret/issues/4071) | [BUG]: TypeError: create_model() got multiple values for argument 'estimator' | bug | 2024-09-11 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4051](https://github.com/pycaret/pycaret/issues/4051) | [BUG]: Feature selection in pycaret.classification setup function error | bug | 2025-08-03 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4049](https://github.com/pycaret/pycaret/issues/4049) | [BUG]: test_data in setup not work despite having same structure as data | bug | 2024-10-30 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4028](https://github.com/pycaret/pycaret/issues/4028) | [BUG]: compare_models in parallel mode fails when include parameter is set to empty list  | bug | 2024-08-01 | Body mentions killed feature(s): \bdask\b, \bfugue\b |
| [#4026](https://github.com/pycaret/pycaret/issues/4026) | [BUG]: tune_model use search_library='optuna' can not limit cpu usage | bug | 2024-07-26 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \bsklearnex\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4023](https://github.com/pycaret/pycaret/issues/4023) | [BUG]: AttributeError for `log_profile=True` | bug | 2024-07-25 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#4018](https://github.com/pycaret/pycaret/issues/4018) | [BUG]:  finalize_model does not work when test_data and groupKfold are used. | bug | 2025-08-01 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3999](https://github.com/pycaret/pycaret/issues/3999) | [BUG]: blend_models() and stack_models() Fail with Certain Models Above 1000 Samples | bug | 2026-04-08 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3995](https://github.com/pycaret/pycaret/issues/3995) | [BUG]: not all models compared in `compare_models` | bug | 2024-10-28 | Body mentions killed feature(s): \bsklearnex\b |
| [#3993](https://github.com/pycaret/pycaret/issues/3993) | [BUG]: ValueError: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead. | bug | 2024-06-17 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3990](https://github.com/pycaret/pycaret/issues/3990) | The check drift functions doesn`t generate a drift report file using the evidently library[BUG]:  | bug | 2024-06-02 | Title mentions killed feature(s): \bevidently\b |
| [#3989](https://github.com/pycaret/pycaret/issues/3989) | [BUG]: Inability to control automatic evaluation metrics, makes clustering infeasible for large datasets | bug | 2024-06-17 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3986](https://github.com/pycaret/pycaret/issues/3986) |  create_api does not annotate the data model correctly | bug | 2024-06-22 | Title mentions killed feature(s): \bcreate_api\b |
| [#3979](https://github.com/pycaret/pycaret/issues/3979) | [BUG]: Cannot use predict_proba() on a finalized model | bug | 2024-04-19 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3973](https://github.com/pycaret/pycaret/issues/3973) | [BUG]: Custom metrics all report 0.0000 for classification in Pycaret 3.3.1 | bug, priority_high | 2024-08-16 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3961](https://github.com/pycaret/pycaret/issues/3961) | [BUG]: compare_models much slower in version 3.3.0 than in 2.X | bug | 2026-04-06 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3957](https://github.com/pycaret/pycaret/issues/3957) | [BUG]: Allow Custom Logger Configuration to Handle Prophet Dependency Warning | bug | 2024-04-04 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3947](https://github.com/pycaret/pycaret/issues/3947) | [BUG]: fix_imbalance_method does not work for SMOTENC | bug | 2024-11-08 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3945](https://github.com/pycaret/pycaret/issues/3945) | [BUG]:  Pydantic model generation issues. | bug | 2024-07-14 | Body mentions killed feature(s): \bcreate_api\b, \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3929](https://github.com/pycaret/pycaret/issues/3929) | [BUG]: Problems with Numeric features in setup() | bug | 2024-02-29 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3928](https://github.com/pycaret/pycaret/issues/3928) | [BUG]: When using compare_models(), some models do not have default hyperparameters and cannot be used as Baselines. | bug | 2024-02-29 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3925](https://github.com/pycaret/pycaret/issues/3925) | [BUG]: compare_models() runs forever when cross-validating on lgbm using n_jobs = -1 | bug | 2024-02-28 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3915](https://github.com/pycaret/pycaret/issues/3915) | [BUG]: ClassificationExperiment().optimize_threshold() raises error 'CustomProbabilityThresholdClassifier' has no len() | bug | 2024-06-12 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3907](https://github.com/pycaret/pycaret/issues/3907) | [BUG]: `experiment.automl()` fails when called on the experiment where training is distributed (with spark) | bug | 2024-02-20 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3901](https://github.com/pycaret/pycaret/issues/3901) | ImportError: cannot import name '_PredictScorer' from 'sklearn.metrics._scorer' | bug | 2024-06-07 | Body mentions killed feature(s): \bcheck_drift\b, \bcheck_fairness\b, \bconvert_model\b, \bcreate_api\b, \bcreate_app\b, \bcreate_docker\b, \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3900](https://github.com/pycaret/pycaret/issues/3900) | [BUG]: incorrect precision, recall, and f1-score for object dtype target | bug | 2024-02-20 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3898](https://github.com/pycaret/pycaret/issues/3898) | [BUG]: For some unrecognized reason, clustering always stops at 67% | bug | 2024-02-21 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3888](https://github.com/pycaret/pycaret/issues/3888) | [BUG]: Not Able to Log-metrics In Mlflow  | bug | 2024-01-26 | Title mentions killed feature(s): \bmlflow\b |
| [#3877](https://github.com/pycaret/pycaret/issues/3877) | [BUG]: return_train_score not properly propagated in tune_model | bug | 2024-01-08 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3873](https://github.com/pycaret/pycaret/issues/3873) | [ENH]: Add signature when logging model to the mlflow server | enhancement | 2024-02-28 | Title mentions killed feature(s): \bmlflow\b |
| [#3872](https://github.com/pycaret/pycaret/issues/3872) | [BUG]: create_api with nan in dataset does not work | bug | 2024-01-03 | Title mentions killed feature(s): \bcreate_api\b |
| [#3865](https://github.com/pycaret/pycaret/issues/3865) | [BUG]: AttributeError: 'LogisticRegression' object has no attribute 'ax' | bug | 2024-01-12 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3853](https://github.com/pycaret/pycaret/issues/3853) | [BUG]: failed to import because of `AttributeError: module 'matplotlib'` | bug | 2023-12-19 | Body mentions killed feature(s): \bcheck_drift\b, \bcheck_fairness\b, \bconvert_model\b, \bcreate_api\b, \bcreate_app\b, \bcreate_docker\b |
| [#3852](https://github.com/pycaret/pycaret/issues/3852) | [BUG]: wrong parameter name in `plot_model` in oop | bug | 2023-12-13 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3836](https://github.com/pycaret/pycaret/issues/3836) | [BUG]: predict_model not working for anomaly | bug | 2023-12-08 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3805](https://github.com/pycaret/pycaret/issues/3805) | [BUG]: MLFLow Transformation Pipeline and Model not saved together | bug | 2023-12-01 | Title mentions killed feature(s): \bmlflow\b |
| [#3796](https://github.com/pycaret/pycaret/issues/3796) | [BUG]: generated API with `create_api` does not annotate the data model correctly / not at all | bug | 2023-11-29 | Title mentions killed feature(s): \bcreate_api\b |
| [#3779](https://github.com/pycaret/pycaret/issues/3779) | [BUG]: tune_model not working with search_algorithm="optuna" and search_library="tune-sklearn" | bug | 2023-10-13 | Title mentions killed feature(s): \btune[_-]sklearn\b |
| [#3774](https://github.com/pycaret/pycaret/issues/3774) | [BUG]: KMeans.predict() raises a  ValueError: Buffer dtype mismatch | bug | 2024-07-23 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \bydata[_-]profiling\b, \byellowbrick\b |
| [#3765](https://github.com/pycaret/pycaret/issues/3765) | [BUG]: `tune_model` fails with `ValueError: No experiment checkpoint file` | bug | 2023-10-17 | Body mentions killed feature(s): \bray[\s_/-]+tune\b, \btune[_-]sklearn\b |
| [#3764](https://github.com/pycaret/pycaret/issues/3764) | [BUG]: Error in FugueBackend or TSForecastingExperiment | bug, time_series, parallelization | 2023-10-09 | Body mentions killed feature(s): \bfugue\b |
| [#3746](https://github.com/pycaret/pycaret/issues/3746) | [BUG]: plot_kwargs not passed to yellowbrick visualizers in function plot_model() | bug | 2024-04-14 | Title mentions killed feature(s): \byellowbrick\b |
| [#3724](https://github.com/pycaret/pycaret/issues/3724) | [BUG]: PyCaret does not inform shap for Shapley value calculations that independent variables had been normalized when using 'normalize=True' in data setup for regression. | bug | 2023-09-03 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3716](https://github.com/pycaret/pycaret/issues/3716) | [BUG]: convert_model raises an error | bug | 2023-08-28 | Title mentions killed feature(s): \bconvert_model\b |
| [#3691](https://github.com/pycaret/pycaret/issues/3691) | [BUG]: Load and Tune Models | bug | 2024-02-02 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3653](https://github.com/pycaret/pycaret/issues/3653) | [BUG]: evaluate and plot function builds model based on default threshold despite threshold being manually set in create model function  | bug | 2024-02-02 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3644](https://github.com/pycaret/pycaret/issues/3644) | [BUG]: AttributeError: 'ClassificationExperiment' object has no attribute 'logging_param' | bug | 2023-07-20 | Body mentions killed feature(s): \bcreate_api\b, \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3625](https://github.com/pycaret/pycaret/issues/3625) | [ENH]: Improving Parallelization Efficiency | enhancement, parallelization | 2023-07-14 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \bplotly[_-]resampler\b, \bschemdraw\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3624](https://github.com/pycaret/pycaret/issues/3624) | [ENH]: Poetry and Pycaret don't work together | enhancement, installation | 2025-01-19 | Body mentions killed feature(s): \bplotly[_-]resampler\b |
| [#3602](https://github.com/pycaret/pycaret/issues/3602) | [BUG]: Predict with External Regressors | bug, time_series | 2024-02-02 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3578](https://github.com/pycaret/pycaret/issues/3578) | [BUG]: Classification "AttributeError: module 'h11' has no attribute 'Event'" | bug, classification, missing_info | 2023-06-09 | Body mentions killed feature(s): \bgradio\b |
| [#3501](https://github.com/pycaret/pycaret/issues/3501) | [BUG]: Some Clustering plots have .png extension instead of .html when saved to disk | enhancement, clustering | 2024-03-04 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3468](https://github.com/pycaret/pycaret/issues/3468) | [BUG]: "interpret_model" not working correctly | bug, interpret_model | 2024-02-02 | Body mentions killed feature(s): \bmlflow\b |
| [#3420](https://github.com/pycaret/pycaret/issues/3420) | [BUG]: Error: Kernel dies when creating a random forest model with use_gpu=True. | bug, gpu | 2024-02-02 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3417](https://github.com/pycaret/pycaret/issues/3417) | [BUG]: predict_model output show unusual high number in error metrics | bug, regression, time_series, metrics | 2024-02-02 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3371](https://github.com/pycaret/pycaret/issues/3371) | [BUG]: Long runtime for Pycaret setup function on large dataset | bug, priority_high, long_run_time | 2024-11-07 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3177](https://github.com/pycaret/pycaret/issues/3177) | [BUG]: compare_models fills master_model_container with unfitted estimators and get_leaderboard takes very long | bug, compare_models, leaderboard | 2022-12-15 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#3011](https://github.com/pycaret/pycaret/issues/3011) | [BUG]: Number of folds not reported correctly when passing ExpandingWindowSplitter | bug, good first issue, time_series | 2025-05-21 | Body mentions killed feature(s): \byellowbrick\b |
| [#2978](https://github.com/pycaret/pycaret/issues/2978) | [ENH]: experiment_custom_tags for tune_model, etc. | enhancement | 2025-01-16 | Body mentions killed feature(s): \bmlflow\b |
| [#2933](https://github.com/pycaret/pycaret/issues/2933) | [ENH]: ExplainerDashboard: add function to get explainer | enhancement | 2022-09-02 | Title mentions killed feature(s): \bexplainerdashboard\b |
| [#2836](https://github.com/pycaret/pycaret/issues/2836) | [BUG]: fit_kwargs parameters break pycaret.time_series.compare_models() | enhancement, regression, classification, time_series, compare_models | 2024-02-02 | Body mentions killed feature(s): \byellowbrick\b |
| [#2738](https://github.com/pycaret/pycaret/issues/2738) | [BUG]: [BUG]: tune_sklearn not working | bug | 2022-10-11 | Title mentions killed feature(s): \btune[_-]sklearn\b |
| [#2694](https://github.com/pycaret/pycaret/issues/2694) | [ENH]: Enhancements for `plotly-resampler` | enhancement, time_series, plot_model | 2024-02-02 | Title mentions killed feature(s): \bplotly[_-]resampler\b |
| [#2610](https://github.com/pycaret/pycaret/issues/2610) | [ENH]: Log additional parameters to mlflow | enhancement, mlflow | 2022-05-31 | Title mentions killed feature(s): \bmlflow\b |
| [#2533](https://github.com/pycaret/pycaret/issues/2533) | [BUG]: Custom pipeline not working (classification) | bug, classification, preprocessing | 2024-02-02 | Body mentions killed feature(s): \bevidently\b, \bexplainerdashboard\b, \bfairlearn\b, \bfugue\b, \bgradio\b, \bm2cgen\b, \bmlflow\b, \btune[_-]sklearn\b, \byellowbrick\b, pandas[_-]profiling |
| [#2181](https://github.com/pycaret/pycaret/issues/2181) | Influential data points - get indexes | enhancement, regression, plot_model | 2022-02-21 | Body mentions killed feature(s): \byellowbrick\b |
| [#2144](https://github.com/pycaret/pycaret/issues/2144) | [BUG] ValueError: Found input variables with inconsistent numbers of samples: [13204, 26408] | bug, classification | 2022-08-10 | Body mentions killed feature(s): \byellowbrick\b |
| [#2001](https://github.com/pycaret/pycaret/issues/2001) | to_csv is slow and cost lot storage. Should use to_feather(), use arrow ipc format? | enhancement, good first issue, priority_low | 2022-11-23 | Body mentions killed feature(s): \bmlflow\b |
| [#1955](https://github.com/pycaret/pycaret/issues/1955) | Option for Kubeflow as MLOps | enhancement, mlops | 2022-03-06 | Body mentions killed feature(s): \bmlflow\b |
| [#1780](https://github.com/pycaret/pycaret/issues/1780) | [BUG] Plots are being saved without labels and titles. | bug, plot_model | 2022-05-10 | Body mentions killed feature(s): \byellowbrick\b |
| [#1648](https://github.com/pycaret/pycaret/issues/1648) | Time Series Module Features & Roadmap | time_series, roadmap, release | 2023-03-12 | Body mentions killed feature(s): \bmlflow\b |
| [#1637](https://github.com/pycaret/pycaret/issues/1637) | Add FLAML as a search_library in tune_model function | enhancement, tune_model, priority_low | 2024-02-02 | Body mentions killed feature(s): \bray[\s_/-]+tune\b |
| [#1610](https://github.com/pycaret/pycaret/issues/1610) | Save plots in other formats, vectorial and image | enhancement | 2021-11-10 | Body mentions killed feature(s): \byellowbrick\b |
| [#1448](https://github.com/pycaret/pycaret/issues/1448) | Model Monitoring feature integration with Pycaret | enhancement, mlflow, logging | 2021-12-29 | Body mentions killed feature(s): \bevidently\b |
| [#1411](https://github.com/pycaret/pycaret/issues/1411) | [BUG]Can we store mlflow artifacts in the Azure blob storage for new experiments in pycaret? | enhancement | 2022-11-22 | Title mentions killed feature(s): \bmlflow\b |
| [#1268](https://github.com/pycaret/pycaret/issues/1268) | Support for "whylogs" in PyCaret | enhancement, logging | 2021-10-22 | Body mentions killed feature(s): \bmlflow\b |
| [#1186](https://github.com/pycaret/pycaret/issues/1186) | Set MLFLOW output directory | enhancement, mlflow | 2021-05-04 | Title mentions killed feature(s): \bmlflow\b |
| [#833](https://github.com/pycaret/pycaret/issues/833) | Pycaret model monitoring | enhancement, open for contribution | 2021-07-24 | Body mentions killed feature(s): \bmlflow\b |
| [#421](https://github.com/pycaret/pycaret/issues/421) | pycaret 2.0: artifacts not getting logged | help wanted, mlflow | 2022-12-06 | Body mentions killed feature(s): \bmlflow\b |

## `stale` — 123 issue(s)

**Action:** Reopener ping: "Does this still reproduce on PyCaret 4.0?" Auto-close after 30 days of silence.

| # | Title | Labels | Updated | Reason |
|---:|---|---|---|---|
| [#3215](https://github.com/pycaret/pycaret/issues/3215) | [ENH]: Handling sessions with Pycaret | enhancement, mlflow | 2022-12-29 | No update since 2022-12-29 (> 2 years) |
| [#3204](https://github.com/pycaret/pycaret/issues/3204) | [ENH]: use_holdout argument for get_leaderboard() function in regression and classification | enhancement | 2022-12-28 | No update since 2022-12-28 (> 2 years) |
| [#3203](https://github.com/pycaret/pycaret/issues/3203) | [ENH]: model_only arg for automl() in regression and classification | enhancement | 2022-12-26 | No update since 2022-12-26 (> 2 years) |
| [#3168](https://github.com/pycaret/pycaret/issues/3168) | [ENH]: Add a new model to the Time Series module: TemporalFusionTransformer | enhancement, time_series | 2022-12-11 | No update since 2022-12-11 (> 2 years) |
| [#3160](https://github.com/pycaret/pycaret/issues/3160) | [BUG]: Stratify split for regression using pycaret | bug, regression | 2022-12-10 | No update since 2022-12-10 (> 2 years) |
| [#3089](https://github.com/pycaret/pycaret/issues/3089) | [ENH]: Integration of new algorithm: LambdaMart | enhancement | 2022-12-05 | No update since 2022-12-05 (> 2 years) |
| [#3088](https://github.com/pycaret/pycaret/issues/3088) | Anomaly detection reverse!? | enhancement | 2022-11-15 | No update since 2022-11-15 (> 2 years) |
| [#3025](https://github.com/pycaret/pycaret/issues/3025) | [ENH]: LightGBMError: Do not support special JSON characters in feature name. | enhancement | 2022-10-08 | No update since 2022-10-08 (> 2 years) |
| [#3024](https://github.com/pycaret/pycaret/issues/3024) | [ENH]: Line thickness in ROC plot | enhancement | 2022-10-07 | No update since 2022-10-07 (> 2 years) |
| [#3015](https://github.com/pycaret/pycaret/issues/3015) | [BUG]: The LR model does not predict correctly after tune_model with search_library='optuna' | bug | 2022-11-21 | No update since 2022-11-21 (> 2 years) |
| [#3014](https://github.com/pycaret/pycaret/issues/3014) | [ENH]: Plots with several models | enhancement, priority_low | 2022-11-08 | No update since 2022-11-08 (> 2 years) |
| [#2990](https://github.com/pycaret/pycaret/issues/2990) | [BUG]: skopt custom search grid throws ValueError  | bug | 2022-12-21 | No update since 2022-12-21 (> 2 years) |
| [#2983](https://github.com/pycaret/pycaret/issues/2983) | [UPDATE]: parameters update | enhancement | 2022-11-21 | No update since 2022-11-21 (> 2 years) |
| [#2965](https://github.com/pycaret/pycaret/issues/2965) | [ENH]: TUNE_MODEL seems to run with n_jobs=1. Is it possible to run with more cpu? | enhancement | 2022-09-15 | No update since 2022-09-15 (> 2 years) |
| [#2950](https://github.com/pycaret/pycaret/issues/2950) | [ENH]: timeseries Compare_model with/without outliers | enhancement, anomaly_detection, time_series | 2022-09-21 | No update since 2022-09-21 (> 2 years) |
| [#2949](https://github.com/pycaret/pycaret/issues/2949) | [ENH]: For timeseries Compare_model has a plot  | enhancement, time_series, plot_model | 2022-09-10 | No update since 2022-09-10 (> 2 years) |
| [#2948](https://github.com/pycaret/pycaret/issues/2948) | [ENH]: Addition of PyThresh to the Anomaly library | enhancement, anomaly_detection | 2022-09-14 | No update since 2022-09-14 (> 2 years) |
| [#2936](https://github.com/pycaret/pycaret/issues/2936) | [ENH]: ability to return top n best models from holdout set score results | enhancement, automl | 2022-09-14 | No update since 2022-09-14 (> 2 years) |
| [#2921](https://github.com/pycaret/pycaret/issues/2921) | [ENH]: Completed model drawing | enhancement | 2022-08-29 | No update since 2022-08-29 (> 2 years) |
| [#2917](https://github.com/pycaret/pycaret/issues/2917) | [ENH]: Deep Q-Learning | enhancement | 2022-08-27 | No update since 2022-08-27 (> 2 years) |
| [#2916](https://github.com/pycaret/pycaret/issues/2916) | [ENH]: New feature engine: Feature-Engine | enhancement | 2022-08-26 | No update since 2022-08-26 (> 2 years) |
| [#2915](https://github.com/pycaret/pycaret/issues/2915) | [ENH]: New algorithm: PyGad | enhancement | 2022-08-26 | No update since 2022-08-26 (> 2 years) |
| [#2897](https://github.com/pycaret/pycaret/issues/2897) | [ENH]: time for tune model | enhancement, tune_model | 2022-08-21 | No update since 2022-08-21 (> 2 years) |
| [#2895](https://github.com/pycaret/pycaret/issues/2895) | [ENH]: Sample Weights for classification | enhancement | 2022-08-21 | No update since 2022-08-21 (> 2 years) |
| [#2890](https://github.com/pycaret/pycaret/issues/2890) | [ENH]: How to use a triple dataset split for a calibrated classifier | enhancement | 2022-08-22 | No update since 2022-08-22 (> 2 years) |
| [#2862](https://github.com/pycaret/pycaret/issues/2862) | [ENH]: Make minimum python version >= 3.8 | enhancement | 2022-08-19 | No update since 2022-08-19 (> 2 years) |
| [#2786](https://github.com/pycaret/pycaret/issues/2786) | [ENH]: Remove multicollinearity using VIF technique | enhancement, preprocessing | 2022-07-22 | No update since 2022-07-22 (> 2 years) |
| [#2778](https://github.com/pycaret/pycaret/issues/2778) | [ENH]: create useful features for machine learning | enhancement | 2022-07-19 | No update since 2022-07-19 (> 2 years) |
| [#2772](https://github.com/pycaret/pycaret/issues/2772) | [ENH]: interpret_model takes TOO long in big dataset (request auto sampling or parameter) | enhancement, interpret_model | 2022-07-27 | No update since 2022-07-27 (> 2 years) |
| [#2763](https://github.com/pycaret/pycaret/issues/2763) | [ENH]: class weight or sample weight for compare_models | enhancement | 2022-07-17 | No update since 2022-07-17 (> 2 years) |
| [#2755](https://github.com/pycaret/pycaret/issues/2755) | [ENH]: suggest add function for difference Feature combination in setup | enhancement | 2022-07-15 | No update since 2022-07-15 (> 2 years) |
| [#2746](https://github.com/pycaret/pycaret/issues/2746) | [ENH]: Pull data from ODBC on Quickstart page; PyODBC? | enhancement | 2022-07-12 | No update since 2022-07-12 (> 2 years) |
| [#2744](https://github.com/pycaret/pycaret/issues/2744) | [ENH]: Add parameter choose_better for calibrate_model() | enhancement | 2022-07-20 | No update since 2022-07-20 (> 2 years) |
| [#2713](https://github.com/pycaret/pycaret/issues/2713) | Pycaret and Dash Plotly Integration [ENH]:  | enhancement | 2022-07-05 | No update since 2022-07-05 (> 2 years) |
| [#2681](https://github.com/pycaret/pycaret/issues/2681) | Plot model (fold based) | enhancement | 2022-07-08 | No update since 2022-07-08 (> 2 years) |
| [#2651](https://github.com/pycaret/pycaret/issues/2651) | Feature Inverse Transform for better readability and expainability | enhancement, shapley | 2022-07-01 | No update since 2022-07-01 (> 2 years) |
| [#2631](https://github.com/pycaret/pycaret/issues/2631) | [ENH]: Automatic reboot kernel after pycaret installation | enhancement, installation | 2022-06-06 | No update since 2022-06-06 (> 2 years) |
| [#2617](https://github.com/pycaret/pycaret/issues/2617) | [ENH]:  Visualization of Model Comparison | enhancement | 2022-07-06 | No update since 2022-07-06 (> 2 years) |
| [#2526](https://github.com/pycaret/pycaret/issues/2526) | [ENH] Add Rocket/Mini-Rocket and MTS (Multiple Time Series) functionality for Classification, Regression, Clustering, and Anomaly Detection | enhancement, regression, classification, clustering, time_series | 2022-05-10 | No update since 2022-05-10 (> 2 years) |
| [#2525](https://github.com/pycaret/pycaret/issues/2525) | [ENH] How to cluster multiple time series? | enhancement, clustering, time_series | 2022-05-10 | No update since 2022-05-10 (> 2 years) |
| [#2491](https://github.com/pycaret/pycaret/issues/2491) | [ENH]: Plot interpret_model(reason) as .png | enhancement | 2022-05-03 | No update since 2022-05-03 (> 2 years) |
| [#2454](https://github.com/pycaret/pycaret/issues/2454) | [ENH]: mlnotify | enhancement | 2022-04-25 | No update since 2022-04-25 (> 2 years) |
| [#2403](https://github.com/pycaret/pycaret/issues/2403) | Show variable name in setup overview.  | enhancement, setup | 2022-04-14 | No update since 2022-04-14 (> 2 years) |
| [#2361](https://github.com/pycaret/pycaret/issues/2361) | pass cross_val = False tune_model method and use test_data for eval | enhancement, tune_model | 2022-04-02 | No update since 2022-04-02 (> 2 years) |
| [#2340](https://github.com/pycaret/pycaret/issues/2340) | Numeric confusion matrix does not match with percentage confusion matrix | bug, classification, plot_model | 2022-04-02 | No update since 2022-04-02 (> 2 years) |
| [#2315](https://github.com/pycaret/pycaret/issues/2315) | Add option to use only ONNX compatible models | enhancement, onnx | 2022-04-14 | No update since 2022-04-14 (> 2 years) |
| [#2310](https://github.com/pycaret/pycaret/issues/2310) | Exponential Smoothing Parameter Search Space Improvement | enhancement, time_series, models | 2022-03-21 | No update since 2022-03-21 (> 2 years) |
| [#2288](https://github.com/pycaret/pycaret/issues/2288) | Improve Code Quality | enhancement, priority_medium, refactor | 2022-09-12 | No update since 2022-09-12 (> 2 years) |
| [#2187](https://github.com/pycaret/pycaret/issues/2187) | Consider adding Brier score as default metric | enhancement, classification, metrics | 2022-08-21 | No update since 2022-08-21 (> 2 years) |
| [#2180](https://github.com/pycaret/pycaret/issues/2180) | [BUG] Save=True not working with clustering module. | bug, clustering, plot_model | 2022-02-19 | No update since 2022-02-19 (> 2 years) |
| [#2165](https://github.com/pycaret/pycaret/issues/2165) | [ENH] Support for non integer seasonal periods | enhancement, time_series, models | 2022-07-23 | No update since 2022-07-23 (> 2 years) |
| [#2157](https://github.com/pycaret/pycaret/issues/2157) | GitHub Actions: cancel old but running workflows of a PR when pushing again | enhancement, unit_tests | 2022-03-15 | No update since 2022-03-15 (> 2 years) |
| [#2156](https://github.com/pycaret/pycaret/issues/2156) | Is is possible to combine two ROC plots? | enhancement, classification, plot_model | 2022-02-24 | No update since 2022-02-24 (> 2 years) |
| [#2138](https://github.com/pycaret/pycaret/issues/2138) | Hill Climbing | enhancement | 2022-02-15 | No update since 2022-02-15 (> 2 years) |
| [#2128](https://github.com/pycaret/pycaret/issues/2128) | Issue with optimizes probability threshold Function  | bug | 2022-02-23 | No update since 2022-02-23 (> 2 years) |
| [#2096](https://github.com/pycaret/pycaret/issues/2096) | Add Support for Multivariate Time Series | enhancement, time_series, backlog, multivariate | 2022-01-26 | No update since 2022-01-26 (> 2 years) |
| [#2071](https://github.com/pycaret/pycaret/issues/2071) | Multiple classification models in pycaret | enhancement, priority_low | 2022-01-19 | No update since 2022-01-19 (> 2 years) |
| [#2022](https://github.com/pycaret/pycaret/issues/2022) | PyCaret + SKORCH : Issues in the step of 'predict_model' | bug | 2022-01-07 | No update since 2022-01-07 (> 2 years) |
| [#2014](https://github.com/pycaret/pycaret/issues/2014) | [BUG] ValueError: could not broadcast input array from shape (169,1) into shape (169,) | bug | 2022-04-15 | No update since 2022-04-15 (> 2 years) |
| [#2004](https://github.com/pycaret/pycaret/issues/2004) | tune_model(**kwargs) | bug | 2022-01-04 | No update since 2022-01-04 (> 2 years) |
| [#1986](https://github.com/pycaret/pycaret/issues/1986) | plot_model for hold-out test dataset | enhancement, plot_model | 2022-01-02 | No update since 2022-01-02 (> 2 years) |
| [#1957](https://github.com/pycaret/pycaret/issues/1957) | Average metrics for class_report plot | enhancement | 2021-12-15 | No update since 2021-12-15 (> 2 years) |
| [#1947](https://github.com/pycaret/pycaret/issues/1947) | Deep learning | enhancement | 2021-12-14 | No update since 2021-12-14 (> 2 years) |
| [#1937](https://github.com/pycaret/pycaret/issues/1937) | cross_validation = False scores | enhancement | 2021-12-12 | No update since 2021-12-12 (> 2 years) |
| [#1919](https://github.com/pycaret/pycaret/issues/1919) | Addition of new function create_report | enhancement | 2021-12-09 | No update since 2021-12-09 (> 2 years) |
| [#1907](https://github.com/pycaret/pycaret/issues/1907) | Create functionality to disable metrics | enhancement, metrics | 2022-01-31 | No update since 2022-01-31 (> 2 years) |
| [#1905](https://github.com/pycaret/pycaret/issues/1905) | Online learning features for Classification & Regression modules  | enhancement | 2021-11-30 | No update since 2021-11-30 (> 2 years) |
| [#1899](https://github.com/pycaret/pycaret/issues/1899) | How to save pictures in vector format | enhancement, plot_model | 2021-12-04 | No update since 2021-12-04 (> 2 years) |
| [#1898](https://github.com/pycaret/pycaret/issues/1898) | How to draw multiple models in an AUC curve chart? | enhancement, plot_model | 2022-01-11 | No update since 2022-01-11 (> 2 years) |
| [#1889](https://github.com/pycaret/pycaret/issues/1889) | Display multiple CI at plot_model() when plot='forecast' for ts module | enhancement, time_series, plot_model | 2022-03-16 | No update since 2022-03-16 (> 2 years) |
| [#1868](https://github.com/pycaret/pycaret/issues/1868) | Time Series \| Move Auto-ARIMA to turbo=False bucket | enhancement, time_series, models | 2022-04-18 | No update since 2022-04-18 (> 2 years) |
| [#1819](https://github.com/pycaret/pycaret/issues/1819) | Time Series - Single call to plot all plots... | enhancement, time_series, plot_model | 2022-03-16 | No update since 2022-03-16 (> 2 years) |
| [#1775](https://github.com/pycaret/pycaret/issues/1775) | White noise test lags in time series | enhancement, time_series | 2022-03-16 | No update since 2022-03-16 (> 2 years) |
| [#1754](https://github.com/pycaret/pycaret/issues/1754) | Adding Miss Classification table and plot | enhancement | 2021-10-25 | No update since 2021-10-25 (> 2 years) |
| [#1753](https://github.com/pycaret/pycaret/issues/1753) | Include interpretable models from imodels package | enhancement | 2021-12-04 | No update since 2021-12-04 (> 2 years) |
| [#1750](https://github.com/pycaret/pycaret/issues/1750) | Seasonality Box Plots & Heatmap | enhancement, time_series, plot_model | 2022-03-16 | No update since 2022-03-16 (> 2 years) |
| [#1721](https://github.com/pycaret/pycaret/issues/1721) | Add Global Forecasting | enhancement, time_series, models | 2022-05-01 | No update since 2022-05-01 (> 2 years) |
| [#1670](https://github.com/pycaret/pycaret/issues/1670) | AutoML for Time Series | enhancement, time_series, automl | 2021-12-04 | No update since 2021-12-04 (> 2 years) |
| [#1669](https://github.com/pycaret/pycaret/issues/1669) | Backtesting Time Series Strategies | enhancement, time_series, automl | 2021-12-02 | No update since 2021-12-02 (> 2 years) |
| [#1642](https://github.com/pycaret/pycaret/issues/1642) | remove_perfect_collinearity: bool, default = True | enhancement, preprocessing | 2022-05-16 | No update since 2022-05-16 (> 2 years) |
| [#1639](https://github.com/pycaret/pycaret/issues/1639) | [Feat] sklearn upgrade | enhancement, maintenance | 2021-10-10 | No update since 2021-10-10 (> 2 years) |
| [#1631](https://github.com/pycaret/pycaret/issues/1631) | Faster TSNE | enhancement | 2021-09-28 | No update since 2021-09-28 (> 2 years) |
| [#1552](https://github.com/pycaret/pycaret/issues/1552) | IllegalMonthError for any NaT values | enhancement | 2021-09-08 | No update since 2021-09-08 (> 2 years) |
| [#1549](https://github.com/pycaret/pycaret/issues/1549) | [BUG]  Cannot clone object DataTypes_Auto_infer as the constructor either does not set or modifies parameter categorical_features | bug | 2022-06-21 | No update since 2022-06-21 (> 2 years) |
| [#1517](https://github.com/pycaret/pycaret/issues/1517) | Statistical tests on Classification and Regression data and models | enhancement, regression, classification | 2022-09-06 | No update since 2022-09-06 (> 2 years) |
| [#1516](https://github.com/pycaret/pycaret/issues/1516) | Add DAE and AutoLGB from Kaggler library | enhancement | 2021-08-17 | No update since 2021-08-17 (> 2 years) |
| [#1504](https://github.com/pycaret/pycaret/issues/1504) | Optimal Number of Clusters | enhancement, clustering | 2021-09-20 | No update since 2021-09-20 (> 2 years) |
| [#1446](https://github.com/pycaret/pycaret/issues/1446) | feature dependant threshold optimization | enhancement | 2021-07-12 | No update since 2021-07-12 (> 2 years) |
| [#1422](https://github.com/pycaret/pycaret/issues/1422) | append algorithm names to create_model_container | enhancement | 2021-07-03 | No update since 2021-07-03 (> 2 years) |
| [#1413](https://github.com/pycaret/pycaret/issues/1413) | Add "TabNet" model to regression and classification algorithms | enhancement | 2021-11-08 | No update since 2021-11-08 (> 2 years) |
| [#1392](https://github.com/pycaret/pycaret/issues/1392) | Custom log file path in the setup function | enhancement, logging | 2021-06-23 | No update since 2021-06-23 (> 2 years) |
| [#1374](https://github.com/pycaret/pycaret/issues/1374) | Cleanup up default search space for all time_series models | enhancement, time_series, tune_model, priority_low | 2021-10-06 | No update since 2021-10-06 (> 2 years) |
| [#1373](https://github.com/pycaret/pycaret/issues/1373) | Incorporate window_length as a tuning parameter in tune_model | enhancement, time_series, tune_model, priority_low | 2021-10-06 | No update since 2021-10-06 (> 2 years) |
| [#1343](https://github.com/pycaret/pycaret/issues/1343) | [Feature Selection] Boruta run-time display  | enhancement | 2021-06-10 | No update since 2021-06-10 (> 2 years) |
| [#1338](https://github.com/pycaret/pycaret/issues/1338) | Feature Interaction Extension | enhancement | 2021-06-07 | No update since 2021-06-07 (> 2 years) |
| [#1280](https://github.com/pycaret/pycaret/issues/1280) | [FEAT] Set logistic trend for prophet container | enhancement, time_series, models, priority_low | 2022-03-31 | No update since 2022-03-31 (> 2 years) |
| [#1237](https://github.com/pycaret/pycaret/issues/1237) | Automl for clustering | enhancement, clustering | 2022-08-21 | No update since 2022-08-21 (> 2 years) |
| [#1233](https://github.com/pycaret/pycaret/issues/1233) | Remove preprocess when using fix_imbalance | enhancement, classification | 2021-05-07 | No update since 2021-05-07 (> 2 years) |
| [#1215](https://github.com/pycaret/pycaret/issues/1215) | Support neural network based timeseries forecasting | enhancement, time_series, backlog | 2021-09-19 | No update since 2021-09-19 (> 2 years) |
| [#1193](https://github.com/pycaret/pycaret/issues/1193) | How to set dpi or scale when using interpret_model() function? | enhancement | 2021-04-30 | No update since 2021-04-30 (> 2 years) |
| [#1176](https://github.com/pycaret/pycaret/issues/1176) | Scoring on train set with `predict_model` | enhancement | 2022-02-07 | No update since 2022-02-07 (> 2 years) |
| [#1141](https://github.com/pycaret/pycaret/issues/1141) | Pycaret model deploy to aws subfolder | enhancement | 2021-10-22 | No update since 2021-10-22 (> 2 years) |
| [#1104](https://github.com/pycaret/pycaret/issues/1104) | Budget time for tune_model | enhancement | 2021-04-07 | No update since 2021-04-07 (> 2 years) |
| [#928](https://github.com/pycaret/pycaret/issues/928) | Feature Selection Methods | enhancement | 2021-05-22 | No update since 2021-05-22 (> 2 years) |
| [#921](https://github.com/pycaret/pycaret/issues/921) | Adding a step to unsupervised learning | enhancement, clustering | 2022-08-21 | No update since 2022-08-21 (> 2 years) |
| [#918](https://github.com/pycaret/pycaret/issues/918) | Features removed through multicollinearity | enhancement | 2022-01-14 | No update since 2022-01-14 (> 2 years) |
| [#839](https://github.com/pycaret/pycaret/issues/839) | Defining positive class in classification | enhancement, classification | 2022-08-21 | No update since 2022-08-21 (> 2 years) |
| [#771](https://github.com/pycaret/pycaret/issues/771) | Implement test coverage and get code coverage badge | enhancement, unit_tests | 2022-12-05 | No update since 2022-12-05 (> 2 years) |
| [#742](https://github.com/pycaret/pycaret/issues/742) | Add advanced categorical encodings methods | enhancement | 2021-05-22 | No update since 2021-05-22 (> 2 years) |
| [#616](https://github.com/pycaret/pycaret/issues/616) | Can Pycaret predict multiple taget? | enhancement, regression, multivariate | 2022-02-24 | No update since 2022-02-24 (> 2 years) |
| [#563](https://github.com/pycaret/pycaret/issues/563) | How to modifiy the plot (name of titel etc.) | enhancement | 2022-07-30 | No update since 2022-07-30 (> 2 years) |
| [#543](https://github.com/pycaret/pycaret/issues/543) | Change contributors list to all-contributors | documentation, enhancement | 2021-07-24 | No update since 2021-07-24 (> 2 years) |
| [#492](https://github.com/pycaret/pycaret/issues/492) | New function create_webservice | enhancement | 2022-04-20 | No update since 2022-04-20 (> 2 years) |
| [#429](https://github.com/pycaret/pycaret/issues/429) | [Feature Request] Support for Featuretools and Compose | enhancement, open for contribution, preprocessing | 2022-12-06 | No update since 2022-12-06 (> 2 years) |
| [#419](https://github.com/pycaret/pycaret/issues/419) | Bootstrap Validation | enhancement, open for contribution | 2020-08-18 | No update since 2020-08-18 (> 2 years) |
| [#408](https://github.com/pycaret/pycaret/issues/408) | Request to add anomaly detection plots for anomaly detection module | enhancement, open for contribution, anomaly_detection | 2022-05-24 | No update since 2022-05-24 (> 2 years) |
| [#377](https://github.com/pycaret/pycaret/issues/377) | UnicodeDecodeError while running the example ( Interpret Model)    What can I do? | help wanted | 2021-05-24 | No update since 2021-05-24 (> 2 years) |
| [#370](https://github.com/pycaret/pycaret/issues/370) | Data profiling and over fitting Detector enhancement proposal  | enhancement, open for contribution | 2020-10-06 | No update since 2020-10-06 (> 2 years) |
| [#350](https://github.com/pycaret/pycaret/issues/350) | Please include a model sensitivity analysis module | enhancement, open for contribution | 2020-12-05 | No update since 2020-12-05 (> 2 years) |
| [#345](https://github.com/pycaret/pycaret/issues/345) | option to deploy to JVM environments | enhancement, help wanted, open for contribution | 2020-08-13 | No update since 2020-08-13 (> 2 years) |
| [#339](https://github.com/pycaret/pycaret/issues/339) | KeyError: 'Only the Series name can be used for the key in Series dtype mappings.' | help wanted | 2021-05-24 | No update since 2021-05-24 (> 2 years) |
| [#216](https://github.com/pycaret/pycaret/issues/216) | Return Important Variables and importance as DataFrames | enhancement, interpret_model | 2022-12-05 | No update since 2022-12-05 (> 2 years) |
| [#25](https://github.com/pycaret/pycaret/issues/25) | Multi target columns | enhancement, regression, classification, multivariate | 2022-02-24 | No update since 2022-02-24 (> 2 years) |

## `still_relevant_bug` — 58 issue(s)

**Action:** Label `4.0-candidate`. Triage into Phase 5 repair queue.

| # | Title | Labels | Updated | Reason |
|---:|---|---|---|---|
| [#4096](https://github.com/pycaret/pycaret/issues/4096) | [BUG]: Different behaviour between PyCaret and Sklearn | bug | 2024-11-12 | Labeled bug; not kill-listed; recent |
| [#4094](https://github.com/pycaret/pycaret/issues/4094) | [BUG]: compare_models with fit_kwargs bug - still not working | bug | 2025-01-22 | Labeled bug; not kill-listed; recent |
| [#4064](https://github.com/pycaret/pycaret/issues/4064) | [tests failing]: ValueError: report is required by not given | bug | 2025-02-17 | Labeled bug; not kill-listed; recent |
| [#4038](https://github.com/pycaret/pycaret/issues/4038) | [BUG]: conda 3.3.2 | bug | 2024-07-27 | Labeled bug; not kill-listed; recent |
| [#4031](https://github.com/pycaret/pycaret/issues/4031) | [BUG]: AXIS Y OF FEATUE | bug | 2024-07-21 | Labeled bug; not kill-listed; recent |
| [#4006](https://github.com/pycaret/pycaret/issues/4006) | [BUG]: Import error in Pycaret 3.3.2 | bug | 2024-06-18 | Labeled bug; not kill-listed; recent |
| [#4000](https://github.com/pycaret/pycaret/issues/4000) | [BUG]: %pip install pycaret[full]  give an error.  | bug | 2024-08-04 | Labeled bug; not kill-listed; recent |
| [#3983](https://github.com/pycaret/pycaret/issues/3983) | [BUG]: pycaret compare_models MAE increase when add installed tpot | bug | 2024-06-17 | Labeled bug; not kill-listed; recent |
| [#3975](https://github.com/pycaret/pycaret/issues/3975) | [BUG]: Failed to set the disabled "data_split_shuffle" option in classification. | bug | 2024-04-17 | Labeled bug; not kill-listed; recent |
| [#3974](https://github.com/pycaret/pycaret/issues/3974) | [BUG]: AUC metrics doesn`t work  in pycaret 3.3.0 and 3.3.1 | bug | 2025-05-16 | Labeled bug; not kill-listed; recent |
| [#3971](https://github.com/pycaret/pycaret/issues/3971) | is:issue is:open ERROR cannot import name  '_format_load_msg' from 'joblib.memory' (py310\lib\site-packages\joblib\memory.py) Hello, I need help for the following error, I have the following error  I have AN ENVIRONMENT WITH pyhton 3.10 and pycaret 3.3.0 What version do I have to have to not have this problem? And by when will the error be resolved. Thank you in advance, and I look forward to your response, because I am working on a project and I am a newbie. [BUG]:  | bug | 2024-05-30 | Labeled bug; not kill-listed; recent |
| [#3969](https://github.com/pycaret/pycaret/issues/3969) | [BUG]: Linux ubuntu file permission error | bug | 2024-04-16 | Labeled bug; not kill-listed; recent |
| [#3943](https://github.com/pycaret/pycaret/issues/3943) | [BUG]:  | bug | 2025-01-14 | Labeled bug; not kill-listed; recent |
| [#3941](https://github.com/pycaret/pycaret/issues/3941) | [BUG]: ImportError: cannot import name 'TSForecastingExperiment' from 'pycaret.time_series'  | bug | 2024-03-12 | Labeled bug; not kill-listed; recent |
| [#3940](https://github.com/pycaret/pycaret/issues/3940) | [BUG]: ImportError: cannot import name 'get_columns_to_stratify_by' from 'pycaret.internal.utils' | bug | 2024-03-11 | Labeled bug; not kill-listed; recent |
| [#3939](https://github.com/pycaret/pycaret/issues/3939) | [BUG]: ImportError: cannot import name 'get_columns_to_stratify_by' | bug | 2024-03-11 | Labeled bug; not kill-listed; recent |
| [#3938](https://github.com/pycaret/pycaret/issues/3938) | [BUG]: Pycaret : target column-Misisng values Handling Issue | bug | 2024-08-04 | Labeled bug; not kill-listed; recent |
| [#3931](https://github.com/pycaret/pycaret/issues/3931) | [BUG]: plot_model(et, plot='tree') not displaying Tree | bug | 2025-09-17 | Labeled bug; not kill-listed; recent |
| [#3930](https://github.com/pycaret/pycaret/issues/3930) | [BUG]: Training fails in the compare model | bug | 2024-03-02 | Labeled bug; not kill-listed; recent |
| [#3909](https://github.com/pycaret/pycaret/issues/3909) | [BUG]: Time Series \| compare_models()  | bug | 2024-02-20 | Labeled bug; not kill-listed; recent |
| [#3899](https://github.com/pycaret/pycaret/issues/3899) | [BUG]: Custom Metric in tune_model Does Not Subset kwargs 'offset' Appropriately for Each Fold | bug | 2024-02-20 | Labeled bug; not kill-listed; recent |
| [#3897](https://github.com/pycaret/pycaret/issues/3897) | [BUG]: LightGBM >= v4 hangs during cross validation | bug | 2024-02-23 | Labeled bug; not kill-listed; recent |
| [#3893](https://github.com/pycaret/pycaret/issues/3893) | [BUG]: AUC score is inaccurate. Model overfits after tuning? | bug | 2024-02-21 | Labeled bug; not kill-listed; recent |
| [#3889](https://github.com/pycaret/pycaret/issues/3889) | [BUG]: got error when interpret_model | bug | 2024-01-21 | Labeled bug; not kill-listed; recent |
| [#3887](https://github.com/pycaret/pycaret/issues/3887) | [BUG]:  xgboost not available | bug | 2024-05-16 | Labeled bug; not kill-listed; recent |
| [#3867](https://github.com/pycaret/pycaret/issues/3867) | [BUG]: 1. When using the plot_model function, it is not possible to plot the distribution of errors on the traini_data. 2. And trying to get the predictions obtained by the regression model on the training set, there doesn't seem to be a corresponding function call. | bug | 2023-12-27 | Labeled bug; not kill-listed; recent |
| [#3861](https://github.com/pycaret/pycaret/issues/3861) | [BUG]: time series - `predict_model()` does not work with loaded model and experiment during future | bug, time_series | 2024-05-28 | Labeled bug; not kill-listed; recent |
| [#3860](https://github.com/pycaret/pycaret/issues/3860) | [BUG]:  interpret_model error,  ExplainerError: Additivity check failed in TreeExplainer | bug | 2023-12-20 | Labeled bug; not kill-listed; recent |
| [#3859](https://github.com/pycaret/pycaret/issues/3859) | [BUG]: Behavior of PyCaret with Multiple Options: Feature Generation, Automatic Variable Selection, Polynomial Features, and PCA Dimensionality Reduction | bug | 2024-01-19 | Labeled bug; not kill-listed; recent |
| [#3851](https://github.com/pycaret/pycaret/issues/3851) | [BUG]:output features names and values of interpret_model function in clssification  | bug | 2023-12-10 | Labeled bug; not kill-listed; recent |
| [#3812](https://github.com/pycaret/pycaret/issues/3812) | [BUG]: Polynomial_Features with Ignore_features | bug | 2024-02-15 | Labeled bug; not kill-listed; recent |
| [#3810](https://github.com/pycaret/pycaret/issues/3810) | [BUG]: the model giving wrong R2   | bug | 2023-11-28 | Labeled bug; not kill-listed; recent |
| [#3794](https://github.com/pycaret/pycaret/issues/3794) | [BUG]:  predict_model on Anomaly detection for 'sos' model is getting stuck for hours. | bug | 2023-10-27 | Labeled bug; not kill-listed; recent |
| [#3793](https://github.com/pycaret/pycaret/issues/3793) | [BUG]: LightGBM pollutes JupyterLab notebook with info, warning and fatal msgs | bug | 2023-10-26 | Labeled bug; not kill-listed; recent |
| [#3788](https://github.com/pycaret/pycaret/issues/3788) | [BUG]: `compare_models` unable to fit all available models after feature engineering on target and exog variables.  | bug, time_series | 2024-02-02 | Labeled bug; not kill-listed; recent |
| [#3787](https://github.com/pycaret/pycaret/issues/3787) | [BUG]: TypeError: can only concatenate str (not "numpy.float32") to str | bug, classification | 2023-11-30 | Labeled bug; not kill-listed; recent |
| [#3785](https://github.com/pycaret/pycaret/issues/3785) | [BUG]: Metrics calculated appear to be incorrect | bug | 2024-11-19 | Labeled bug; not kill-listed; recent |
| [#3783](https://github.com/pycaret/pycaret/issues/3783) | [BUG]: Error converting PyCaret model to ONNX | bug | 2023-10-16 | Labeled bug; not kill-listed; recent |
| [#3719](https://github.com/pycaret/pycaret/issues/3719) | [BUG]: TypeError: field() got an unexpected keyword argument 'alias' | bug | 2023-09-01 | Labeled bug; not kill-listed; recent |
| [#3697](https://github.com/pycaret/pycaret/issues/3697) | [BUG]: Future exogenous data has no impact on prediction | bug, time_series | 2024-02-02 | Labeled bug; not kill-listed; recent |
| [#3692](https://github.com/pycaret/pycaret/issues/3692) | [BUG]: Not allow to use the "use_train_data" option on plot_model function | bug | 2024-09-12 | Labeled bug; not kill-listed; recent |
| [#3690](https://github.com/pycaret/pycaret/issues/3690) | [BUG]: Run compare_models but always return empty list | bug | 2024-02-02 | Labeled bug; not kill-listed; recent |
| [#3681](https://github.com/pycaret/pycaret/issues/3681) | [BUG]: Model setup error on Snowflake | bug | 2024-02-02 | Labeled bug; not kill-listed; recent |
| [#3668](https://github.com/pycaret/pycaret/issues/3668) | [BUG]: Models cannot be compared. | bug, missing_info | 2023-08-26 | Labeled bug; not kill-listed; recent |
| [#3647](https://github.com/pycaret/pycaret/issues/3647) | [BUG]: TypeError: __init__() got an unexpected keyword argument 'predict_fn' | bug, interpret_model | 2023-07-21 | Labeled bug; not kill-listed; recent |
| [#3634](https://github.com/pycaret/pycaret/issues/3634) | [BUG]: When using scale parameter in plot_model function, after saving, the figure is still not high-resolution/quality | bug | 2023-07-07 | Labeled bug; not kill-listed; recent |
| [#3579](https://github.com/pycaret/pycaret/issues/3579) | Multiple classification / compare_models() / metrics results of trainning set. | bug, missing_info | 2023-06-30 | Labeled bug; not kill-listed; recent |
| [#3472](https://github.com/pycaret/pycaret/issues/3472) | [BUG]: maximum recursion depth exceeded while calling a Python object | bug, clustering, missing_info | 2024-02-02 | Labeled bug; not kill-listed; recent |
| [#3357](https://github.com/pycaret/pycaret/issues/3357) | [BUG]: Loaded Model not working  | bug, missing_info, load_models | 2023-03-23 | Labeled bug; not kill-listed; recent |
| [#3348](https://github.com/pycaret/pycaret/issues/3348) | [BUG]: GroupTimeSeriesSplit CV Splitter not working as expected in PyCaret Classification | bug, classification | 2024-02-02 | Labeled bug; not kill-listed; recent |
| [#3305](https://github.com/pycaret/pycaret/issues/3305) | [BUG]: DecisionTreeClassifier feature_importances_ mismatch | bug, missing_info | 2023-07-28 | Labeled bug; not kill-listed; recent |
| [#3266](https://github.com/pycaret/pycaret/issues/3266) | [BUG]: Too many anomalies found, makes no sense | bug, anomaly_detection | 2023-02-06 | Labeled bug; not kill-listed; recent |
| [#3265](https://github.com/pycaret/pycaret/issues/3265) | [BUG]: st.pyplot(plot_model(..., display_format= 'streamlit')) creates second empty window below the plot | bug, plot_model | 2023-01-21 | Labeled bug; not kill-listed; recent |
| [#3192](https://github.com/pycaret/pycaret/issues/3192) | Original data shape is showing 10 million records instead of 300k records. setup is taking too much time to run. | bug, classification, big_data | 2023-03-10 | Labeled bug; not kill-listed; recent |
| [#3154](https://github.com/pycaret/pycaret/issues/3154) | [BUG]: Each run becomes a nested run when using FugueBackend | bug, mlflow, parallelization, fugue | 2024-02-02 | Labeled bug; not kill-listed; recent |
| [#3135](https://github.com/pycaret/pycaret/issues/3135) | [BUG]: AttributeError: 'Make_Time_Features' object has no attribute 'list_of_features' | bug | 2023-03-05 | Labeled bug; not kill-listed; recent |
| [#2996](https://github.com/pycaret/pycaret/issues/2996) | [BUG]: Can't save model to .txt? Only can save to .pkl? It's buggy when I transform the saved .pkl to txt. | bug | 2025-02-19 | Labeled bug; not kill-listed; recent |
| [#1633](https://github.com/pycaret/pycaret/issues/1633) | [BUG]'DataFrame' object has no attribute 'predict' | bug, classification | 2024-02-02 | Labeled bug; not kill-listed; recent |

## `still_relevant_enhancement` — 107 issue(s)

**Action:** Label `4.0-candidate`. Decide per-issue whether to accept, defer to 4.1+, or close.

| # | Title | Labels | Updated | Reason |
|---:|---|---|---|---|
| [#4174](https://github.com/pycaret/pycaret/issues/4174) | [Feature Request] Implementation of Time-Series Exploratory Data Analysis (EDA) Module | enhancement | 2025-12-31 | Not kill-listed; recent |
| [#4163](https://github.com/pycaret/pycaret/issues/4163) | [ENH]: Support Snowflake Notebooks | enhancement | 2025-05-01 | Not kill-listed; recent |
| [#4156](https://github.com/pycaret/pycaret/issues/4156) | [ENH]: support for StratifiedGroupKFold folding strategy | enhancement | 2025-03-19 | Not kill-listed; recent |
| [#4134](https://github.com/pycaret/pycaret/issues/4134) | [MNT] please switch merge modus to squash | enhancement | 2025-02-19 | Not kill-listed; recent |
| [#4133](https://github.com/pycaret/pycaret/issues/4133) | [INSTALL]: | installation | 2025-02-18 | Not kill-listed; recent |
| [#4077](https://github.com/pycaret/pycaret/issues/4077) | [ENH]: PR-AUC Score Metric in Comparison Result | enhancement | 2024-12-23 | Not kill-listed; recent |
| [#4070](https://github.com/pycaret/pycaret/issues/4070) | [ENH]: Classification | enhancement | 2024-09-10 | Not kill-listed; recent |
| [#4067](https://github.com/pycaret/pycaret/issues/4067) | [ENH]: Publish pycaret-cpu package. | enhancement | 2024-09-03 | Not kill-listed; recent |
| [#4066](https://github.com/pycaret/pycaret/issues/4066) | [ENH]: Make plotting dependencies optional | enhancement | 2024-09-03 | Not kill-listed; recent |
| [#4056](https://github.com/pycaret/pycaret/issues/4056) | [ENH]: Add early_stopping to tune_model in pycaret.time_series.tune_model | enhancement | 2024-08-29 | Not kill-listed; recent |
| [#4055](https://github.com/pycaret/pycaret/issues/4055) | [ENH]: Add PerpetualBooster | enhancement | 2024-09-23 | Not kill-listed; recent |
| [#4045](https://github.com/pycaret/pycaret/issues/4045) | [ENH]: Accept train and test datasets with similar indices | enhancement | 2024-08-16 | Not kill-listed; recent |
| [#4027](https://github.com/pycaret/pycaret/issues/4027) | [ENH]: Kernel Approximation for large dataset | enhancement | 2024-07-18 | Not kill-listed; recent |
| [#4003](https://github.com/pycaret/pycaret/issues/4003) | [ENH]: Call estimator in custom metric function | enhancement | 2024-06-07 | Not kill-listed; recent |
| [#3951](https://github.com/pycaret/pycaret/issues/3951) | ARDL model | enhancement | 2024-06-27 | Not kill-listed; recent |
| [#3919](https://github.com/pycaret/pycaret/issues/3919) | [ENH]: add support for polars | enhancement | 2024-02-26 | Not kill-listed; recent |
| [#3912](https://github.com/pycaret/pycaret/issues/3912) | [ENH]:  | enhancement | 2024-02-20 | Not kill-listed; recent |
| [#3891](https://github.com/pycaret/pycaret/issues/3891) | [ENH]: black 23.3.0 | enhancement | 2024-02-28 | Not kill-listed; recent |
| [#3883](https://github.com/pycaret/pycaret/issues/3883) | [ENH]: compare_models() for clustering | enhancement | 2024-12-23 | Not kill-listed; recent |
| [#3864](https://github.com/pycaret/pycaret/issues/3864) | [DOC]: define responsible for each theme | enhancement | 2023-12-26 | Not kill-listed; recent |
| [#3863](https://github.com/pycaret/pycaret/issues/3863) | [ENH]: define template for header of the files | enhancement | 2023-12-21 | Not kill-listed; recent |
| [#3847](https://github.com/pycaret/pycaret/issues/3847) | [ENH]: In pycaret version 3.0.1 have xgboost, but in pycaret 3.20.0 not have xgboost | enhancement | 2023-12-09 | Not kill-listed; recent |
| [#3802](https://github.com/pycaret/pycaret/issues/3802) | Darts Multivariate time series in pycaret or add multivariate time series in pycaret  | enhancement | 2023-10-28 | Not kill-listed; recent |
| [#3797](https://github.com/pycaret/pycaret/issues/3797) | [ENH] Time Series Multiple(multi variate) Time Series Forecasting | enhancement, time_series | 2023-12-23 | Not kill-listed; recent |
| [#3776](https://github.com/pycaret/pycaret/issues/3776) | [ENH]: enable GPU support for LighGBM by installing replacing it with lightgbm_ray | enhancement | 2023-10-11 | Not kill-listed; recent |
| [#3775](https://github.com/pycaret/pycaret/issues/3775) | [ENH]: Allow plot_model to accept pass though an axis to scatter plot | enhancement | 2023-10-11 | Not kill-listed; recent |
| [#3744](https://github.com/pycaret/pycaret/issues/3744) | [ENH]: Mechanism Cross validation | enhancement | 2023-09-19 | Not kill-listed; recent |
| [#3696](https://github.com/pycaret/pycaret/issues/3696) | [ENH]: EDA for time series module | enhancement, good first issue, time_series | 2025-10-24 | Not kill-listed; recent |
| [#3651](https://github.com/pycaret/pycaret/issues/3651) | [ENH]: manually setting probability threshold for already trained model (classification) | enhancement | 2023-07-22 | Not kill-listed; recent |
| [#3641](https://github.com/pycaret/pycaret/issues/3641) | [ENH]: models with equal metrics should be ordered by time | enhancement | 2023-07-20 | Not kill-listed; recent |
| [#3640](https://github.com/pycaret/pycaret/issues/3640) | [ENH]: Prepare for XGBoost 2.0 | enhancement | 2024-12-02 | Not kill-listed; recent |
| [#3637](https://github.com/pycaret/pycaret/issues/3637) | [ENH]: get result of interpret_model() as text. | enhancement | 2023-07-20 | Not kill-listed; recent |
| [#3633](https://github.com/pycaret/pycaret/issues/3633) | [ENH]: time_series save_model() should accept **kwargs and pass to joblib.dump() like other modeling types | enhancement | 2023-07-06 | Not kill-listed; recent |
| [#3627](https://github.com/pycaret/pycaret/issues/3627) | [ENH]: Detailed tutorial for Time Series Forecasting using ML models including analysis and feature engineering / extractions / selection | enhancement, time_series | 2023-07-05 | Not kill-listed; recent |
| [#3622](https://github.com/pycaret/pycaret/issues/3622) | [ENH]: XGBoost gblinear and dart booster support | enhancement | 2024-12-23 | Not kill-listed; recent |
| [#3589](https://github.com/pycaret/pycaret/issues/3589) | Pycaret forecasting svm | enhancement, good first issue, time_series | 2023-09-09 | Not kill-listed; recent |
| [#3585](https://github.com/pycaret/pycaret/issues/3585) | [ENH]:  SVM with radial kernel is so slow | enhancement, missing_info | 2023-06-07 | Not kill-listed; recent |
| [#3571](https://github.com/pycaret/pycaret/issues/3571) | [ENH]: Yellow+white color problems | enhancement | 2023-09-30 | Not kill-listed; recent |
| [#3550](https://github.com/pycaret/pycaret/issues/3550) | [ENH]: time series models - add crucial parameter sets as test cases in `sktime` | enhancement, time_series | 2023-05-15 | Not kill-listed; recent |
| [#3529](https://github.com/pycaret/pycaret/issues/3529) | Request for credit score card model development | enhancement | 2023-05-03 | Not kill-listed; recent |
| [#3512](https://github.com/pycaret/pycaret/issues/3512) | [ENH]: Add support for separate validation and test sets in setup function. | enhancement | 2023-04-26 | Not kill-listed; recent |
| [#3511](https://github.com/pycaret/pycaret/issues/3511) | [BUG]: confusion_matrix | enhancement, plot_model | 2023-06-30 | Not kill-listed; recent |
| [#3495](https://github.com/pycaret/pycaret/issues/3495) | [ENH]: Re-enable tests that were failing and that were disabled | enhancement, unit_tests | 2024-02-02 | Not kill-listed; recent |
| [#3492](https://github.com/pycaret/pycaret/issues/3492) | [ENH]: smaller deployment install | enhancement | 2024-02-02 | Not kill-listed; recent |
| [#3461](https://github.com/pycaret/pycaret/issues/3461) | [ENH]: "log_plot" to include SHAP values ("interpret_model" plots) | enhancement | 2023-04-05 | Not kill-listed; recent |
| [#3443](https://github.com/pycaret/pycaret/issues/3443) | [ENH]: Easy way to get confidence interval with compare_model | enhancement | 2023-03-29 | Not kill-listed; recent |
| [#3429](https://github.com/pycaret/pycaret/issues/3429) | [ENH]: Receive the calculated Shapley values as a return | enhancement | 2023-03-24 | Not kill-listed; recent |
| [#3399](https://github.com/pycaret/pycaret/issues/3399) | multilabel | duplicate, enhancement | 2023-03-21 | Not kill-listed; recent |
| [#3370](https://github.com/pycaret/pycaret/issues/3370) | [BUG]: html parameter in the setup | enhancement, priority_medium, refactor | 2024-02-02 | Not kill-listed; recent |
| [#3369](https://github.com/pycaret/pycaret/issues/3369) | [BUG]: Behavior of ignore_features | enhancement | 2024-02-02 | Not kill-listed; recent |
| [#3368](https://github.com/pycaret/pycaret/issues/3368) | [ENH]: want to adjust dot size and transparency in 'error' and 'residuals' plot_model | enhancement | 2023-03-07 | Not kill-listed; recent |
| [#3341](https://github.com/pycaret/pycaret/issues/3341) | [ENH]: Prepare for `sktime` split | enhancement, time_series, roadmap | 2026-02-03 | Not kill-listed; recent |
| [#3338](https://github.com/pycaret/pycaret/issues/3338) | [ENH]: Add support for python 3.11 | enhancement, installation | 2024-02-02 | Not kill-listed; recent |
| [#3291](https://github.com/pycaret/pycaret/issues/3291) | [ENH]: Other ways to save model than pickle | enhancement | 2023-01-31 | Not kill-listed; recent |
| [#3281](https://github.com/pycaret/pycaret/issues/3281) | Ability to Continue the Optimization (Tuning, Blending, Ensembling) Pipeline Upon Unexpected Kernel Interruption | enhancement | 2023-01-30 | Not kill-listed; recent |
| [#3239](https://github.com/pycaret/pycaret/issues/3239) | [ENH]: Time series unsupervised learning module | enhancement, anomaly_detection, clustering, time_series | 2023-01-14 | Not kill-listed; recent |
| [#3237](https://github.com/pycaret/pycaret/issues/3237) | [ENH]: Design documentation for Time Series Clustering Module | enhancement, clustering, time_series | 2023-01-10 | Not kill-listed; recent |
| [#3218](https://github.com/pycaret/pycaret/issues/3218) | Please add evaluate_model for visualizing all the plots in the time_seris like train_test_split, ts, forecast,decomposition.  | enhancement, time_series, plot_model | 2024-02-02 | Not kill-listed; recent |
| [#3201](https://github.com/pycaret/pycaret/issues/3201) | [ENH]: Add a Simple Moving Average Forecaster to the time series module. | enhancement, good first issue, time_series | 2024-02-02 | Not kill-listed; recent |
| [#3183](https://github.com/pycaret/pycaret/issues/3183) | Improve inheritance in regards to properties in Base Experiments | enhancement, refactor | 2024-02-02 | Not kill-listed; recent |
| [#3055](https://github.com/pycaret/pycaret/issues/3055) | [ENH]: CPU and GPU benchmarks | enhancement, benchmark | 2023-06-16 | Not kill-listed; recent |
| [#2920](https://github.com/pycaret/pycaret/issues/2920) | [ENH]: Add back clustering tests to pycaret | enhancement, unit_tests | 2024-02-02 | Not kill-listed; recent |
| [#2902](https://github.com/pycaret/pycaret/issues/2902) | [ENH]: `use_gpu` to use engines interface | enhancement, models, gpu | 2024-02-02 | Not kill-listed; recent |
| [#2686](https://github.com/pycaret/pycaret/issues/2686) | [ENH]: Add more Statistical Tests to Time Series `.check_stats()` method | enhancement, time_series, stats | 2024-02-02 | Not kill-listed; recent |
| [#2539](https://github.com/pycaret/pycaret/issues/2539) | [ENH]: Default behavior of predict_model in classification | enhancement, classification, discussion | 2024-02-02 | Not kill-listed; recent |
| [#2529](https://github.com/pycaret/pycaret/issues/2529) | Pycaret should support Neural Network or Deep   | enhancement | 2023-01-26 | Not kill-listed; recent |
| [#2512](https://github.com/pycaret/pycaret/issues/2512) | [ENH]: Multi Output Regression and Classification | enhancement, block | 2024-02-02 | Not kill-listed; recent |
| [#2483](https://github.com/pycaret/pycaret/issues/2483) | [ENH]: Add setup argument for `test_alpha` that is used in the rest of the methods for tests and plots | enhancement, time_series, plot_model, stats, setup | 2024-02-02 | Not kill-listed; recent |
| [#2463](https://github.com/pycaret/pycaret/issues/2463) | [ENH]: Add sktime conformal prediction to time series module | enhancement, time_series | 2024-02-02 | Not kill-listed; recent |
| [#2424](https://github.com/pycaret/pycaret/issues/2424) | Unit tests to run without optional dependencies | enhancement, plot_model, priority_medium, unit_tests | 2024-02-02 | Not kill-listed; recent |
| [#2419](https://github.com/pycaret/pycaret/issues/2419) | Time series CV plot enhancement | enhancement, time_series, plot_model | 2024-02-02 | Not kill-listed; recent |
| [#2399](https://github.com/pycaret/pycaret/issues/2399) | Time Series \| Bias metric | enhancement, good first issue, time_series, metrics | 2024-02-02 | Not kill-listed; recent |
| [#2331](https://github.com/pycaret/pycaret/issues/2331) | Add GPU support for ReducedRegressionModels | enhancement, good first issue, time_series, big_data, gpu | 2024-02-02 | Not kill-listed; recent |
| [#2323](https://github.com/pycaret/pycaret/issues/2323) | Same model from multiple libraries | enhancement, time_series, models | 2024-02-02 | Not kill-listed; recent |
| [#2287](https://github.com/pycaret/pycaret/issues/2287) | Refactor large functions into smaller functions | enhancement, refactor | 2024-02-02 | Not kill-listed; recent |
| [#2286](https://github.com/pycaret/pycaret/issues/2286) | Additional Unit Tests - MAC | enhancement, unit_tests, refactor | 2024-02-02 | Not kill-listed; recent |
| [#2283](https://github.com/pycaret/pycaret/issues/2283) | Additional Unit Tests - prophet | enhancement, unit_tests, refactor | 2024-02-02 | Not kill-listed; recent |
| [#2282](https://github.com/pycaret/pycaret/issues/2282) | Remove experiment specific references from Base Experiments | enhancement, refactor | 2024-02-02 | Not kill-listed; recent |
| [#2265](https://github.com/pycaret/pycaret/issues/2265) | Benchmark Time Series Results against Darts | enhancement, good first issue, time_series, benchmark | 2024-02-02 | Not kill-listed; recent |
| [#2264](https://github.com/pycaret/pycaret/issues/2264) | Replicate TSstudio plots from R | enhancement, time_series, plot_model | 2024-02-02 | Not kill-listed; recent |
| [#2247](https://github.com/pycaret/pycaret/issues/2247) | Determine max_p and max_q for AutoARIMA models | enhancement, time_series, models | 2024-02-02 | Not kill-listed; recent |
| [#2246](https://github.com/pycaret/pycaret/issues/2246) | Allow ability to run tests such as ADF on the seasonal components of decomposition | enhancement, time_series, stats | 2024-02-02 | Not kill-listed; recent |
| [#2245](https://github.com/pycaret/pycaret/issues/2245) | Allow ability to run ACF and PACF on the seasonal components of decomposition - Useful to determine P and Q | enhancement, models, plot_model | 2024-02-02 | Not kill-listed; recent |
| [#2230](https://github.com/pycaret/pycaret/issues/2230) | Incorporate `FeatureSelector` in time series | enhancement, time_series, exogenous | 2024-02-02 | Not kill-listed; recent |
| [#2147](https://github.com/pycaret/pycaret/issues/2147) | [Forecasting] Add setup argument for `lags` that is used in the rest of the methods | enhancement, time_series, plot_model, priority_low | 2024-02-02 | Not kill-listed; recent |
| [#2136](https://github.com/pycaret/pycaret/issues/2136) | [INSTALL] Release a docker image with pycaret already installed | enhancement, installation | 2023-02-02 | Not kill-listed; recent |
| [#1944](https://github.com/pycaret/pycaret/issues/1944) | parallel back-end | enhancement, parallelization, fugue | 2023-10-04 | Not kill-listed; recent |
| [#1911](https://github.com/pycaret/pycaret/issues/1911) | Request for get cross validation predict result y_foldN_pred | enhancement | 2024-02-02 | Not kill-listed; recent |
| [#1836](https://github.com/pycaret/pycaret/issues/1836) | Time Series \| Add insample prediction intervals for Anomaly Detection | enhancement, anomaly_detection, time_series, plot_model, prediction | 2024-02-19 | Not kill-listed; recent |
| [#1833](https://github.com/pycaret/pycaret/issues/1833) | Time Series Predictions \| Add Alpha value in dataframe | enhancement, time_series, prediction | 2024-02-02 | Not kill-listed; recent |
| [#1766](https://github.com/pycaret/pycaret/issues/1766) | Add lower limit and upper limit for time series forecasting | enhancement, time_series, priority_medium | 2024-02-02 | Not kill-listed; recent |
| [#1731](https://github.com/pycaret/pycaret/issues/1731) | Add support for VAR model in time series module | enhancement, time_series, backlog, multivariate, models | 2024-01-10 | Not kill-listed; recent |
| [#1724](https://github.com/pycaret/pycaret/issues/1724) | Add `neuralprophet` to time_series module | enhancement, time_series, backlog, models | 2025-06-13 | Not kill-listed; recent |
| [#1688](https://github.com/pycaret/pycaret/issues/1688) | Support chaining operation in OOP API | enhancement, oop | 2024-02-02 | Not kill-listed; recent |
| [#1606](https://github.com/pycaret/pycaret/issues/1606) | Add add_model and remove_model functionality | enhancement, priority_medium | 2024-02-02 | Not kill-listed; recent |
| [#1505](https://github.com/pycaret/pycaret/issues/1505) | Adding support for multi-label classification? | enhancement, classification, multivariate | 2025-03-20 | Not kill-listed; recent |
| [#1436](https://github.com/pycaret/pycaret/issues/1436) | [Time Series] compare_models top N by category | enhancement, time_series, priority_medium | 2024-02-02 | Not kill-listed; recent |
| [#1405](https://github.com/pycaret/pycaret/issues/1405) | Visualize Error Distribution & backtesting results across folds  | enhancement, time_series, priority_high, plot_model | 2023-06-09 | Not kill-listed; recent |
| [#1350](https://github.com/pycaret/pycaret/issues/1350) | Add sample_weights to setup | enhancement | 2024-06-11 | Not kill-listed; recent |
| [#1298](https://github.com/pycaret/pycaret/issues/1298) | pycaret FAQ's - commonly asked questions | documentation | 2023-05-11 | Not kill-listed; recent |
| [#1235](https://github.com/pycaret/pycaret/issues/1235) | Support for T SNE  for dimentionality reduction | enhancement, preprocessing | 2025-08-04 | Not kill-listed; recent |
| [#1136](https://github.com/pycaret/pycaret/issues/1136) | pycaret.classification setup(): How do I display all rows of the 'Data Types' DataFrame? | enhancement | 2023-03-05 | Not kill-listed; recent |
| [#1117](https://github.com/pycaret/pycaret/issues/1117) | Imblearn Pipeline for fix_imbalance_method | enhancement | 2023-01-27 | Not kill-listed; recent |
| [#861](https://github.com/pycaret/pycaret/issues/861) | Bayesian optimization (skopt) : is there a way to pass the callback parameter ? | enhancement, tune_model | 2023-05-02 | Not kill-listed; recent |
| [#708](https://github.com/pycaret/pycaret/issues/708) | Has plot_model reset matplotlib.pyplot? | enhancement | 2023-06-26 | Not kill-listed; recent |
| [#382](https://github.com/pycaret/pycaret/issues/382) | (Suggestion) Feature Engineering: Use tsfresh to create features for time-series data | enhancement, time_series, priority_medium, exogenous | 2024-02-02 | Not kill-listed; recent |
| [#165](https://github.com/pycaret/pycaret/issues/165) | evaluate_model not generating User Interface | help wanted, evaluate_model | 2024-07-15 | Not kill-listed; recent |

