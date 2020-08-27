# Changelog
All notable changes to this project will be documented in this file.

#### Release: PyCaret 2.1 | Release Date: August 28, 2020
### Summary of Changes

- **Model Deployment** Model deployment support for `gcp` and `azure` has been added in `deploy_model` function for all modules. See `documentation` for details.
- **Compare Models Budget Time** new parameter `budget_time` added in `compare_models` function. To set the upper limit on `compare_models` training time, `budget_time` parameter can be used.
- **Feature Selection** New feature selection method `boruta` has been added for feature selection. By default, `feature_selection_method` parameter in the `setup` function is set to `classic` but can be set to `boruta` for feature selection using boruta algorithm. This change is applicable for `pycaret.classification` and `pycaret.regression`.
- **Numeric Imputation** New method `zero` has been added in the `numeric_imputation` in the `setup` function. When method is set to `zero`, missing values are replaced with constant 0. Default behavior of `numeric_imputation` is unchanged.
- **Plot Model** New parameter `scale` has been added in `plot_model` for all modules to enable high quality images for research publications.
- **User Defined Loss Function** You can now pass `custom_scorer` for optimizing user defined loss function in `tune_model` for `pycaret.classification` and `pycaret.regression`. You must use `make_scorer` from `sklearn` to create custom loss function that can be passed into `custom_scorer` for the `tune_model` function.
- **Change in Pipeline Behavior** When using `save_model` the `model` object is appended into `Pipeline`, as such the behavior of `Pipeline` and `predict_model` is now changed. Instead of saving a `list`, `save_model` now saves `Pipeline` object where trained model is on last position. The user functionality on front-end for `predict_model` remains same.
- **Compare Models** parameter `blacklist` and `whitelist` is now renamed to  `exclude` and `include` with no change in functionality.
- **Predict Model Labels** The `Label` column returned by `predict_model` function in `pycaret.classification` now returns the original label instead of encoded value. This change is made to make output from `predict_model` more human-readable. A new parameter `encoded_labels` is added, which is `False` by default. When set to `True`, it will return encoded labels.
- **Model Logging** Model persistence in the backend when `log_experiment` is set to `True` is now changed. Instead of using internal `save_model` functionality, it now adopts to `mlflow.sklearn.save_model` to allow the use of Model Registry and `MLFlow` native deployment functionalities.
- **CatBoost Compatibility** `CatBoostClassifier` is now compatible with `blend_models` in `pycaret.classification`. As such `blend_models` without any `estimator_list` will now result in blending total of `15` estimators including `CatBoostClassifier`.
- **Stack Models** `stack_models` in `pycaret.classification` and `pycaret.regression` now adopts to `StackingClassifier()` and `StackingRegressor` from `sklearn`. As such the `stack_models` function now returns `sklearn` object instead of custom `list` in previous versions.
- **Create Stacknet** `create_stacknet` in `pycaret.classification` and `pycaret.regression` is now removed.
- **Tune Model** `tune_model` in `pycaret.classification` and `pycaret.regression` now inherits params from the input `estimator`. As such if you have trained `xgboost`, `lightgbm` or `catboost` on gpu will not inherits training method from `estimator`.
- **Interpret Model** `**kwargs` argument now added in `interpret_model`. 
- **Pandas Categorical Type** All modules are now compatible with `pandas.Categorical` object. Internally they are converted into object and are treated as the same way as `object` or `bool` is treated. 
- **use_gpu** A new parameter added in the `setup` function for `pycaret.classification` and `pycaret.regression`. In `2.1` it was added to prepare for the backend work required to make this change in future releases. As such using `use_gpu` param in `2.1` has no impact.  
- **Unit Tests** Unit testing enhanced. Continious improvement in progress https://github.com/pycaret/pycaret/tree/master/pycaret/tests
- **Automated Documentation Added** Automated documentation now added. Documentation on Website will only update for `major` releases 0.X. For all minor monthly releases, documentation will be available on: https://pycaret.readthedocs.io/en/latest/
- **Introduction of GitHub Actions** CI/CD build testing is now moved from `travis-ci` to `github-actions`. `pycaret-nightly` is now being published every 24 hours automatically. 
- **Tutorials** All tutorials are now updated using `pycaret==2.0`. https://github.com/pycaret/pycaret/tree/master/tutorials
- **Resources** New resources added under `/pycaret/resources/` https://github.com/pycaret/pycaret/tree/master/resources
- **Example Notebook** Many example notebooks added under `/pycaret/examples/` https://github.com/pycaret/pycaret/tree/master/examples
___


#### Release: PyCaret 2.0 | Release Date: July 31, 2020

### Summary of Changes
- **Experiment Logging** MLFlow logging backend added. New parameters `log_experiment` `experiment_name` `log_profile`  `log_data` added in `setup`. Available in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`<br/> 
- **Save / Load Experiment** `save_experiment` and `load_experiment` function from `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp` is removed in PyCaret 2.0<br/>
- **System Logging** System log files now generated when `setup` is executed. `logs.log` file is saved in current working directory. Function `get_system_logs` can be used to access log file in notebook. <br/>
- **Command Line Support** When using PyCaret 2.0 outside of Notebook, `html` parameter in `setup` must be set to False. <br/>
- **Imbalance Dataset** `fix_imbalance` and `fix_imbalance_method` parameter added in `setup` for `pycaret.classification`. When set to True, SMOTE is applied by default to create synthetic datapoints for minority class. To change the method pass any class from `imblearn` that supports `fit_resample` method in `fix_imbalance_method` parameter. <br/>
- **Save Plot** `save` parameter added in `plot_model`. When set to True, it saves the plot as `png` or `html` in current working directory. <br/>
- **kwargs** `kwargs**` added in `create_model` for `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` <br/>
- **choose_better** `choose_better` and `optimize` parameter added in `tune_model` `ensemble_model` `blend_models` `stack_models` `create_stacknet` in `pycaret.classification` and `pycaret.regression`. Read the details below to learn more about thi added in `create_model` for `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` <br/>
- **Training Time** `TT (Sec)` added in `compare_models` function for `pycaret.classification` and `pycaret.regression` <br/>
- **New Metric: MCC** `MCC` metric added in score grid for `pycaret.classification` <br/>
- **NEW FUNCTION: automl()** New function `automl` added in `pycaret.classification` `pycaret.regression` <br/>
- **NEW FUNCTION: pull()** New function `pull` added in `pycaret.classification` `pycaret.regression` <br/>
- **NEW FUNCTION: models()** New function `models` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp` <br/>
- **NEW FUNCTION: get_logs()** New function `get_logs` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp` <br/>
- **NEW FUNCTION: get_config()** New function `get_config` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp` <br/>
- **NEW FUNCTION: set_config()** New function `set_config` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp` <br/>
- **NEW FUNCTION: get_system_logs** New function `get_logs` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp` <br/>
- **CHANGE IN BEHAVIOR: compare_models** `compare_models` now returns top_n models defined by `n_select` parameter, by default set to 1. <br/>
- **CHANGE IN BEHAVIOR: tune_model** `tune_model` function in `pycaret.classification` and `pycaret.regression` now requires trained model object to be passed as `estimator` instead of string abbreviation / ID. <br/>
- **REMOVED DEPENDENCIES** `awscli` and `shap` removed from requirements.txt. To use `interpret_model` function in `pycaret.classification` `pycaret.regression` and `deploy_model` function in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`, these libraries will have to be installed separately. <br/>

# <span style="color:red"> setup </span>
**`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`** <br/>

- **`remove_perfect_collinearity`** parameter added in `setup()`. Default set to False. <br/> 
When set to True, perfect collinearity (features with correlation = 1) is removed from the dataset, When two features are 100% correlated, one of it is randomly dropped from the dataset. <br/><br/>
- **`fix_imbalance`** parameter added in `setup()`. Default set to False. <br/>
When dataset has unequal distribution of target class it can be fixed using fix_imbalance parameter. When set to True, SMOTE (Synthetic Minority Over-sampling Technique) is applied by default to create synthetic datapoints for minority class. <br/><br/>
- **`fix_imbalance_method`** parameter added in `setup()`. Default set to None. <br/>
When fix_imbalance is set to True and fix_imbalance_method is None, 'smote' is applied by default to oversample minority class during cross validation. This parameter accepts any module from 'imblearn' that supports 'fit_resample' method. <br/><br/>
- **`data_split_shuffle`** parameter added in `setup()`. Default set to True. <br/>
If set to False, prevents shuffling of rows when splitting data. <br/><br/>
- **`folds_shuffle`** parameter added in `setup()`. Default set to False. <br/>
If set to False, prevents shuffling of rows when using cross validation. <br/><br/>
- **`n_jobs`** parameter added in `setup()`. Default set to -1. <br/>
The number of jobs to run in parallel (for functions that supports parallel processing) -1 means using all processors. To run all functions on single processor set n_jobs to None. <br/><br/>
- **`html`** parameter added in `setup()`. Default set to True. <br/>
If set to False, prevents runtime display of monitor. This must be set to False when using environment that doesnt support HTML. <br/><br/>
- **`log_experiment`** parameter added in `setup()`. Default set to False. <br/>
When set to True, all metrics and parameters are logged on MLFlow server. <br/><br/>
- **`experiment_name`** parameter added in `setup()`. Default set to None. <br/>
Name of experiment for logging. When set to None, 'clf' is by default used as alias for the experiment name. <br/><br/>
- **`log_plots`** parameter added in `setup()`. Default set to False. <br/>
When set to True, specific plots are logged in MLflow as a png file. <br/><br/>
- **`log_profile`** parameter added in `setup()`. Default set to False. <br/>
When set to True, data profile is also logged on MLflow as a html file. <br/><br/>
- **`log_data`** parameter added in `setup()`. Default set to False. <br/>
When set to True, train and test dataset are logged as csv. <br/><br/>
- **`verbose`** parameter added in `setup()`. Default set to True. <br/>
Information grid is not printed when verbose is set to False.

# <span style="color:red"> compare_models </span>
**`pycaret.classification` `pycaret.regression`** <br/>

- **`whitelist`** parameter added in `compare_models`. Default set to None. <br/> 
In order to run only certain models for the comparison, the model ID's can be passed as a list of strings in whitelist param.  <br/><br/>
- **`n_select`** parameter added in `compare_models`. Default set to 1. <br/> 
Number of top_n models to return. use negative argument for bottom selection. For example, n_select = -3 means bottom 3 models. <br/><br/>
- **`verbose`** parameter added in `compare_models`. Default set to True. <br/> 
Score grid is not printed when verbose is set to False.

# <span style="color:red"> create_model </span>
**`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`** <br/>

- **`cross_validation`** parameter added in `create_model`. Default set to True. <br/> 
When cross_validation set to False fold parameter is ignored and model is trained on entire training dataset. No metric evaluation is returned. Only applicable in `pycaret.classification` and `pycaret.regression`  <br/><br/>
- **`system`** parameter added in `create_model`. Default set to True. <br/> 
Must remain True all times. Only to be changed by internal functions. <br/><br/>
- **`ground_truth`** parameter added in `create_model`. Default set to None. <br/> 
When ground_truth is provided, Homogeneity Score, Rand Index, and Completeness Score is evaluated and printer along with other metrics. This is only available in **`pycaret.clustering`**  <br/><br/>
- **`kwargs`** parameter added in `create_model`. <br/> 
Additional keyword arguments to pass to the estimator.

# <span style="color:red"> tune_model </span>
**`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`** <br/>

- **`custom_grid`** parameter added in `tune_model`. Default set to None. <br/> 
To use custom hyperparameters for tuning pass a dictionary with parameter name and values to be iterated. When set to None it uses pre-defined tuning grid. For `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`, custom_grid param must be a list of values to iterate over. <br/><br/>
- **`choose_better`** parameter added in `tune_model`. Default set to False. <br/> 
When set to set to True, base estimator is returned when the performance doesn't improve by tune_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create_model or model returned by compare_models.

# <span style="color:red"> ensemble_model </span>
**`pycaret.classification` `pycaret.regression`** <br/>

- **`choose_better`** parameter added in `ensemble_model`. Default set to False. <br/> 
When set to set to True, base estimator is returned when the performance doesn't improve by tune_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create_model or model returned by compare_models. <br/><br/>
- **`optimize`** parameter added in `ensemble_model`. Default set to **`Accuracy`** for `pycaret.classification` and **`R2`** for `pycaret.regression`. <br/> 
Only used when choose_better is set to True. optimize parameter is used to compare emsembled model with base estimator. Values accepted in optimize parameter for `pycaret.classification` are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC' and for `pycaret.regression` are 'MAE', 'MSE', 'RMSE' 'R2', 'RMSLE' and 'MAPE'.

# <span style="color:red"> blend_models </span>
**`pycaret.classification` `pycaret.regression`** <br/>

- **`choose_better`** parameter added in `blend_models`. Default set to False. <br/> 
When set to set to True, base estimator is returned when the performance doesn't improve by tune_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create_model or model returned by compare_models. <br/><br/>
- **`optimize`** parameter added in `blend_models`. Default set to **`Accuracy`** for `pycaret.classification` and **`R2`** for `pycaret.regression`. <br/> 
Only used when choose_better is set to True. optimize parameter is used to compare emsembled model with base estimator. Values accepted in optimize parameter for `pycaret.classification` are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC' and for `pycaret.regression` are 'MAE', 'MSE', 'RMSE' 'R2', 'RMSLE' and 'MAPE'.

# <span style="color:red"> stack_models </span>
**`pycaret.classification` `pycaret.regression`** <br/>

- **`choose_better`** parameter added in `stack_models`. Default set to False. <br/> 
When set to set to True, base estimator is returned when the performance doesn't improve by tune_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create_model or model returned by compare_models. <br/><br/>
- **`optimize`** parameter added in `stack_models`. Default set to **`Accuracy`** for `pycaret.classification` and **`R2`** for `pycaret.regression`. <br/> 
Only used when choose_better is set to True. optimize parameter is used to compare emsembled model with base estimator. Values accepted in optimize parameter for `pycaret.classification` are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC' and for `pycaret.regression` are 'MAE', 'MSE', 'RMSE' 'R2', 'RMSLE' and 'MAPE'.

# <span style="color:red"> create_stacknet </span>
**`pycaret.classification` `pycaret.regression`** <br/>

- **`choose_better`** parameter added in `create_stacknet`. Default set to False. <br/> 
When set to set to True, base estimator is returned when the performance doesn't improve by tune_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create_model or model returned by compare_models. <br/><br/>
- **`optimize`** parameter added in `create_stacknet`. Default set to **`Accuracy`** for `pycaret.classification` and **`R2`** for `pycaret.regression`. <br/> 
Only used when choose_better is set to True. optimize parameter is used to compare emsembled model with base estimator. Values accepted in optimize parameter for `pycaret.classification` are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC' and for `pycaret.regression` are 'MAE', 'MSE', 'RMSE' 'R2', 'RMSLE' and 'MAPE'.

# <span style="color:red"> predict_model </span>
**`pycaret.classification` `pycaret.regression`** <br/>

- **`verbose`** parameter added in `predict_model`. Default set to True. <br/> 
Holdout score grid is not printed when verbose is set to False.

# <span style="color:red"> plot_model </span>
**`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`** <br/>

- **`save`** parameter added in `plot_model`. Default set to False. <br/> 
When set to True, Plot is saved as a 'png' file in current working directory. <br/> <br/>
- **`verbose`** parameter added in `plot_model`. Default set to True. <br/> 
Progress bar not shown when verbose set to False. <br/> <br/>
- **`system`** parameter added in `plot_model`. Default set to True. <br/> 
Must remain True all times. Only to be changed by internal functions.

# <span style="color:red"> NEW FUNCTION: automl </span>
**`pycaret.classification` `pycaret.regression`** <br/>

- This function returns the best model out of all models created in current active environment based on metric defined in optimize parameter.
### Parameters: <br/>
- **`optimize`** string, default = 'Accuracy' for `pycaret.classification` and 'R2' for `pycaret.regression` <br/>
Other values you can pass in optimize param are 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', and 'MCC' for `pycaret.classification` and 'MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', and 'MAPE' for `pycaret.regression` <br/><br/>
- **`use_holdout`** bool, default = False <br/>
When set to True, metrics are evaluated on holdout set instead of CV.

# <span style="color:red"> NEW FUNCTION: pull </span>
**`pycaret.classification` `pycaret.regression`** <br/>

- This function returns the last printed score grid as pandas dataframe.

# <span style="color:red"> NEW FUNCTION: models </span>
**`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`** <br/>

- This function Returns the table of models available in model library.

### Parameters: 
- **`type`** string, default = None <br/>
linear : filters and only return linear models <br/>
tree : filters and only return tree based models <br/>
ensemble : filters and only return ensemble models <br/>

`type` parameter only available in `pycaret.classification` and `pycaret.regression`

# <span style="color:red"> NEW FUNCTION: get_logs </span>
**`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`** <br/>

- This function returns a table with experiment logs consisting run details, parameter, metrics and tags. 

### Parameters: 
- **`experiment_name`** string, default = None <br/>
When set to None current active run is used. <br/><br/>

- **`save`** bool, default = False <br/>
When set to True, csv file is saved in current directory.

# <span style="color:red"> NEW FUNCTION: get_config </span>
**`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`** <br/>

- This function is used to access global environment variables. Check docstring for the list of global var accessible.

# <span style="color:red"> NEW FUNCTION: set_config </span>
**`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`** <br/>

- This function is used to reset global environment variables. Check docstring for the list of global var accessible.

# <span style="color:red"> NEW FUNCTION: get_system_logs </span>
**`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`** <br/>

- This function is reads and print 'logs.log' file from current active directory. logs.log is generated from `setup` is initialized in any module.
