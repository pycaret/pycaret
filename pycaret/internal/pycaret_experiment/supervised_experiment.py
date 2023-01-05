import datetime
import gc
import os
import time
import traceback
import warnings
from copy import copy, deepcopy
from functools import partial
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np  # type: ignore
import pandas as pd  # type ignore
import pandas.io.formats.style
from sklearn.base import clone  # type: ignore
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.utils.validation import check_is_fitted as check_fitted

import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
import pycaret.internal.preprocess
from pycaret.containers.metrics import (
    get_all_class_metric_containers,
    get_all_reg_metric_containers,
)
from pycaret.internal.display import CommonDisplay, DummyDisplay
from pycaret.internal.distributions import (
    CategoricalDistribution,
    Distribution,
    UniformDistribution,
    get_base_distributions,
    get_CS_distributions,
    get_hyperopt_distributions,
    get_optuna_distributions,
    get_skopt_distributions,
    get_tune_distributions,
)
from pycaret.internal.logging import get_logger, redirect_output
from pycaret.internal.meta_estimators import (
    CustomProbabilityThresholdClassifier,
    get_estimator_from_meta_estimator,
)
from pycaret.internal.parallel.parallel_backend import ParallelBackend
from pycaret.internal.pipeline import (
    Pipeline,
    estimator_pipeline,
    get_pipeline_estimator_label,
    get_pipeline_fit_kwargs,
)
from pycaret.internal.pycaret_experiment.tabular_experiment import _TabularExperiment
from pycaret.internal.tunable import TunableMixin
from pycaret.internal.validation import is_fitted, is_sklearn_cv_generator
from pycaret.utils._dependencies import _check_soft_dependencies
from pycaret.utils.constants import DATAFRAME_LIKE, LABEL_COLUMN, SCORE_COLUMN
from pycaret.utils.generic import (
    MLUsecase,
    can_early_stop,
    color_df,
    get_label_encoder,
    get_ml_task,
    id_or_display_name,
    nullcontext,
    true_warm_start,
)

try:
    from collections.abc import Iterable
except Exception:
    from collections import Iterable

LOGGER = get_logger()


class _SupervisedExperiment(_TabularExperiment):
    _create_app_predict_kwargs = {}

    def __init__(self) -> None:
        super().__init__()
        self.transform_target_param = False  # Default False for both class/reg
        self._variable_keys = self._variable_keys.union(
            {
                "X",
                "y",
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "target_param",
                "fold_shuffle_param",
                "fold_generator",
                "fold_groups_param",
            }
        )

    def _calculate_metrics(
        self,
        y_test,
        pred,
        pred_prob,
        weights: Optional[list] = None,
        **additional_kwargs,
    ) -> dict:
        """
        Calculate all metrics in _all_metrics.
        """
        from pycaret.utils.generic import calculate_metrics

        with redirect_output(self.logger):
            try:
                return calculate_metrics(
                    metrics=self._all_metrics,
                    y_test=y_test,
                    pred=pred,
                    pred_proba=pred_prob,
                    weights=weights,
                    **additional_kwargs,
                )
            except Exception:
                ml_usecase = get_ml_task(y_test)
                if ml_usecase == MLUsecase.CLASSIFICATION:
                    metrics = get_all_class_metric_containers(self.variables, True)
                elif ml_usecase == MLUsecase.REGRESSION:
                    metrics = get_all_reg_metric_containers(self.variables, True)
                return calculate_metrics(
                    metrics=metrics,  # type: ignore
                    y_test=y_test,
                    pred=pred,
                    pred_proba=pred_prob,
                    weights=weights,
                    **additional_kwargs,
                )

    def _is_unsupervised(self) -> bool:
        return False

    def _get_final_model_from_pipeline(
        self,
        pipeline: Pipeline,
        check_is_fitted: bool = False,
    ) -> Any:
        """Extracts and returns the final model from the pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline with a final model

        check_is_fitted : bool, default=False
            If True, will check if final model is fitted and raise an exception
            if it is not, by default False.

        Returns
        -------
        Model
            The final model in the pipeline.

        """
        model = pipeline._final_estimator
        if check_is_fitted:
            check_fitted(model)

        return model

    def _choose_better(
        self,
        models_and_results: list,
        compare_dimension: str,
        fold: int,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        display: Optional[CommonDisplay] = None,
    ):
        """
        When choose_better is set to True, optimize metric in scoregrid is
        compared with base model created using create_model so that the
        functions return the model with better score only. This will ensure
        model performance is at least equivalent to what is seen in compare_models
        """

        self.logger.info("choose_better activated")
        if display is not None:
            display.update_monitor(1, "Compiling Final Results")

        if not fit_kwargs:
            fit_kwargs = {}

        for i, x in enumerate(models_and_results):
            if not isinstance(x, tuple):
                models_and_results[i] = (x, None)
            elif isinstance(x[0], str):
                models_and_results[i] = (x[1], None)
            elif len(x) != 2:
                raise ValueError(f"{x} must have length 2 but has {len(x)}")

        metric = self._get_metric_by_name_or_id(compare_dimension)

        best_result = None
        best_model = None
        for model, result in models_and_results:
            if result is not None and is_fitted(model):
                try:
                    indices = self._get_return_train_score_indices_for_logging(
                        return_train_score=True
                    )
                    result = result.loc[indices][compare_dimension]
                except KeyError:
                    indices = self._get_return_train_score_indices_for_logging(
                        return_train_score=False
                    )
                    result = result.loc[indices][compare_dimension]
            else:
                self.logger.info(
                    "SubProcess create_model() called =================================="
                )
                model, _ = self._create_model(
                    model,
                    verbose=False,
                    system=False,
                    fold=fold,
                    fit_kwargs=fit_kwargs,
                    groups=groups,
                )
                self.logger.info(
                    "SubProcess create_model() end =================================="
                )
                result = self.pull(pop=True).loc[
                    self._get_return_train_score_indices_for_logging(
                        return_train_score=False
                    )
                ][compare_dimension]
            self.logger.info(f"{model} result for {compare_dimension} is {result}")
            if not metric.greater_is_better:
                result *= -1
            if best_result is None or best_result < result:
                best_result = result
                best_model = model

        self.logger.info(f"{best_model} is best model")

        self.logger.info("choose_better completed")
        return best_model

    def _get_cv_n_folds(self, fold, X, y=None, groups=None):
        import pycaret.utils.generic

        return pycaret.utils.generic.get_cv_n_folds(
            fold, default=self.fold_generator, X=X, y=y, groups=groups
        )

    def _set_up_logging(
        self, runtime, log_data, log_profile, experiment_custom_tags=None
    ):
        # experiment custom tags
        if experiment_custom_tags is not None:
            if not isinstance(experiment_custom_tags, dict):
                raise TypeError(
                    "experiment_custom_tags parameter must be dict if not None"
                )

        if self.logging_param:
            self.logging_param.log_experiment(
                self,
                log_profile,
                log_data,
                experiment_custom_tags,
                runtime,
            )

    def _parallel_compare_models(
        self,
        parallel: Optional[ParallelBackend],
        caller_params: Optional[dict],
        turbo: bool,
    ) -> List[Any]:
        params = dict(caller_params)
        parallel.attach(self)
        if params.get("include", None) is None:
            _models = self.models()
            if turbo:
                _models = _models[_models.Turbo]
            params["include"] = _models.index.tolist()
        del params["self"]
        del params["__class__"]
        del params["parallel"]
        return parallel.compare_models(self, params)

    def _get_greater_is_worse_columns(self) -> Set[str]:
        input_ml_usecase = self._ml_usecase
        target_ml_usecase = MLUsecase.TIME_SERIES

        greater_is_worse_columns = {
            id_or_display_name(v, input_ml_usecase, target_ml_usecase).upper()
            for k, v in self._all_metrics.items()
            if not v.greater_is_better
        }
        greater_is_worse_columns.add("TT (Sec)")
        return greater_is_worse_columns

    def _highlight_models(self, master_display_: Any) -> Any:
        def highlight_max(s):
            to_highlight = s == s.max()
            return ["background-color: yellow" if v else "" for v in to_highlight]

        def highlight_min(s):
            to_highlight = s == s.min()
            return ["background-color: yellow" if v else "" for v in to_highlight]

        def highlight_cols(s):
            color = "lightgrey"
            return f"background-color: {color}"

        greater_is_worse_columns = self._get_greater_is_worse_columns()

        if master_display_ is not None:
            return (
                master_display_.apply(
                    highlight_max,
                    subset=[
                        x
                        for x in master_display_.columns[1:]
                        if x not in greater_is_worse_columns
                    ],
                )
                .apply(
                    highlight_min,
                    subset=[
                        x
                        for x in master_display_.columns[1:]
                        if x in greater_is_worse_columns
                    ],
                )
                .applymap(highlight_cols, subset=["TT (Sec)"])
            )
        else:
            return pd.DataFrame().style

    def _process_sort(self, sort: Any) -> Tuple[str, bool]:
        """This function is extracted from different parts from the
        compare_models function, and it is used for parallel compare_models
        """
        input_ml_usecase = self._ml_usecase
        target_ml_usecase = MLUsecase.TIME_SERIES

        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort = self._get_metric_by_name_or_id(sort)
            if sort is None:
                raise ValueError(
                    "Sort method not supported. See docstring for list of available parameters."
                )

        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort_ascending = not sort.greater_is_better
            sort = id_or_display_name(sort, input_ml_usecase, target_ml_usecase)
        else:
            sort_ascending = True
            sort = "TT (Sec)"

        if self._ml_usecase == MLUsecase.TIME_SERIES:
            sort = sort.upper()

        return sort, sort_ascending

    def compare_models(
        self,
        include: Optional[
            List[Union[str, Any]]
        ] = None,  # changed whitelist to include in pycaret==2.1
        exclude: Optional[
            List[str]
        ] = None,  # changed blacklist to exclude in pycaret==2.1
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        sort: str = "Accuracy",
        n_select: int = 1,
        budget_time: Optional[float] = None,  # added in pycaret==2.1.0
        turbo: bool = True,
        errors: str = "ignore",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
        parallel: Optional[ParallelBackend] = None,
        caller_params: Optional[dict] = None,
    ) -> List[Any]:

        """
        This function train all the models available in the model library and scores them
        using Cross Validation. The output prints a score grid with Accuracy,
        AUC, Recall, Precision, F1, Kappa and MCC (averaged across folds).

        This function returns all of the models compared, sorted by the value of the selected metric.

        When turbo is set to True ('rbfsvm', 'gpc' and 'mlp') are excluded due to longer
        training time. By default turbo parameter is set to True.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> best_model = compare_models()

        This will return the averaged score grid of all the models except 'rbfsvm', 'gpc'
        and 'mlp'. When turbo parameter is set to False, all models including 'rbfsvm', 'gpc'
        and 'mlp' are used but this may result in longer training time.

        >>> best_model = compare_models( exclude = [ 'knn', 'gbc' ] , turbo = False)

        This will return a comparison of all models except K Nearest Neighbour and
        Gradient Boosting Classifier.

        >>> best_model = compare_models( exclude = [ 'knn', 'gbc' ] , turbo = True)

        This will return comparison of all models except K Nearest Neighbour,
        Gradient Boosting Classifier, SVM (RBF), Gaussian Process Classifier and
        Multi Level Perceptron.


        >>> tuned_model = tune_model(create_model('lr'))
        >>> best_model = compare_models( include = [ 'lr', tuned_model ])

        This will compare a tuned Linear Regression model with an untuned one.

        Parameters
        ----------
        exclude: list of strings, default = None
            In order to omit certain models from the comparison model ID's can be passed as
            a list of strings in exclude param.

        include: list of strings or objects, default = None
            In order to run only certain models for the comparison, the model ID's can be
            passed as a list of strings in include param. The list can also include estimator
            objects to be compared.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        cross_validation: bool, default = True
            When cross_validation set to False fold parameter is ignored and models are trained
            on entire training dataset, returning metrics calculated using the train (holdout) set.

        sort: str, default = 'Accuracy'
            The scoring measure specified is used for sorting the average score grid
            Other options are 'AUC', 'Recall', 'Precision', 'F1', 'Kappa' and 'MCC'.

        n_select: int, default = 1
            Number of top_n models to return. use negative argument for bottom selection.
            for example, n_select = -3 means bottom 3 models.

        budget_time: int or float, default = None
            If not 0 or None, will terminate execution of the function after budget_time
            minutes have passed and return results up to that point.

        turbo: bool, default = True
            When turbo is set to True, it excludes estimators that have longer
            training time.

        errors: str, default = 'ignore'
            If 'ignore', will suppress model exceptions and continue.
            If 'raise', will allow exceptions to be raised.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model. The parameters will be applied to all models,
            therefore it is recommended to set errors parameter to 'ignore'.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        parallel: pycaret.internal.parallel.parallel_backend.ParallelBackend, default = None
            A ParallelBackend instance. For example if you have a SparkSession ``session``,
            you can use ``FugueBackend(session)`` to make this function running using
            Spark. For more details, see
            :class:`~pycaret.parallel.fugue_backend.FugueBackend`


        caller_params: dict, default = None
            The parameters used to call this function in the subclass. There are inconsistencies
            in this function's signature between this base class and the subclasses, so this is
            used to prevent inconsistencies. It must be set when ``parallel`` is not None


        extra_params: Any
            Extra parameters used to call the same method in the derived class. These parameters
            are mainly used when ``parallel`` is not None.


        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds.
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1,
            Kappa and MCC. Mean and standard deviation of the scores across
            the folds are also returned.

        list
            List of fitted model objects that were compared.

        Warnings
        --------
        - compare_models() though attractive, might be time consuming with large
        datasets. By default turbo is set to True, which excludes models that
        have longer training times. Changing turbo parameter to False may result
        in very high training times with datasets where number of samples exceed
        10,000.

        - If target variable is multiclass (more than 2 classes), AUC will be
        returned as zero (0.0)

        - If cross_validation parameter is set to False, no models will be logged with MLFlow.

        """
        self._check_setup_ran()

        if parallel is not None:
            return self._parallel_compare_models(parallel, caller_params, turbo=turbo)

        # No extra code should be added above this line
        # --------------------------------------------------------------

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing compare_models()")
        self.logger.info(f"compare_models({function_params_str})")

        self.logger.info("Checking exceptions")

        if not fit_kwargs:
            fit_kwargs = {}

        # checking error for exclude (string)
        available_estimators = self._all_models

        if include is not None:
            for i in include:
                if isinstance(i, str):
                    if i not in available_estimators:
                        raise ValueError(
                            f"Estimator {i} Not Available. Please see docstring for list of available estimators."
                        )
                elif not hasattr(i, "fit"):
                    raise ValueError(
                        f"Estimator {i} does not have the required fit() method."
                    )

        # include and exclude together check
        if include is not None and exclude is not None:
            raise TypeError(
                "Cannot use exclude parameter when include is used to compare models."
            )

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking budget_time parameter
        if (
            budget_time
            and type(budget_time) is not int
            and type(budget_time) is not float
        ):
            raise TypeError(
                "budget_time parameter only accepts integer or float values."
            )

        # checking sort parameter
        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort = self._get_metric_by_name_or_id(sort)
            if sort is None:
                raise ValueError(
                    "Sort method not supported. See docstring for list of available parameters."
                )

        # checking errors parameter
        possible_errors = ["ignore", "raise"]
        if errors not in possible_errors:
            raise ValueError(
                f"errors parameter must be one of: {', '.join(possible_errors)}."
            )

        # checking optimize parameter for multiclass
        if self.is_multiclass:
            if not sort.is_multiclass:
                raise TypeError(
                    f"{sort} metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                )

        """

        ERROR HANDLING ENDS HERE

        """

        if self._ml_usecase != MLUsecase.TIME_SERIES:
            fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        pd.set_option("display.max_columns", 500)

        self.logger.info("Preparing display monitor")

        len_mod = (
            len({k: v for k, v in self._all_models.items() if v.is_turbo})
            if turbo
            else len(self._all_models)
        )

        if include:
            len_mod = len(include)
        elif exclude:
            len_mod -= len(exclude)

        progress_args = {"max": (4 * len_mod) + 4 + min(len_mod, abs(n_select))}
        master_display_columns = (
            ["Model"]
            + [v.display_name for k, v in self._all_metrics.items()]
            + ["TT (Sec)"]
        )
        master_display = pd.DataFrame(columns=master_display_columns)
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            [
                "Status",
                ". . . . . . . . . . . . . . . . . .",
                "Loading Dependencies",
            ],
            [
                "Estimator",
                ". . . . . . . . . . . . . . . . . .",
                "Compiling Library",
            ],
        ]
        display = (
            DummyDisplay()
            if self._remote
            else CommonDisplay(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                monitor_rows=monitor_rows,
            )
        )
        if display.can_update_text:
            display.display(master_display, final_display=False)

        input_ml_usecase = self._ml_usecase
        target_ml_usecase = MLUsecase.TIME_SERIES

        np.random.seed(self.seed)

        display.move_progress()

        # defining sort parameter (making Precision equivalent to Prec. )

        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort_ascending = not sort.greater_is_better
            sort = id_or_display_name(sort, input_ml_usecase, target_ml_usecase)
        else:
            sort_ascending = True
            sort = "TT (Sec)"

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Loading Estimator")

        """
        MONITOR UPDATE ENDS
        """

        if include:
            model_library = include
        else:
            if turbo:
                model_library = [k for k, v in self._all_models.items() if v.is_turbo]
            else:
                model_library = list(self._all_models.keys())
            if exclude:
                model_library = [x for x in model_library if x not in exclude]

        if self._ml_usecase == MLUsecase.TIME_SERIES:
            if "ensemble_forecaster" in model_library:
                warnings.warn(
                    "Unsupported estimator `ensemble_forecaster` for method `compare_models()`, removing from model_library"
                )
                model_library.remove("ensemble_forecaster")

        display.move_progress()

        # create URI (before loop)
        import secrets

        URI = secrets.token_hex(nbytes=4)

        master_display = None
        master_display_ = None

        total_runtime_start = time.time()
        total_runtime = 0
        over_time_budget = False
        if budget_time and budget_time > 0:
            self.logger.info(f"Time budget is {budget_time} minutes")

        for i, model in enumerate(model_library):

            model_id = (
                model
                if (
                    isinstance(model, str)
                    and all(isinstance(m, str) for m in model_library)
                )
                else str(i)
            )
            model_name = self._get_model_name(model)

            if isinstance(model, str):
                self.logger.info(f"Initializing {model_name}")
            else:
                self.logger.info(f"Initializing custom model {model_name}")

            # run_time
            runtime_start = time.time()
            total_runtime += (runtime_start - total_runtime_start) / 60
            self.logger.info(f"Total runtime is {total_runtime} minutes")
            over_time_budget = (
                budget_time and budget_time > 0 and total_runtime > budget_time
            )
            if over_time_budget:
                self.logger.info(
                    f"Total runtime {total_runtime} is over time budget by {total_runtime - budget_time}, breaking loop"
                )
                break
            total_runtime_start = runtime_start

            """
            MONITOR UPDATE STARTS
            """

            display.update_monitor(2, model_name)

            """
            MONITOR UPDATE ENDS
            """

            self.logger.info(
                "SubProcess create_model() called =================================="
            )
            create_model_args = dict(
                estimator=model,
                system=False,
                verbose=False,
                display=display,
                fold=fold,
                round=round,
                cross_validation=cross_validation,
                fit_kwargs=fit_kwargs,
                groups=groups,
                probability_threshold=probability_threshold,
                refit=False,
            )
            results_columns_to_ignore = ["Object", "runtime", "cutoff"]
            if errors == "raise":
                model, model_fit_time = self._create_model(**create_model_args)
                model_results = self.pull(pop=True)
            else:
                try:
                    model, model_fit_time = self._create_model(**create_model_args)
                    model_results = self.pull(pop=True)
                    assert (
                        np.sum(
                            model_results.drop(
                                results_columns_to_ignore, axis=1, errors="ignore"
                            ).iloc[0]
                        )
                        != 0.0
                    )
                except Exception:
                    self.logger.warning(
                        f"create_model() for {model} raised an exception or returned all 0.0, trying without fit_kwargs:"
                    )
                    self.logger.warning(traceback.format_exc())
                    try:
                        model, model_fit_time = self._create_model(**create_model_args)
                        model_results = self.pull(pop=True)
                        assert (
                            np.sum(
                                model_results.drop(
                                    results_columns_to_ignore, axis=1, errors="ignore"
                                ).iloc[0]
                            )
                            != 0.0
                        )
                    except Exception:
                        self.logger.error(
                            f"create_model() for {model} raised an exception or returned all 0.0:"
                        )
                        self.logger.error(traceback.format_exc())
                        continue
            self.logger.info(
                "SubProcess create_model() end =================================="
            )

            if model is None:
                over_time_budget = True
                self.logger.info(
                    "Time budged exceeded in create_model(), breaking loop"
                )
                break

            runtime_end = time.time()
            runtime = np.array(runtime_end - runtime_start).round(2)

            self.logger.info("Creating metrics dataframe")
            if cross_validation:
                # cutoff only present in time series and when cv = True
                if "cutoff" in model_results.columns:
                    model_results.drop("cutoff", axis=1, errors="ignore")
                compare_models_ = pd.DataFrame(
                    model_results.loc[
                        self._get_return_train_score_indices_for_logging(
                            return_train_score=False
                        )
                    ]
                ).T.reset_index(drop=True)
            else:
                compare_models_ = pd.DataFrame(model_results.iloc[0]).T
            compare_models_.insert(
                len(compare_models_.columns), "TT (Sec)", model_fit_time
            )
            compare_models_.insert(0, "Model", model_name)
            compare_models_.insert(0, "Object", [model])
            compare_models_.insert(0, "runtime", runtime)
            compare_models_.index = [model_id]
            if master_display is None:
                master_display = compare_models_
            else:
                master_display = pd.concat(
                    [master_display, compare_models_], ignore_index=False
                )
            master_display = master_display.round(round)
            if self._ml_usecase != MLUsecase.TIME_SERIES:
                master_display = master_display.sort_values(
                    by=sort, ascending=sort_ascending
                )
            else:
                master_display = master_display.sort_values(
                    by=sort.upper(), ascending=sort_ascending
                )

            master_display_ = master_display.drop(
                results_columns_to_ignore, axis=1, errors="ignore"
            ).style.format(precision=round)
            master_display_ = master_display_.set_properties(**{"text-align": "left"})
            master_display_ = master_display_.set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]
            )

            if display.can_update_text:
                display.display(master_display_, final_display=False)

        display.move_progress()

        compare_models_ = self._highlight_models(master_display_)

        display.update_monitor(1, "Compiling Final Models")

        display.move_progress()

        sorted_models = []

        if master_display is not None:
            clamped_n_select = min(len(master_display), abs(n_select))
            if n_select < 0:
                n_select_range = range(
                    len(master_display) - clamped_n_select, len(master_display)
                )
            else:
                n_select_range = range(0, clamped_n_select)

            if self.logging_param:
                self.logging_param.log_model_comparison(
                    master_display, "compare_models"
                )

            for index, row in enumerate(master_display.iterrows()):
                _, row = row
                model = row["Object"]

                results = row.to_frame().T.drop(
                    ["Object", "Model", "runtime", "TT (Sec)"], errors="ignore", axis=1
                )

                avgs_dict_log = {k: v for k, v in results.iloc[0].items()}

                full_logging = False

                if index in n_select_range:
                    display.update_monitor(2, self._get_model_name(model))
                    create_model_args = dict(
                        estimator=model,
                        system=False,
                        verbose=False,
                        fold=fold,
                        round=round,
                        cross_validation=False,
                        predict=False,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        probability_threshold=probability_threshold,
                    )
                    if errors == "raise":
                        model, model_fit_time = self._create_model(**create_model_args)
                        sorted_models.append(model)
                    else:
                        try:
                            model, model_fit_time = self._create_model(
                                **create_model_args
                            )
                            sorted_models.append(model)
                            assert (
                                np.sum(
                                    model_results.drop(
                                        results_columns_to_ignore,
                                        axis=1,
                                        errors="ignore",
                                    ).iloc[0]
                                )
                                != 0.0
                            )
                        except Exception:
                            self.logger.error(
                                f"create_model() for {model} raised an exception or returned all 0.0:"
                            )
                            self.logger.error(traceback.format_exc())
                            model = None
                            display.move_progress()
                            continue
                    display.move_progress()
                    full_logging = True

                if self.logging_param and cross_validation and model is not None:

                    self._log_model(
                        model=model,
                        model_results=results,
                        score_dict=avgs_dict_log,
                        source="compare_models",
                        runtime=row["runtime"],
                        model_fit_time=row["TT (Sec)"],
                        pipeline=self.pipeline,
                        log_plots=self.log_plots_param if full_logging else [],
                        log_holdout=full_logging,
                        URI=URI,
                        display=display,
                        experiment_custom_tags=experiment_custom_tags,
                    )

        if len(sorted_models) == 1:
            sorted_models = sorted_models[0]

        display.display(compare_models_, final_display=True)

        pd.reset_option("display.max_columns")

        # store in display container
        self._display_container.append(compare_models_.data)

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(sorted_models))
        self.logger.info(
            "compare_models() successfully completed......................................"
        )

        return sorted_models

    def _create_model_without_cv(
        self,
        model,
        data_X,
        data_y,
        fit_kwargs,
        round,
        predict,
        system,
        display: CommonDisplay,
        model_only: bool = True,
        return_train_score: bool = False,
    ):
        with estimator_pipeline(self.pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            self.logger.info("Cross validation set to False")

            self.logger.info("Fitting Model")
            model_fit_start = time.time()
            with redirect_output(self.logger):
                pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
            model_fit_end = time.time()

            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

            display.move_progress()

            if predict:
                if return_train_score:
                    # call class explicitly to get access to preprocess arg
                    # in subclasses
                    _SupervisedExperiment.predict_model(
                        self,
                        pipeline_with_model,
                        data=pd.concat([data_X, data_y], axis=1),
                        verbose=False,
                    )
                    train_results = self.pull(pop=True).drop("Model", axis=1)
                    train_results.index = ["Train"]
                else:
                    train_results = None

                self.predict_model(pipeline_with_model, verbose=False)
                model_results = self.pull(pop=True).drop("Model", axis=1)
                model_results.index = ["Test"]
                if train_results is not None:
                    model_results = pd.concat([model_results, train_results])

                self._display_container.append(model_results)

                model_results = model_results.style.format(precision=round)

                if system:
                    display.display(model_results)

                self.logger.info(f"_display_container: {len(self._display_container)}")

            if not model_only:
                return pipeline_with_model, model_fit_time

        return model, model_fit_time

    def _create_model_with_cv(
        self,
        model,
        data_X,
        data_y,
        fit_kwargs,
        round,
        cv,
        groups,
        metrics,
        refit,
        system,
        display,
        return_train_score: bool = False,
    ):
        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(
            1,
            f"Fitting {self._get_cv_n_folds(cv, data_X, y=data_y, groups=groups)} Folds",
        )
        """
        MONITOR UPDATE ENDS
        """

        from sklearn.model_selection import cross_validate

        metrics_dict = dict([(k, v.scorer) for k, v in metrics.items()])

        self.logger.info("Starting cross validation")

        n_jobs = self.gpu_n_jobs_param
        from sklearn.gaussian_process import (
            GaussianProcessClassifier,
            GaussianProcessRegressor,
        )

        # special case to prevent running out of memory
        if isinstance(model, (GaussianProcessClassifier, GaussianProcessRegressor)):
            n_jobs = 1

        with estimator_pipeline(self.pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            self.logger.info(f"Cross validating with {cv}, n_jobs={n_jobs}")

            model_fit_start = time.time()
            with redirect_output(self.logger):
                scores = cross_validate(
                    pipeline_with_model,
                    data_X,
                    data_y,
                    cv=cv,
                    groups=groups,
                    scoring=metrics_dict,
                    fit_params=fit_kwargs,
                    n_jobs=n_jobs,
                    return_train_score=return_train_score,
                    error_score=0,
                )
            model_fit_end = time.time()
            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

            score_dict = {}
            for k, v in metrics.items():
                score_dict[v.display_name] = []
                if return_train_score:
                    train_score = scores[f"train_{k}"] * (
                        1 if v.greater_is_better else -1
                    )
                    train_score = train_score.tolist()
                    score_dict[v.display_name] = train_score
                test_score = scores[f"test_{k}"] * (1 if v.greater_is_better else -1)
                test_score = test_score.tolist()
                score_dict[v.display_name] += test_score

            self.logger.info("Calculating mean and std")

            avgs_dict = {}
            for k, v in metrics.items():
                avgs_dict[v.display_name] = []
                if return_train_score:
                    train_score = scores[f"train_{k}"] * (
                        1 if v.greater_is_better else -1
                    )
                    train_score = train_score.tolist()
                    avgs_dict[v.display_name] = [
                        np.mean(train_score),
                        np.std(train_score),
                    ]
                test_score = scores[f"test_{k}"] * (1 if v.greater_is_better else -1)
                test_score = test_score.tolist()
                avgs_dict[v.display_name] += [np.mean(test_score), np.std(test_score)]

            display.move_progress()

            self.logger.info("Creating metrics dataframe")

            if hasattr(cv, "n_splits"):
                fold = cv.n_splits
            elif hasattr(cv, "get_n_splits"):
                fold = cv.get_n_splits()
            else:
                raise ValueError(
                    "The cross validation class should implement a n_splits "
                    f"attribute or a get_n_splits method. {cv.__class__.__name__} "
                    "has neither."
                )

            if return_train_score:
                model_results = pd.DataFrame(
                    {
                        "Split": ["CV-Train"] * fold
                        + ["CV-Val"] * fold
                        + ["CV-Train"] * 2
                        + ["CV-Val"] * 2,
                        "Fold": np.arange(fold).tolist()
                        + np.arange(fold).tolist()
                        + ["Mean", "Std"] * 2,
                    }
                )
            else:
                model_results = pd.DataFrame(
                    {
                        "Fold": np.arange(fold).tolist() + ["Mean", "Std"],
                    }
                )

            model_scores = pd.concat(
                [pd.DataFrame(score_dict), pd.DataFrame(avgs_dict)]
            ).reset_index(drop=True)

            model_results = pd.concat([model_results, model_scores], axis=1)
            model_results.set_index(
                self._get_return_train_score_columns_for_display(return_train_score),
                inplace=True,
            )

            if refit:
                # refitting the model on complete X_train, y_train
                display.update_monitor(1, "Finalizing Model")
                model_fit_start = time.time()
                self.logger.info("Finalizing model")
                with redirect_output(self.logger):
                    pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
                    model_fit_end = time.time()

                # calculating metrics on predictions of complete train dataset
                if return_train_score:
                    # call class explicitly to get access to preprocess arg
                    # in subclasses
                    _SupervisedExperiment.predict_model(
                        self,
                        pipeline_with_model,
                        data=pd.concat([data_X, data_y], axis=1),
                        verbose=False,
                    )
                    metrics = self.pull(pop=True).drop("Model", axis=1)
                    df_score = pd.DataFrame({"Split": ["Train"], "Fold": [None]})
                    df_score = pd.concat([df_score, metrics], axis=1)
                    df_score.set_index(["Split", "Fold"], inplace=True)

                    # concatenating train results to cross-validation socre dataframe
                    model_results = pd.concat([model_results, df_score])

                model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
            else:
                model_fit_time /= self._get_cv_n_folds(
                    cv, data_X, y=data_y, groups=groups
                )

        model_results = model_results.round(round)

        return model, model_fit_time, model_results, avgs_dict

    def _get_return_train_score_columns_for_display(
        self, return_train_score: bool
    ) -> List[str]:
        if return_train_score:
            columns = ["Split", "Fold"]
        else:
            columns = ["Fold"]
        return columns

    def _get_return_train_score_indices_for_logging(self, return_train_score: bool):
        if return_train_score:
            indices = ("CV-Val", "Mean")
        else:
            indices = "Mean"
        return indices

    def _highlight_and_round_model_results(
        self, model_results: pd.DataFrame, return_train_score: bool, round: int
    ) -> pandas.io.formats.style.Styler:
        # yellow the mean
        if return_train_score:
            indices = [("CV-Val", "Mean"), ("CV-Train", "Mean")]
        else:
            indices = ["Mean"]
        model_results = color_df(model_results, "yellow", indices, axis=1)
        model_results = model_results.format(precision=round)
        return model_results

    def _create_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        predict: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        refit: bool = True,
        probability_threshold: Optional[float] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        system: bool = True,
        add_to_model_list: bool = True,
        X_train_data: Optional[pd.DataFrame] = None,  # added in pycaret==2.2.0
        y_train_data: Optional[pd.DataFrame] = None,  # added in pycaret==2.2.0
        metrics=None,
        display: Optional[CommonDisplay] = None,  # added in pycaret==2.2.0
        model_only: bool = True,
        return_train_score: bool = False,
        **kwargs,
    ) -> Any:

        """
        Internal version of ``create_model`` with private arguments.
        """
        self._check_setup_ran()

        function_params_str = ", ".join(
            [
                f"{k}={v}"
                for k, v in locals().items()
                if k not in ("X_train_data", "y_train_data")
            ]
        )

        self.logger.info("Initializing create_model()")
        self.logger.info(f"create_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        available_estimators = set(self._all_models_internal.keys())

        if not fit_kwargs:
            fit_kwargs = {}

        # only raise exception of estimator is of type string.
        if isinstance(estimator, str):
            if estimator not in available_estimators:
                raise ValueError(
                    f"Estimator {estimator} not available. Please see docstring for list of available estimators."
                )
        elif not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        # checking system parameter
        if type(system) is not bool:
            raise TypeError("System parameter can only take argument as True or False.")

        # checking cross_validation parameter
        if type(cross_validation) is not bool:
            raise TypeError(
                "cross_validation parameter can only take argument as True or False."
            )

        # checking return_train_score parameter
        if type(return_train_score) is not bool:
            raise TypeError(
                "return_train_score can only take argument as True or False"
            )

        """

        ERROR HANDLING ENDS HERE

        """

        if not display:
            progress_args = {"max": 4}
            timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
            monitor_rows = [
                ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
                [
                    "Status",
                    ". . . . . . . . . . . . . . . . . .",
                    "Loading Dependencies",
                ],
                [
                    "Estimator",
                    ". . . . . . . . . . . . . . . . . .",
                    "Compiling Library",
                ],
            ]
            display = CommonDisplay(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                monitor_rows=monitor_rows,
            )

        self.logger.info("Importing libraries")

        # general dependencies

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")

        # Storing X_train and y_train in data_X and data_y parameter
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            data_X = self.X_train if X_train_data is None else X_train_data.copy()
            data_y = self.y_train if y_train_data is None else y_train_data.copy()
        else:
            if X_train_data is not None:
                data_X = X_train_data.copy()
            else:
                if self.X_train is None:
                    data_X = None
                else:
                    data_X = self.X_train
            data_y = self.y_train if y_train_data is None else y_train_data.copy()

        groups = self._get_groups(groups, data=data_X)

        if metrics is None:
            metrics = self._all_metrics

        display.move_progress()

        self.logger.info("Defining folds")

        # cross validation setup starts here
        if self._ml_usecase == MLUsecase.TIME_SERIES:
            cv = self.get_fold_generator(fold=fold)

            # Add forecast horizon
            fit_kwargs = self.update_fit_kwargs_with_fh_from_cv(
                fit_kwargs=fit_kwargs, cv=cv
            )

        else:
            cv = self._get_cv_splitter(fold)

        self.logger.info("Declaring metric variables")

        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(1, "Selecting Estimator")
        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Importing untrained model")

        if isinstance(estimator, str) and estimator in available_estimators:
            model_definition = self._all_models_internal[estimator]
            model_args = model_definition.args
            model_args = {**model_args, **kwargs}
            model = model_definition.class_def(**model_args)
            full_name = model_definition.name
        else:
            self.logger.info("Declaring custom model")

            model = clone(estimator)
            model.set_params(**kwargs)

            full_name = self._get_model_name(model)

        # workaround for an issue with set_params in cuML
        model = clone(model)

        display.update_monitor(2, full_name)

        if (
            probability_threshold
            and self._ml_usecase == MLUsecase.CLASSIFICATION
            and not self.is_multiclass
        ):
            if not isinstance(model, CustomProbabilityThresholdClassifier):
                model = CustomProbabilityThresholdClassifier(
                    classifier=model,
                    probability_threshold=probability_threshold,
                )
            elif probability_threshold is not None:
                model.set_params(probability_threshold=probability_threshold)
        self.logger.info(f"{full_name} Imported successfully")

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """
        if not cross_validation:
            display.update_monitor(1, f"Fitting {str(full_name)}")
        else:
            display.update_monitor(1, "Initializing CV")

        """
        MONITOR UPDATE ENDS
        """

        if not cross_validation:
            model, model_fit_time = self._create_model_without_cv(
                model,
                data_X,
                data_y,
                fit_kwargs,
                round,
                predict,
                system,
                display,
                model_only=model_only,
                return_train_score=return_train_score,
            )

            display.move_progress()

            self.logger.info(str(model))
            self.logger.info(
                "create_model() successfully completed......................................"
            )

            gc.collect()

            if not system:
                return model, model_fit_time
            return model

        model, model_fit_time, model_results, _ = self._create_model_with_cv(
            model,
            data_X,
            data_y,
            fit_kwargs,
            round,
            cv,
            groups,
            metrics,
            refit,
            system,
            display,
            return_train_score=return_train_score,
        )

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # dashboard logging
        if self.logging_param and system and refit:
            indices = self._get_return_train_score_indices_for_logging(
                return_train_score
            )
            avgs_dict_log = {k: v for k, v in model_results.loc[indices].items()}

            self._log_model(
                model=model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="create_model",
                runtime=runtime,
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                experiment_custom_tags=experiment_custom_tags,
                display=display,
            )

        display.move_progress()

        self.logger.info("Uploading results into container")

        if not self._ml_usecase == MLUsecase.TIME_SERIES:
            model_results.drop("cutoff", axis=1, inplace=True, errors="ignore")

        self._display_container.append(model_results)

        # storing results in _master_model_container
        if add_to_model_list:
            self.logger.info("Uploading model into container now")
            self._master_model_container.append(
                {"model": model, "scores": model_results, "cv": cv}
            )

        # yellow the mean
        model_results = self._highlight_and_round_model_results(
            model_results, return_train_score, round
        )
        if system:
            display.display(model_results)

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "create_model() successfully completed......................................"
        )
        gc.collect()

        if not system:
            return model, model_fit_time

        return model

    def create_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        predict: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        refit: bool = True,
        probability_threshold: Optional[float] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        return_train_score: bool = False,
        **kwargs,
    ) -> Any:

        """
        This function creates a model and scores it using Cross Validation.
        The output prints a score grid that shows Accuracy, AUC, Recall, Precision,
        F1, Kappa and MCC by fold (default = 10 Fold).

        This function returns a trained model object.

        setup() function must be called before using create_model()

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')

        This will create a trained Logistic Regression model.

        Parameters
        ----------
        estimator : str / object, default = None
            Enter ID of the estimators available in model library or pass an untrained model
            object consistent with fit / predict API to train and evaluate model. All
            estimators support binary or multiclass problem. List of estimators in model
            library (ID - Name):

            * 'lr' - Logistic Regression
            * 'knn' - K Nearest Neighbour
            * 'nb' - Naive Bayes
            * 'dt' - Decision Tree Classifier
            * 'svm' - SVM - Linear Kernel
            * 'rbfsvm' - SVM - Radial Kernel
            * 'gpc' - Gaussian Process Classifier
            * 'mlp' - Multi Level Perceptron
            * 'ridge' - Ridge Classifier
            * 'rf' - Random Forest Classifier
            * 'qda' - Quadratic Discriminant Analysis
            * 'ada' - Ada Boost Classifier
            * 'gbc' - Gradient Boosting Classifier
            * 'lda' - Linear Discriminant Analysis
            * 'et' - Extra Trees Classifier
            * 'xgboost' - Extreme Gradient Boosting
            * 'lightgbm' - Light Gradient Boosting
            * 'catboost' - CatBoost Classifier

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        cross_validation: bool, default = True
            When cross_validation set to False fold parameter is ignored and model is trained
            on entire training dataset.

        predict: bool, default = True
            Whether to predict model on holdout if cross_validation == False.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        refit: bool, default = True
            Whether to refit the model on the entire dataset after CV. Ignored if cross_validation == False.

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        system: bool, default = True
            Must remain True all times. Only to be changed by internal functions.
            If False, method will return a tuple of model and the model fit time.

        add_to_model_list: bool, default = True
            Whether to save model and results in _master_model_container.

        X_train_data: pandas.DataFrame, default = None
            If not None, will use this dataframe as training features.
            Intended to be only changed by internal functions.

        y_train_data: pandas.DataFrame, default = None
            If not None, will use this dataframe as training target.
            Intended to be only changed by internal functions.

        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.

        **kwargs:
            Additional keyword arguments to pass to the estimator.

        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds.
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1,
            Kappa and MCC. Mean and standard deviation of the scores across
            the folds are highlighted in yellow.

        model
            trained model object

        Warnings
        --------
        - 'svm' and 'ridge' doesn't support predict_proba method. As such, AUC will be
        returned as zero (0.0)

        - If target variable is multiclass (more than 2 classes), AUC will be returned
        as zero (0.0)

        - 'rbfsvm' and 'gpc' uses non-linear kernel and hence the fit time complexity is
        more than quadratic. These estimators are hard to scale on datasets with more
        than 10,000 samples.

        - If cross_validation parameter is set to False, model will not be logged with MLFlow.

        """

        # TODO improve error message
        assert not any(
            x
            in (
                "system",
                "add_to_model_list",
                "X_train_data",
                "y_train_data",
                "metrics",
            )
            for x in kwargs
        )
        return self._create_model(
            estimator=estimator,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            predict=predict,
            fit_kwargs=fit_kwargs,
            groups=groups,
            refit=refit,
            probability_threshold=probability_threshold,
            experiment_custom_tags=experiment_custom_tags,
            verbose=verbose,
            return_train_score=return_train_score,
            **kwargs,
        )

    def tune_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[Union[Dict[str, list], Any]] = None,
        optimize: str = "Accuracy",
        custom_scorer=None,  # added in pycaret==2.1 - depreciated
        search_library: str = "scikit-learn",
        search_algorithm: Optional[str] = None,
        early_stopping: Any = False,
        early_stopping_max_iters: int = 10,
        choose_better: bool = False,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        return_tuner: bool = False,
        verbose: bool = True,
        tuner_verbose: Union[int, bool] = True,
        return_train_score: bool = False,
        **kwargs,
    ) -> Any:

        """
        This function tunes the hyperparameters of a model and scores it using Cross Validation.
        The output prints a score grid that shows Accuracy, AUC, Recall
        Precision, F1, Kappa and MCC by fold (by default = 10 Folds).

        This function returns a trained model object.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> xgboost = create_model('xgboost')
        >>> tuned_xgboost = tune_model(xgboost)

        This will tune the hyperparameters of Extreme Gradient Boosting Classifier.


        Parameters
        ----------
        estimator : object, default = None

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        n_iter: integer, default = 10
            Number of iterations within the Random Grid Search. For every iteration,
            the model randomly selects one value from the pre-defined grid of
            hyperparameters.

        custom_grid: dictionary, default = None
            To use custom hyperparameters for tuning pass a dictionary with parameter name
            and values to be iterated. When set to None it uses pre-defined tuning grid.
            Custom grids must be in a format supported by the chosen search library.

        optimize: str, default = 'Accuracy'
            Measure used to select the best model through hyperparameter tuning.
            Can be either a string representing a metric or a custom scorer object
            created using sklearn.make_scorer.

        custom_scorer: object, default = None
            Will be eventually depreciated.
            custom_scorer can be passed to tune hyperparameters of the model. It must be
            created using sklearn.make_scorer.

        search_library: str, default = 'scikit-learn'
            The search library used to tune hyperparameters.
            Possible values:

            - 'scikit-learn' - default, requires no further installation
            - 'scikit-optimize' - scikit-optimize. ``pip install scikit-optimize`` https://scikit-optimize.github.io/stable/
            - 'tune-sklearn' - Ray Tune scikit API. Does not support GPU models.
            ``pip install tune-sklearn ray[tune]`` https://github.com/ray-project/tune-sklearn
            - 'optuna' - Optuna. ``pip install optuna`` https://optuna.org/

        search_algorithm: str, default = None
            The search algorithm depends on the ``search_library`` parameter.
            Some search algorithms require additional libraries to be installed.
            If None, will use search library-specific default algorithm.

            - 'scikit-learn' possible values:
                - 'random' : random grid search (default)
                - 'grid' : grid search

            - 'scikit-optimize' possible values:
                - 'bayesian' : Bayesian search (default)

            - 'tune-sklearn' possible values:
                - 'random' : random grid search (default)
                - 'grid' : grid search
                - 'bayesian' : ``pip install scikit-optimize``
                - 'hyperopt' : ``pip install hyperopt``
                - 'optuna' : ``pip install optuna``
                - 'bohb' : ``pip install hpbandster ConfigSpace``

            - 'optuna' possible values:
                - 'random' : randomized search
                - 'tpe' : Tree-structured Parzen Estimator search (default)

        early_stopping: bool or str or object, default = False
            Use early stopping to stop fitting to a hyperparameter configuration
            if it performs poorly. Ignored if search_library is ``scikit-learn``, or
            if the estimator doesn't have partial_fit attribute.
            If False or None, early stopping will not be used.
            Can be either an object accepted by the search library or one of the
            following:

            - 'asha' for Asynchronous Successive Halving Algorithm
            - 'hyperband' for Hyperband
            - 'median' for median stopping rule
            - If False or None, early stopping will not be used.

            More info for Optuna - https://optuna.readthedocs.io/en/stable/reference/pruners.html
            More info for Ray Tune (tune-sklearn) - https://docs.ray.io/en/master/tune/api_docs/schedulers.html

        early_stopping_max_iters: int, default = 10
            Maximum number of epochs to run for each sampled configuration.
            Ignored if early_stopping is False or None.

        choose_better: bool, default = False
            When set to set to True, base estimator is returned when the performance doesn't
            improve by tune_model. This guarantees the returned object would perform at least
            equivalent to base estimator created using create_model or model returned by
            compare_models.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the tuner.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        return_tuner: bool, default = False
            If True, will return a tuple of (model, tuner_object). Otherwise,
            will return just the best model.

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        tuner_verbose: bool or in, default = True
            If True or above 0, will print messages from the tuner. Higher values
            print more messages. Ignored if verbose parameter is False.

        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.

        **kwargs:
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds.
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1,
            Kappa and MCC. Mean and standard deviation of the scores across
            the folds are also returned.

        model
            Trained and tuned model object.

        tuner_object
            Only if return_tuner parameter is True. The object used for tuning.

        Notes
        -----

        - If a StackingClassifier is passed, the hyperparameters of the meta model (final_estimator)
        will be tuned.

        - If a VotingClassifier is passed, the weights will be tuned.

        Warnings
        --------

        - Using 'Grid' search algorithm with default parameter grids may result in very
        long computation.


        """
        self._check_setup_ran()

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing tune_model()")
        self.logger.info(f"tune_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # checking estimator if string
        if type(estimator) is str:
            raise TypeError(
                "The behavior of tune_model in version 1.0.1 is changed. Please pass trained model object."
            )

        # Check for estimator
        if not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking n_iter parameter
        if type(n_iter) is not int:
            raise TypeError("n_iter parameter only accepts integer value.")

        # checking early_stopping parameter
        possible_early_stopping = ["asha", "Hyperband", "Median"]
        if (
            isinstance(early_stopping, str)
            and early_stopping not in possible_early_stopping
        ):
            raise TypeError(
                f"early_stopping parameter must be one of {', '.join(possible_early_stopping)}"
            )

        # checking early_stopping_max_iters parameter
        if type(early_stopping_max_iters) is not int:
            raise TypeError(
                "early_stopping_max_iters parameter only accepts integer value."
            )

        # checking return_train_score parameter
        if type(return_train_score) is not bool:
            raise TypeError(
                "return_train_score can only take argument as True or False"
            )

        # checking search_library parameter
        possible_search_libraries = [
            "scikit-learn",
            "scikit-optimize",
            "tune-sklearn",
            "optuna",
        ]
        search_library = search_library.lower()
        if search_library not in possible_search_libraries:
            raise ValueError(
                f"search_library parameter must be one of {', '.join(possible_search_libraries)}"
            )

        if search_library == "scikit-optimize":
            _check_soft_dependencies(
                "skopt",
                extra="tuners",
                severity="error",
                install_name="scikit-optimize",
            )
            import skopt

            if not search_algorithm:
                search_algorithm = "bayesian"

            possible_search_algorithms = ["bayesian"]
            if search_algorithm not in possible_search_algorithms:
                raise ValueError(
                    f"For 'scikit-optimize' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
                )

        elif search_library == "tune-sklearn":
            _check_soft_dependencies(
                "tune_sklearn",
                extra="tuners",
                severity="error",
                install_name="tune-sklearn ray[tune]",
            )

            if not search_algorithm:
                search_algorithm = "random"

            possible_search_algorithms = [
                "random",
                "grid",
                "bayesian",
                "hyperopt",
                "optuna",
                "bohb",
            ]
            if search_algorithm not in possible_search_algorithms:
                raise ValueError(
                    f"For 'tune-sklearn' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
                )

            if search_algorithm == "bohb":
                _check_soft_dependencies("ConfigSpace", extra=None, severity="error")
                _check_soft_dependencies("hpbandster", extra=None, severity="error")
                _check_soft_dependencies(
                    "ray", extra="tuners", severity="error", install_name="ray[tune]"
                )

            elif search_algorithm == "hyperopt":
                _check_soft_dependencies("hyperopt", extra="tuners", severity="error")
                _check_soft_dependencies(
                    "ray", extra="tuners", severity="error", install_name="ray[tune]"
                )

            elif search_algorithm == "bayesian":
                _check_soft_dependencies(
                    "skopt",
                    extra="tuners",
                    severity="error",
                    install_name="scikit-optimize",
                )
                import skopt

            elif search_algorithm == "optuna":
                _check_soft_dependencies("optuna", extra="tuners", severity="error")
                import optuna

        elif search_library == "optuna":
            _check_soft_dependencies("optuna", extra="tuners", severity="error")
            import optuna

            if not search_algorithm:
                search_algorithm = "tpe"

            possible_search_algorithms = ["random", "tpe"]
            if search_algorithm not in possible_search_algorithms:
                raise ValueError(
                    f"For 'optuna' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
                )
        else:
            if not search_algorithm:
                search_algorithm = "random"

            possible_search_algorithms = ["random", "grid"]
            if search_algorithm not in possible_search_algorithms:
                raise ValueError(
                    f"For 'scikit-learn' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
                )

        if custom_scorer is not None:
            optimize = custom_scorer
            warnings.warn(
                "custom_scorer parameter will be depreciated, use optimize instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(optimize, str):
            # checking optimize parameter
            optimize = self._get_metric_by_name_or_id(optimize)
            if optimize is None:
                raise ValueError(
                    "Optimize method not supported. See docstring for list of available parameters."
                )

            # checking optimize parameter for multiclass
            if self.is_multiclass:
                if not optimize.is_multiclass:
                    raise TypeError(
                        "Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                    )
        else:
            self.logger.info(f"optimize set to user defined function {optimize}")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "verbose parameter can only take argument as True or False."
            )

        # checking verbose parameter
        if type(return_tuner) is not bool:
            raise TypeError(
                "return_tuner parameter can only take argument as True or False."
            )

        if not verbose:
            tuner_verbose = 0

        if type(tuner_verbose) not in (bool, int):
            raise TypeError("tuner_verbose parameter must be a bool or an int.")

        tuner_verbose = int(tuner_verbose)

        if tuner_verbose < 0:
            tuner_verbose = 0
        elif tuner_verbose > 2:
            tuner_verbose = 2

        """

        ERROR HANDLING ENDS HERE

        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        progress_args = {"max": 3 + 4}
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            [
                "Status",
                ". . . . . . . . . . . . . . . . . .",
                "Loading Dependencies",
            ],
            [
                "Estimator",
                ". . . . . . . . . . . . . . . . . .",
                "Compiling Library",
            ],
        ]
        display = CommonDisplay(
            verbose=verbose,
            html_param=self.html_param,
            progress_args=progress_args,
            monitor_rows=monitor_rows,
        )

        import logging

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")
        # Storing X_train and y_train in data_X and data_y parameter
        data_X = self.X_train
        data_y = self.y_train

        display.move_progress()

        # setting optimize parameter

        compare_dimension = optimize.display_name
        optimize = optimize.scorer

        # convert trained estimator into string name for grids

        self.logger.info("Checking base model")

        is_stacked_model = False

        if hasattr(estimator, "final_estimator"):
            self.logger.info("Model is stacked, using the definition of the meta-model")
            is_stacked_model = True
            estimator_id = self._get_model_id(estimator.final_estimator)
        else:
            estimator_id = self._get_model_id(estimator)
        if estimator_id is None:
            if custom_grid is None:
                raise ValueError(
                    "When passing a model not in PyCaret's model library, the custom_grid parameter must be provided."
                )
            estimator_name = self._get_model_name(estimator)
            estimator_definition = None
            self.logger.info("A custom model has been passed")
        else:
            estimator_definition = self._all_models_internal[estimator_id]
            estimator_name = estimator_definition.name
        self.logger.info(f"Base model : {estimator_name}")

        if estimator_definition is None or estimator_definition.tunable is None:
            model = clone(estimator)
        else:
            self.logger.info("Model has a special tunable class, using that")
            model = clone(estimator_definition.tunable(**estimator.get_params()))

        base_estimator = model

        display.update_monitor(2, estimator_name)

        display.move_progress()

        self.logger.info("Declaring metric variables")

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Searching Hyperparameters")

        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Defining Hyperparameters")

        from pycaret.internal.tunable import VotingClassifier, VotingRegressor

        def total_combinations_in_grid(grid):
            nc = 1

            def get_iter(x):
                if isinstance(x, dict):
                    return x.values()
                return x

            for v in get_iter(grid):
                if isinstance(v, dict):
                    for v2 in get_iter(v):
                        nc *= len(v2)
                else:
                    nc *= len(v)
            return nc

        if custom_grid is not None:
            if not isinstance(custom_grid, dict):
                raise TypeError(f"custom_grid must be a dict, got {type(custom_grid)}.")
            param_grid = custom_grid
            if not (
                search_library == "scikit-learn"
                or (
                    search_library == "tune-sklearn"
                    and (search_algorithm == "grid" or search_algorithm == "random")
                )
            ):
                param_grid = {
                    k: CategoricalDistribution(v) if isinstance(v, Iterable) else v
                    for k, v in param_grid.items()
                }
            elif any(isinstance(v, Distribution) for k, v in param_grid.items()):
                raise TypeError(
                    f"For the combination of search_library {search_library} and search_algorithm {search_algorithm}, PyCaret Distribution objects are not supported. Pass a list or other object supported by the search library (in most cases, an object with a 'rvs' function)."
                )
        elif search_library == "scikit-learn" or (
            search_library == "tune-sklearn"
            and (search_algorithm == "grid" or search_algorithm == "random")
        ):
            param_grid = estimator_definition.tune_grid
            if isinstance(base_estimator, (VotingClassifier, VotingRegressor)):
                # special case to handle VotingClassifier, as weights need to be
                # generated dynamically
                param_grid = {
                    f"weight_{i}": np.arange(0.01, 1, 0.01)
                    for i, e in enumerate(base_estimator.estimators)
                }

            if search_algorithm != "grid":
                tc = total_combinations_in_grid(param_grid)
                if tc <= n_iter:
                    self.logger.info(
                        f"{n_iter} is bigger than total combinations {tc}, setting search algorithm to grid"
                    )
                    search_algorithm = "grid"
        else:
            param_grid = estimator_definition.tune_distribution

            if isinstance(base_estimator, (VotingClassifier, VotingRegressor)):
                # special case to handle VotingClassifier, as weights need to be
                # generated dynamically
                param_grid = {
                    f"weight_{i}": UniformDistribution(0.000000001, 1)
                    for i, e in enumerate(base_estimator.estimators)
                }

        if not param_grid:
            raise ValueError(
                "parameter grid for tuning is empty. If passing custom_grid, make sure that it is not empty. If not passing custom_grid, the passed estimator does not have a built-in tuning grid."
            )

        suffixes = []

        if is_stacked_model:
            self.logger.info(
                "Stacked model passed, will tune meta model hyperparameters"
            )
            suffixes.append("final_estimator")

        gc.collect()

        with estimator_pipeline(self.pipeline, model) as pipeline_with_model:
            extra_params = {}

            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)

            actual_estimator_label = get_pipeline_estimator_label(pipeline_with_model)

            suffixes.append(actual_estimator_label)

            suffixes = "__".join(reversed(suffixes))

            param_grid = {f"{suffixes}__{k}": v for k, v in param_grid.items()}

            if estimator_definition is not None:
                search_kwargs = {**estimator_definition.tune_args, **kwargs}
                n_jobs = (
                    self.gpu_n_jobs_param
                    if estimator_definition.is_gpu_enabled
                    else self.n_jobs_param
                )
            else:
                search_kwargs = {}
                n_jobs = self.n_jobs_param

            if custom_grid is not None:
                self.logger.info(f"custom_grid: {param_grid}")

            from sklearn.gaussian_process import GaussianProcessClassifier

            # special case to prevent running out of memory
            if isinstance(pipeline_with_model.steps[-1][1], GaussianProcessClassifier):
                n_jobs = 1

            self.logger.info(f"Tuning with n_jobs={n_jobs}")

            def get_optuna_tpe_sampler():
                try:
                    tpe_sampler = optuna.samplers.TPESampler(
                        seed=self.seed, multivariate=True, constant_liar=True
                    )
                except TypeError:
                    # constant_liar added in 2.8.0
                    tpe_sampler = optuna.samplers.TPESampler(
                        seed=self.seed, multivariate=True
                    )
                return tpe_sampler

            if search_library == "optuna":
                # suppress output
                logging.getLogger("optuna").setLevel(logging.WARNING)

                pruner_translator = {
                    "asha": optuna.pruners.SuccessiveHalvingPruner(),  # type: ignore
                    "hyperband": optuna.pruners.HyperbandPruner(),  # type: ignore
                    "median": optuna.pruners.MedianPruner(),  # type: ignore
                    False: optuna.pruners.NopPruner(),  # type: ignore
                    None: optuna.pruners.NopPruner(),  # type: ignore
                }
                pruner = early_stopping
                if pruner in pruner_translator:
                    pruner = pruner_translator[early_stopping]

                sampler_translator = {
                    "tpe": get_optuna_tpe_sampler(),  # type: ignore
                    "random": optuna.samplers.RandomSampler(seed=self.seed),  # type: ignore
                }
                sampler = sampler_translator[search_algorithm]

                try:
                    param_grid = get_optuna_distributions(param_grid)
                except Exception:
                    self.logger.warning(
                        "Couldn't convert param_grid to specific library distributions. Exception:"
                    )
                    self.logger.warning(traceback.format_exc())

                study = optuna.create_study(  # type: ignore
                    direction="maximize", sampler=sampler, pruner=pruner
                )

                self.logger.info("Initializing optuna.integration.OptunaSearchCV")
                model_grid = optuna.integration.OptunaSearchCV(  # type: ignore
                    estimator=pipeline_with_model,
                    param_distributions=param_grid,
                    cv=fold,
                    enable_pruning=early_stopping
                    and can_early_stop(
                        pipeline_with_model, True, False, False, param_grid
                    ),
                    max_iter=early_stopping_max_iters,
                    n_jobs=n_jobs,
                    n_trials=n_iter,
                    random_state=self.seed,
                    scoring=optimize,
                    study=study,
                    refit=False,
                    verbose=tuner_verbose,
                    error_score="raise",
                    **search_kwargs,
                )

            elif search_library == "tune-sklearn":

                early_stopping_translator = {
                    "asha": "ASHAScheduler",
                    "hyperband": "HyperBandScheduler",
                    "median": "MedianStoppingRule",
                }
                if early_stopping in early_stopping_translator:
                    early_stopping = early_stopping_translator[early_stopping]

                do_early_stop = early_stopping and can_early_stop(
                    pipeline_with_model, True, True, True, param_grid
                )

                if not do_early_stop and search_algorithm == "bohb":
                    raise ValueError(
                        "'bohb' requires early_stopping = True and the estimator to support early stopping (has partial_fit, warm_start or is an XGBoost model)."
                    )

                elif early_stopping and can_early_stop(
                    pipeline_with_model, False, True, False, param_grid
                ):
                    if "actual_estimator__n_estimators" in param_grid:
                        if custom_grid is None:
                            extra_params[
                                "actual_estimator__n_estimators"
                            ] = pipeline_with_model.get_params()[
                                "actual_estimator__n_estimators"
                            ]
                            param_grid.pop("actual_estimator__n_estimators")
                        else:
                            raise ValueError(
                                "parameter grid cannot contain n_estimators or max_iter if early_stopping is True and the model is warm started. Use early_stopping_max_iters params to set the upper bound of n_estimators or max_iter."
                            )
                    if "actual_estimator__max_iter" in param_grid:
                        if custom_grid is None:
                            param_grid.pop("actual_estimator__max_iter")
                        else:
                            raise ValueError(
                                "parameter grid cannot contain n_estimators or max_iter if early_stopping is True and the model is warm started. Use early_stopping_max_iters params to set the upper bound of n_estimators or max_iter."
                            )

                from tune_sklearn import TuneGridSearchCV, TuneSearchCV

                with true_warm_start(
                    pipeline_with_model
                ) if do_early_stop else nullcontext():
                    if search_algorithm == "grid":

                        self.logger.info("Initializing tune_sklearn.TuneGridSearchCV")
                        model_grid = TuneGridSearchCV(
                            estimator=pipeline_with_model,
                            param_grid=param_grid,
                            early_stopping=do_early_stop,
                            scoring=optimize,
                            cv=fold,
                            max_iters=early_stopping_max_iters,
                            n_jobs=n_jobs,
                            use_gpu=self.gpu_param,
                            refit=False,
                            verbose=tuner_verbose,
                            pipeline_auto_early_stop=True,
                            **search_kwargs,
                        )
                    else:
                        if search_algorithm == "hyperopt":
                            try:
                                param_grid = get_hyperopt_distributions(param_grid)
                            except Exception:
                                self.logger.warning(
                                    "Couldn't convert param_grid to specific library distributions. Exception:"
                                )
                                self.logger.warning(traceback.format_exc())
                        elif search_algorithm == "bayesian":
                            try:
                                param_grid = get_skopt_distributions(param_grid)
                            except Exception:
                                self.logger.warning(
                                    "Couldn't convert param_grid to specific library distributions. Exception:"
                                )
                                self.logger.warning(traceback.format_exc())
                        elif search_algorithm == "bohb":
                            try:
                                param_grid = get_CS_distributions(param_grid)
                            except Exception:
                                self.logger.warning(
                                    "Couldn't convert param_grid to specific library distributions. Exception:"
                                )
                                self.logger.warning(traceback.format_exc())
                        elif search_algorithm != "random":
                            try:
                                param_grid = get_tune_distributions(param_grid)
                            except Exception:
                                self.logger.warning(
                                    "Couldn't convert param_grid to specific library distributions. Exception:"
                                )
                                self.logger.warning(traceback.format_exc())
                        self.logger.info(
                            f"Initializing tune_sklearn.TuneSearchCV, {search_algorithm}"
                        )
                        if (
                            search_algorithm == "optuna"
                            and "sampler" not in search_kwargs
                        ):
                            import optuna

                            search_kwargs["sampler"] = get_optuna_tpe_sampler()
                        model_grid = TuneSearchCV(
                            estimator=pipeline_with_model,
                            search_optimization=search_algorithm,
                            param_distributions=param_grid,
                            n_trials=n_iter,
                            early_stopping=do_early_stop,
                            scoring=optimize,
                            cv=fold,
                            random_state=self.seed,
                            max_iters=early_stopping_max_iters,
                            n_jobs=n_jobs,
                            use_gpu=self.gpu_param,
                            refit=True,
                            verbose=tuner_verbose,
                            pipeline_auto_early_stop=True,
                            search_kwargs=search_kwargs,
                        )

            elif search_library == "scikit-optimize":
                try:
                    param_grid = get_skopt_distributions(param_grid)
                except Exception:
                    self.logger.warning(
                        "Couldn't convert param_grid to specific library distributions. Exception:"
                    )
                    self.logger.warning(traceback.format_exc())

                self.logger.info("Initializing skopt.BayesSearchCV")
                model_grid = skopt.BayesSearchCV(
                    estimator=pipeline_with_model,
                    search_spaces=param_grid,
                    scoring=optimize,
                    n_iter=n_iter,
                    cv=fold,
                    random_state=self.seed,
                    refit=False,
                    n_jobs=n_jobs,
                    verbose=tuner_verbose,
                    **search_kwargs,
                )
            else:
                # needs to be imported like that for the monkeypatch
                import sklearn.model_selection._search

                try:
                    param_grid = get_base_distributions(param_grid)
                except Exception:
                    self.logger.warning(
                        "Couldn't convert param_grid to specific library distributions. Exception:"
                    )
                    self.logger.warning(traceback.format_exc())

                if search_algorithm == "grid":
                    self.logger.info("Initializing GridSearchCV")
                    model_grid = sklearn.model_selection._search.GridSearchCV(
                        estimator=pipeline_with_model,
                        param_grid=param_grid,
                        scoring=optimize,
                        cv=fold,
                        refit=False,
                        n_jobs=n_jobs,
                        verbose=tuner_verbose,
                        **search_kwargs,
                    )
                else:
                    self.logger.info("Initializing RandomizedSearchCV")
                    model_grid = sklearn.model_selection._search.RandomizedSearchCV(
                        estimator=pipeline_with_model,
                        param_distributions=param_grid,
                        scoring=optimize,
                        n_iter=n_iter,
                        cv=fold,
                        random_state=self.seed,
                        refit=False,
                        n_jobs=n_jobs,
                        verbose=tuner_verbose,
                        **search_kwargs,
                    )

            if search_library == "scikit-learn":
                # monkey patching to fix overflows on Windows
                with patch(
                    "sklearn.model_selection._search.sample_without_replacement",
                    pycaret.internal.patches.sklearn._mp_sample_without_replacement,
                ):
                    with patch(
                        "sklearn.model_selection._search.ParameterGrid.__getitem__",
                        pycaret.internal.patches.sklearn._mp_ParameterGrid_getitem,
                    ):
                        model_grid.fit(data_X, data_y, groups=groups, **fit_kwargs)
            else:
                model_grid.fit(data_X, data_y, groups=groups, **fit_kwargs)
            best_params = model_grid.best_params_
            self.logger.info(f"best_params: {best_params}")
            best_params = {**best_params, **extra_params}
            if actual_estimator_label:
                best_params = {
                    k.replace(f"{actual_estimator_label}__", ""): v
                    for k, v in best_params.items()
                }
            cv_results = None
            try:
                cv_results = model_grid.cv_results_
            except Exception:
                self.logger.warning(
                    "Couldn't get cv_results from model_grid. Exception:"
                )
                self.logger.warning(traceback.format_exc())

        display.move_progress()

        self.logger.info("Hyperparameter search completed")

        if isinstance(model, TunableMixin):
            self.logger.info("Getting base sklearn object from tunable")
            best_params = {
                k: v
                for k, v in model.get_params().items()
                if k in model.get_base_sklearn_params().keys()
            }
            model = model.get_base_sklearn_object()

        self.logger.info(
            "SubProcess create_model() called =================================="
        )
        best_model, model_fit_time = self._create_model(
            estimator=model,
            system=False,
            display=display,
            fold=fold,
            round=round,
            groups=groups,
            fit_kwargs=fit_kwargs,
            return_train_score=return_train_score,
            **best_params,
        )
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        if choose_better:
            new_best_model = self._choose_better(
                [estimator, (best_model, model_results)],
                compare_dimension,
                fold,
                groups=groups,
                fit_kwargs=fit_kwargs,
                display=display,
            )
            if new_best_model is not best_model:
                msg = (
                    "Original model was better than the tuned model, hence it will be returned. "
                    "NOTE: The display metrics are for the tuned model (not the original one)."
                )
                if verbose:
                    print(msg)
                self.logger.info(msg)
            best_model = new_best_model

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # dashboard logging
        if self.logging_param:
            indices = self._get_return_train_score_indices_for_logging(
                return_train_score=return_train_score
            )
            avgs_dict_log = {k: v for k, v in model_results.loc[indices].items()}
            self.logging_param.log_model_comparison(model_results, "tune_model")

            self._log_model(
                model=best_model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="tune_model",
                runtime=runtime,
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                tune_cv_results=cv_results,
                display=display,
            )

        model_results = self._highlight_and_round_model_results(
            model_results, return_train_score, round
        )
        display.display(model_results)

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(best_model))
        self.logger.info(
            "tune_model() successfully completed......................................"
        )

        gc.collect()
        if return_tuner:
            return (best_model, model_grid)
        return best_model

    def ensemble_model(
        self,
        estimator,
        method: str = "Bagging",
        fold: Optional[Union[int, Any]] = None,
        n_estimators: int = 10,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
        return_train_score: bool = False,
    ) -> Any:
        """
        This function ensembles the trained base estimator using the method defined in
        'method' parameter (default = 'Bagging'). The output prints a score grid that shows
        Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by fold (default = 10 Fold).

        This function returns a trained model object.

        Model must be created using create_model() or tune_model().

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> dt = create_model('dt')
        >>> ensembled_dt = ensemble_model(dt)

        This will return an ensembled Decision Tree model using 'Bagging'.

        Parameters
        ----------
        estimator : object, default = None

        method: str, default = 'Bagging'
            Bagging method will create an ensemble meta-estimator that fits base
            classifiers each on random subsets of the original dataset. The other
            available method is 'Boosting' which will create a meta-estimators by
            fitting a classifier on the original dataset and then fits additional
            copies of the classifier on the same dataset but where the weights of
            incorrectly classified instances are adjusted such that subsequent
            classifiers focus more on difficult cases.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        n_estimators: integer, default = 10
            The number of base estimators in the ensemble.
            In case of perfect fit, the learning procedure is stopped early.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        choose_better: bool, default = False
            When set to set to True, base estimator is returned when the metric doesn't
            improve by ensemble_model. This guarantees the returned object would perform
            at least equivalent to base estimator created using create_model or model
            returned by compare_models.

        optimize: str, default = 'Accuracy'
            Only used when choose_better is set to True. optimize parameter is used
            to compare ensembled model with base estimator. Values accepted in
            optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1',
            'Kappa', 'MCC'.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.

        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds.
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1,
            Kappa and MCC. Mean and standard deviation of the scores across
            the folds are also returned.

        model
            Trained ensembled model object.

        Warnings
        --------
        - If target variable is multiclass (more than 2 classes), AUC will be returned
        as zero (0.0).


        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing ensemble_model()")
        self.logger.info(f"ensemble_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # Check for estimator
        if not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

        # Check for allowed method
        available_method = ["Bagging", "Boosting"]
        if method not in available_method:
            raise ValueError(
                "Method parameter only accepts two values 'Bagging' or 'Boosting'."
            )

        # check boosting conflict
        if method == "Boosting":

            boosting_model_definition = self._all_models_internal["ada"]

            check_model = estimator

            try:
                check_model = boosting_model_definition.class_def(
                    check_model,
                    n_estimators=n_estimators,
                    **boosting_model_definition.args,
                )
                with redirect_output(self.logger):
                    check_model.fit(self.X_train_transformed, self.y_train_transformed)
            except Exception:
                raise TypeError(
                    "Estimator not supported for the Boosting method. Change the estimator or method to 'Bagging'."
                )

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking n_estimators parameter
        if type(n_estimators) is not int:
            raise TypeError("n_estimators parameter only accepts integer value.")

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        # checking optimize parameter
        optimize = self._get_metric_by_name_or_id(optimize)
        if optimize is None:
            raise ValueError(
                "Optimize method not supported. See docstring for list of available parameters."
            )

        # checking optimize parameter for multiclass
        if self.is_multiclass:
            if not optimize.is_multiclass:
                raise TypeError(
                    "Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                )

        # checking return_train_score parameter
        if type(return_train_score) is not bool:
            raise TypeError(
                "return_train_score can only take argument as True or False"
            )

        """

        ERROR HANDLING ENDS HERE

        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        progress_args = {"max": 2 + 4}
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            [
                "Status",
                ". . . . . . . . . . . . . . . . . .",
                "Loading Dependencies",
            ],
            [
                "Estimator",
                ". . . . . . . . . . . . . . . . . .",
                "Compiling Library",
            ],
        ]
        display = CommonDisplay(
            verbose=verbose,
            html_param=self.html_param,
            progress_args=progress_args,
            monitor_rows=monitor_rows,
        )

        self.logger.info("Importing libraries")

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")

        display.move_progress()

        # setting optimize parameter

        compare_dimension = optimize.display_name
        optimize = optimize.scorer

        self.logger.info("Checking base model")

        _estimator_ = estimator

        estimator_id = self._get_model_id(estimator)

        if estimator_id is None:
            estimator_name = self._get_model_name(estimator)
            self.logger.info("A custom model has been passed")
        else:
            estimator_definition = self._all_models_internal[estimator_id]
            estimator_name = estimator_definition.name

        self.logger.info(f"Base model : {estimator_name}")

        display.update_monitor(2, estimator_name)

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Selecting Estimator")

        """
        MONITOR UPDATE ENDS
        """

        model = get_estimator_from_meta_estimator(_estimator_)

        self.logger.info("Importing untrained ensembler")

        if method == "Bagging":
            self.logger.info("Ensemble method set to Bagging")
            bagging_model_definition = self._all_models_internal["Bagging"]

            model = bagging_model_definition.class_def(
                model,
                bootstrap=True,
                n_estimators=n_estimators,
                **bagging_model_definition.args,
            )

        else:
            self.logger.info("Ensemble method set to Boosting")
            boosting_model_definition = self._all_models_internal["ada"]
            model = boosting_model_definition.class_def(
                model, n_estimators=n_estimators, **boosting_model_definition.args
            )

        display.move_progress()

        self.logger.info(
            "SubProcess create_model() called =================================="
        )
        model, model_fit_time = self._create_model(
            estimator=model,
            system=False,
            display=display,
            fold=fold,
            round=round,
            fit_kwargs=fit_kwargs,
            groups=groups,
            probability_threshold=probability_threshold,
            return_train_score=return_train_score,
        )
        best_model = model
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # dashboard logging
        if self.logging_param:
            indices = self._get_return_train_score_indices_for_logging(
                return_train_score=return_train_score
            )
            avgs_dict_log = {k: v for k, v in model_results.loc[indices].items()}
            self.logging_param.log_model_comparison(model_results, "ensemble_model")

            self._log_model(
                model=best_model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="ensemble_model",
                runtime=runtime,
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                display=display,
            )

        if choose_better:
            new_model = self._choose_better(
                [_estimator_, (best_model, model_results)],
                compare_dimension,
                fold,
                groups=groups,
                fit_kwargs=fit_kwargs,
                display=display,
            )
            if new_model is not best_model:
                msg = (
                    "Original model was better than the ensembled model, hence it will be returned. "
                    "NOTE: The display metrics are for the ensembled model (not the original one)."
                )
                if verbose:
                    print(msg)
                self.logger.info(msg)
            model = new_model

        model_results = self._highlight_and_round_model_results(
            model_results, return_train_score, round
        )
        display.display(model_results)

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "ensemble_model() successfully completed......................................"
        )

        gc.collect()
        return model

    def blend_models(
        self,
        estimator_list: list,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        method: str = "auto",
        weights: Optional[List[float]] = None,  # added in pycaret==2.2.0
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
        return_train_score: bool = False,
    ) -> Any:

        """
        This function creates a Soft Voting / Majority Rule classifier for all the
        estimators in the model library (excluding the few when turbo is True) or
        for specific trained estimators passed as a list in estimator_list param.
        It scores it using Cross Validation. The output prints a score
        grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by
        fold (default CV = 10 Folds).

        This function returns a trained model object.

        Example
        -------
        >>> lr = create_model('lr')
        >>> rf = create_model('rf')
        >>> knn = create_model('knn')
        >>> blend_three = blend_models(estimator_list = [lr,rf,knn])

        This will create a VotingClassifier of lr, rf and knn.

        Parameters
        ----------
        estimator_list : list of objects

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        choose_better: bool, default = False
            When set to set to True, base estimator is returned when the metric doesn't
            improve by ensemble_model. This guarantees the returned object would perform
            at least equivalent to base estimator created using create_model or model
            returned by compare_models.

        optimize: str, default = 'Accuracy'
            Only used when choose_better is set to True. optimize parameter is used
            to compare ensembled model with base estimator. Values accepted in
            optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1',
            'Kappa', 'MCC'.

        method: str, default = 'auto'
            'hard' uses predicted class labels for majority rule voting. 'soft', predicts
            the class label based on the argmax of the sums of the predicted probabilities,
            which is recommended for an ensemble of well-calibrated classifiers. Default value,
            'auto', will try to use 'soft' and fall back to 'hard' if the former is not supported.

        weights: list, default = None
            Sequence of weights (float or int) to weight the occurrences of predicted class labels (hard voting)
            or class probabilities before averaging (soft voting). Uses uniform weights if None.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.

        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds.
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1,
            Kappa and MCC. Mean and standard deviation of the scores across
            the folds are also returned.

        model
            Trained Voting Classifier model object.

        Warnings
        --------
        - When passing estimator_list with method set to 'soft'. All the models in the
        estimator_list must support predict_proba function. 'svm' and 'ridge' does not
        support the predict_proba and hence an exception will be raised.

        - When estimator_list is set to 'All' and method is forced to 'soft', estimators
        that does not support the predict_proba function will be dropped from the estimator
        list.

        - If target variable is multiclass (more than 2 classes), AUC will be returned as
        zero (0.0).



        """
        self._check_setup_ran()

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing blend_models()")
        self.logger.info(f"blend_models({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # checking method parameter
        available_method = ["auto", "soft", "hard", "mean", "median", "voting"]
        if method not in available_method:
            raise ValueError(
                "Method parameter only accepts 'auto', 'soft', 'hard', 'mean', 'median' or 'voting' as a parameter. See Docstring for details."
            )

        # checking error for estimator_list (skip for timeseries)
        if not self._ml_usecase == MLUsecase.TIME_SERIES:
            for i in estimator_list:
                if not hasattr(i, "fit"):
                    raise ValueError(
                        f"Estimator {i} does not have the required fit() method."
                    )
                if self._ml_usecase == MLUsecase.CLASSIFICATION:
                    # checking method parameter with estimator list
                    if method != "hard":

                        for i in estimator_list:
                            if not hasattr(i, "predict_proba"):
                                if method != "auto":
                                    raise TypeError(
                                        f"Estimator list contains estimator {i} that doesn't support probabilities and method is forced to 'soft'. Either change the method or drop the estimator."
                                    )
                                else:
                                    self.logger.info(
                                        f"Estimator {i} doesn't support probabilities, falling back to 'hard'."
                                    )
                                    method = "hard"
                                    break

                        if method == "auto":
                            method = "soft"

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        if weights is not None:
            num_estimators = len(estimator_list)
            # checking weights parameter
            if len(weights) != num_estimators:
                raise ValueError(
                    "weights parameter must have the same length as the estimator_list."
                )
            if not all((isinstance(x, int) or isinstance(x, float)) for x in weights):
                raise TypeError("weights must contain only ints or floats.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        # checking optimize parameter
        optimize = self._get_metric_by_name_or_id(optimize)
        if optimize is None:
            raise ValueError(
                "Optimize method not supported. See docstring for list of available parameters."
            )

        # checking optimize parameter for multiclass
        if self.is_multiclass:
            if not optimize.is_multiclass:
                raise TypeError(
                    "Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                )

        # checking return_train_score parameter
        if type(return_train_score) is not bool:
            raise TypeError(
                "return_train_score can only take argument as True or False"
            )

        """

        ERROR HANDLING ENDS HERE

        """
        if self._ml_usecase == MLUsecase.TIME_SERIES:
            # Just return the fold to create_model. It will do the rest
            # fold = fold
            pass
        else:
            fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        progress_args = {"max": 2 + 4}
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            [
                "Status",
                ". . . . . . . . . . . . . . . . . .",
                "Loading Dependencies",
            ],
            [
                "Estimator",
                ". . . . . . . . . . . . . . . . . .",
                "Compiling Library",
            ],
        ]
        display = CommonDisplay(
            verbose=verbose,
            html_param=self.html_param,
            progress_args=progress_args,
            monitor_rows=monitor_rows,
        )

        self.logger.info("Importing libraries")

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")

        # setting optimize parameter
        compare_dimension = optimize.display_name
        optimize = optimize.scorer

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Compiling Estimators")

        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Getting model names")
        estimator_dict = {}
        for x in estimator_list:
            x = get_estimator_from_meta_estimator(x)
            name = self._get_model_name(x)
            suffix = 1
            original_name = name
            while name in estimator_dict:
                name = f"{original_name}_{suffix}"
                suffix += 1
            estimator_dict[name] = x

        estimator_list = list(estimator_dict.items())

        if self._ml_usecase == MLUsecase.TIME_SERIES:
            voting_model_definition = self._all_models_internal["ensemble_forecaster"]
        else:
            voting_model_definition = self._all_models_internal["Voting"]

        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            model = voting_model_definition.class_def(
                estimators=estimator_list, voting=method, n_jobs=self.gpu_n_jobs_param
            )
        elif self._ml_usecase == MLUsecase.TIME_SERIES:
            model = voting_model_definition.class_def(
                forecasters=estimator_list,
                method=method,
                weights=weights,
                n_jobs=self.gpu_n_jobs_param,
            )
        else:
            model = voting_model_definition.class_def(
                estimators=estimator_list, n_jobs=self.gpu_n_jobs_param
            )

        display.update_monitor(2, voting_model_definition.name)

        display.move_progress()

        self.logger.info(
            "SubProcess create_model() called =================================="
        )
        model, model_fit_time = self._create_model(
            estimator=model,
            system=False,
            display=display,
            fold=fold,
            round=round,
            fit_kwargs=fit_kwargs,
            groups=groups,
            probability_threshold=probability_threshold,
            return_train_score=return_train_score,
        )

        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # dashboard logging
        if self.logging_param:
            indices = self._get_return_train_score_indices_for_logging(
                return_train_score=return_train_score
            )
            avgs_dict_log = {k: v for k, v in model_results.loc[indices].items()}
            self.logging_param.log_model_comparison(model_results, "blend_models")

            self._log_model(
                model=model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="blend_models",
                runtime=runtime,
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                display=display,
            )

        if choose_better:
            new_model = self._choose_better(
                [(model, model_results)] + estimator_list,
                compare_dimension,
                fold,
                groups=groups,
                fit_kwargs=fit_kwargs,
                display=display,
            )
            if new_model is not model:
                msg = (
                    "Original model was better than the blended model, hence it will be returned. "
                    "NOTE: The display metrics are for the blended model (not the original one)."
                )
                if verbose:
                    print(msg)
                self.logger.info(msg)
            model = new_model

        model_results = self._highlight_and_round_model_results(
            model_results, return_train_score, round
        )
        display.display(model_results)

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "blend_models() successfully completed......................................"
        )

        gc.collect()
        return model

    def stack_models(
        self,
        estimator_list: list,
        meta_model=None,
        meta_model_fold: Optional[Union[int, Any]] = 5,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        method: str = "auto",
        restack: bool = False,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
        return_train_score: bool = False,
    ) -> Any:

        """
        This function trains a meta model and scores it using Cross Validation.
        The predictions from the base level models as passed in the estimator_list parameter
        are used as input features for the meta model. The restacking parameter controls
        the ability to expose raw features to the meta model when set to True
        (default = False).

        The output prints the score grid that shows Accuracy, AUC, Recall, Precision,
        F1, Kappa and MCC by fold (default = 10 Folds).

        This function returns a trained model object.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> dt = create_model('dt')
        >>> rf = create_model('rf')
        >>> ada = create_model('ada')
        >>> ridge = create_model('ridge')
        >>> knn = create_model('knn')
        >>> stacked_models = stack_models(estimator_list=[dt,rf,ada,ridge,knn])

        This will create a meta model that will use the predictions of all the
        models provided in estimator_list param. By default, the meta model is
        Logistic Regression but can be changed with meta_model param.

        Parameters
        ----------
        estimator_list : list of objects

        meta_model : object, default = None
            If set to None, Logistic Regression is used as a meta model.

        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        method: string, default = 'auto'
            - if ‘auto’, it will try to invoke, for each estimator, 'predict_proba',
            'decision_function' or 'predict' in that order.
            - otherwise, one of 'predict_proba', 'decision_function' or 'predict'.
            If the method is not implemented by the estimator, it will raise an error.

        restack: bool, default = False
            When restack is set to True, raw data will be exposed to meta model when
            making predictions, otherwise when False, only the predicted label or
            probabilities is passed to meta model when making final predictions.

        choose_better: bool, default = False
            When set to set to True, base estimator is returned when the metric doesn't
            improve by ensemble_model. This guarantees the returned object would perform
            at least equivalent to base estimator created using create_model or model
            returned by compare_models.

        optimize: str, default = 'Accuracy'
            Only used when choose_better is set to True. optimize parameter is used
            to compare ensembled model with base estimator. Values accepted in
            optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1',
            'Kappa', 'MCC'.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.

        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds.
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1,
            Kappa and MCC. Mean and standard deviation of the scores across
            the folds are also returned.

        model
            Trained model object.

        Warnings
        --------
        -  If target variable is multiclass (more than 2 classes), AUC will be returned
        as zero (0.0).

        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing stack_models()")
        self.logger.info(f"stack_models({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # checking error for estimator_list
        for i in estimator_list:
            if not hasattr(i, "fit"):
                raise ValueError(
                    f"Estimator {i} does not have the required fit() method."
                )

        # checking meta model
        if meta_model is not None:
            if not hasattr(meta_model, "fit"):
                raise ValueError(
                    f"Meta Model {meta_model} does not have the required fit() method."
                )

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking method parameter
        available_method = ["auto", "predict_proba", "decision_function", "predict"]
        if method not in available_method:
            raise ValueError(
                "Method parameter not acceptable. It only accepts 'auto', 'predict_proba', 'decision_function', 'predict'."
            )

        # checking restack parameter
        if type(restack) is not bool:
            raise TypeError(
                "Restack parameter can only take argument as True or False."
            )

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        # checking optimize parameter
        optimize = self._get_metric_by_name_or_id(optimize)
        if optimize is None:
            raise ValueError(
                "Optimize method not supported. See docstring for list of available parameters."
            )

        # checking optimize parameter for multiclass
        if self.is_multiclass:
            if not optimize.is_multiclass:
                raise TypeError(
                    "Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                )

        # checking return_train_score parameter
        if type(return_train_score) is not bool:
            raise TypeError(
                "return_train_score can only take argument as True or False"
            )

        """

        ERROR HANDLING ENDS HERE

        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        self.logger.info("Defining meta model")
        if meta_model is None:
            estimator = "lr"
            meta_model_definition = self._all_models_internal[estimator]
            meta_model_args = meta_model_definition.args
            meta_model = meta_model_definition.class_def(**meta_model_args)
        else:
            meta_model = clone(get_estimator_from_meta_estimator(meta_model))

        progress_args = {"max": 2 + 4}
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            [
                "Status",
                ". . . . . . . . . . . . . . . . . .",
                "Loading Dependencies",
            ],
            [
                "Estimator",
                ". . . . . . . . . . . . . . . . . .",
                "Compiling Library",
            ],
        ]
        display = CommonDisplay(
            verbose=verbose,
            html_param=self.html_param,
            progress_args=progress_args,
            monitor_rows=monitor_rows,
        )

        np.random.seed(self.seed)

        # setting optimize parameter
        compare_dimension = optimize.display_name
        optimize = optimize.scorer

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Compiling Estimators")

        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Getting model names")
        estimator_dict = {}
        for x in estimator_list:
            x = get_estimator_from_meta_estimator(x)
            name = self._get_model_name(x)
            suffix = 1
            original_name = name
            while name in estimator_dict:
                name = f"{original_name}_{suffix}"
                suffix += 1
            estimator_dict[name] = x

        estimator_list = list(estimator_dict.items())

        self.logger.info(estimator_list)

        stacking_model_definition = self._all_models_internal["Stacking"]
        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            model = stacking_model_definition.class_def(
                estimators=estimator_list,
                final_estimator=meta_model,
                cv=meta_model_fold,
                stack_method=method,
                n_jobs=self.gpu_n_jobs_param,
                passthrough=restack,
            )
        else:
            model = stacking_model_definition.class_def(
                estimators=estimator_list,
                final_estimator=meta_model,
                cv=meta_model_fold,
                n_jobs=self.gpu_n_jobs_param,
                passthrough=restack,
            )

        display.update_monitor(2, stacking_model_definition.name)

        display.move_progress()

        self.logger.info(
            "SubProcess create_model() called =================================="
        )
        model, model_fit_time = self._create_model(
            estimator=model,
            system=False,
            display=display,
            fold=fold,
            round=round,
            fit_kwargs=fit_kwargs,
            groups=groups,
            probability_threshold=probability_threshold,
            return_train_score=return_train_score,
        )
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # dashboard logging
        if self.logging_param:
            indices = self._get_return_train_score_indices_for_logging(
                return_train_score=return_train_score
            )
            avgs_dict_log = {k: v for k, v in model_results.loc[indices].items()}
            self.logging_param.log_model_comparison(model_results, "stack_model")

            self._log_model(
                model=model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="stack_models",
                runtime=runtime,
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                display=display,
            )

        if choose_better:
            new_model = self._choose_better(
                [(model, model_results)] + estimator_list,
                compare_dimension,
                fold,
                groups=groups,
                fit_kwargs=fit_kwargs,
                display=display,
            )
            if new_model is not model:
                msg = (
                    "Original model was better than the stacked model, hence it will be returned. "
                    "NOTE: The display metrics are for the stacked model (not the original one)."
                )
                if verbose:
                    print(msg)
                self.logger.info(msg)
            model = new_model

        model_results = self._highlight_and_round_model_results(
            model_results, return_train_score, round
        )
        display.display(model_results)

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "stack_models() successfully completed......................................"
        )

        gc.collect()
        return model

    def interpret_model(
        self,
        estimator,
        plot: str = "summary",
        feature: Optional[str] = None,
        observation: Optional[int] = None,
        use_train_data: bool = False,
        X_new_sample: Optional[pd.DataFrame] = None,
        y_new_sample: Optional[pd.DataFrame] = None,  # add for pfi explainer
        save: Union[str, bool] = False,
        **kwargs,  # added in pycaret==2.1
    ):

        """
        This function takes a trained model object and returns an interpretation plot
        based on the test / hold-out set. It only supports tree based algorithms.

        This function is implemented based on the SHAP (SHapley Additive exPlanations),
        which is a unified approach to explain the output of any machine learning model.
        SHAP connects game theory with local explanations.

        For more information : https://shap.readthedocs.io/en/latest/

        For Partial Dependence Plot : https://github.com/SauceCat/PDPbox

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> dt = create_model('dt')
        >>> interpret_model(dt)

        This will return a summary interpretation plot of Decision Tree model.

        Parameters
        ----------
        estimator : object, default = none
            A trained model object to be passed as an estimator. Only tree-based
            models are accepted when plot type is 'summary', 'correlation', or
            'reason'. 'pdp' plot is model agnostic.

        plot : str, default = 'summary'
            Abbreviation of type of plot. The current list of plots supported
            are (Plot - Name):
            * 'summary' - Summary Plot using SHAP
            * 'correlation' - Dependence Plot using SHAP
            * 'reason' - Force Plot using SHAP
            * 'pdp' - Partial Dependence Plot
            * 'msa' - Morris Sensitivity Analysis
            * 'pfi' - Permutation Feature Importance

        feature: str, default = None
            This parameter is only needed when plot = 'correlation' or 'pdp'.
            By default feature is set to None which means the first column of the
            dataset will be used as a variable. A feature parameter must be passed
            to change this.

        observation: integer, default = None
            This parameter only comes into effect when plot is set to 'reason'. If no
            observation number is provided, it will return an analysis of all observations
            with the option to select the feature on x and y axes through drop down
            interactivity. For analysis at the sample level, an observation parameter must
            be passed with the index value of the observation in test / hold-out set.

        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.

        X_new_sample: pd.DataFrame, default = None
            Row from an out-of-sample dataframe (neither train nor test data) to be plotted.
            The sample must have the same columns as the raw input train data, and it is transformed
            by the preprocessing pipeline automatically before plotting.

        y_new_sample: pd.DataFrame, default = None
            Row from an out-of-sample dataframe (neither train nor test data) to be plotted.
            The sample must have the same columns as the raw input label data, and it is transformed
            by the preprocessing pipeline automatically before plotting.

        save: string or bool, default = False
            When set to True, Plot is saved as a 'png' file in current working directory.
            When a path destination is given, Plot is saved as a 'png' file the given path to the directory of choice.

        **kwargs:
            Additional keyword arguments to pass to the plot.

        Returns
        -------
        Visual_Plot
            Returns the visual plot.
            Returns the interactive JS plot when plot = 'reason'.

        Warnings
        --------
        - interpret_model doesn't support multiclass problems.

        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing interpret_model()")
        self.logger.info(f"interpret_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # checking if shap available
        if plot in ["summary", "correlation", "reason"]:
            _check_soft_dependencies("shap", extra="analysis", severity="error")
            import shap

        # checking if pdpbox is available
        if plot == "pdp":
            _check_soft_dependencies("interpret", extra="analysis", severity="error")

        # checking interpret is available
        if plot == "msa":
            _check_soft_dependencies("interpret", extra="analysis", severity="error")

        # checking interpret-community is available
        if plot == "pfi":
            _check_soft_dependencies(
                "interpret_community",
                extra=None,
                severity="error",
                install_name="interpret-community",
            )

        # get estimator from meta estimator
        estimator = get_estimator_from_meta_estimator(estimator)

        # allowed models
        model_id = self._get_model_id(estimator)

        shap_models = {k: v for k, v in self._all_models_internal.items() if v.shap}
        shap_models_ids = set(shap_models.keys())

        if plot in ["summary", "correlation", "reason"] and (
            model_id not in shap_models_ids
        ):
            raise TypeError(
                f"This function only supports tree based models for binary classification: {', '.join(shap_models_ids)}."
            )

        # plot type
        allowed_types = ["summary", "correlation", "reason", "pdp", "msa", "pfi"]
        if plot not in allowed_types:
            raise ValueError(
                f"type parameter only accepts {', '.join(list(allowed_types) + str(None))}."
            )

        if X_new_sample is not None and (observation is not None or use_train_data):
            raise ValueError(
                "Specifying 'X_new_sample' and ('observation' or 'use_train_data') is ambiguous."
            )

        """
        Error Checking Ends here

        """

        # Storing X_train and y_train in data_X and data_y parameter
        if X_new_sample is not None:
            test_X = self.pipeline.transform(X_new_sample)
            if plot == "pfi":
                test_y = self.pipeline.transform(y_new_sample)  # add for pfi explainer
        else:
            # Storing X_train and y_train in data_X and data_y parameter
            if use_train_data:
                test_X = self.X_train_transformed
            else:
                test_X = self.X_test_transformed
            if plot == "pfi":
                if use_train_data:
                    test_y = self.y_train_transformed
                else:
                    test_y = self.y_test_transformed

        np.random.seed(self.seed)

        # storing estimator in model variable
        model = estimator

        # defining type of classifier
        shap_models_type1 = {k for k, v in shap_models.items() if v.shap == "type1"}
        shap_models_type2 = {k for k, v in shap_models.items() if v.shap == "type2"}

        self.logger.info(f"plot type: {plot}")

        shap_plot = None

        def summary(show: bool = True):

            self.logger.info("Creating TreeExplainer")
            explainer = shap.TreeExplainer(model)
            self.logger.info("Compiling shap values")
            shap_values = explainer.shap_values(test_X)
            try:
                assert len(shap_values) == 2
                shap_plot = shap.summary_plot(
                    shap_values[1], test_X, show=show, **kwargs
                )
            except Exception:
                shap_plot = shap.summary_plot(shap_values, test_X, show=show, **kwargs)
            if save:
                plot_filename = f"SHAP {plot}.png"
                if not isinstance(save, bool):
                    plot_filename = os.path.join(save, plot_filename)
                self.logger.info(f"Saving '{plot_filename}'")
                plt.savefig(plot_filename, bbox_inches="tight")
                plt.close()
            return shap_plot

        def correlation(show: bool = True):

            if feature is None:

                self.logger.warning(
                    f"No feature passed. Default value of feature used for correlation plot: {test_X.columns[0]}"
                )
                dependence = test_X.columns[0]

            else:

                self.logger.warning(
                    f"feature value passed. Feature used for correlation plot: {feature}"
                )
                dependence = feature

            self.logger.info("Creating TreeExplainer")
            explainer = shap.TreeExplainer(model)
            self.logger.info("Compiling shap values")
            shap_values = explainer.shap_values(test_X)

            if model_id in shap_models_type1:
                self.logger.info("model type detected: type 1")
                shap.dependence_plot(
                    dependence, shap_values[1], test_X, show=show, **kwargs
                )
            elif model_id in shap_models_type2:
                self.logger.info("model type detected: type 2")
                shap.dependence_plot(
                    dependence, shap_values, test_X, show=show, **kwargs
                )
            if save:
                plot_filename = f"SHAP {plot}.png"
                if not isinstance(save, bool):
                    plot_filename = os.path.join(save, plot_filename)
                self.logger.info(f"Saving '{plot_filename}'")
                plt.savefig(plot_filename, bbox_inches="tight")
                plt.close()
            return None

        def reason(show: bool = True):
            shap_plot = None
            if model_id in shap_models_type1:
                self.logger.info("model type detected: type 1")

                self.logger.info("Creating TreeExplainer")
                explainer = shap.TreeExplainer(model)
                self.logger.info("Compiling shap values")

                if observation is None:
                    self.logger.warning(
                        "Observation set to None. Model agnostic plot will be rendered."
                    )
                    shap_values = explainer.shap_values(test_X)
                    shap.initjs()
                    shap_plot = shap.force_plot(
                        explainer.expected_value[1], shap_values[1], test_X, **kwargs
                    )

                else:
                    row_to_show = observation
                    data_for_prediction = test_X.iloc[row_to_show]

                    if model_id == "lightgbm":
                        self.logger.info("model type detected: LGBMClassifier")
                        shap_values = explainer.shap_values(test_X)
                        shap.initjs()
                        shap_plot = shap.force_plot(
                            explainer.expected_value[1],
                            shap_values[0][row_to_show],
                            data_for_prediction,
                            show=show,
                            **kwargs,
                        )

                    else:
                        self.logger.info("model type detected: Unknown")

                        shap_values = explainer.shap_values(data_for_prediction)
                        shap.initjs()
                        shap_plot = shap.force_plot(
                            explainer.expected_value[1],
                            shap_values[1],
                            data_for_prediction,
                            show=show,
                            **kwargs,
                        )

            elif model_id in shap_models_type2:
                self.logger.info("model type detected: type 2")

                self.logger.info("Creating TreeExplainer")
                explainer = shap.TreeExplainer(model)
                self.logger.info("Compiling shap values")
                shap_values = explainer.shap_values(test_X)
                shap.initjs()

                if observation is None:
                    self.logger.warning(
                        "Observation set to None. Model agnostic plot will be rendered."
                    )

                    shap_plot = shap.force_plot(
                        explainer.expected_value,
                        shap_values,
                        test_X,
                        show=show,
                        **kwargs,
                    )

                else:

                    row_to_show = observation
                    data_for_prediction = test_X.iloc[row_to_show]

                    shap_plot = shap.force_plot(
                        explainer.expected_value,
                        shap_values[row_to_show, :],
                        test_X.iloc[row_to_show, :],
                        show=show,
                        **kwargs,
                    )
            if save:
                plot_filename = f"SHAP {plot}.html"
                if not isinstance(save, bool):
                    plot_filename = os.path.join(save, plot_filename)
                self.logger.info(f"Saving '{plot_filename}'")
                shap.save_html(plot_filename, shap_plot)
            return shap_plot

        def pdp(show: bool = True):

            self.logger.info("Checking feature parameter passed")
            if feature is None:

                self.logger.warning(
                    f"No feature passed. Default value of feature used for pdp : {test_X.columns[0]}"
                )
                pdp_feature = test_X.columns[0]

            else:

                self.logger.warning(
                    f"feature value passed. Feature used for correlation plot: {feature}"
                )
                pdp_feature = feature

            from interpret.blackbox import PartialDependence

            try:
                pdp = PartialDependence(
                    predict_fn=model.predict_proba, data=test_X
                )  # classification
            except AttributeError:
                pdp = PartialDependence(
                    predict_fn=model.predict, data=test_X
                )  # regression

            pdp_global = pdp.explain_global()
            pdp_plot = pdp_global.visualize(list(test_X.columns).index(pdp_feature))
            if save:
                import plotly.io as pio

                plot_filename = f"PDP {plot}.html"
                if not isinstance(save, bool):
                    plot_filename = os.path.join(save, plot_filename)
                self.logger.info(f"Saving '{plot_filename}'")
                pio.write_html(pdp_plot, plot_filename)
            return pdp_plot

        def msa(show: bool = True):
            from interpret.blackbox import MorrisSensitivity

            try:
                msa = MorrisSensitivity(
                    predict_fn=model.predict_proba, data=test_X
                )  # classification
            except AttributeError:
                msa = MorrisSensitivity(
                    predict_fn=model.predict, data=test_X
                )  # regression
            msa_global = msa.explain_global()
            msa_plot = msa_global.visualize()
            if save:
                import plotly.io as pio

                plot_filename = f"MSA {plot}.html"
                if not isinstance(save, bool):
                    plot_filename = os.path.join(save, plot_filename)
                self.logger.info(f"Saving '{plot_filename}'")
                pio.write_html(msa_plot, plot_filename)
            return msa_plot

        def pfi(show: bool = True):
            from interpret.ext.blackbox import PFIExplainer

            pfi = PFIExplainer(model)
            pfi_global = pfi.explain_global(test_X, true_labels=test_y)
            pfi_plot = pfi_global.visualize()
            if save:
                import plotly.io as pio

                plot_filename = f"PFI {plot}.html"
                if not isinstance(save, bool):
                    plot_filename = os.path.join(save, plot_filename)
                self.logger.info(f"Saving '{plot_filename}'")
                pio.write_html(pfi_plot, plot_filename)
            return pfi_plot

        shap_plot = locals()[plot](show=not save)

        self.logger.info("Visual Rendered Successfully")

        self.logger.info(
            "interpret_model() successfully completed......................................"
        )

        gc.collect()
        return shap_plot

    def models(
        self,
        type: Optional[str] = None,
        internal: bool = False,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        """
        Returns table of models available in model library.

        Example
        -------
        >>> _all_models = models()

        This will return pandas dataframe with all available
        models and their metadata.

        Parameters
        ----------
        type : str, default = None
            - linear : filters and only return linear models
            - tree : filters and only return tree based models
            - ensemble : filters and only return ensemble models

        internal: bool, default = False
            If True, will return extra columns and rows used internally.

        raise_errors: bool, default = True
            If False, will suppress all exceptions, ignoring models
            that couldn't be created.

        Returns
        -------
        pandas.DataFrame

        """

        model_type = {
            "linear": [
                "lr",
                "ridge",
                "svm",
                "lasso",
                "en",
                "lar",
                "llar",
                "omp",
                "br",
                "ard",
                "par",
                "ransac",
                "tr",
                "huber",
                "kr",
            ],
            "tree": ["dt"],
            "ensemble": [
                "rf",
                "et",
                "gbc",
                "gbr",
                "xgboost",
                "lightgbm",
                "catboost",
                "ada",
            ],
        }

        def filter_model_df_by_type(df):
            if not type:
                return df
            return df[df.index.isin(model_type[type])]

        # Check if type is valid
        if type not in list(model_type) + [None]:
            raise ValueError(
                f"type parameter only accepts {', '.join(list(model_type) + str(None))}."
            )

        self.logger.info(f"gpu_param set to {self.gpu_param}")

        _, model_containers = self._get_models(raise_errors)

        rows = [
            v.get_dict(internal)
            for k, v in model_containers.items()
            if (internal or not v.is_special)
        ]

        df = pd.DataFrame(rows)
        df.set_index("ID", inplace=True, drop=True)

        return filter_model_df_by_type(df)

    def get_metrics(
        self,
        reset: bool = False,
        include_custom: bool = True,
        raise_errors: bool = True,
    ) -> pd.DataFrame:
        """
        Returns table of metrics available.

        Example
        -------
        >>> metrics = get_metrics()

        This will return pandas dataframe with all available
        metrics and their metadata.

        Parameters
        ----------
        reset: bool, default = False
            If True, will reset all changes made using add_metric() and get_metric().
        include_custom: bool, default = True
            Whether to include user added (custom) metrics or not.
        raise_errors: bool, default = True
            If False, will suppress all exceptions, ignoring models
            that couldn't be created.

        Returns
        -------
        pandas.DataFrame

        """

        if reset and not self._setup_ran:
            raise ValueError("setup() needs to be ran first.")

        np.random.seed(self.seed)

        if reset:
            self._all_metrics = self._get_metrics(raise_errors=raise_errors)

        metric_containers = self._all_metrics
        rows = [v.get_dict() for k, v in metric_containers.items()]

        df = pd.DataFrame(rows)
        df.set_index("ID", inplace=True, drop=True)

        if not include_custom:
            df = df[df["Custom"] is False]

        return df

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: type,
        target: str = "pred",
        greater_is_better: bool = True,
        multiclass: bool = True,
        **kwargs,
    ) -> pd.Series:
        """
        Adds a custom metric to be used in all functions.

        Parameters
        ----------
        id: str
            Unique id for the metric.

        name: str
            Display name of the metric.

        score_func: type
            Score function (or loss function) with signature score_func(y, y_pred, **kwargs).

        target: str, default = 'pred'
            The target of the score function.
            - 'pred' for the prediction table
            - 'pred_proba' for pred_proba
            - 'threshold' for decision_function or predict_proba

        greater_is_better: bool, default = True
            Whether score_func is a score function (default), meaning high is good,
            or a loss function, meaning low is good. In the latter case, the
            scorer object will sign-flip the outcome of the score_func.

        multiclass: bool, default = True
            Whether the metric supports multiclass problems.

        **kwargs:
            Arguments to be passed to score function.

        Returns
        -------
        pandas.Series
            The created row as Series.

        """

        if not self._setup_ran:
            raise ValueError("setup() needs to be ran first.")

        if id in self._all_metrics:
            raise ValueError("id already present in metrics dataframe.")

        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            new_metric = (
                pycaret.containers.metrics.classification.ClassificationMetricContainer(
                    id=id,
                    name=name,
                    score_func=score_func,
                    target=target,
                    args=kwargs,
                    display_name=name,
                    greater_is_better=greater_is_better,
                    is_multiclass=bool(multiclass),
                    is_custom=True,
                )
            )
        elif self._ml_usecase == MLUsecase.TIME_SERIES:
            new_metric = (
                pycaret.containers.metrics.time_series.TimeSeriesMetricContainer(
                    id=id,
                    name=name,
                    score_func=score_func,
                    args=kwargs,
                    display_name=name,
                    greater_is_better=greater_is_better,
                    is_custom=True,
                )
            )
        else:
            new_metric = (
                pycaret.containers.metrics.regression.RegressionMetricContainer(
                    id=id,
                    name=name,
                    score_func=score_func,
                    args=kwargs,
                    display_name=name,
                    greater_is_better=greater_is_better,
                    is_custom=True,
                )
            )

        self._all_metrics[id] = new_metric

        new_metric = new_metric.get_dict()

        new_metric = pd.Series(new_metric, name=id.replace(" ", "_")).drop("ID")

        return new_metric

    def remove_metric(self, name_or_id: str):
        """
        Removes a metric used in all functions.

        Parameters
        ----------
        name_or_id: str
            Display name or ID of the metric.

        """
        if not self._setup_ran:
            raise ValueError("setup() needs to be ran first.")

        try:
            self._all_metrics.pop(name_or_id)
            return
        except Exception:
            pass

        try:
            k_to_remove = next(
                k for k, v in self._all_metrics.items() if v.name == name_or_id
            )
            self._all_metrics.pop(k_to_remove)
            return
        except Exception:
            pass

        raise ValueError(
            f"No metric 'Display Name' or 'ID' (index) {name_or_id} present in the metrics repository."
        )

    def finalize_model(
        self,
        estimator,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        model_only: bool = False,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
    ) -> Any:  # added in pycaret==2.2.0

        """
        This function fits the complete pipeline with the estimator on the
        complete dataset passed during the setup() stage. The purpose of
        this function is to prepare for final model deployment after
        experimentation.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> final_lr = finalize_model(lr)

        This will return the final model object fitted to complete dataset.

        Parameters
        ----------
        estimator : object, default = none
            A trained model object should be passed as an estimator.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        model_only : bool, default = False
            Whether to return the complete fitted pipeline or only the fitted model.

        experiment_custom_tags: dict, default = None
            Dictionary of tag_name: String -> value: (String, but will be string-ified if
            not) passed to the mlflow.set_tags to add new custom tags for the experiment.

        Returns
        -------
            Trained pipeline or model object fitted on complete dataset.

        Warnings
        --------
        - If the model returned by finalize_model(), is used on predict_model() without
        passing a new unseen dataset, then the information grid printed is misleading
        as the model is trained on the complete dataset including test / hold-out sample.
        Once finalize_model() is used, the model is considered ready for deployment and
        should be used on new unseen dataset only.

        """
        self._check_setup_ran()

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing finalize_model()")
        self.logger.info(f"finalize_model({function_params_str})")

        runtime_start = time.time()

        display = CommonDisplay(
            verbose=False,
            html_param=self.html_param,
        )

        np.random.seed(self.seed)

        self.logger.info(f"Finalizing {estimator}")
        pipeline_final, model_fit_time = self._create_model(
            estimator=estimator,
            cross_validation=False,
            verbose=False,
            system=False,
            X_train_data=self.X,
            y_train_data=self.y,
            fit_kwargs=fit_kwargs or {},
            predict=False,
            groups=self._get_groups(groups, data=self.X),
            add_to_model_list=False,
            model_only=False,
        )

        # dashboard logging
        if self.logging_param:
            self._log_model(
                model=pipeline_final,
                model_results=None,
                score_dict={},
                source="finalize_model",
                runtime=np.array(time.time() - runtime_start).round(2),
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                experiment_custom_tags=experiment_custom_tags,
                display=display,
            )

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(pipeline_final))
        self.logger.info(
            "finalize_model() successfully completed......................................"
        )

        gc.collect()

        if model_only:
            return self._get_final_model_from_pipeline(pipeline_final)

        return pipeline_final

    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,  # added in pycaret==2.1.0
        raw_score: bool = False,
        drift_report: bool = False,
        round: int = 4,  # added in pycaret==2.2.0
        verbose: bool = True,
        ml_usecase: Optional[MLUsecase] = None,
        preprocess: Union[bool, str] = True,
    ) -> pd.DataFrame:

        """
        This function is used to predict label and probability score on the new dataset
        using a trained estimator. New unseen data can be passed to data parameter as pandas
        Dataframe. If data is not passed, the test / hold-out set separated at the time of
        setup() is used to generate predictions.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> lr_predictions_holdout = predict_model(lr)

        Parameters
        ----------
        estimator : object, default = none
            A trained model object / pipeline should be passed as an estimator.

        data : pandas.DataFrame
            Shape (n_samples, n_features) where n_samples is the number of samples
            and n_features is the number of features. All features used during training
            must be present in the new dataset.

        probability_threshold : float, default = None
            Threshold used to convert probability values into binary outcome. By default
            the probability threshold for all binary classifiers is 0.5 (50%). This can be
            changed using probability_threshold param.

        encoded_labels: Boolean, default = False
            If True, will return labels encoded as an integer.

        raw_score: bool, default = False
            When set to True, scores for all labels will be returned.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        verbose: bool, default = True
            Holdout score grid is not printed when verbose is set to False.

        preprocess: bool or 'features', default = True
            Whether to preprocess unseen data. If 'features', will not
            preprocess labels.

        Returns
        -------
        Predictions
            Predictions (label and score) columns are attached to the original
            dataset and returned as pandas dataframe.

        score_grid
            A table containing the scoring metrics on hold-out / test set.

        Warnings
        --------
        - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
        As such, the pipelines trained using the version (<= 2.0), may not work for inference
        with version >= 2.1. You can either retrain your models with a newer version or downgrade
        the version for inference.

        """

        def replace_labels_in_column(pipeline, labels: pd.Series) -> pd.Series:
            # Check if there is a LabelEncoder in the pipeline
            le = get_label_encoder(pipeline)
            if le:
                return pd.Series(
                    data=le.inverse_transform(labels),
                    name=labels.name,
                    index=labels.index,
                )
            else:
                return labels

        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if k != "data"]
        )

        self.logger.info("Initializing predict_model()")
        self.logger.info(f"predict_model({function_params_str})")

        self.logger.info("Checking exceptions")

        """
        exception checking starts here
        """

        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        if data is None and not self._setup_ran:
            raise ValueError(
                "data parameter may not be None without running setup() first."
            )

        if probability_threshold is not None:
            # probability_threshold allowed types
            allowed_types = [int, float]
            if (
                type(probability_threshold) not in allowed_types
                or probability_threshold > 1
                or probability_threshold < 0
            ):
                raise TypeError(
                    "probability_threshold parameter only accepts value between 0 to 1."
                )

        """
        exception checking ends here
        """

        self.logger.info("Preloading libraries")

        try:
            np.random.seed(self.seed)
            display = CommonDisplay(
                verbose=verbose,
                html_param=self.html_param,
            )
        except Exception:
            display = CommonDisplay(
                verbose=False,
                html_param=False,
            )

        if isinstance(estimator, skPipeline):
            if not hasattr(estimator, "feature_names_in_"):
                raise ValueError(
                    "If estimator is a Pipeline, it must implement `feature_names_in_`."
                )
            # We use copy instead of deepcopy because of https://github.com/pycaret/pycaret/issues/2769
            # Catboost behaves strange when deep copied. Using copy is fine
            # since the underlying estimators are only used for transform
            pipeline = copy(estimator)

            # Temporarily remove final estimator so it's not used for transform
            final_step = pipeline.steps[-1]
            estimator = final_step[-1]
            pipeline.steps = pipeline.steps[:-1]
        elif not self._setup_ran:
            raise ValueError(
                "If estimator is not a Pipeline, you must run setup() first."
            )
        else:
            pipeline = self.pipeline
            final_step = None

        X_columns = pipeline.feature_names_in_[:-1]
        y_name = pipeline.feature_names_in_[-1]
        y_test_ = None
        if data is None:
            X_test_, y_test_ = self.X_test_transformed, self.y_test_transformed
            X_test_untransformed, y_test_untransformed = self.X_test, self.y_test
        else:
            if y_name in data.columns:
                data = self._set_index(self._prepare_dataset(data, y_name))
                target = data[y_name]
                data = data.drop(y_name, axis=1)
            else:
                data = self._set_index(self._prepare_dataset(data))
                target = None
            data = data[X_columns]  # Ignore all columns but the originals
            if preprocess:
                X_test_ = pipeline.transform(
                    X=data,
                    y=(target if preprocess != "features" else None),
                )
                if final_step:
                    pipeline.steps.append(final_step)

                if isinstance(X_test_, tuple):
                    X_test_, y_test_ = X_test_
                elif target is not None:
                    y_test_ = target
            else:
                X_test_ = data
                y_test_ = target
            X_test_untransformed = data
            y_test_untransformed = target

        # generate drift report
        if drift_report:
            _check_soft_dependencies("evidently", extra="mlops", severity="error")
            from evidently.dashboard import Dashboard
            from evidently.pipeline.column_mapping import ColumnMapping
            from evidently.tabs import CatTargetDriftTab, DataDriftTab

            column_mapping = ColumnMapping()
            column_mapping.target = self.target_param
            column_mapping.prediction = None
            column_mapping.datetime = None
            column_mapping.numerical_features = self._fxs["Numeric"]
            column_mapping.categorical_features = self._fxs["Categorical"]
            column_mapping.datetime_features = self._fxs["Date"]

            drift_data = data if data is not None else self.test

            if y_name not in drift_data.columns:
                raise ValueError(
                    f"The dataset must contain a label column {y_name} "
                    "in order to create a drift report."
                )

            dashboard = Dashboard(tabs=[DataDriftTab(), CatTargetDriftTab()])
            dashboard.calculate(self.train, drift_data, column_mapping=column_mapping)
            report_name = f"{self._get_model_name(estimator)}_Drift_Report.html"
            dashboard.save(report_name)
            if verbose:
                print(f"{report_name} saved successfully.")

        # prediction starts here
        if isinstance(estimator, CustomProbabilityThresholdClassifier):
            if probability_threshold is None:
                probability_threshold = estimator.probability_threshold
            estimator = get_estimator_from_meta_estimator(estimator)

        pred = np.nan_to_num(estimator.predict(X_test_))

        try:
            score = estimator.predict_proba(X_test_)

            if len(np.unique(pred)) <= 2:
                pred_prob = score[:, 1]
            else:
                pred_prob = score

        except Exception:
            score = None
            pred_prob = None

        if probability_threshold is not None and pred_prob is not None:
            try:
                pred = (pred_prob >= probability_threshold).astype(int)
            except Exception:
                pass

        if pred_prob is None:
            pred_prob = pred

        df_score = None
        if y_test_ is not None and self._setup_ran:
            # model name
            full_name = self._get_model_name(estimator)
            metrics = self._calculate_metrics(y_test_, pred, pred_prob)  # type: ignore
            df_score = pd.DataFrame(metrics, index=[0])
            df_score.insert(0, "Model", full_name)
            df_score = df_score.round(round)
            display.display(df_score.style.format(precision=round))

        label = pd.DataFrame(
            pred, columns=[LABEL_COLUMN], index=X_test_untransformed.index
        )
        if ml_usecase == MLUsecase.CLASSIFICATION:
            try:
                label[LABEL_COLUMN] = label[LABEL_COLUMN].astype(int)
            except Exception:
                pass

        if not encoded_labels:
            label[LABEL_COLUMN] = replace_labels_in_column(
                pipeline, label[LABEL_COLUMN]
            )
        else:
            y_test_untransformed = y_test_
        old_index = X_test_untransformed.index
        X_test_ = pd.concat([X_test_untransformed, y_test_untransformed, label], axis=1)
        X_test_.index = old_index

        if score is not None:
            pred = pred.astype(int)

            if not raw_score:
                score = pd.DataFrame(
                    data=[s[pred[i]] for i, s in enumerate(score)],
                    index=X_test_.index,
                    columns=[SCORE_COLUMN],
                )
            else:
                if not encoded_labels:
                    le = get_label_encoder(pipeline)
                    if le:
                        columns = le.classes_
                    else:
                        columns = range(score.shape[1])
                else:
                    columns = range(score.shape[1])

                score = pd.DataFrame(
                    data=score,
                    index=X_test_.index,
                    columns=[f"{SCORE_COLUMN}_{col}" for col in columns],
                )

            score = score.round(round)
            X_test_ = pd.concat((X_test_, score), axis=1)

        # store predictions on hold-out in _display_container
        if df_score is not None:
            self._display_container.append(df_score)

        gc.collect()
        return X_test_

    def get_leaderboard(
        self,
        finalize_models: bool = False,
        model_only: bool = False,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
    ):
        """
        generates leaderboard for all models run in current run.
        """
        model_container = self._master_model_container

        progress_args = {"max": len(model_container) + 1}
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            [
                "Status",
                ". . . . . . . . . . . . . . . . . .",
                "Loading Dependencies",
            ],
            [
                "Estimator",
                ". . . . . . . . . . . . . . . . . .",
                "Compiling Library",
            ],
        ]
        display = CommonDisplay(
            verbose=verbose,
            html_param=self.html_param,
            progress_args=progress_args,
            monitor_rows=monitor_rows,
        )

        result_container_mean = []
        finalized_models = []

        display.update_monitor(
            1, "Finalizing models" if finalize_models else "Collecting models"
        )
        for i, model_results_tuple in enumerate(model_container):

            model_results = model_results_tuple["scores"]
            model = model_results_tuple["model"]
            mean_scores = model_results[-2:-1]
            model_name = self._get_model_name(model)
            mean_scores["Index"] = i
            mean_scores["Model Name"] = model_name
            display.update_monitor(2, model_name)
            if finalize_models:
                model = self.finalize_model(
                    model,
                    fit_kwargs=fit_kwargs,
                    groups=groups,
                    model_only=model_only,
                )
            else:
                model = deepcopy(model)
                if not is_fitted(model):
                    model, _ = self._create_model(
                        estimator=model,
                        verbose=False,
                        system=False,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        add_to_model_list=False,
                    )
                if not model_only:
                    pipeline = deepcopy(self.pipeline)
                    pipeline.steps.append(["trained_model", model])
                    model = pipeline
            display.move_progress()
            finalized_models.append(model)
            result_container_mean.append(mean_scores)

        display.update_monitor(1, "Creating dataframe")
        results = pd.concat(result_container_mean)
        results["Model"] = list(range(len(results)))
        results["Model"] = results["Model"].astype("object")
        model_loc = results.columns.get_loc("Model")
        for x in range(len(results)):
            results.iat[x, model_loc] = finalized_models[x]
        rearranged_columns = list(results.columns)
        rearranged_columns.remove("Model")
        rearranged_columns.remove("Model Name")
        rearranged_columns = ["Model Name", "Model"] + rearranged_columns
        results = results[rearranged_columns]
        results.set_index("Index", inplace=True, drop=True)
        display.close()
        # display.clear_output()
        return results

    def check_fairness(
        self, estimator, sensitive_features: list, plot_kwargs: dict = {}
    ):

        """
        There are many approaches to conceptualizing fairness. This function follows
        the approach known as group fairness, which asks: Which groups of individuals
        are at risk for experiencing harms. This function provides fairness-related
        metrics between different groups (also called subpopulation).


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> income = get_data('income')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = income,  target = 'income >50K')
        >>> lr = create_model('lr')
        >>> lr_fairness = check_fairness(lr, sensitive_features = ['sex', 'race'])


        estimator: scikit-learn compatible object
            Trained model object


        sensitive_features: list
            List of column names as present in the original dataset before any
            transformations.


        plot_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the matplotlib plot.


        Returns:
            pandas.DataFrame

        """

        _check_soft_dependencies("fairlearn", extra="analysis", severity="error")
        from fairlearn.metrics import MetricFrame, count, selection_rate

        all_metrics = self.get_metrics()[["Name", "Score Function", "Args"]].set_index(
            "Name"
        )
        metric_dict = {}
        metric_dict["Samples"] = count
        for i in all_metrics.index:
            metric_dict[i] = partial(all_metrics.loc[i][0], **all_metrics.loc[i][1])

        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            metric_dict["Selection Rate"] = selection_rate

        y_pred = self.predict_model(estimator)[LABEL_COLUMN]
        y_true = self.y_test

        try:
            multi_metric = MetricFrame(
                metrics=metric_dict,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=self.X_test[sensitive_features],
            )
        except Exception:
            if MLUsecase.CLASSIFICATION:
                metric_dict.pop("AUC")
                multi_metric = MetricFrame(
                    metrics=metric_dict,
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=self.X_test[sensitive_features],
                )

        multi_metric.by_group.plot.bar(
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[16, 8],
            title="Performance Metrics by Sensitive Features",
            **plot_kwargs,
        )

        return pd.DataFrame(multi_metric.by_group)

    def automl(
        self,
        optimize: str = "Accuracy",
        use_holdout: bool = False,
        turbo: bool = True,
        return_train_score: bool = False,
    ) -> Any:

        """
        This function returns the best model out of all models created in
        current active environment based on metric defined in optimize parameter.

        Parameters
        ----------
        optimize : str, default = 'Accuracy'
            Other values you can pass in optimize parameter are 'AUC', 'Recall', 'Precision',
            'F1', 'Kappa', and 'MCC'.

        use_holdout: bool, default = False
            When set to True, metrics are evaluated on holdout set instead of CV.

        turbo: bool, default = True
            When set to True and use_holdout is False, only models created with default fold
            parameter will be considered. If set to False, models created with a non-default
            fold parameter will be scored again using default fold settings, so that they can be
            compared.

        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.

        Returns:
            Trained Model
        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing automl()")
        self.logger.info(f"automl({function_params_str})")

        # checking optimize parameter
        optimize = self._get_metric_by_name_or_id(optimize)
        if optimize is None:
            raise ValueError(
                "Optimize method not supported. See docstring for list of available parameters."
            )

        # checking optimize parameter for multiclass
        if self.is_multiclass:
            if not optimize.is_multiclass:
                raise TypeError(
                    "Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                )

        # checking return_train_score parameter
        if type(return_train_score) is not bool:
            raise TypeError(
                "return_train_score can only take argument as True or False"
            )

        compare_dimension = optimize.display_name
        greater_is_better = optimize.greater_is_better
        optimize = optimize.scorer

        best_model = None
        best_score = None

        def compare_score(new, best):
            if not best:
                return True
            if greater_is_better:
                return new > best
            else:
                return new < best

        if use_holdout:
            self.logger.info("Model Selection Basis : Holdout set")
            for i in self._master_model_container:
                self.logger.info(f"Checking model {i}")
                model = i["model"]
                try:
                    self.predict_model(model, verbose=False)  # type: ignore
                except Exception:
                    self.logger.warning(
                        f"Model {model} is not fitted, running create_model"
                    )
                    model, _ = self._create_model(  # type: ignore
                        estimator=model,
                        system=False,
                        verbose=False,
                        cross_validation=False,
                        predict=False,
                        groups=self.fold_groups_param,
                        return_train_score=return_train_score,
                    )
                    self.predict_model(model, verbose=False)  # type: ignore

                p = self.pull(pop=True)
                p = p[compare_dimension][0]
                if compare_score(p, best_score):
                    best_model = model
                    best_score = p

        else:
            self.logger.info("Model Selection Basis : CV Results on Training set")
            for i in range(len(self._master_model_container)):
                model = self._master_model_container[i]
                scores = None
                if model["cv"] is not self.fold_generator:
                    if turbo or self._is_unsupervised():
                        continue
                    self._create_model(  # type: ignore
                        estimator=model["model"],
                        system=False,
                        verbose=False,
                        cross_validation=True,
                        predict=False,
                        groups=self.fold_groups_param,
                        return_train_score=return_train_score,
                    )
                    scores = self.pull(pop=True)
                    self._master_model_container.pop()
                self.logger.info(f"Checking model {i}")
                if scores is None:
                    scores = model["scores"]
                r = scores[compare_dimension][-2:][0]
                if compare_score(r, best_score):
                    best_model = model["model"]
                    best_score = r

        automl_model, _ = self._create_model(  # type: ignore
            estimator=best_model,
            system=False,
            verbose=False,
            cross_validation=False,
            predict=False,
            groups=self.fold_groups_param,
            return_train_score=return_train_score,
        )

        gc.collect()

        self.logger.info(str(automl_model))
        self.logger.info(
            "automl() successfully completed......................................"
        )

        return automl_model

    def create_app(self, estimator, app_kwargs: Optional[dict]):
        """
        This function creates a basic gradio app for inference.
        It will later be expanded for other app types such as
        Streamlit.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> create_app(lr)


        estimator: scikit-learn compatible object
            Trained model object


        app_kwargs: dict, default = {} (empty dict)
            arguments to be passed to app class.


        Returns:
            None
        """

        _check_soft_dependencies("gradio", extra="mlops", severity="error")
        import gradio as gr

        all_inputs = []
        app_kwargs = app_kwargs or {}

        for i in self.X.columns:
            if i in self._fxs["Categorical"] or i in self._fxs["Ordinal"]:
                all_inputs.append(gr.inputs.Dropdown(list(self.X[i].unique()), label=i))
            else:
                all_inputs.append(gr.inputs.Textbox(label=i))

        def predict(*dict_input):

            input_df = pd.DataFrame.from_dict([dict_input])
            input_df.columns = list(self.X.columns)
            return (
                self.predict_model(
                    estimator, data=input_df, **self._create_app_predict_kwargs
                )
                .iloc[0]
                .to_dict()
            )

        return gr.Interface(
            fn=predict,
            inputs=all_inputs,
            outputs="text",
            live=False,
            **app_kwargs,
        ).launch()

    def dashboard(
        self,
        estimator,
        display_format: str = "dash",
        dashboard_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        This function generates the interactive dashboard for a trained model. The
        dashboard is implemented using ExplainerDashboard (explainerdashboard.readthedocs.io)


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> dashboard(lr)


        estimator: scikit-learn compatible object
            Trained model object


        display_format: str, default = 'dash'
            Render mode for the dashboard. The default is set to ``dash`` which will
            render a dashboard in browser. There are four possible options:

            - 'dash' - displays the dashboard in browser
            - 'inline' - displays the dashboard in the jupyter notebook cell.
            - 'jupyterlab' - displays the dashboard in jupyterlab pane.
            - 'external' - displays the dashboard in a separate tab. (use in Colab)


        dashboard_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ``ExplainerDashboard`` class.


        run_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ``run`` method of ``ExplainerDashboard``.


        **kwargs:
            Additional keyword arguments to pass to the ``ClassifierExplainer`` or
            ``RegressionExplainer`` class.


        Returns:
            None
        """

        _check_soft_dependencies(
            "explainerdashboard", extra="analysis", severity="error"
        )

    def deep_check(self, estimator, check_kwargs: Optional[dict]):
        """
        This function runs a full suite check over a trained model
        using deepchecks library.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> deep_check(lr)


        estimator: scikit-learn compatible object
            Trained model object


        check_kwargs: dict, default = {} (empty dict)
            arguments to be passed to deepchecks full_suite class.


        Returns:
            Results of deepchecks.suites.full_suite.run
        """

        _check_soft_dependencies("deepchecks", extra="analysis", severity="error")
        check_kwargs = check_kwargs or {}

        from deepchecks import Dataset

        ds_train = Dataset(
            self.X_train_transformed, label=self.y_train_transformed, cat_features=[]
        )
        ds_test = Dataset(
            self.X_test_transformed, label=self.y_test_transformed, cat_features=[]
        )

        from deepchecks.tabular.suites import full_suite

        suite = full_suite(**check_kwargs)
        return suite.run(train_dataset=ds_train, test_dataset=ds_test, model=estimator)

    @classmethod
    def load_experiment(
        cls,
        path_or_file: Union[str, os.PathLike, BinaryIO],
        data: Optional[DATAFRAME_LIKE] = None,
        data_func: Optional[Callable[[], DATAFRAME_LIKE]] = None,
        test_data: Optional[DATAFRAME_LIKE] = None,
        preprocess_data: bool = True,
        **cloudpickle_kwargs,
    ) -> "_SupervisedExperiment":
        """
        Load an experiment saved with ``save_experiment`` from path
        or file.

        The data (and test data) is NOT saved with the experiment
        and will need to be specified again.


        path_or_file: str or BinaryIO (file pointer)
            The path/file pointer to load the experiment from.
            The pickle file must be created through ``save_experiment``.


        data: dataframe-like
            Data set with shape (n_samples, n_features), where n_samples is the
            number of samples and n_features is the number of features. If data
            is not a pandas dataframe, it's converted to one using default column
            names.


        data_func: Callable[[], DATAFRAME_LIKE] = None
            The function that generate ``data`` (the dataframe-like input). This
            is useful when the dataset is large, and you need parallel operations
            such as ``compare_models``. It can avoid broadcasting large dataset
            from driver to workers. Notice one and only one of ``data`` and
            ``data_func`` must be set.


        test_data: dataframe-like or None, default = None
            If not None, test_data is used as a hold-out set and `train_size` parameter
            is ignored. The columns of data and test_data must match.


        preprocess_data: bool, default = True
            If True, the data will be preprocessed again (through running ``setup``
            internally). If False, the data will not be preprocessed. This means
            you can save the value of the ``data`` attribute of an experiment
            separately, and then load it separately and pass it here with
            ``preprocess_data`` set to False. This is an advanced feature.
            We recommend leaving it set to True and passing the same data
            as passed to the initial ``setup`` call.


        **cloudpickle_kwargs:
            Kwargs to pass to the ``cloudpickle.load`` call.


        Returns:
            loaded experiment

        """

        return cls._load_experiment(
            path_or_file,
            cloudpickle_kwargs=cloudpickle_kwargs,
            preprocess_data=preprocess_data,
            data=data,
            data_func=data_func,
            test_data=test_data,
        )

    @property
    def X(self):
        """Feature set."""
        return self.dataset.drop(self.target_param, axis=1)

    @property
    def dataset_transformed(self):
        """Transformed dataset."""
        return pd.concat([self.train_transformed, self.test_transformed])

    @property
    def X_train_transformed(self):
        """Transformed feature set of the training set."""
        return self.pipeline.transform(
            X=self.X_train,
            y=self.y_train,
            filter_train_only=False,
        )[0]

    @property
    def train_transformed(self):
        """Transformed training set."""
        return pd.concat(
            [self.X_train_transformed, self.y_train_transformed],
            axis=1,
        )

    @property
    def X_transformed(self):
        """Transformed feature set."""
        return pd.concat([self.X_train_transformed, self.X_test_transformed])

    @property
    def y(self):
        """Target column."""
        return self.dataset[self.target_param]

    @property
    def X_train(self):
        """Feature set of the training set."""
        return self.train.drop(self.target_param, axis=1)

    @property
    def X_test(self):
        """Feature set of the test set."""
        return self.test.drop(self.target_param, axis=1)

    @property
    def train(self):
        """Training set."""
        return self.dataset.loc[self.idx[0], :]

    @property
    def test(self):
        """Test set."""
        return self.dataset.loc[self.idx[1], :]

    @property
    def y_train(self):
        """Target column of the training set."""
        return self.train[self.target_param]

    @property
    def y_test(self):
        """Target column of the test set."""
        return self.test[self.target_param]

    @property
    def test_transformed(self):
        """Transformed test set."""
        return pd.concat(
            [self.X_test_transformed, self.y_test_transformed],
            axis=1,
        )

    @property
    def y_transformed(self):
        """Transformed target column."""
        return pd.concat([self.y_train_transformed, self.y_test_transformed])

    @property
    def X_test_transformed(self):
        """Transformed feature set of the test set."""
        return self.pipeline.transform(self.X_test)

    @property
    def y_train_transformed(self):
        """Transformed target column of the training set."""
        return self.pipeline.transform(
            X=self.X_train,
            y=self.y_train,
            filter_train_only=False,
        )[1]

    @property
    def y_test_transformed(self):
        """Transformed target column of the test set."""
        return self.pipeline.transform(y=self.y_test)
