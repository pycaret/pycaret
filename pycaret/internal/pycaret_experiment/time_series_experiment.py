from copy import deepcopy
import os

from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)


from pycaret.internal.pycaret_experiment.utils import highlight_setup, MLUsecase
from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)
from pycaret.internal.pipeline import (
    estimator_pipeline,
    get_pipeline_fit_kwargs,
)
from pycaret.internal.utils import color_df, SeasonalPeriod, TSModelTypes
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
from pycaret.internal.logging import get_logger
from pycaret.internal.Display import Display

from pycaret.internal.distributions import *
from pycaret.internal.validation import *
from pycaret.internal.tunable import TunableMixin

import pycaret.containers.metrics.time_series
import pycaret.containers.models.time_series
import pycaret.internal.preprocess
import pycaret.internal.persistence
import pandas as pd  # type: ignore
from pandas.io.formats.style import Styler
import numpy as np  # type: ignore
import datetime
import time
import gc
from sklearn.base import clone  # type: ignore
from typing import List, Tuple, Any, Union, Optional, Dict, Generator
import warnings
from IPython.utils import io
import traceback
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import logging
from sklearn.base import clone  # type: ignore
from sklearn.model_selection._validation import _aggregate_score_dicts  # type: ignore
from sklearn.model_selection import check_cv, ParameterGrid, ParameterSampler  # type: ignore
from sklearn.model_selection._search import _check_param_grid  # type: ignore
from sklearn.metrics._scorer import get_scorer, _PredictScorer  # type: ignore
from collections import defaultdict
from functools import partial
from scipy.stats import rankdata  # type: ignore
from joblib import Parallel, delayed  # type: ignore


from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation.forecasting import check_y_X  # type: ignore
from sktime.forecasting.model_selection import SlidingWindowSplitter  # type: ignore

from pycaret.internal.tests.time_series import test_
from pycaret.internal.plots.time_series import plot_


warnings.filterwarnings("ignore")
LOGGER = get_logger()


# def _get_cv_n_folds(y, cv) -> int:
#     """
#     Get the number of folds for time series
#     cv must be of type SlidingWindowSplitter or ExpandingWindowSplitter
#     TODO: Fix this inside sktime and replace this with sktime method [1]

#     Ref:
#     [1] https://github.com/alan-turing-institute/sktime/issues/632
#     """
#     n_folds = int((len(y) - cv.initial_window) / cv.step_length)
#     return n_folds


def get_folds(cv, y) -> Generator[Tuple[pd.Series, pd.Series], None, None]:
    """
    Returns the train and test indices for the time series data
    """
    # https://github.com/alan-turing-institute/sktime/blob/main/examples/window_splitters.ipynb
    for train_indices, test_indices in cv.split(y):
        # print(f"Train Indices: {train_indices}, Test Indices: {test_indices}")
        yield train_indices, test_indices


def cross_validate_ts(
    forecaster,
    y: pd.Series,
    X: Optional[Union[pd.Series, pd.DataFrame]],
    cv,
    scoring: Dict[str, Union[str, _PredictScorer]],
    fit_params,
    n_jobs,
    return_train_score,
    error_score=0,
    verbose: int = 0,
) -> Dict[str, np.array]:
    """Performs Cross Validation on time series data

    Parallelization is based on `sklearn` cross_validate function [1]
    Ref:
    [1] https://github.com/scikit-learn/scikit-learn/blob/0.24.1/sklearn/model_selection/_validation.py#L246


    Parameters
    ----------
    forecaster : [type]
        Time Series Forecaster that is compatible with sktime
    y : pd.Series
        The variable of interest for forecasting
    X : Optional[Union[pd.Series, pd.DataFrame]]
        Exogenous Variables
    cv : [type]
        [description]
    scoring : Dict[str, Union[str, _PredictScorer]]
        Scoring Dictionary. Values can be valid strings that can be converted to
        callable metrics or the callable metrics directly
    fit_params : [type]
        Fit parameters to be used when training
    n_jobs : [type]
        Number of cores to use to parallelize. Refer to sklearn for details
    return_train_score : [type]
        Should the training scores be returned. Unused for now.
    error_score : int, optional
        Unused for now, by default 0
    verbose : int
        Sets the verbosity level. Unused for now

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    Error
        If fit and score raises any exceptions
    """
    try:
        # # For Debug
        # n_jobs = 1
        scoring = _get_metrics_dict_ts(scoring)
        parallel = Parallel(n_jobs=n_jobs)

        out = parallel(
            delayed(_fit_and_score)(
                forecaster=clone(forecaster),
                y=y,
                X=X,
                scoring=scoring,
                train=train,
                test=test,
                parameters=None,
                fit_params=fit_params,
                return_train_score=return_train_score,
                error_score=error_score,
            )
            for train, test in get_folds(cv, y)
        )
    # raise key exceptions
    except Exception:
        raise

    # Similar to parts of _format_results in BaseGridSearch
    (test_scores_dict, fit_time, score_time, cutoffs) = zip(*out)
    test_scores = _aggregate_score_dicts(test_scores_dict)

    return test_scores, cutoffs


def _get_metrics_dict_ts(
    metrics_dict: Dict[str, Union[str, _PredictScorer]]
) -> Dict[str, _PredictScorer]:
    """Returns a metrics dictionary in which all values are callables
    of type _PredictScorer

    Parameters
    ----------
    metrics_dict : A metrics dictionary in which some values can be strings.
        If the value is a string, the corresponding callable metric is returned
        e.g. Dictionary Value of 'neg_mean_absolute_error' will return
        make_scorer(mean_absolute_error, greater_is_better=False)
    """
    return_metrics_dict = {}
    for k, v in metrics_dict.items():
        if isinstance(v, str):
            return_metrics_dict[k] = get_scorer(v)
        else:
            return_metrics_dict[k] = v
    return return_metrics_dict


def _fit_and_score(
    forecaster,
    y: pd.Series,
    X: Optional[Union[pd.Series, pd.DataFrame]],
    scoring: Dict[str, Union[str, _PredictScorer]],
    train,
    test,
    parameters,
    fit_params,
    return_train_score,
    error_score=0,
):
    """Fits the forecaster on a single train split and scores on the test split
    Similar to _fit_and_score from `sklearn` [1] (and to some extent `sktime` [2]).
    Difference is that [1] operates on a single fold only, whereas [2] operates on all cv folds.
    Ref:
    [1] https://github.com/scikit-learn/scikit-learn/blob/0.24.1/sklearn/model_selection/_validation.py#L449
    [2] https://github.com/alan-turing-institute/sktime/blob/v0.5.3/sktime/forecasting/model_selection/_tune.py#L95

    Parameters
    ----------
    forecaster : [type]
        Time Series Forecaster that is compatible with sktime
    y : pd.Series
        The variable of interest for forecasting
    X : Optional[Union[pd.Series, pd.DataFrame]]
        Exogenous Variables
    scoring : Dict[str, Union[str, _PredictScorer]]
        Scoring Dictionary. Values can be valid strings that can be converted to
        callable metrics or the callable metrics directly
    train : [type]
        Indices of training samples.
    test : [type]
        Indices of test samples.
    parameters : [type]
        Parameter to set for the forecaster
    fit_params : [type]
        Fit parameters to be used when training
    return_train_score : [type]
        Should the training scores be returned. Unused for now.
    error_score : int, optional
        Unused for now, by default 0

    Raises
    ------
    ValueError
        When test indices do not match predicted indices. This is only for
        for internal checks and should not be raised when used by external users
    """
    if parameters is not None:
        forecaster.set_params(**parameters)

    y_train, y_test = y[train], y[test]
    X_train = None if X is None else X[train]
    X_test = None if X is None else X[test]

    #### Fit the forecaster ----
    start = time.time()
    try:
        forecaster.fit(y_train, X_train, **fit_params)
    except ValueError as error:
        ## Currently only catching ValueError. Can catch more later if needed.
        logging.error(f"Fit failed on {forecaster}")
        logging.error(error)

    fit_time = time.time() - start

    #### Determine Cutoff ----
    # NOTE: Cutoff is available irrespective of whether fit passed or failed
    cutoff = forecaster.cutoff

    #### Score the model ----
    if forecaster.is_fitted:
        y_pred = forecaster.predict(X_test)

        if (y_test.index.values != y_pred.index.values).any():
            print(
                f"\t y_train: {y_train.index.values},"
                f"\n\t y_test: {y_test.index.values}"
            )
            print(f"\t y_pred: {y_pred.index.values}")
            raise ValueError(
                "y_test indices do not match y_pred_indices or split/prediction "
                "length does not match forecast horizon."
            )

    start = time.time()
    fold_scores = {}
    scoring = _get_metrics_dict_ts(scoring)
    for scorer_name, scorer in scoring.items():
        if forecaster.is_fitted:
            metric = scorer._score_func(y_true=y_test, y_pred=y_pred, **scorer._kwargs)
        else:
            metric = None
        fold_scores[scorer_name] = metric
    score_time = time.time() - start

    return fold_scores, fit_time, score_time, cutoff


class BaseGridSearch:
    """
    Parallelization is based predominantly on [1]. Also similar to [2]

    Ref:
    [1] https://github.com/scikit-learn/scikit-learn/blob/0.24.1/sklearn/model_selection/_search.py#L795
    [2] https://github.com/scikit-optimize/scikit-optimize/blob/v0.8.1/skopt/searchcv.py#L410
    """

    def __init__(
        self,
        forecaster,
        cv,
        n_jobs=None,
        pre_dispatch=None,
        refit: bool = False,
        refit_metric: str = "smape",
        scoring=None,
        verbose=0,
        error_score=None,
        return_train_score=None,
    ):
        self.forecaster = forecaster
        self.cv = cv
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.refit = refit
        self.refit_metric = refit_metric
        self.scoring = scoring
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score

        self.best_params_ = {}
        self.cv_results_ = {}

    def fit(self, y, X=None, **fit_params):
        y, X = check_y_X(y, X)

        # validate cross-validator
        cv = check_cv(self.cv)
        base_forecaster = clone(self.forecaster)

        # This checker is sktime specific and only support 1 metric
        # Removing for now since we can have multiple metrics
        # TODO: Add back later if it supports multiple metrics
        # scoring = check_scoring(self.scoring)
        # Multiple metrics supported
        scorers = self.scoring  # Dict[str, Union[str, scorer]]  Not metrics container
        scorers = _get_metrics_dict_ts(scorers)
        refit_metric = self.refit_metric
        if refit_metric not in list(scorers.keys()):
            raise ValueError(
                f"Refit Metric: '{refit_metric}' is not available. ",
                f"Available Values are: {list(scorers.keys())}",
            )

        results = {}
        all_candidate_params = []
        all_out = []

        def evaluate_candidates(candidate_params):
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)
            n_splits = cv.get_n_splits(y)

            if self.verbose > 0:
                print(  # noqa
                    f"Fitting {n_splits} folds for each of {n_candidates} "
                    f"candidates, totalling {n_candidates * n_splits} fits"
                )

            parallel = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch
            )
            out = parallel(
                delayed(_fit_and_score)(
                    forecaster=clone(base_forecaster),
                    y=y,
                    X=X,
                    scoring=scorers,
                    train=train,
                    test=test,
                    parameters=parameters,
                    fit_params=fit_params,
                    return_train_score=self.return_train_score,
                    error_score=self.error_score,
                )
                for parameters in candidate_params
                for train, test in get_folds(cv, y)
            )

            if len(out) < 1:
                raise ValueError(
                    "No fits were performed. "
                    "Was the CV iterator empty? "
                    "Were there no candidates?"
                )

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

            nonlocal results
            results = self._format_results(
                all_candidate_params, scorers, all_out, n_splits
            )
            return results

        self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
        self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        self.best_forecaster_ = clone(base_forecaster).set_params(**self.best_params_)

        if self.refit:
            refit_start_time = time.time()
            self.best_forecaster_.fit(y, X, **fit_params)
            self.refit_time_ = time.time() - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = cv.get_n_splits(y)

        self._is_fitted = True
        return self

    @staticmethod
    def _format_results(candidate_params, scorers, out, n_splits):
        """From sklearn and sktime"""
        n_candidates = len(candidate_params)
        (test_scores_dict, fit_time, score_time, cutoffs) = zip(*out)
        test_scores_dict = _aggregate_score_dicts(test_scores_dict)

        results = {}

        # From sklearn (with the addition of greater_is_better from sktime)
        # INFO: For some reason, sklearn func does not work with sktime metrics
        # without passing greater_is_better (as done in sktime) and processing
        # it as such.
        def _store(
            key_name,
            array,
            weights=None,
            splits=False,
            rank=False,
            greater_is_better=False,
        ):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s" % (split_idx, key_name)] = array[:, split_idx]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(
                ~np.isfinite(array_means)
            ):
                warnings.warn(
                    f"One or more of the {key_name.split('_')[0]} scores "
                    f"are non-finite: {array_means}",
                    category=UserWarning,
                )

            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average(
                    (array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights
                )
            )
            results["std_%s" % key_name] = array_stds

            if rank:
                # This section is taken from sktime
                array_means = -array_means if greater_is_better else array_means
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(array_means, method="min"), dtype=np.int32
                )

        _store("fit_time", fit_time)
        _store("score_time", score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                np.ma.MaskedArray, np.empty(n_candidates,), mask=True, dtype=object,
            )
        )
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key "params"
        results["params"] = candidate_params

        for scorer_name, scorer in scorers.items():
            # Computed the (weighted) mean and std for test scores alone
            _store(
                "test_%s" % scorer_name,
                test_scores_dict[scorer_name],
                splits=True,
                rank=True,
                weights=None,
                greater_is_better=True if scorer._sign == 1 else False,
            )

        return results


class ForecastingGridSearchCV(BaseGridSearch):
    def __init__(
        self,
        forecaster,
        cv,
        param_grid,
        scoring=None,
        n_jobs=None,
        refit=True,
        refit_metric: str = "smape",
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super(ForecastingGridSearchCV, self).__init__(
            forecaster=forecaster,
            cv=cv,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            refit=refit,
            refit_metric=refit_metric,
            scoring=scoring,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))


class ForecastingRandomizedSearchCV(BaseGridSearch):
    def __init__(
        self,
        forecaster,
        cv,
        param_distributions,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        refit_metric: str = "smape",
        verbose=0,
        random_state=None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super(ForecastingRandomizedSearchCV, self).__init__(
            forecaster=forecaster,
            cv=cv,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            refit=refit,
            refit_metric=refit_metric,
            scoring=scoring,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        return evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state
            )
        )


class TimeSeriesExperiment(_SupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.TIME_SERIES
        self.exp_name_log = "ts-default-name"

        # Values in variable_keys are accessible in globals
        self.variable_keys = self.variable_keys.difference(
            {
                "target_param",
                "iterative_imputation_iters_param",
                "imputation_regressor",
                "imputation_classifier",
                "fold_shuffle_param",
                "stratify_param",
                "fold_groups_param",
            }
        )
        self.variable_keys = self.variable_keys.union(
            {
                "fh",
                "seasonal_period",
                "seasonality_present",
                "strictly_positive",
                "enforce_pi",
            }
        )
        self._available_plots = {
            "ts": "Time Series Plot",
            "train_test_split": "Train Test Split",
            "cv": "Cross Validation",
            "acf": "Auto Correlation (ACF)",
            "pacf": "Partial Auto Correlation (PACF)",
            "decomp_classical": "Decomposition Classical",
            "decomp_stl": "Decomposition STL",
            "diagnostics": "Diagnostics Plot",
            "forecast": "Out-of-Sample Forecast Plot",
            "insample": "In-Sample Forecast Plot",
            "residuals": "Residuals Plot",
        }

        self._available_plots_data_keys = [
            "ts",
            "train_test_split",
            "cv",
            "acf",
            "pacf",
            "decomp_classical",
            "decomp_stl",
            "diagnostics",
        ]

        self._available_plots_estimator_keys = [
            "ts",
            "train_test_split",
            "cv",
            "acf",
            "pacf",
            "decomp_classical",
            "decomp_stl",
            "diagnostics",
            "forecast",
            "insample",
            "residuals",
        ]

    def _get_setup_display(self, **kwargs) -> Styler:
        # define highlight function for function grid to display

        functions = pd.DataFrame(
            [
                ["session_id", self.seed],
                # ["Target", self.target_param],
                ["Original Data", self.data_before_preprocess.shape],
                ["Missing Values", kwargs["missing_flag"]],
            ]
            + (
                [
                    ["Transformed Train Set", self.y_train.shape],
                    ["Transformed Test Set", self.y_test.shape],
                    ["Fold Generator", type(self.fold_generator).__name__],
                    ["Fold Number", self.fold_param],
                    ["Enforce Prediction Interval", self.enforce_pi],
                    ["Seasonal Period Tested", self.seasonal_period],
                    ["Seasonality Detected", self.seasonality_present],
                    ["Target Strictly Positive", self.strictly_positive],
                    ["Target White Noise", self.white_noise],
                    ["Recommended d", self.lowercase_d],
                    ["Recommended Seasonal D", self.uppercase_d],
                    ["CPU Jobs", self.n_jobs_param],
                    ["Use GPU", self.gpu_param],
                    ["Log Experiment", self.logging_param],
                    ["Experiment Name", self.exp_name_log],
                    ["USI", self.USI],
                ]
            )
            + (
                [["Imputation Type", kwargs["imputation_type"]],]
                if self.preprocess
                else []
            ),
            # + (
            #    [
            #        ["Transform Target", self.transform_target_param],
            #        ["Transform Target Method", self.transform_target_method_param],
            #    ]
            # ),
            columns=["Description", "Value"],
        )
        return functions.style.apply(highlight_setup)

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in pycaret.containers.models.time_series.get_all_model_containers(
                self.variables, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = pycaret.containers.models.time_series.get_all_model_containers(
            self.variables, raise_errors=raise_errors
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        """Gets the metrics for the Time Series Module

        Parameters
        ----------
        raise_errors : bool, optional
            [description], by default True

        Returns
        -------
        dict
            [description]
        """
        return pycaret.containers.metrics.time_series.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
        )

    def _get_default_plots_to_log(self) -> List[str]:
        return ["forecast", "residuals", "diagnostics"]

    def check_fh(self, fh: Union[List[int], int, np.array]) -> np.array:
        """
        Checks fh for validity and converts fh into an appropriate forecasting
        horizon compatible with sktime (if necessary)

        Parameters
        ----------
        fh : Union[List[int], int, np.array]
            Forecasting Horizon

        Returns
        -------
        np.array
            Forecast Horizon (possibly updated to made compatible with sktime)

        Raises
        ------
        ValueError
            (1) When forecast horizon is an integer < 1
            (2) When forecast horizon is not the correct type
        """
        if isinstance(fh, int):
            if fh >= 1:
                fh = np.arange(1, fh + 1)
            else:
                raise ValueError(
                    f"If Forecast Horizon `fh` is an integer, it must be >= 1. You provided fh = '{fh}'!"
                )
        elif isinstance(fh, List):
            fh = np.array(fh)
        elif isinstance(fh, np.ndarray):
            # Good to go
            pass
        else:
            raise ValueError(
                f"Horizon `fh` must be a of type int, list, or numpy array, got object of {type(fh)} type!"
            )
        return fh

    def setup(
        self,
        data: Union[pd.Series, pd.DataFrame],
        preprocess: bool = True,
        imputation_type: str = "simple",
        #        transform_target: bool = False,
        #        transform_target_method: str = "box-cox",
        fold_strategy: Union[str, Any] = "expanding",
        fold: int = 3,
        fh: Union[List[int], int, np.array] = 1,
        seasonal_period: Optional[Union[int, str]] = None,
        enforce_pi: bool = False,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        custom_pipeline: Union[
            Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]
        ] = None,
        html: bool = True,
        session_id: Optional[int] = None,
        system_log: Union[bool, logging.Logger] = True,
        log_experiment: bool = False,
        experiment_name: Optional[str] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        verbose: bool = True,
        profile: bool = False,
        profile_kwargs: Dict[str, Any] = None,
    ):
        """
        This function initializes the training environment and creates the transformation
        pipeline. Setup function must be called before executing any other function. It takes
        one mandatory parameters: ``data``. All the other parameters are optional.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)


        data : pandas.Series or pandas.DataFrame
            Shape (n_samples, 1), when pandas.DataFrame, otherwise (n_samples, ).


        preprocess: bool, default = True
            Parameter not in use for now. Behavior may change in future.


        imputation_type: str, default = 'simple'
            Parameter not in use for now. Behavior may change in future.


        fold_strategy: str or sklearn CV generator object, default = 'expanding'
            Choice of cross validation strategy. Possible values are:

            * 'expanding'
            * 'rolling' (same as/aliased to 'expanding')
            * 'sliding'

            You can also pass an sktime compatible cross validation object such
            as ``SlidingWindowSplitter`` or ``ExpandingWindowSplitter``. In this case,
            the `fold` and `fh` parameters will be ignored and these values will
            be extracted from the ``fold_strategy`` object directly.


        fold: int, default = 3
            Number of folds to be used in cross validation. Must be at least 2. This is
            a global setting that can be over-written at function level by using ``fold``
            parameter. Ignored when ``fold_strategy`` is a custom object.


        fh: int or list or np.array, default = 1
            The forecast horizon to be used for forecasting. Default is set to ``1`` i.e.
            forecast one point ahead. When integer is passed it means N continious points 
            in the future without any gap. If you want to forecast values with gaps, you 
            must pass an array e.g. np.array([2, 5]) will forecast 2 and 5 points ahead.


        seasonal_period: int or str, default = None
            Seasonal period in timeseries data. If not provided the frequency of the data
            index is map to a seasonal period as follows:

            * 'S': 60
            * 'T': 60
            * 'H': 24
            * 'D': 7
            * 'W': 52
            * 'M': 12
            * 'Q': 4
            * 'A': 1
            * 'Y': 1

            Alternatively you can provide a custom `seasonal_parameter` by passing
            it as an integer.


        enforce_pi: bool, default = False
            When set to True, only models that support prediction intervals are
            loaded in the environment. 


        n_jobs: int, default = -1
            The number of jobs to run in parallel (for functions that supports parallel
            processing) -1 means using all processors. To run all functions on single
            processor set n_jobs to None.


        use_gpu: bool or str, default = False
            Parameter not in use for now. Behavior may change in future.


        custom_pipeline: (str, transformer) or list of (str, transformer), default = None
            Parameter not in use for now. Behavior may change in future.


        html: bool, default = True
            When set to False, prevents runtime display of monitor. This must be set to False
            when the environment does not support IPython. For example, command line terminal,
            Databricks Notebook, Spyder and other similar IDEs.


        session_id: int, default = None
            Controls the randomness of experiment. It is equivalent to 'random_state' in
            scikit-learn. When None, a pseudo random number is generated. This can be used
            for later reproducibility of the entire experiment.


        system_log: bool or logging.Logger, default = True
            Whether to save the system logging file (as logs.log). If the input already is a 
            logger object, that one is used instead.


        log_experiment: bool, default = False
            When set to True, all metrics and parameters are logged on the ``MLflow`` server.


        experiment_name: str, default = None
            Name of the experiment for logging. Ignored when ``log_experiment`` is not True.


        log_plots: bool or list, default = False
            When set to True, certain plots are logged automatically in the ``MLFlow`` server.
            To change the type of plots to be logged, pass a list containing plot IDs. Refer
            to documentation of ``plot_model``. Ignored when ``log_experiment`` is not True.


        log_profile: bool, default = False
            When set to True, data profile is logged on the ``MLflow`` server as a html file.
            Ignored when ``log_experiment`` is not True.


        log_data: bool, default = False
            When set to True, dataset is logged on the ``MLflow`` server as a csv file.
            Ignored when ``log_experiment`` is not True.


        verbose: bool, default = True
            When set to False, Information grid is not printed.


        profile: bool, default = False
            When set to True, an interactive EDA report is displayed.


        profile_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ProfileReport method used
            to create the EDA report. Ignored if ``profile`` is False.


        Returns:
            Global variables that can be changed using the ``set_config`` function.


        """
        from sktime.utils.seasonality import (
            autocorrelation_seasonality_test,
        )  # only needed in setup

        ## Make a local copy so as not to perfrom inplace operation on the
        ## original dataset
        data_ = data.copy()

        if isinstance(data_, pd.Series) and data_.name is None:
            data_.name = "Time Series"

        # Forecast Horizon Checks
        if fh is None and isinstance(fold_strategy, str):
            raise ValueError(
                f"The forecast horizon `fh` must be provided when fold_strategy is of type 'string'"
            )

        # Check Fold Strategy
        if not isinstance(fold_strategy, str):
            self.logger.info(
                f"fh parameter {fh} will be ignored since fold_strategy has been provided. "
                f"fh from fold_strategy will be used instead."
            )
            fh = fold_strategy.fh
            self.logger.info(
                f"fold parameter {fold} will be ignored since fold_strategy has been provided. "
                f"fold based on fold_strategy will be used instead."
            )
            # fold value will be reset after the data is split in the parent class setup

        fh = self.check_fh(fh)
        self.fh = fh

        # Check Index
        allowed_freq_index_types = (pd.PeriodIndex, pd.DatetimeIndex)
        if (
            not isinstance(data_.index, allowed_freq_index_types)
            and seasonal_period is None
        ):
            # https://stackoverflow.com/questions/3590165/join-a-list-of-items-with-different-types-as-string-in-python
            raise ValueError(
                f"The index of your 'data' is of type '{type(data_.index)}'. "
                "If the 'data' index is not of one of the following types: "
                f"{', '.join(str(type) for type in allowed_freq_index_types)}, "
                "then 'seasonal_period' must be provided. Refer to docstring for options."
            )

        if isinstance(data_.index, pd.DatetimeIndex):
            data_.index = data_.index.to_period()

        if seasonal_period is None:

            index_freq = data_.index.freqstr
            index_freq = index_freq.split("-")[0] or index_freq

            if index_freq in SeasonalPeriod.__members__:
                self.seasonal_period = SeasonalPeriod[index_freq].value
            else:
                raise ValueError(
                    f"Unsupported Period frequency: {index_freq}, valid Period frequencies: {', '.join(SeasonalPeriod.__members__.keys())}"
                )

        else:

            if not isinstance(seasonal_period, (int, str)):
                raise ValueError(
                    f"seasonal_period parameter must be an int or str, got {type(seasonal_period)}"
                )

            if isinstance(seasonal_period, str):
                try:
                    self.seasonal_period = SeasonalPeriod[seasonal_period]
                except KeyError:
                    raise ValueError(
                        f"Unsupported Period frequency: {seasonal_period}, valid Period frequencies: {', '.join(SeasonalPeriod.__members__.keys())}"
                    )
            else:
                self.seasonal_period = seasonal_period

        if isinstance(data_, (pd.Series, pd.DataFrame)):
            if isinstance(data_, pd.DataFrame):
                if data_.shape[1] != 1:
                    raise ValueError(
                        f"data must be a pandas Series or DataFrame with one column, got {data_.shape[1]} columns!"
                    )
                data_ = data_.copy()
            else:
                data_ = pd.DataFrame(data_)  # Force convertion to DataFrame
        else:
            raise ValueError(
                f"data must be a pandas Series or DataFrame, got object of {type(data_)} type!"
            )

        data_.columns = [str(x) for x in data_.columns]

        target_name = data_.columns[0]
        if not np.issubdtype(data_[target_name].dtype, np.number):
            raise TypeError(
                f"Data must be of 'numpy.number' subtype, got {data_[target_name].dtype}!"
            )

        if len(data_.index) != len(set(data_.index)):
            raise ValueError("Index may not have duplicate values!")

        # check valid seasonal parameter
        valid_seasonality = autocorrelation_seasonality_test(
            data_[target_name], self.seasonal_period
        )

        self.seasonality_present = True if valid_seasonality else False

        # Should multiplicative components be allowed in models that support it
        self.strictly_positive = np.all(data_[target_name] > 0)

        self.enforce_pi = enforce_pi

        return super().setup(
            data=data_,
            target=data_.columns[0],
            test_data=None,
            preprocess=preprocess,
            imputation_type=imputation_type,
            categorical_features=None,
            ordinal_features=None,
            high_cardinality_features=None,
            numeric_features=None,
            date_features=None,
            ignore_features=None,
            normalize=False,
            transformation=False,
            handle_unknown_categorical=False,
            pca=False,
            ignore_low_variance=False,
            combine_rare_levels=False,
            bin_numeric_features=None,
            remove_outliers=False,
            remove_multicollinearity=False,
            remove_perfect_collinearity=False,
            create_clusters=False,
            polynomial_features=False,
            trigonometry_features=False,
            group_features=None,
            feature_selection=False,
            feature_interaction=False,
            transform_target=False,
            data_split_shuffle=False,
            data_split_stratify=False,
            fold_strategy=fold_strategy,
            fold=fold,
            fh=fh,
            seasonal_period=seasonal_period,
            fold_shuffle=False,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            custom_pipeline=custom_pipeline,
            html=html,
            session_id=session_id,
            system_log=system_log,
            log_experiment=log_experiment,
            experiment_name=experiment_name,
            log_plots=log_plots,
            log_profile=log_profile,
            log_data=log_data,
            silent=True,
            verbose=verbose,
            profile=profile,
            profile_kwargs=profile_kwargs,
        )

    def compare_models(
        self,
        include: Optional[List[Union[str, Any]]] = None,
        exclude: Optional[List[str]] = None,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        sort: str = "smape",
        n_select: int = 1,
        budget_time: Optional[float] = None,
        turbo: bool = True,
        errors: str = "ignore",
        fit_kwargs: Optional[dict] = None,
        verbose: bool = True,
    ):

        """
        This function trains and evaluates performance of all estimators available in the
        model library using cross validation. The output of this function is a score grid
        with average cross validated scores. Metrics evaluated during CV can be accessed
        using the ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)
        >>> best_model = compare_models()


        include: list of str or sktime compatible object, default = None
            To train and evaluate select models, list containing model ID or scikit-learn
            compatible object can be passed in include param. To see a list of all models
            available in the model library use the ``models`` function.


        exclude: list of str, default = None
            To omit certain models from training and evaluation, pass a list containing
            model id in the exclude parameter. To see a list of all models available
            in the model library use the ``models`` function.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        cross_validation: bool, default = True
            When set to False, metrics are evaluated on holdout set. ``fold`` param
            is ignored when cross_validation is set to False.


        sort: str, default = 'SMAPE'
            The sort order of the score grid. It also accepts custom metrics that are
            added through the ``add_metric`` function.


        n_select: int, default = 1
            Number of top_n models to return. For example, to select top 3 models use
            n_select = 3.


        budget_time: int or float, default = None
            If not None, will terminate execution of the function after budget_time
            minutes have passed and return results up to that point.


        turbo: bool, default = True
            When set to True, it excludes estimators with longer training times. To
            see which algorithms are excluded use the ``models`` function.


        errors: str, default = 'ignore'
            When set to 'ignore', will skip the model with exceptions and continue.
            If 'raise', will break the function when exceptions are raised.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained model or list of trained models, depending on the ``n_select`` param.


        Warnings
        --------
        - Changing turbo parameter to False may result in very high training times.

        - No models are logged in ``MLflow`` when ``cross_validation`` parameter is False.

        """

        return super().compare_models(
            include=include,
            exclude=exclude,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            sort=sort,
            n_select=n_select,
            budget_time=budget_time,
            turbo=turbo,
            errors=errors,
            fit_kwargs=fit_kwargs,
            verbose=verbose,
        )

    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        verbose: bool = True,
        **kwargs,
    ):

        """
        This function trains and evaluates the performance of a given estimator
        using cross validation. The output of this function is a score grid with
        CV scores by fold. Metrics evaluated during CV can be accessed using the
        ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function. All the available models
        can be accessed using the ``models`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)
        >>> naive = create_model('naive')

        estimator: str or sktime compatible object
            ID of an estimator available in model library or pass an untrained
            model object consistent with scikit-learn API. Estimators available
            in the model library (ID - Name):

            * 'naive' - Naive Forecaster
            * 'grand_means' - Grand Means Forecaster
            * 'snaive' - Seasonal Naive Forecaster
            * 'polytrend' - Polynomial Trend Forecaster
            * 'arima' - ARIMA
            * 'auto_arima' - Auto ARIMA
            * 'arima' - ARIMA
            * 'exp_smooth' - Exponential Smoothing
            * 'ets' - ETS
            * 'theta' - Theta Forecaster
            * 'tbats' - TBATS
            * 'bats' - BATS
            * 'prophet' - Prophet Forecaster
            * 'lr_cds_dt' - Linear w/ Cond. Deseasonalize & Detrending
            * 'en_cds_dt' - Elastic Net w/ Cond. Deseasonalize & Detrending
            * 'ridge_cds_dt' - Ridge w/ Cond. Deseasonalize & Detrending
            * 'lasso_cds_dt' - Lasso w/ Cond. Deseasonalize & Detrending
            * 'lar_cds_dt' -   Least Angular Regressor w/ Cond. Deseasonalize & Detrending
            * 'llar_cds_dt' - Lasso Least Angular Regressor w/ Cond. Deseasonalize & Detrending
            * 'br_cds_dt' - Bayesian Ridge w/ Cond. Deseasonalize & Deseasonalize & Detrending
            * 'huber_cds_dt' - Huber w/ Cond. Deseasonalize & Detrending
            * 'par_cds_dt' - Passive Aggressive w/ Cond. Deseasonalize & Detrending
            * 'omp_cds_dt' - Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending
            * 'knn_cds_dt' - K Neighbors w/ Cond. Deseasonalize & Detrending
            * 'dt_cds_dt' - Decision Tree w/ Cond. Deseasonalize & Detrending
            * 'rf_cds_dt' - Random Forest w/ Cond. Deseasonalize & Detrending
            * 'et_cds_dt' - Extra Trees w/ Cond. Deseasonalize & Detrending
            * 'gbr_cds_dt' - Gradient Boosting w/ Cond. Deseasonalize & Detrending
            * 'ada_cds_dt' - AdaBoost w/ Cond. Deseasonalize & Detrending
            * 'lightgbm_cds_dt' - Light Gradient Boosting w/ Cond. Deseasonalize & Detrending


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        cross_validation: bool, default = True
            When set to False, metrics are evaluated on holdout set. ``fold`` param
            is ignored when cross_validation is set to False.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        **kwargs:
            Additional keyword arguments to pass to the estimator.


        Returns:
            Trained Model


        Warnings
        --------
        - Models are not logged on the ``MLFlow`` server when ``cross_validation`` param
        is set to False.

        """
        return super().create_model(
            estimator=estimator,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            fit_kwargs=fit_kwargs,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def update_fit_kwargs_with_fh_from_cv(fit_kwargs: Optional[Dict], cv) -> Dict:
        """Updated the fit_ kwargs to include the fh parameter from cv

        Parameters
        ----------
        fit_kwargs : Optional[Dict]
            Original fit kwargs
        cv : [type]
            cross validation object

        Returns
        -------
        Dict[Any]
            Updated fit kwargs
        """
        fh_param = {"fh": cv.fh}
        if fit_kwargs is None:
            fit_kwargs = fh_param
        else:
            fit_kwargs.update(fh_param)
        return fit_kwargs

    def _create_model_without_cv(
        self, model, data_X, data_y, fit_kwargs, predict, system, display: Display
    ):
        # with estimator_pipeline(self._internal_pipeline, model) as pipeline_with_model:

        self.logger.info(
            "Support for Exogenous variables not yet supported. Switching X, y order"
        )
        data_X, data_y = data_y, data_X

        fit_kwargs = get_pipeline_fit_kwargs(model, fit_kwargs)
        self.logger.info("Cross validation set to False")

        self.logger.info("Fitting Model")
        model_fit_start = time.time()
        with io.capture_output():
            model.fit(data_X, data_y, **fit_kwargs)
        model_fit_end = time.time()

        model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

        display.move_progress()

        if predict:
            self.predict_model(model, verbose=False)
            model_results = self.pull(pop=True).drop("Model", axis=1)

            self.display_container.append(model_results)

            display.display(
                model_results, clear=system, override=False if not system else None,
            )

            self.logger.info(f"display_container: {len(self.display_container)}")

        return model, model_fit_time

    def _create_model_with_cv(
        self,
        model,
        data_X,
        data_y,
        fit_kwargs,
        round,
        cv,
        groups,  # TODO: See if we can remove groups
        metrics,
        refit,
        system,
        display,
    ):
        """
        MONITOR UPDATE STARTS
        """

        # display.update_monitor(
        #     1, f"Fitting {_get_cv_n_folds(data_y, cv)} Folds",
        # )
        display.update_monitor(
            1, f"Fitting {cv.get_n_splits(data_y)} Folds",
        )
        display.display_monitor()
        """
        MONITOR UPDATE ENDS
        """
        metrics_dict = {k: v.scorer for k, v in metrics.items()}

        self.logger.info("Starting cross validation")

        n_jobs = self._gpu_n_jobs_param

        # fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)

        self.logger.info(f"Cross validating with {cv}, n_jobs={n_jobs}")

        # Cross Validate time series
        # fh_param = {"fh": cv.fh}

        # if fit_kwargs is None:
        #     fit_kwargs = fh_param
        # else:
        #     fit_kwargs.update(fh_param)
        fit_kwargs = self.update_fit_kwargs_with_fh_from_cv(
            fit_kwargs=fit_kwargs, cv=cv
        )

        model_fit_start = time.time()

        scores, cutoffs = cross_validate_ts(
            # Commented out since supervised_experiment also does not clone
            # when doing cross_validate
            # forecaster=clone(model),
            forecaster=model,
            y=data_y,
            X=data_X,
            scoring=metrics_dict,
            cv=cv,
            n_jobs=n_jobs,
            verbose=0,
            fit_params=fit_kwargs,
            return_train_score=False,
            error_score=0,
        )

        model_fit_end = time.time()
        model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

        # Scores has metric names in lowercase, scores_dict has metric names in uppercase
        score_dict = {v.display_name: scores[f"{k}"] for k, v in metrics.items()}

        self.logger.info("Calculating mean and std")

        avgs_dict = {k: [np.mean(v), np.std(v)] for k, v in score_dict.items()}

        display.move_progress()

        self.logger.info("Creating metrics dataframe")

        model_results = pd.DataFrame(score_dict)
        model_results.insert(0, "cutoff", cutoffs)

        model_avgs = pd.DataFrame(avgs_dict, index=["Mean", "SD"],)
        model_avgs.insert(0, "cutoff", np.nan)

        model_results = model_results.append(model_avgs)
        # Round the results
        model_results = model_results.round(round)

        # yellow the mean (converts model_results from dataframe to dataframe styler)
        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)

        if refit:
            # refitting the model on complete X_train, y_train
            display.update_monitor(1, "Finalizing Model")
            display.display_monitor()
            model_fit_start = time.time()
            self.logger.info("Finalizing model")
            with io.capture_output():
                model.fit(y=data_y, X=data_X, **fit_kwargs)
            model_fit_end = time.time()

            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
        else:
            # Set fh explicitly since we are not fitting explicitly
            # This is needed so that the model can be used later to predict, etc.
            model._set_fh(fit_kwargs.get("fh"))

            # model_fit_time /= _get_cv_n_folds(data_y, cv)
            model_fit_time /= cv.get_n_splits(data_y)

        # return model, model_fit_time, model_results, avgs_dict
        return model, model_fit_time, model_results, avgs_dict

    def tune_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[Union[Dict[str, list], Any]] = None,
        optimize: str = "SMAPE",
        custom_scorer=None,
        search_algorithm: Optional[str] = None,
        choose_better: bool = True,
        fit_kwargs: Optional[dict] = None,
        return_tuner: bool = False,
        verbose: bool = True,
        tuner_verbose: Union[int, bool] = True,
        display: Optional[Display] = None,
        **kwargs,
    ):

        """
        This function tunes the hyperparameters of a given estimator. The output of
        this function is a score grid with CV scores by fold of the best selected
        model based on ``optimize`` parameter. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)
        >>> dt = create_model('dt_cds_dt')
        >>> tuned_dt = tune_model(dt)


        estimator: sktime compatible object
            Trained model object


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        n_iter: int, default = 10
            Number of iterations in the grid search. Increasing 'n_iter' may improve
            model performance but also increases the training time.


        custom_grid: dictionary, default = None
            To define custom search space for hyperparameters, pass a dictionary with
            parameter name and values to be iterated. Custom grids must be in a format
            supported by the defined ``search_library``.


        optimize: str, default = 'SMAPE'
            Metric name to be evaluated for hyperparameter tuning. It also accepts custom
            metrics that are added through the ``add_metric`` function.


        custom_scorer: object, default = None
            custom scoring strategy can be passed to tune hyperparameters of the model.
            It must be created using ``sklearn.make_scorer``. It is equivalent of adding
            custom metric using the ``add_metric`` function and passing the name of the
            custom metric in the ``optimize`` parameter.
            Will be deprecated in future.


        search_algorithm: str, default = 'random'
            use 'random' for random grid search and 'grid' for complete grid search. 


        choose_better: bool, default = True
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the tuner.


        return_tuner: bool, default = False
            When set to True, will return a tuple of (model, tuner_object).


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        tuner_verbose: bool or in, default = True
            If True or above 0, will print messages from the tuner. Higher values
            print more messages. Ignored when ``verbose`` param is False.


        **kwargs:
            Additional keyword arguments to pass to the optimizer.


        Returns:
            Trained Model and Optional Tuner Object when ``return_tuner`` is True.

        """

        search_library = "pycaret"  # only 1 library supported right now

        _allowed_search_algorithms = []
        if search_library == "pycaret":
            _allowed_search_algorithms = [None, "random", "grid"]
            if search_algorithm not in _allowed_search_algorithms:
                raise ValueError(
                    "`search_algorithm` must be one of "
                    f"'{', '.join(str(allowed_type) for allowed_type in _allowed_search_algorithms)}'. "
                    f"You passed '{search_algorithm}'."
                )

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

        if isinstance(optimize, str):
            # checking optimize parameter
            # TODO: Changed with reference to other ML Usecases. Check with Antoni
            # optimize = self._get_metric_by_name_or_id(optimize)
            # if optimize is None:
            #     raise ValueError(
            #         "Optimize method not supported. See docstring for list of available parameters."
            #     )
            optimize_container = self._get_metric_by_name_or_id(optimize)
            if optimize_container is None:
                raise ValueError(
                    "Optimize method not supported. See docstring for list of available parameters."
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

        # cross validation setup starts here
        cv = self.get_fold_generator(fold=fold)

        if not display:
            progress_args = {"max": 3 + 4}
            master_display_columns = [
                v.display_name for k, v in self._all_metrics.items()
            ]
            if self._ml_usecase == MLUsecase.TIME_SERIES:
                master_display_columns.insert(0, "cutoff")
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
            display = Display(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                master_display_columns=master_display_columns,
                monitor_rows=monitor_rows,
            )

            display.display_progress()
            display.display_monitor()
            display.display_master_display()

        # ignore warnings

        warnings.filterwarnings("ignore")

        import logging

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")
        # Storing X_train and y_train in data_X and data_y parameter
        data_X = self.X_train.copy()
        data_y = self.y_train.copy()

        # Replace Empty DataFrame with None as empty DataFrame causes issues
        if (data_X.shape[0] == 0) or (data_X.shape[1] == 0):
            data_X = None

        display.move_progress()

        # setting optimize parameter
        # TODO: Changed compared to other PyCaret UseCases (Check with Antoni)
        # optimize = optimize.scorer
        compare_dimension = optimize_container.display_name
        optimize_metric_dict = {optimize_container.id: optimize_container.scorer}

        # Returns a dictionary of all metric containers (disabled for now since
        # we only need optimize metric)
        # {'mae': <pycaret.containers....783DEB0C8>, 'rmse': <pycaret.containers....783DEB148> ...}
        #  all_metric_containers = self._all_metrics

        # # Returns a dictionary of all metric scorers (disabled for now since
        # we only need optimize metric)
        # {'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error' ...}
        # all_metrics_dict = {
        #     all_metric_containers[metric_id].id: all_metric_containers[metric_id].scorer
        #     for metric_id in all_metric_containers
        # }

        refit_metric = optimize_container.id  # Name of the metric: e.g. 'mae'

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
            estimator_definition = self._all_models_internal[estimator_id]  # Container
            estimator_name = estimator_definition.name
        self.logger.info(f"Base model : {estimator_name}")

        # If no special tunable class is defined inside PyCaret then just clone the estimator
        if estimator_definition is None or estimator_definition.tunable is None:
            model = clone(estimator)
        # If special tunable class is defined, then use that instead
        else:
            self.logger.info("Model has a special tunable class, using that")
            model = clone(estimator_definition.tunable(**estimator.get_params()))
        is_stacked_model = False

        base_estimator = model

        display.update_monitor(2, estimator_name)
        display.display_monitor()

        display.move_progress()

        self.logger.info("Declaring metric variables")

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Searching Hyperparameters")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Defining Hyperparameters")

        if search_algorithm is None:
            search_algorithm = "random"  # Defaults to Random

        param_grid = None
        if search_library == "pycaret":
            if search_algorithm == "grid":
                param_grid = estimator_definition.tune_grid
            elif search_algorithm == "random":
                param_grid = estimator_definition.tune_distribution

        if not param_grid:
            raise ValueError(
                "parameter grid for tuning is empty. If passing custom_grid, "
                "make sure that it is not empty. If not passing custom_grid, "
                "the passed estimator does not have a built-in tuning grid."
            )

        suffixes = []

        if is_stacked_model:
            self.logger.info(
                "Stacked model passed, will tune meta model hyperparameters"
            )
            suffixes.append("final_estimator")

        gc.collect()

        # with estimator_pipeline(self._internal_pipeline, model) as pipeline_with_model:
        if True:

            # fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)

            # fh_param = {"fh": cv.fh}
            # if fit_kwargs is None:
            #     fit_kwargs = fh_param
            # else:
            #     fit_kwargs.update(fh_param)
            fit_kwargs = self.update_fit_kwargs_with_fh_from_cv(
                fit_kwargs=fit_kwargs, cv=cv
            )

            # actual_estimator_label = get_pipeline_estimator_label(pipeline_with_model)
            actual_estimator_label = ""

            # suffixes.append(actual_estimator_label)

            # suffixes = "__".join(reversed(suffixes))

            # param_grid = {f"{suffixes}__{k}": v for k, v in param_grid.items()}

            if estimator_definition is not None:
                search_kwargs = {**estimator_definition.tune_args, **kwargs}
                n_jobs = (
                    self._gpu_n_jobs_param
                    if estimator_definition.is_gpu_enabled
                    else self.n_jobs_param
                )
            else:
                search_kwargs = {}
                n_jobs = self.n_jobs_param

            if custom_grid is not None:
                param_grid = custom_grid
                self.logger.info(f"custom_grid: {param_grid}")

            self.logger.info(f"Tuning with n_jobs={n_jobs}")

            if search_library == "pycaret":
                if search_algorithm == "random":
                    try:
                        param_grid = get_base_distributions(param_grid)
                    except:
                        self.logger.warning(
                            "Couldn't convert param_grid to specific library distributions. Exception:"
                        )
                        self.logger.warning(traceback.format_exc())

            if search_library == "pycaret":
                if search_algorithm == "grid":
                    self.logger.info("Initializing ForecastingGridSearchCV")

                    model_grid = ForecastingGridSearchCV(
                        forecaster=model,
                        cv=cv,
                        param_grid=param_grid,
                        scoring=optimize_metric_dict,
                        refit_metric=refit_metric,
                        n_jobs=n_jobs,
                        verbose=tuner_verbose,
                        refit=False,  # since we will refit afterwards anyway
                        **search_kwargs,
                    )
                elif search_algorithm == "random":
                    self.logger.info("Initializing ForecastingRandomizedGridSearchCV")

                    model_grid = ForecastingRandomizedSearchCV(
                        forecaster=model,
                        cv=cv,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        scoring=optimize_metric_dict,
                        refit_metric=refit_metric,
                        n_jobs=n_jobs,
                        verbose=tuner_verbose,
                        random_state=self.seed,
                        refit=False,  # since we will refit afterwards anyway
                        **search_kwargs,
                    )
                else:
                    raise NotImplementedError(
                        f"Search type '{search_algorithm}' is not supported"
                    )

            model_grid.fit(y=data_y, X=data_X, **fit_kwargs)

            best_params = model_grid.best_params_
            self.logger.info(f"best_params: {best_params}")
            best_params = {**best_params}
            if actual_estimator_label:
                best_params = {
                    k.replace(f"{actual_estimator_label}__", ""): v
                    for k, v in best_params.items()
                }
            cv_results = None
            try:
                cv_results = model_grid.cv_results_
            except:
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

        best_model, model_fit_time = self.create_model(
            estimator=model,
            system=False,
            display=display,
            fold=fold,
            round=round,
            fit_kwargs=fit_kwargs,
            **best_params,
        )
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        if choose_better:
            best_model = self._choose_better(
                [estimator, (best_model, model_results)],
                compare_dimension,
                fold,
                fit_kwargs=fit_kwargs,
                display=display,
            )

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # mlflow logging
        if self.logging_param:

            avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

            try:
                self._mlflow_log_model(
                    model=best_model,
                    model_results=model_results,
                    score_dict=avgs_dict_log,
                    source="tune_model",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    tune_cv_results=cv_results,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {best_model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)
        display.display(model_results, clear=True)

        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(best_model))
        self.logger.info(
            "tune_model() succesfully completed......................................"
        )

        gc.collect()
        if return_tuner:
            return (best_model, model_grid)
        return best_model

    # def ensemble_model(
    #     self,
    #     estimator,
    #     method: str = "Bagging",
    #     fold: Optional[Union[int, Any]] = None,
    #     n_estimators: int = 10,
    #     round: int = 4,
    #     choose_better: bool = False,
    #     optimize: str = "R2",
    #     fit_kwargs: Optional[dict] = None,
    #     verbose: bool = True,
    # ) -> Any:

    #     """
    #         This function ensembles a given estimator. The output of this function is
    #         a score grid with CV scores by fold. Metrics evaluated during CV can be
    #         accessed using the ``get_metrics`` function. Custom metrics can be added
    #         or removed using ``add_metric`` and ``remove_metric`` function.

    #         Example
    #         --------
    #         >>> from pycaret.datasets import get_data
    #         >>> boston = get_data('boston')
    #         >>> from pycaret.regression import *
    #         >>> exp_name = setup(data = boston,  target = 'medv')
    #         >>> dt = create_model('dt')
    #         >>> bagged_dt = ensemble_model(dt, method = 'Bagging')

    #     estimator: scikit-learn compatible object
    #             Trained model object

    #         method: str, default = 'Bagging'
    #             Method for ensembling base estimator. It can be 'Bagging' or 'Boosting'.

    #         fold: int or scikit-learn compatible CV generator, default = None
    #             Controls cross-validation. If None, the CV generator in the ``fold_strategy``
    #             parameter of the ``setup`` function is used. When an integer is passed,
    #             it is interpreted as the 'n_splits' parameter of the CV generator in the
    #             ``setup`` function.

    #         n_estimators: int, default = 10
    #             The number of base estimators in the ensemble. In case of perfect fit, the
    #             learning procedure is stopped early.

    #         round: int, default = 4
    #             Number of decimal places the metrics in the score grid will be rounded to.

    #         choose_better: bool, default = False
    #             When set to True, the returned object is always better performing. The
    #             metric used for comparison is defined by the ``optimize`` parameter.

    #         optimize: str, default = 'R2'
    #             Metric to compare for model selection when ``choose_better`` is True.

    #         fit_kwargs: dict, default = {} (empty dict)
    #             Dictionary of arguments passed to the fit method of the model.

    #         verbose: bool, default = True
    #             Score grid is not printed when verbose is set to False.

    #         Returns:
    #             Trained Model

    #     """

    #     return super().ensemble_model(
    #         estimator=estimator,
    #         method=method,
    #         fold=fold,
    #         n_estimators=n_estimators,
    #         round=round,
    #         choose_better=choose_better,
    #         optimize=optimize,
    #         fit_kwargs=fit_kwargs,
    #         verbose=verbose,
    #     )

    def blend_models(
        self,
        estimator_list: list,
        method: str = "mean",
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "SMAPE",
        weights: Optional[List[float]] = None,
        fit_kwargs: Optional[dict] = None,
        verbose: bool = True,
    ):

        """
        This function trains a EnsembleForecaster for select models passed in the
        ``estimator_list`` param. The output of this function is a score grid with
        CV scores by fold. Metrics evaluated during CV can be accessed using the
        ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)
        >>> top3 = compare_models(n_select = 3)
        >>> blender = blend_models(top3)


        estimator_list: list of sktime compatible estimators
            List of model objects


        method: str, default = 'mean'
            Method to average the individual predictions to form a final prediction.
            Available Methods:

            * 'mean' - Mean of individual predictions
            * 'median' - Median of individual predictions
            * 'voting' - Vote individual predictions based on the provided weights.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'SMAPE'
            Metric to compare for model selection when ``choose_better`` is True.


        weights: list, default = None
            Sequence of weights (float or int) to weight the occurrences of predicted class
            labels (hard voting) or class probabilities before averaging (soft voting). Uses
            uniform weights when None.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained Model


        """

        return super().blend_models(
            estimator_list=estimator_list,
            fold=fold,
            round=round,
            choose_better=choose_better,
            optimize=optimize,
            method=method,
            weights=weights,
            fit_kwargs=fit_kwargs,
            verbose=verbose,
        )

    # def stack_models(
    #     self,
    #     estimator_list: list,
    #     meta_model=None,
    #     fold: Optional[Union[int, Any]] = None,
    #     round: int = 4,
    #     restack: bool = False,
    #     choose_better: bool = False,
    #     optimize: str = "R2",
    #     fit_kwargs: Optional[dict] = None,
    #     verbose: bool = True,
    # ):

    #     """
    #     This function trains a meta model over select estimators passed in
    #     the ``estimator_list`` parameter. The output of this function is a
    #     score grid with CV scores by fold. Metrics evaluated during CV can
    #     be accessed using the ``get_metrics`` function. Custom metrics
    #     can be added or removed using ``add_metric`` and ``remove_metric``
    #     function.

    #     Example
    #     --------
    #     >>> from pycaret.datasets import get_data
    #     >>> boston = get_data('boston')
    #     >>> from pycaret.regression import *
    #     >>> exp_name = setup(data = boston,  target = 'medv')
    #     >>> top3 = compare_models(n_select = 3)
    #     >>> stacker = stack_models(top3)

    #     estimator_list: list of scikit-learn compatible objects
    #         List of trained model objects

    #     meta_model: scikit-learn compatible object, default = None
    #         When None, Linear Regression is trained as a meta model.

    #     fold: int or scikit-learn compatible CV generator, default = None
    #         Controls cross-validation. If None, the CV generator in the ``fold_strategy``
    #         parameter of the ``setup`` function is used. When an integer is passed,
    #         it is interpreted as the 'n_splits' parameter of the CV generator in the
    #         ``setup`` function.

    #     round: int, default = 4
    #         Number of decimal places the metrics in the score grid will be rounded to.

    #     restack: bool, default = False
    #         When set to False, only the predictions of estimators will be used as
    #         training data for the ``meta_model``.

    #     choose_better: bool, default = False
    #         When set to True, the returned object is always better performing. The
    #         metric used for comparison is defined by the ``optimize`` parameter.

    #     optimize: str, default = 'R2'
    #         Metric to compare for model selection when ``choose_better`` is True.

    #     fit_kwargs: dict, default = {} (empty dict)
    #         Dictionary of arguments passed to the fit method of the model.

    #     verbose: bool, default = True
    #         Score grid is not printed when verbose is set to False.

    #     Returns:
    #         Trained Model

    #     """

    #     return super().stack_models(
    #         estimator_list=estimator_list,
    #         meta_model=meta_model,
    #         fold=fold,
    #         round=round,
    #         method="auto",
    #         restack=restack,
    #         choose_better=choose_better,
    #         optimize=optimize,
    #         fit_kwargs=fit_kwargs,
    #         verbose=verbose,
    #     )

    def plot_model(
        self,
        estimator: Optional[Any] = None,
        plot: Optional[str] = None,
        return_data: bool = False,
        verbose: bool = False,
        display_format: Optional[str] = None,
        data_kwargs: Optional[Dict] = None,
        fig_kwargs: Optional[Dict] = None,
        system: bool = True,
        save: Union[str, bool] = False,
    ) -> Tuple[str, Any]:

        """
        This function analyzes the performance of a trained model on holdout set.
        When used without any estimator, this function generates plots on the 
        original data set. When used with an estimator, it will generate plots on 
        the model residuals.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)
        >>> arima = create_model('arima')
        >>> plot_model(plot = 'ts')
        >>> plot_model(plot = 'decomp_classical', data_kwargs = {'type' : 'multiplicative'})
        >>> plot_model(estimator = arima, plot = 'forecast', data_kwargs = {'fh' : 24})


        estimator: sktime compatible object, default = None
            Trained model object


        plot: str, default = None
            Default is 'ts' when estimator is None, When estimator is not None,
            default is changed to 'forecast'. List of available plots (ID - Name):

            * 'ts' - Time Series Plot
            * 'train_test_split' - Train Test Split
            * 'cv' - Cross Validation
            * 'acf' - Auto Correlation (ACF)
            * 'pacf' - Partial Auto Correlation (PACF)
            * 'decomp_classical' - Decomposition Classical
            * 'decomp_stl' - Decomposition STL
            * 'diagnostics' - Diagnostics Plot
            * 'forecast' - "Out-of-Sample" Forecast Plot
            * 'insample' - "In-Sample" Forecast Plot
            * 'residuals' - Residuals Plot


        return_data: bool, default = False
            When set to True, it returns the data for plotting.


        verbose: bool, default = True
            Unused for now


        display_format: str, default = None
            To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
            Currently, not all plots are supported.


        data_kwargs: dict, default = None
            Dictionary of arguments passed to the data for plotting.


        fig_kwargs: dict, default = None
            Dictionary of arguments passed to the figure object of plotly. Example:
            * fig_kwargs = {'fig_size' : [800, 500], 'fig_template' : 'simple_white'}


        save: string or bool, default = False
            When set to True, Plot is saved as a 'png' file in current working directory.
            When a path destination is given, Plot is saved as a 'png' file the given path to the directory of choice.


        Returns:
            None

        """
        # checking display_format parameter
        self.plot_model_check_display_format_(display_format=display_format)

        # Import required libraries ----
        if display_format == "streamlit":
            try:
                import streamlit as st
            except ImportError:
                raise ImportError(
                    "It appears that streamlit is not installed. Do: pip install hpbandster ConfigSpace"
                )

        if data_kwargs is None:
            data_kwargs = {}
        if fig_kwargs is None:
            fig_kwargs = {}

        available_plots_common = [
            "ts",
            "train_test_split",
            "cv",
            "acf",
            "pacf",
            "diagnostics",
            "decomp_classical",
            "decomp_stl",
        ]
        available_plots_data = available_plots_common
        available_plots_model = available_plots_common + [
            "forecast",
            "insample",
            "residuals",
        ]

        return_pred_int = False

        # Type checks
        if estimator is not None and isinstance(estimator, str):
            raise ValueError(
                "Estimator must be a trained object. "
                f"You have passed a string: '{estimator}'"
            )

        # Default plot when no model is specified is the time series plot
        # Default plot when model is specified is the forecast plot
        if plot is None and estimator is None:
            plot = "ts"
        elif plot is None and estimator is not None:
            plot = "forecast"

        data, train, test, predictions, cv, model_name = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

        if plot == "ts":
            data = self._get_y_data(split="all")
        elif plot == "train_test_split":
            train = self._get_y_data(split="train")
            test = self._get_y_data(split="test")
        elif plot == "cv":
            data = self._get_y_data(split="train")
            cv = self.get_fold_generator()
        elif estimator is None:
            require_full_data = [
                "acf",
                "pacf",
                "diagnostics",
                "decomp_classical",
                "decomp_stl",
            ]
            if plot in require_full_data:
                data = self._get_y_data(split="all")
            else:
                plots_formatted_data = [f"'{plot}'" for plot in available_plots_data]
                raise ValueError(
                    f"Plot type '{plot}' is not supported when estimator is not provided. Available plots are: {', '.join(plots_formatted_data)}"
                )
        else:
            # Estimator is Provided

            if hasattr(self, "_get_model_name") and hasattr(
                self, "_all_models_internal"
            ):
                model_name = self._get_model_name(estimator)
            else:
                # If the model is saved and loaded afterwards,
                # it will not have self._get_model_name
                model_name = estimator.__class__.__name__

            require_insample_predictions = ["insample"]
            require_residuals = [
                "residuals",
                "diagnostics",
                "acf",
                "pacf",
                "decomp_classical",
                "decomp_stl",
            ]
            if plot == "forecast":
                data = self._get_y_data(split="all")

                fh = data_kwargs.get("fh", None)
                alpha = data_kwargs.get("alpha", 0.05)
                return_pred_int = estimator.get_tag("capability:pred_int")
                predictions = self.predict_model(
                    estimator,
                    fh=fh,
                    alpha=alpha,
                    return_pred_int=return_pred_int,
                    verbose=False,
                )
            elif plot in require_insample_predictions:
                # Try to get insample forecasts if possible
                insample_predictions = self.get_insample_predictions(
                    estimator=estimator
                )
                if insample_predictions is None:
                    return
                predictions = insample_predictions
                data = self._get_y_data(split="all")
                # Do not plot prediction interval for insample predictions
                return_pred_int = False

            elif plot in require_residuals:
                resid = self.get_residuals(estimator=estimator)
                if resid is None:
                    return
                resid = self.check_and_clean_resid(resid=resid)
                data = resid
            else:
                plots_formatted_model = [f"'{plot}'" for plot in available_plots_model]
                raise ValueError(
                    f"Plot type '{plot}' is not supported when estimator is provided. Available plots are: {', '.join(plots_formatted_model)}"
                )

        fig, plot_data = plot_(
            plot=plot,
            data=data,
            train=train,
            test=test,
            predictions=predictions,
            cv=cv,
            model_name=model_name,
            return_pred_int=return_pred_int,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )

        plot_name = self._available_plots[plot]
        plot_filename = f"{plot_name}.html"

        # Per https://github.com/pycaret/pycaret/issues/1699#issuecomment-962460539
        if save:
            if not isinstance(save, bool):
                plot_filename = os.path.join(save, plot_filename)

            self.logger.info(f"Saving '{plot_filename}'")
            fig.write_html(plot_filename)

            if return_data:
                return plot_filename, plot_data
            else:
                return plot_filename

        elif system:
            if display_format == "streamlit":
                st.write(fig)
                return fig
            # else:
            #     fig.show()
            self.logger.info("Visual Rendered Successfully")

            if return_data:
                return fig, plot_data
            else:
                return fig

        # if return_data:
        #     return plot_filename, plot_data
        # else:
        #     return plot_filename

    # def evaluate_model(
    #     self,
    #     estimator,
    #     fold: Optional[Union[int, Any]] = None,
    #     fit_kwargs: Optional[dict] = None,
    #     use_train_data: bool = False,
    # ):

    #     """
    #     This function displays a user interface for analyzing performance of a trained
    #     model. It calls the ``plot_model`` function internally.

    #     Example
    #     --------
    #     >>> from pycaret.datasets import get_data
    #     >>> boston = get_data('boston')
    #     >>> from pycaret.regression import *
    #     >>> exp_name = setup(data = boston,  target = 'medv')
    #     >>> lr = create_model('lr')
    #     >>> evaluate_model(lr)

    #     estimator: scikit-learn compatible object
    #         Trained model object

    #     fold: int or scikit-learn compatible CV generator, default = None
    #         Controls cross-validation. If None, the CV generator in the ``fold_strategy``
    #         parameter of the ``setup`` function is used. When an integer is passed,
    #         it is interpreted as the 'n_splits' parameter of the CV generator in the
    #         ``setup`` function.

    #     fit_kwargs: dict, default = {} (empty dict)
    #         Dictionary of arguments passed to the fit method of the model.

    #     use_train_data: bool, default = False
    #         When set to true, train data will be used for plots, instead
    #         of test data.

    #     Returns:
    #         None

    #     Warnings
    #     --------
    #     -   This function only works in IPython enabled Notebook.

    #     """

    #     return super().evaluate_model(
    #         estimator=estimator,
    #         fold=fold,
    #         fit_kwargs=fit_kwargs,
    #         use_train_data=use_train_data,
    #     )

    # def interpret_model(
    #     self,
    #     estimator,
    #     plot: str = "summary",
    #     feature: Optional[str] = None,
    #     observation: Optional[int] = None,
    #     use_train_data: bool = False,
    #     **kwargs,
    # ):

    #     """
    #     This function analyzes the predictions generated from a trained model. Most plots
    #     in this function are implemented based on the SHAP (SHapley Additive exPlanations).
    #     For more info on this, please see https://shap.readthedocs.io/en/latest/

    #     Example
    #     --------
    #     >>> from pycaret.datasets import get_data
    #     >>> boston = get_data('boston')
    #     >>> from pycaret.regression import *
    #     >>> exp = setup(data = boston,  target = 'medv')
    #     >>> xgboost = create_model('xgboost')
    #     >>> interpret_model(xgboost)

    #     estimator: scikit-learn compatible object
    #         Trained model object

    #     plot: str, default = 'summary'
    #         List of available plots (ID - Name):
    #         * 'summary' - Summary Plot using SHAP
    #         * 'correlation' - Dependence Plot using SHAP
    #         * 'reason' - Force Plot using SHAP
    #         * 'pdp' - Partial Dependence Plot

    #     feature: str, default = None
    #         Feature to check correlation with. This parameter is only required when ``plot``
    #         type is 'correlation' or 'pdp'. When set to None, it uses the first column from
    #         the dataset.

    #     observation: int, default = None
    #         Observation index number in holdout set to explain. When ``plot`` is not
    #         'reason', this parameter is ignored.

    #     use_train_data: bool, default = False
    #         When set to true, train data will be used for plots, instead
    #         of test data.

    #     **kwargs:
    #         Additional keyword arguments to pass to the plot.

    #     Returns:
    #         None

    #     """

    #     return super().interpret_model(
    #         estimator=estimator,
    #         plot=plot,
    #         feature=feature,
    #         observation=observation,
    #         use_train_data=use_train_data,
    #         **kwargs,
    #     )

    def predict_model(
        self,
        estimator,
        fh=None,
        return_pred_int=False,
        alpha=0.05,
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:

        """
        This function forecast using a trained model. When ``fh`` is None, 
        it forecasts using the same forecast horizon used during the 
        training.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)
        >>> arima = create_model('arima')
        >>> pred_holdout = predict_model(arima)
        >>> pred_unseen = predict_model(finalize_model(arima), fh = 24)


        estimator: sktime compatible object
            Trained model object


        fh: int, default = None
            Number of points from the last date of training to forecast.
            When fh is None, it forecasts using the same forecast horizon 
            used during the training.


        return_pred_int: bool, default = False
            When set to True, it returns lower bound and upper bound
            prediction interval, in addition to the point prediction.


        alpha: float, default = 0.05
            alpha for prediction interval. CI = 1 - alpha.


        round: int, default = 4
            Number of decimal places to round predictions to.


        verbose: bool, default = True
            When set to False, holdout score grid is not printed.


        Returns:
            pandas.DataFrame


        """

        data = None  # TODO: Add back when we have support for multivariate TS

        estimator_ = deep_clone(estimator)

        loaded_in_same_env = True
        # Check if loaded in a different environment
        if not hasattr(self, "X_test") or fh is not None:
            # If the model is saved and loaded afterwards,
            # it will not have self.X_test

            # Also do not display metrics if user provides own fh
            # (even if it is same as test set horizon) per
            # https://github.com/pycaret/pycaret/issues/1702
            loaded_in_same_env = False
            verbose = False

        if fh is not None:
            # Do not display metrics if user provides own fh
            # (even if it is same as test set horizon) per
            # https://github.com/pycaret/pycaret/issues/1702
            verbose = False

        if fh is None:
            if not hasattr(self, "fh"):
                # If the model is saved and loaded afterwards,
                # it will not have self.fh
                fh = estimator_.fh
        else:
            # Get the fh in the right format for sktime
            fh = self.check_fh(fh)

        try:
            return_vals = estimator_.predict(
                X=data, fh=fh, return_pred_int=return_pred_int, alpha=alpha
            )
        except NotImplementedError as error:
            self.logger.warning(error)
            self.logger.warning(
                "Most likely, prediction intervals has not been implemented for this "
                "algorithm. Predcition will be run with `return_pred_int` = False, and "
                "NaN values will be returned for the prediction intervals instead."
            )
            return_vals = estimator_.predict(
                X=data, fh=fh, return_pred_int=False, alpha=alpha
            )
        if isinstance(return_vals, tuple):
            # Prediction Interval is returned
            #   First Value is a series of predictions
            #   Second Value is a dataframe of lower and upper bounds
            # result = pd.DataFrame(return_vals[0], columns=["y_pred"])
            # result = result.join(return_vals[1])
            result = pd.concat(return_vals, axis=1)
            result.columns = ["y_pred", "lower", "upper"]
        else:
            # Prediction interval is not returned (not implemented)
            if return_pred_int:
                result = pd.DataFrame(return_vals, columns=["y_pred"])
                result["lower"] = np.nan
                result["upper"] = np.nan
            else:
                # Leave as series
                result = return_vals
                if result.name is None:
                    if hasattr(self, "y"):
                        result.name = self.y.name
                    else:
                        # If the model is saved and loaded afterwards,
                        # it will not have self.y
                        pass

        # Converting to float since rounding does not support int
        result = result.astype(float).round(round)

        if isinstance(result.index, pd.DatetimeIndex):
            result.index = (
                result.index.to_period()
            )  # Prophet with return_pred_int = True returns datetime index.

        #################
        #### Metrics ####
        #################
        # Only display if loaded in same environment

        # This is not technically y_test_pred in all cases.
        # If the model has not been finalized, y_test_pred will match the indices from y_test
        # If the model has been finalized, y_test_pred will not match the indices from y_test
        # Also, the user can use a different fh length in predict in which case the length
        # of y_test_pred will not match y_test.

        if loaded_in_same_env:
            X_test_ = self.X_test.copy()
            # Some predict methods in sktime expect None (not an empty dataframe as
            # returned by pycaret). Hence converting to None.
            if X_test_.shape[0] == 0 or X_test_.shape[1] == 0:
                X_test_ = None
            y_test_ = self.y_test.copy()

            y_test_pred = estimator_.predict(
                X=X_test_, fh=fh, return_pred_int=False, alpha=alpha
            )
            if len(y_test_pred) != len(y_test_):
                msg = (
                    "predict_model >> Forecast Horizon does not match the horizon length "
                    "used during training. Metrics will not be displayed."
                )
                self.logger.warning(msg)
                verbose = False

            # concatenates by index
            y_test_and_pred = pd.concat([y_test_pred, y_test_], axis=1)
            # Removes any indices that do not match
            y_test_and_pred.dropna(inplace=True)
            y_test_pred_common = y_test_and_pred[y_test_and_pred.columns[0]]
            y_test_common = y_test_and_pred[y_test_and_pred.columns[1]]

            if len(y_test_and_pred) == 0:
                self.logger.warning(
                    "predict_model >> No indices matched between test set and prediction. "
                    "You are most likely calling predict_model after finalizing model. "
                    "Metrics will not be displayed"
                )
                metrics = self._calculate_metrics(y_test=[], pred=[], pred_prob=None)  # type: ignore
                metrics = {metric_name: np.nan for metric_name, _ in metrics.items()}
                verbose = False
            else:
                metrics = self._calculate_metrics(y_test=y_test_common, pred=y_test_pred_common, pred_prob=None)  # type: ignore

            # Display Test Score
            # model name
            display = None
            try:
                np.random.seed(self.seed)
                if not display:
                    display = Display(verbose=verbose, html_param=self.html_param,)
            except:
                display = Display(verbose=False, html_param=False,)

            full_name = self._get_model_name(estimator_)
            df_score = pd.DataFrame(metrics, index=[0])
            df_score.insert(0, "Model", full_name)
            df_score = df_score.round(round)
            display.display(df_score.style.set_precision(round), clear=False)

            # store predictions on hold-out in display_container
            if df_score is not None:
                self.display_container.append(df_score)

        gc.collect()

        return result

    def finalize_model(
        self, estimator, fit_kwargs: Optional[dict] = None, model_only: bool = True,
    ) -> Any:

        """
        This function trains a given estimator on the entire dataset including the
        holdout set.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> data = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = data, fh = 12)
        >>> arima = create_model('arima')
        >>> final_arima = finalize_model(arima)


        estimator: sktime compatible object
            Trained model object


        fit_kwargs: dict, default = None
            Dictionary of arguments passed to the fit method of the model.


        model_only: bool, default = True
            Parameter not in use for now. Behavior may change in future.


        Returns:
            Trained Model


        """

        return super().finalize_model(
            estimator=estimator, fit_kwargs=fit_kwargs, model_only=model_only,
        )

    def deploy_model(
        self, model, model_name: str, authentication: dict, platform: str = "aws",
    ):

        """
        This function deploys the transformation pipeline and trained model on cloud.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> data = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = data, fh = 12)
        >>> arima = create_model('arima')
        >>> deploy_model(model = arima, model_name = 'arima-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})


        Amazon Web Service (AWS) users:
            To deploy a model on AWS S3 ('aws'), environment variables must be set in your
            local environment. To configure AWS environment variables, type ``aws configure``
            in the command line. Following information from the IAM portal of amazon console
            account is required:

            - AWS Access Key ID
            - AWS Secret Key Access
            - Default Region Name (can be seen under Global settings on your AWS console)

            More info: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html


        Google Cloud Platform (GCP) users:
            To deploy a model on Google Cloud Platform ('gcp'), project must be created
            using command line or GCP console. Once project is created, you must create
            a service account and download the service account key as a JSON file to set
            environment variables in your local environment.

            More info: https://cloud.google.com/docs/authentication/production


        Microsoft Azure (Azure) users:
            To deploy a model on Microsoft Azure ('azure'), environment variables for connection
            string must be set in your local environment. Go to settings of storage account on
            Azure portal to access the connection string required.

            More info: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json


        model: scikit-learn compatible object
            Trained model object


        model_name: str
            Name of model.


        authentication: dict
            Dictionary of applicable authentication tokens.

            When platform = 'aws':
            {'bucket' : 'S3-bucket-name', 'path': (optional) folder name under the bucket}

            When platform = 'gcp':
            {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

            When platform = 'azure':
            {'container': 'azure-container-name'}


        platform: str, default = 'aws'
            Name of the platform. Currently supported platforms: 'aws', 'gcp' and 'azure'.


        Returns:
            None

        """

        return super().deploy_model(
            model=model,
            model_name=model_name,
            authentication=authentication,
            platform=platform,
        )

    def save_model(
        self, model, model_name: str, model_only: bool = True, verbose: bool = True
    ):

        """
        This function saves the transformation pipeline and trained model object
        into the current working directory as a pickle file for later use.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> data = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = data, fh = 12)
        >>> arima = create_model('arima')
        >>> save_model(arima, 'saved_arima_model')


        model: sktime compatible object
            Trained model object


        model_name: str
            Name of the model.


        model_only: bool, default = True
            Parameter not in use for now. Behavior may change in future.


        verbose: bool, default = True
            Success message is not printed when verbose is set to False.


        Returns:
            Tuple of the model object and the filename.

        """

        return super().save_model(
            model=model, model_name=model_name, model_only=model_only, verbose=verbose
        )

    def load_model(
        self,
        model_name,
        platform: Optional[str] = None,
        authentication: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):

        """
        This function loads a previously saved pipeline/model.

        Example
        -------
        >>> from pycaret.time_series import load_model
        >>> saved_arima = load_model('saved_arima_model')


        model_name: str
            Name of the model.


        platform: str, default = None
            Name of the cloud platform. Currently supported platforms:
            'aws', 'gcp' and 'azure'.


        authentication: dict, default = None
            dictionary of applicable authentication tokens.

            when platform = 'aws':
            {'bucket' : 'S3-bucket-name'}

            when platform = 'gcp':
            {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

            when platform = 'azure':
            {'container': 'azure-container-name'}


        verbose: bool, default = True
            Success message is not printed when verbose is set to False.


        Returns:
            Trained Model

        """

        return super().load_model(
            model_name=model_name,
            platform=platform,
            authentication=authentication,
            verbose=verbose,
        )

    # def automl(
    #     self, optimize: str = "R2", use_holdout: bool = False, turbo: bool = True
    # ) -> Any:

    #     """
    #     This function returns the best model out of all trained models in
    #     current session based on the ``optimize`` parameter. Metrics
    #     evaluated can be accessed using the ``get_metrics`` function.

    #     Example
    #     -------
    #     >>> from pycaret.datasets import get_data
    #     >>> boston = get_data('boston')
    #     >>> from pycaret.regression import *
    #     >>> exp_name = setup(data = boston,  target = 'medv')
    #     >>> top3 = compare_models(n_select = 3)
    #     >>> tuned_top3 = [tune_model(i) for i in top3]
    #     >>> blender = blend_models(tuned_top3)
    #     >>> stacker = stack_models(tuned_top3)
    #     >>> best_mae_model = automl(optimize = 'MAE')

    #     optimize: str, default = 'R2'
    #         Metric to use for model selection. It also accepts custom metrics
    #         added using the ``add_metric`` function.

    #     use_holdout: bool, default = False
    #         When set to True, metrics are evaluated on holdout set instead of CV.

    #     turbo: bool, default = True
    #         When set to True and use_holdout is False, only models created with default fold
    #         parameter will be considered. If set to False, models created with a non-default
    #         fold parameter will be scored again using default fold settings, so that they can be
    #         compared.

    #     Returns:
    #         Trained Model

    #     """

    #     return super().automl(optimize=optimize, use_holdout=use_holdout, turbo=turbo)

    def models(
        self,
        type: Optional[str] = None,
        internal: bool = False,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        """
        Returns table of models available in the model library.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> data = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = data, fh = 12)
        >>> models()


        type: str, default = None
            - baseline : filters and only return baseline models
            - classical : filters and only return classical models
            - linear : filters and only return linear models
            - tree : filters and only return tree based models
            - neighbors : filters and only return neighbors models


        internal: bool, default = False
            When True, will return extra columns and rows used internally.


        raise_errors: bool, default = True
            When False, will suppress all exceptions, ignoring models
            that couldn't be created.


        Returns:
            pandas.DataFrame

        """
        self.logger.info(f"gpu_param set to {self.gpu_param}")

        model_types = list(TSModelTypes)

        if type:
            try:
                type = TSModelTypes(type)
            except ValueError:
                raise ValueError(
                    f"type parameter only accepts: {', '.join([x.value for x in TSModelTypes.__members__.values()])}."
                )

            model_types = [type]

        _, model_containers = self._get_models(raise_errors)

        model_containers = {
            k: v for k, v in model_containers.items() if v.model_type in model_types
        }

        rows = [
            v.get_dict(internal)
            for k, v in model_containers.items()
            if (internal or not v.is_special)
        ]

        df = pd.DataFrame(rows)
        df.set_index("ID", inplace=True, drop=True)

        return df

    def get_metrics(
        self,
        reset: bool = False,
        include_custom: bool = True,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        """
        Returns table of available metrics used for CV.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)
        >>> all_metrics = get_metrics()


        reset: bool, default = False
            When True, will reset all changes made using the ``add_metric``
            and ``remove_metric`` function.


        include_custom: bool, default = True
            Whether to include user added (custom) metrics or not.


        raise_errors: bool, default = True
            If False, will suppress all exceptions, ignoring models that
            couldn't be created.


        Returns:
            pandas.DataFrame

        """

        return super().get_metrics(
            reset=reset, include_custom=include_custom, raise_errors=raise_errors,
        )

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: type,
        greater_is_better: bool = True,
        **kwargs,
    ) -> pd.Series:

        """
        Adds a custom metric to be used for CV.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)
        >>> from sklearn.metrics import explained_variance_score
        >>> add_metric('evs', 'EVS', explained_variance_score)


        id: str
            Unique id for the metric.


        name: str
            Display name of the metric.


        score_func: type
            Score function (or loss function) with signature ``score_func(y, y_pred, **kwargs)``.


        greater_is_better: bool, default = True
            Whether ``score_func`` is higher the better or not.


        **kwargs:
            Arguments to be passed to score function.


        Returns:
            pandas.Series

        """

        return super().add_metric(
            id=id,
            name=name,
            score_func=score_func,
            target="pred",
            greater_is_better=greater_is_better,
            **kwargs,
        )

    def remove_metric(self, name_or_id: str):

        """
        Removes a metric from CV.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> data = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = data, fh = 12)
        >>> remove_metric('MAPE')


        name_or_id: str
            Display name or ID of the metric.


        Returns:
            None

        """
        return super().remove_metric(name_or_id=name_or_id)

    def get_logs(
        self, experiment_name: Optional[str] = None, save: bool = False
    ) -> pd.DataFrame:

        """
        Returns a table of experiment logs. Only works when ``log_experiment``
        is True when initializing the ``setup`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> data = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = data, fh = 12)
        >>> best = compare_models()
        >>> exp_logs = get_logs()   


        experiment_name: str, default = None
            When None current active run is used.


        save: bool, default = False
            When set to True, csv file is saved in current working directory.


        Returns:
            pandas.DataFrame

        """

        return super().get_logs(experiment_name=experiment_name, save=save)

    def get_fold_generator(
        self,
        fold: Optional[Union[int, Any]] = None,
        fold_strategy: Optional[str] = None,
    ) -> Union[ExpandingWindowSplitter, SlidingWindowSplitter]:
        """Returns the cv object based on number of folds and fold_strategy

        Parameters
        ----------
        fold : Optional[Union[int, Any]]
            The number of folds (int), by default None which returns the fold generator
            (cv object) defined during setup. Could also be a sktime cross-validation object.
            If it is a sktime cross-validation object, it is simply returned back
        fold_strategy : Optional[str], optional
            The fold strategy - 'expanding' or 'sliding', by default None which
            takes the strategy set during `setup`

        Returns
        -------
        Union[ExpandingWindowSplitter, SlidingWindowSplitter]
            The cross-validation object

        Raises
        ------
        ValueError
            If not enough data points to support the number of folds requested
        """
        # cross validation setup starts here
        if fold is None:
            # Get cv object defined during setup
            if self.fold_generator is None:
                raise ValueError(
                    "Trying to retrieve Fold Generator but this has not been defined yet."
                )
            fold_generator = self.fold_generator
        elif not isinstance(fold, int):
            return fold  # assumes fold is an sktime compatible cross-validation object
        else:
            # Get new cv object based on the fold parameter
            y_size = len(self.y_train)
            window_length = len(self.fh)
            step_length = len(self.fh)
            initial_window = y_size - (fold * window_length)

            if initial_window < 1:
                raise ValueError(
                    "Not Enough Data Points, set a lower number of folds or fh"
                )

            # If None, get the strategy defined in the setup (e.g. `expanding`, 'sliding`, etc.)
            if fold_strategy is None:
                fold_strategy = self.fold_strategy

            if fold_strategy == "expanding" or fold_strategy == "rolling":
                fold_generator = ExpandingWindowSplitter(
                    initial_window=initial_window,
                    step_length=step_length,
                    # window_length=window_length,
                    fh=self.fh,
                    start_with_window=True,
                )

            if fold_strategy == "sliding":
                fold_generator = SlidingWindowSplitter(
                    # initial_window=initial_window,
                    step_length=step_length,
                    window_length=initial_window,
                    fh=self.fh,
                    start_with_window=True,
                )
        return fold_generator

    def check_stats(
        self,
        estimator: Optional[Any] = None,
        test: str = "all",
        alpha: float = 0.05,
        split: str = "all",
    ) -> pd.DataFrame:
        #### Step 1: Get the data to be tested ----
        if estimator is None:
            data = self._get_y_data(split=split)
        else:
            data = self.get_residuals(estimator=estimator)
            if data is None:
                return
            data = self.check_and_clean_resid(resid=data)

        #### Step 2: Test ----
        results = test_(data=data, test=test, alpha=alpha)
        results.reset_index(inplace=True, drop=True)
        return results

    def _get_y_data(self, split="all"):
        if split == "all":
            data = self.y
        elif split == "train":
            data = self.y_train
        elif split == "test":
            data = self.y_test
        else:
            raise ValueError(f"split value: '{split}' is not supported.")
        return data

    def get_residuals(self, estimator) -> Optional[pd.Series]:
        # https://github.com/alan-turing-institute/sktime/issues/1105#issuecomment-932216820
        resid = None

        estimator.check_is_fitted()
        estimator_ = deep_clone(estimator)
        y_used_to_train = estimator_._y
        try:
            resid = y_used_to_train - estimator_.predict(
                ForecastingHorizon(y_used_to_train.index, is_relative=False)
            )
        except NotImplementedError as exception:
            self.logger.warning(exception)
            print(
                "In sample predictions has not been implemented for this estimator "
                f"of type '{estimator_.__class__.__name__}' in `sktime`. When "
                "this is implemented, it will be enabled by default in pycaret."
            )

        return resid

    def get_insample_predictions(self, estimator) -> Optional[pd.Series]:
        # https://github.com/alan-turing-institute/sktime/issues/1105#issuecomment-932216820
        insample_predictions = None

        estimator.check_is_fitted()
        estimator_ = deep_clone(estimator)
        y_used_to_train = estimator_._y
        try:
            insample_predictions = self.predict_model(
                estimator, fh=-np.arange(0, len(y_used_to_train))
            )
        except NotImplementedError as exception:
            self.logger.warning(exception)
            print(
                "In sample predictions has not been implemented for this estimator "
                f"of type '{estimator_.__class__.__name__}' in `sktime`. When "
                "this is implemented, it will be enabled by default in pycaret."
            )

        return insample_predictions

    def check_and_clean_resid(self, resid: pd.Series) -> pd.Series:
        """Checks to see if the residuals matches one of the test set or
        full dataset. If it does, it resturns the residuals without the NA values.

        Parameters
        ----------
        resid : pd.Series
            Residuals from an estimator

        Returns
        -------
        pd.Series
            Cleaned Residuals

        Raises
        ------
        ValueError
          If any one of these 3 conditions are satisfied:
            1. If residual length matches the length of train set but indices do not
            2. If residual length matches the length of full data set but indices do not
            3. If residual length does not match either train OR full dataset
        """
        y_train = self._get_y_data(split="train")
        y_all = self._get_y_data(split="all")

        if len(resid.index) == len(y_train.index):
            if np.all(resid.index != y_train.index):
                raise ValueError(
                    "Residuals match the length of the train set, but indices do not match up..."
                )
        elif len(resid.index) == len(y_all.index):
            if np.all(resid.index != y_all.index):
                raise ValueError(
                    "Residuals match the length of the full data set, but indices do not match up..."
                )
        else:
            raise ValueError(
                "Residuals time points do not match either test set or full dataset."
            )
        resid.dropna(inplace=True)
        return resid


# TODO: Add to pycaret utils or some common location
def deep_clone(estimator):
    # Cloning since setting fh to another value replaces it inplace
    # Note cloning does not copy the fitted model (only model hyperparams)
    # Hence, we need to do deep copy per
    # https://stackoverflow.com/a/33576345/8925915
    estimator_ = deepcopy(estimator)
    return estimator_
