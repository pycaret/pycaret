from pycaret.internal.pycaret_experiment.utils import highlight_setup, MLUsecase
from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)
from pycaret.internal.pipeline import (
    estimator_pipeline,
    get_pipeline_fit_kwargs,
)
from pycaret.internal.utils import color_df
from pycaret.internal.utils import SeasonalPeriod
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
import pandas as pd  # type ignore
from pandas.io.formats.style import Styler
import numpy as np  # type: ignore
import datetime
import time
import gc
from sklearn.base import clone  # type: ignore
from typing import List, Tuple, Any, Union, Optional, Dict
import warnings
from IPython.utils import io
import traceback
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore


warnings.filterwarnings("ignore")
LOGGER = get_logger()


class TimeSeriesExperiment(_SupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.TIME_SERIES
        self.exp_name_log = "ts-default-name"
        self._available_plots = {}
        # Values in variable_keys are accessible in globals
        self.variable_keys = self.variable_keys.difference(
            {
                "X",
                # "X_train",
                "X_test",
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
            {"fh", "seasonal_period", "seasonality_present"}
        )
        return

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
                    ["CPU Jobs", self.n_jobs_param],
                    ["Use GPU", self.gpu_param],
                    ["Log Experiment", self.logging_param],
                    ["Experiment Name", self.exp_name_log],
                    ["USI", self.USI],
                ]
            )
            + (
                [
                    ["Imputation Type", kwargs["imputation_type"]],
                ]
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
        all_models_internal = (
            pycaret.containers.models.time_series.get_all_model_containers(
                self.variables, raise_errors=raise_errors
            )
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
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        custom_pipeline: Union[
            Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]
        ] = None,
        html: bool = True,
        session_id: Optional[int] = None,
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
        two mandatory parameters: ``data`` and ``target``. All the other parameters are
        optional.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = boston,  target = 'medv')


        data : pandas.Series or pandas.DataFrame
            Shape (n_samples, 1), where n_samples is the number of samples.


        fh: np.array, default = None
            The forecast horizon to be used for forecasting. User must specify a value.
            The values of the array must be integers specifying the lookahead points that
            must be forecasted. e.g. np.array([2, 5]) will forecast 2 and 5 points ahead.
            Default value of None will result in an error.


        preprocess: bool, default = True
            When set to False, no transformations are applied except for train_test_split
            and custom transformations passed in ``custom_pipeline`` param. Data must be
            ready for modeling (no missing values, no dates, categorical data encoding),
            when preprocess is set to False.


        imputation_type: str, default = 'simple'
            The type of imputation to use. Can be either 'simple' or 'iterative'.


        fold_strategy: str or sklearn CV generator object, default = 'expanding'
            Choice of cross validation strategy. Possible values are:

            * 'expanding'
            * 'rolling' (same as/aliased to 'expanding')
            * 'sliding'

            You can also pass an sktime compatible cross validation object such
            as SlidingWindowSplitter or ExpandingWindowSplitter. In this case,
            the `fold` and `fh` parameters will be ignored and these values will
            be extracted from the fold_strategy object directly.


        fold: int, default = 3
            Number of folds to be used in cross validation. Must be at least 2. This is
            a global setting that can be over-written at function level by using ``fold``
            parameter. Ignored when ``fold_strategy`` is a custom object.


        fh: int, list or np.array, default = 1
            Number of steps ahead to take to evaluate forecast.


        seasonal_period: int or str, default = None
            Seasonal periods in timeseries data. If not provided the frequency of the data
            index is map to a seasonal period as follows:

            * "S": 60
            * "T": 60
            * 'H': 24
            * 'D': 7
            * 'W': 52
            * 'M': 12
            * 'Q': 4
            * 'A': 1
            * 'Y': 1

            Alternatively you can provide a custom `seasonal_parameter` by passing
            it as an integer.

            NOTE: If data index is not of type pd.core.indexes.period.PeriodIndex,
            then seasonal_period MUST be passed. Refer to the mapping above for
            a guide of what values to use depending on the frequency of the data.
            If your data does not have any seasonality, then set seasonal_period = 1.


        n_jobs: int, default = -1
            The number of jobs to run in parallel (for functions that supports parallel
            processing) -1 means using all processors. To run all functions on single
            processor set n_jobs to None.


        use_gpu: bool or str, default = False
            When set to True, it will use GPU for training with algorithms that support it,
            and fall back to CPU if they are unavailable. When set to 'force', it will only
            use GPU-enabled algorithms and raise exceptions when they are unavailable. When
            False, all algorithms are trained using CPU only.

            GPU enabled algorithms:

            - Extreme Gradient Boosting, requires no further installation

            - CatBoost Regressor, requires no further installation
            (GPU is only enabled when data > 50,000 rows)

            - Light Gradient Boosting Machine, requires GPU installation
            https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html

            - Linear Regression, Lasso Regression, Ridge Regression, K Neighbors Regressor,
            Random Forest, Support Vector Regression, Elastic Net requires cuML >= 0.15
            https://github.com/rapidsai/cuml


        custom_pipeline: (str, transformer) or list of (str, transformer), default = None
            When passed, will append the custom transformers in the preprocessing pipeline
            and are applied on each CV fold separately and on the final fit. All the custom
            transformations are applied after 'train_test_split' and before pycaret's internal
            transformations.


        html: bool, default = True
            When set to False, prevents runtime display of monitor. This must be set to False
            when the environment does not support IPython. For example, command line terminal,
            Databricks Notebook, Spyder and other similar IDEs.


        session_id: int, default = None
            Controls the randomness of experiment. It is equivalent to 'random_state' in
            scikit-learn. When None, a pseudo random number is generated. This can be used
            for later reproducibility of the entire experiment.


        log_experiment: bool, default = False
            When set to True, all metrics and parameters are logged on the ``MLFlow`` server.


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

        # Forecast Horizon Checks
        if fh is None and isinstance(fold_strategy, str):
            raise ValueError(
                f"The forecast horizon `fh` must be provided when fold_strategy is of type 'string'"
            )

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
        self.fh = fh

        allowed_freq_index_types = (pd.PeriodIndex, pd.DatetimeIndex)
        if (
            not isinstance(data.index, allowed_freq_index_types)
            and seasonal_period is None
        ):
            raise ValueError(
                f"The index of your 'data' is of type '{type(data.index)}'. "
                f"If the 'data' index is not of one of the following types: {', '.join(allowed_freq_index_types)}, "
                "then 'seasonal_period' must be provided. Refer to docstring for options."
            )

        if seasonal_period is None:

            index_freq = data.index.freqstr
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

        if isinstance(data, (pd.Series, pd.DataFrame)):
            if isinstance(data, pd.DataFrame):
                if data.shape[1] != 1:
                    raise ValueError(
                        f"data must be a pandas Series or DataFrame with one column, got {data.shape[1]} columns!"
                    )
                data = data.copy()
            else:
                data = pd.DataFrame(data)  # Force convertion to DataFrame
        else:
            raise ValueError(
                f"data must be a pandas Series or DataFrame, got object of {type(data)} type!"
            )

        data.columns = [str(x) for x in data.columns]

        if not np.issubdtype(data[data.columns[0]].dtype, np.number):
            raise TypeError(
                f"Data must be of 'numpy.number' subtype, got {data[data.columns[0]].dtype}!"
            )

        if len(data.index) != len(set(data.index)):
            raise ValueError("Index may not have duplicate values!")

        # check valid seasonal parameter
        valid_seasonality = autocorrelation_seasonality_test(
            data[data.columns[0]], self.seasonal_period
        )

        self.seasonality_present = True if valid_seasonality else False

        return super().setup(
            data=data,
            target=data.columns[0],
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
        groups: Optional[Union[str, Any]] = None,
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
        >>> from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
        >>> airline = get_data('airline', verbose=False)
        >>> fh, fold = np.arange(1,13), 3
        >>> exp = TimeSeriesExperiment()
        >>> exp.setup(data=airline, fh=fh, fold=fold)
        >>> master_display_exp = exp.compare_models(fold=fold, sort='mape')


        include: list of str or scikit-learn compatible object, default = None
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


        sort: str, default = 'smape'
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


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when 'GroupKFold' is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in the training dataset. When string is passed, it is interpreted
            as the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained model or list of trained models, depending on the ``n_select`` param.


        Warnings
        --------
        - Changing turbo parameter to False may result in very high training times with
        datasets exceeding 10,000 rows.

        - No models are logged in ``MLFlow`` when ``cross_validation`` parameter is False.

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
            groups=groups,
            verbose=verbose,
        )

    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
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
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')


        estimator: str or scikit-learn compatible object
            ID of an estimator available in model library or pass an untrained
            model object consistent with scikit-learn API. Estimators available
            in the model library (ID - Name):

            * 'arima' - ARIMA
            * 'naive' - Naive
            * 'poly_trend' - PolyTrend
            * 'exp_smooth' - ExponentialSmoothing
            * 'theta' - Theta


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


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


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
            groups=groups,
            verbose=verbose,
            **kwargs,
        )

    def _create_model_without_cv(
        self, model, data_X, data_y, fit_kwargs, predict, system, display
    ):
        with estimator_pipeline(self._internal_pipeline, model) as pipeline_with_model:

            self.logger.info(
                "Support for Exogenous variables not yet supported. Switching X, y order"
            )
            data_X, data_y = data_y, data_X

            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            self.logger.info("Cross validation set to False")

            self.logger.info("Fitting Model")
            model_fit_start = time.time()
            with io.capture_output():
                pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
            model_fit_end = time.time()

            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

            display.move_progress()

            if predict:
                self.predict_model(pipeline_with_model, verbose=False)
                model_results = self.pull(pop=True).drop("Model", axis=1)

                self.display_container.append(model_results)

                display.display(
                    model_results,
                    clear=system,
                    override=False if not system else None,
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
        groups,
        metrics,
        refit,
        display,
    ):
        """
        MONITOR UPDATE STARTS
        """

        from pycaret.time_series import cross_validate_ts, _get_cv_n_folds

        display.update_monitor(
            1,
            f"Fitting {_get_cv_n_folds(data_y, cv)} Folds",
        )
        display.display_monitor()
        """
        MONITOR UPDATE ENDS
        """
        # TODO: pass metrics dict to cross_validate_ts instead of metrics (also update cross_validate_ts to use scorers directly)
        # Also change to dict comprehension (for efficiency)
        metrics_dict = dict([(k, v.scorer) for k, v in metrics.items()])
        # metrics_dict = {k, v.scorer for k, v in metrics.items()}

        self.logger.info("Starting cross validation")

        n_jobs = self._gpu_n_jobs_param

        # fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)

        self.logger.info(f"Cross validating with {cv}, n_jobs={n_jobs}")

        # Cross Validate time series
        fh_param = {"fh": cv.fh}
        if fit_kwargs is None:
            fit_kwargs = fh_param
        else:
            fit_kwargs.update(fh_param)
        # fit_kwargs.update({'actual_estimator__fh': self.fh})
        # # TODO: Temporarily disabling parallelization for debug (parallelization makes debugging harder)
        # n_jobs=1

        model_fit_start = time.time()

        scores = cross_validate_ts(
            forecaster=clone(model),
            y=data_y,
            X=data_X,
            scoring=metrics_dict,  # metrics,
            cv=cv,
            n_jobs=n_jobs,
            verbose=0,
            fit_params=fit_kwargs,
            return_train_score=False,
            error_score=0,
        )

        model_fit_end = time.time()
        model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

        score_dict = scores

        self.logger.info("Calculating mean and std")

        avgs_dict = {k: [np.mean(v), np.std(v)] for k, v in score_dict.items()}

        display.move_progress()

        self.logger.info("Creating metrics dataframe")

        model_results = pd.DataFrame(score_dict)
        model_avgs = pd.DataFrame(
            avgs_dict,
            index=["Mean", "SD"],
        )
        model_results = model_results.append(model_avgs)
        # Round the results
        model_results = model_results.round(round)

        # yellow the mean
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
            model_fit_time /= _get_cv_n_folds(data_y, cv)

        # return model, model_fit_time, model_results, avgs_dict
        return model, model_fit_time, model_results, avgs_dict

    def tune_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[Union[Dict[str, list], Any]] = None,
        optimize: str = "smape",
        custom_scorer=None,
        search_library: str = "pycaret",
        search_algorithm: Optional[str] = None,
        early_stopping: Any = False,
        early_stopping_max_iters: int = 10,
        choose_better: bool = False,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
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
        >>> y = get_data('airline', verbose=False)
        >>> exp = TimeSeriesExperiment()
        >>> exp.setup(data=y, fh=12 fold_strategy='expandingwindow')
        >>> exp.create_model("arima")
        >>> tuned_arima = tune_model('arima')


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


        optimize: str, default = 'R2'
            Metric name to be evaluated for hyperparameter tuning. It also accepts custom
            metrics that are added through the ``add_metric`` function.


        custom_scorer: object, default = None
            custom scoring strategy can be passed to tune hyperparameters of the model.
            It must be created using ``sklearn.make_scorer``. It is equivalent of adding
            custom metric using the ``add_metric`` function and passing the name of the
            custom metric in the ``optimize`` parameter.
            Will be deprecated in future.


        search_library: str, default = 'scikit-learn'
            The search library used for tuning hyperparameters. Possible values:

            - 'scikit-learn' - default, requires no further installation
                https://github.com/scikit-learn/scikit-learn

            - 'scikit-optimize' - ``pip install scikit-optimize``
                https://scikit-optimize.github.io/stable/

            - 'tune-sklearn' - ``pip install tune-sklearn ray[tune]``
                https://github.com/ray-project/tune-sklearn

            - 'optuna' - ``pip install optuna``
                https://optuna.org/


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
                - 'bohb' : ``pip install hpbandster ConfigSpace``

            - 'optuna' possible values:
                - 'random' : randomized search
                - 'tpe' : Tree-structured Parzen Estimator search (default)


        early_stopping: bool or str or object, default = False
            Use early stopping to stop fitting to a hyperparameter configuration
            if it performs poorly. Ignored when ``search_library`` is scikit-learn,
            or if the estimator does not have 'partial_fit' attribute. If False or
            None, early stopping will not be used. Can be either an object accepted
            by the search library or one of the following:

            - 'asha' for Asynchronous Successive Halving Algorithm
            - 'hyperband' for Hyperband
            - 'median' for Median Stopping Rule
            - If False or None, early stopping will not be used.


        early_stopping_max_iters: int, default = 10
            Maximum number of epochs to run for each sampled configuration.
            Ignored if ``early_stopping`` is False or None.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the tuner.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        return_tuner: bool, default = False
            When set to True, will return a tuple of (model, tuner_object).


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        tuner_verbose: bool or in, default = True
            If True or above 0, will print messages from the tuner. Higher values
            print more messages. Ignored when ``verbose`` parameter is False.


        **kwargs:
            Additional keyword arguments to pass to the optimizer.


        Returns:
            Trained Model and Optional Tuner Object when ``return_tuner`` is True.


        Warnings
        --------
        - Using 'grid' as ``search_algorithm`` may result in very long computation.
        Only recommended with smaller search spaces that can be defined in the
        ``custom_grid`` parameter.

        - ``search_library`` 'tune-sklearn' does not support GPU models.

        """

        # return super().tune_model(
        #     estimator=estimator,
        #     fold=fold,
        #     round=round,
        #     n_iter=n_iter,
        #     custom_grid=custom_grid,
        #     optimize=optimize,
        #     custom_scorer=custom_scorer,
        #     search_library=search_library,
        #     search_algorithm=search_algorithm,
        #     early_stopping=early_stopping,
        #     early_stopping_max_iters=early_stopping_max_iters,
        #     choose_better=choose_better,
        #     fit_kwargs=fit_kwargs,
        #     groups=groups,
        #     return_tuner=return_tuner,
        #     verbose=verbose,
        #     tuner_verbose=tuner_verbose,
        #     **kwargs,
        # )

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
        cv = self.fold_generator

        if not display:
            progress_args = {"max": 3 + 4}
            master_display_columns = [
                v.display_name for k, v in self._all_metrics.items()
            ]
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
        optimize_dict = {optimize: optimize_container.scorer}

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

        # TODO: Replace with time series specific code
        def total_combintaions_in_grid(grid):
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

        if search_algorithm is None:
            search_algorithm = "grid"

        if search_algorithm == "grid":
            param_grid = estimator_definition.tune_grid
        elif search_algorithm == "random":
            param_grid = estimator_definition.tune_distribution

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

        # with estimator_pipeline(self._internal_pipeline, model) as pipeline_with_model:
        if True:

            # fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            fh_param = {"fh": cv.fh}
            if fit_kwargs is None:
                fit_kwargs = fh_param
            else:
                fit_kwargs.update(fh_param)

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
                    from pycaret.time_series import ForecastingGridSearchCV

                    model_grid = ForecastingGridSearchCV(
                        forecaster=model,
                        cv=cv,
                        param_grid=param_grid,
                        scoring=optimize_dict,  # metrics
                        n_jobs=n_jobs,
                        verbose=tuner_verbose,
                        **search_kwargs,
                    )
                elif search_algorithm == "random":
                    self.logger.info("Initializing ForecastingRandomizedGridSearchCV")
                    from pycaret.time_series import ForecastingRandomizedSearchCV

                    model_grid = ForecastingRandomizedSearchCV(
                        forecaster=model,
                        cv=cv,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        scoring=optimize_dict,  # metrics
                        n_jobs=n_jobs,
                        verbose=tuner_verbose,
                        random_state=self.seed,
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
            groups=groups,
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
                groups=groups,
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

    def ensemble_model(
        self,
        estimator,
        method: str = "Bagging",
        fold: Optional[Union[int, Any]] = None,
        n_estimators: int = 10,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "R2",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
    ) -> Any:

        """
            This function ensembles a given estimator. The output of this function is
            a score grid with CV scores by fold. Metrics evaluated during CV can be
            accessed using the ``get_metrics`` function. Custom metrics can be added
            or removed using ``add_metric`` and ``remove_metric`` function.


            Example
            --------
            >>> from pycaret.datasets import get_data
            >>> boston = get_data('boston')
            >>> from pycaret.regression import *
            >>> exp_name = setup(data = boston,  target = 'medv')
            >>> dt = create_model('dt')
            >>> bagged_dt = ensemble_model(dt, method = 'Bagging')


        estimator: scikit-learn compatible object
                Trained model object


            method: str, default = 'Bagging'
                Method for ensembling base estimator. It can be 'Bagging' or 'Boosting'.


            fold: int or scikit-learn compatible CV generator, default = None
                Controls cross-validation. If None, the CV generator in the ``fold_strategy``
                parameter of the ``setup`` function is used. When an integer is passed,
                it is interpreted as the 'n_splits' parameter of the CV generator in the
                ``setup`` function.


            n_estimators: int, default = 10
                The number of base estimators in the ensemble. In case of perfect fit, the
                learning procedure is stopped early.


            round: int, default = 4
                Number of decimal places the metrics in the score grid will be rounded to.


            choose_better: bool, default = False
                When set to True, the returned object is always better performing. The
                metric used for comparison is defined by the ``optimize`` parameter.


            optimize: str, default = 'R2'
                Metric to compare for model selection when ``choose_better`` is True.


            fit_kwargs: dict, default = {} (empty dict)
                Dictionary of arguments passed to the fit method of the model.


            groups: str or array-like, with shape (n_samples,), default = None
                Optional group labels when GroupKFold is used for the cross validation.
                It takes an array with shape (n_samples, ) where n_samples is the number
                of rows in training dataset. When string is passed, it is interpreted as
                the column name in the dataset containing group labels.


            verbose: bool, default = True
                Score grid is not printed when verbose is set to False.


            Returns:
                Trained Model

        """

        return super().ensemble_model(
            estimator=estimator,
            method=method,
            fold=fold,
            n_estimators=n_estimators,
            round=round,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
        )

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
        groups: Optional[Union[str, Any]] = None,
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
        >>> from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
        >>> import numpy as np
        >>> airline_data = get_data('airline', verbose=False)
        >>> fh = np.arange(1,13)
        >>> fold = 3
        >>> exp = TimeSeriesExperiment()
        >>> exp.setup(data=y, fh=fh, fold=fold)
        >>> arima_model = exp.create_model("arima")
        >>> naive_model = exp.create_model("naive")
        >>> ts_blender = exp.blend_models([arima_model, naive_model], optimize='MAPE_ts')


        estimator_list: list of scikit-learn compatible objects
            List of trained model objects


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


        optimize: str, default = 'MAPE_ts'
            Metric to compare for model selection when ``choose_better`` is True.


        weights: list, default = None
            Sequence of weights (float or int) to weight the occurrences of predicted class
            labels (hard voting) or class probabilities before averaging (soft voting). Uses
            uniform weights when None.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


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
            groups=groups,
            verbose=verbose,
        )

    def stack_models(
        self,
        estimator_list: list,
        meta_model=None,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        restack: bool = False,
        choose_better: bool = False,
        optimize: str = "R2",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
    ):

        """
        This function trains a meta model over select estimators passed in
        the ``estimator_list`` parameter. The output of this function is a
        score grid with CV scores by fold. Metrics evaluated during CV can
        be accessed using the ``get_metrics`` function. Custom metrics
        can be added or removed using ``add_metric`` and ``remove_metric``
        function.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> top3 = compare_models(n_select = 3)
        >>> stacker = stack_models(top3)


        estimator_list: list of scikit-learn compatible objects
            List of trained model objects


        meta_model: scikit-learn compatible object, default = None
            When None, Linear Regression is trained as a meta model.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        restack: bool, default = False
            When set to False, only the predictions of estimators will be used as
            training data for the ``meta_model``.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'R2'
            Metric to compare for model selection when ``choose_better`` is True.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained Model

        """

        return super().stack_models(
            estimator_list=estimator_list,
            meta_model=meta_model,
            fold=fold,
            round=round,
            method="auto",
            restack=restack,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
        )

    def plot_model(
        self,
        estimator,
        plot: str = "residuals",
        scale: float = 1,
        save: bool = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
        verbose: bool = True,
        display_format: Optional[str] = None,
    ) -> str:

        """
        This function analyzes the performance of a trained model on holdout set.
        It may require re-training the model in certain cases.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> plot_model(lr, plot = 'residual')


        estimator: scikit-learn compatible object
            Trained model object


        plot: str, default = 'residual'
            List of available plots (ID - Name):

            * 'residuals' - Residuals Plot
            * 'error' - Prediction Error Plot
            * 'cooks' - Cooks Distance Plot
            * 'rfe' - Recursive Feat. Selection
            * 'learning' - Learning Curve
            * 'vc' - Validation Curve
            * 'manifold' - Manifold Learning
            * 'feature' - Feature Importance
            * 'feature_all' - Feature Importance (All)
            * 'parameter' - Model Hyperparameter
            * 'tree' - Decision Tree


        scale: float, default = 1
            The resolution scale of the figure.


        save: bool, default = False
            When set to True, plot is saved in the current working directory.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


        verbose: bool, default = True
            When set to False, progress bar is not displayed.


        display_format: str, default = None
            To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
            Currently, not all plots are supported.


        Returns:
            None

        """

        return super().plot_model(
            estimator=estimator,
            plot=plot,
            scale=scale,
            save=save,
            fold=fold,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            use_train_data=use_train_data,
            system=True,
            display_format=display_format,
        )

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
    ):

        """
        This function displays a user interface for analyzing performance of a trained
        model. It calls the ``plot_model`` function internally.

        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> evaluate_model(lr)


        estimator: scikit-learn compatible object
            Trained model object


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


        Returns:
            None


        Warnings
        --------
        -   This function only works in IPython enabled Notebook.

        """

        return super().evaluate_model(
            estimator=estimator,
            fold=fold,
            fit_kwargs=fit_kwargs,
            groups=groups,
            use_train_data=use_train_data,
        )

    def interpret_model(
        self,
        estimator,
        plot: str = "summary",
        feature: Optional[str] = None,
        observation: Optional[int] = None,
        use_train_data: bool = False,
        **kwargs,
    ):

        """
        This function analyzes the predictions generated from a tree-based model. It is
        implemented based on the SHAP (SHapley Additive exPlanations). For more info on
        this, please see https://shap.readthedocs.io/en/latest/


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp = setup(data = boston,  target = 'medv')
        >>> xgboost = create_model('xgboost')
        >>> interpret_model(xgboost)


        estimator: scikit-learn compatible object
            Trained model object


        plot: str, default = 'summary'
            Type of plot. Available options are: 'summary', 'correlation', and 'reason'.


        feature: str, default = None
            Feature to check correlation with. This parameter is only required when ``plot``
            type is 'correlation'. When set to None, it uses the first column in the train
            dataset.


        observation: int, default = None
            Observation index number in holdout set to explain. When ``plot`` is not
            'reason', this parameter is ignored.


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


        **kwargs:
            Additional keyword arguments to pass to the plot.


        Returns:
            None

        """

        return super().interpret_model(
            estimator=estimator,
            plot=plot,
            feature=feature,
            observation=observation,
            use_train_data=use_train_data,
            **kwargs,
        )

    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:

        """
        This function predicts ``Label`` using a trained model. When ``data`` is
        None, it predicts label on the holdout set.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> pred_holdout = predict_model(lr)
        >>> pred_unseen = predict_model(lr, data = unseen_dataframe)


        estimator: scikit-learn compatible object
            Trained model object


        data : pandas.DataFrame
            Shape (n_samples, n_features). All features used during training
            must be available in the unseen dataset.


        round: int, default = 4
            Number of decimal places to round predictions to.


        verbose: bool, default = True
            When set to False, holdout score grid is not printed.


        Returns:
            pandas.DataFrame


        Warnings
        --------
        - The behavior of the ``predict_model`` is changed in version 2.1 without backward
        compatibility. As such, the pipelines trained using the version (<= 2.0), may not
        work for inference with version >= 2.1. You can either retrain your models with a
        newer version or downgrade the version for inference.


        """

        return super().predict_model(
            estimator=estimator,
            data=data,
            probability_threshold=None,
            encoded_labels=True,
            round=round,
            verbose=verbose,
        )

    def finalize_model(
        self,
        estimator,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        model_only: bool = True,
    ) -> Any:

        """
        This function trains a given estimator on the entire dataset including the
        holdout set.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> final_lr = finalize_model(lr)


        estimator: scikit-learn compatible object
            Trained model object


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        model_only: bool, default = True
            When set to False, only model object is re-trained and all the
            transformations in Pipeline are ignored.


        Returns:
            Trained Model


        """

        return super().finalize_model(
            estimator=estimator,
            fit_kwargs=fit_kwargs,
            groups=groups,
            model_only=model_only,
        )

    def deploy_model(
        self,
        model,
        model_name: str,
        authentication: dict,
        platform: str = "aws",
    ):

        """
        This function deploys the transformation pipeline and trained model on cloud.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> deploy_model(model = lr, model_name = 'lr-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})


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
            {'bucket' : 'S3-bucket-name'}

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
        self, model, model_name: str, model_only: bool = False, verbose: bool = True
    ):

        """
        This function saves the transformation pipeline and trained model object
        into the current working directory as a pickle file for later use.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> save_model(lr, 'saved_lr_model')


        model: scikit-learn compatible object
            Trained model object


        model_name: str
            Name of the model.


        model_only: bool, default = False
            When set to True, only trained model object is saved instead of the
            entire pipeline.


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
        This function loads a previously saved pipeline.

        Example
        -------
        >>> from pycaret.regression import load_model
        >>> saved_lr = load_model('saved_lr_model')


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

    def automl(
        self, optimize: str = "R2", use_holdout: bool = False, turbo: bool = True
    ) -> Any:

        """
        This function returns the best model out of all trained models in
        current session based on the ``optimize`` parameter. Metrics
        evaluated can be accessed using the ``get_metrics`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> top3 = compare_models(n_select = 3)
        >>> tuned_top3 = [tune_model(i) for i in top3]
        >>> blender = blend_models(tuned_top3)
        >>> stacker = stack_models(tuned_top3)
        >>> best_mae_model = automl(optimize = 'MAE')


        optimize: str, default = 'R2'
            Metric to use for model selection. It also accepts custom metrics
            added using the ``add_metric`` function.


        use_holdout: bool, default = False
            When set to True, metrics are evaluated on holdout set instead of CV.


        turbo: bool, default = True
            When set to True and use_holdout is False, only models created with default fold
            parameter will be considered. If set to False, models created with a non-default
            fold parameter will be scored again using default fold settings, so that they can be
            compared.


        Returns:
            Trained Model


        """

        return super().automl(optimize=optimize, use_holdout=use_holdout, turbo=turbo)

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
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> all_models = models()


        type: str, default = None
            - linear : filters and only return linear models
            - tree : filters and only return tree based models
            - ensemble : filters and only return ensemble models


        internal: bool, default = False
            When True, will return extra columns and rows used internally.


        raise_errors: bool, default = True
            When False, will suppress all exceptions, ignoring models
            that couldn't be created.


        Returns:
            pandas.DataFrame

        """
        return super().models(type=type, internal=internal, raise_errors=raise_errors)

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
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
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
            reset=reset,
            include_custom=include_custom,
            raise_errors=raise_errors,
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
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
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
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'mredv')
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
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv', log_experiment = True)
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
