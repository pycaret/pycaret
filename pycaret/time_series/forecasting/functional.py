"""Functional API for Time Series Forecasting Experiment
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pycaret.internal.parallel.parallel_backend import ParallelBackend
from pycaret.time_series.forecasting.oop import TSForecastingExperiment
from pycaret.utils.generic import check_if_global_is_not_none

if TYPE_CHECKING:
    from sktime.forecasting.base import ForecastingHorizon

_EXPERIMENT_CLASS = TSForecastingExperiment
_CURRENT_EXPERIMENT: Optional[TSForecastingExperiment] = None
_CURRENT_EXPERIMENT_EXCEPTION = (
    "_CURRENT_EXPERIMENT global variable is not set. Please run setup() first."
)
_CURRENT_EXPERIMENT_DECORATOR_DICT = {
    "_CURRENT_EXPERIMENT": _CURRENT_EXPERIMENT_EXCEPTION
}


def setup(
    data: Union[pd.Series, pd.DataFrame] = None,
    data_func: Optional[Callable[[], Union[pd.Series, pd.DataFrame]]] = None,
    target: Optional[str] = None,
    index: Optional[str] = None,
    ignore_features: Optional[List] = None,
    numeric_imputation_target: Optional[Union[int, float, str]] = None,
    numeric_imputation_exogenous: Optional[Union[int, float, str]] = None,
    transform_target: Optional[str] = None,
    transform_exogenous: Optional[str] = None,
    scale_target: Optional[str] = None,
    scale_exogenous: Optional[str] = None,
    fold_strategy: Union[str, Any] = "expanding",
    fold: int = 3,
    fh: Optional[Union[List[int], int, np.ndarray, "ForecastingHorizon"]] = 1,
    seasonal_period: Optional[Union[List[Union[int, str]], int, str]] = None,
    point_alpha: Optional[float] = None,
    coverage: Union[float, List[float]] = 0.9,
    enforce_exogenous: bool = True,
    n_jobs: Optional[int] = -1,
    use_gpu: bool = False,
    custom_pipeline: Optional[Any] = None,
    html: bool = True,
    session_id: Optional[int] = None,
    system_log: Union[bool, str, logging.Logger] = True,
    log_experiment: bool = False,
    experiment_name: Optional[str] = None,
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    verbose: bool = True,
    profile: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
    fig_kwargs: Optional[Dict[str, Any]] = None,
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


    data : pandas.Series or pandas.DataFrame = None
        Shape (n_samples, 1), when pandas.DataFrame, otherwise (n_samples, ).


    data_func: Callable[[], Union[pd.Series, pd.DataFrame]] = None
            The function that generate ``data`` (the dataframe-like input). This
            is useful when the dataset is large, and you need parallel operations
            such as ``compare_models``. It can avoid boradcasting large dataset
            from driver to workers. Notice one and only one of ``data`` and
            ``data_func`` must be set.


    target : Optional[str], default = None
        Target name to be forecasted. Must be specified when data is a pandas
        DataFrame with more than 1 column. When data is a pandas Series or
        pandas DataFrame with 1 column, this can be left as None.


    index: Optional[str], default = None
        Column name to be used as the datetime index for modeling. Column is
        internally converted to datetime using `pd.to_datetime()`. If None,
        then the data's index is used as is for modeling.


    ignore_features: Optional[List], default = None
        List of features to ignore for modeling when the data is a pandas
        Dataframe with more than 1 column. Ignored when data is a pandas Series
        or Dataframe with 1 column.


    numeric_imputation_target: Optional[Union[int, float, str]], default = None
        Indicates how to impute missing values in the target.
        If None, no imputation is done.
        If the target has missing values, then imputation is mandatory.
        If str, then value passed as is to the underlying `sktime` imputer.
        Allowed values are:
            "drift", "linear", "nearest", "mean", "median", "backfill",
            "bfill", "pad", "ffill", "random"
        If int or float, imputation method is set to "constant" with the given value.


    numeric_imputation_exogenous: Optional[Union[int, float, str]], default = None
        Indicates how to impute missing values in the exogenous variables.
        If None, no imputation is done.
        If exogenous variables have missing values, then imputation is mandatory.
        If str, then value passed as is to the underlying `sktime` imputer.
        Allowed values are:
            "drift", "linear", "nearest", "mean", "median", "backfill",
            "bfill", "pad", "ffill", "random"
        If int or float, imputation method is set to "constant" with the given value.


    transform_target: Optional[str], default = None
        Indicates how the target variable should be transformed.
        If None, no transformation is performed. Allowed values are
            "box-cox", "log", "sqrt", "exp", "cos"


    transform_exogenous: Optional[str], default = None
        Indicates how the exogenous variables should be transformed.
        If None, no transformation is performed. Allowed values are
            "box-cox", "log", "sqrt", "exp", "cos"


    scale_target: Optional[str], default = None
        Indicates how the target variable should be scaled.
        If None, no scaling is performed. Allowed values are
            "zscore", "minmax", "maxabs", "robust"


    scale_exogenous: Optional[str], default = None
        Indicates how the exogenous variables should be scaled.
        If None, no scaling is performed. Allowed values are
            "zscore", "minmax", "maxabs", "robust"


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


    fh: Optional[int or list or np.array or ForecastingHorizon], default = 1
        The forecast horizon to be used for forecasting. Default is set to ``1``
        i.e. forecast one point ahead. Valid options are:
        (1) Integer: When integer is passed it means N continuous points in
            the future without any gap.
        (2) List or np.array: Indicates points to predict in the future. e.g.
            fh = [1, 2, 3, 4] or np.arange(1, 5) will predict 4 points in the future.
        (3) If you want to forecast values with gaps, you can pass an list or array
            with gaps. e.g. np.arange([13, 25]) will skip the first 12 future points
            and forecast from the 13th point till the 24th point ahead (note in numpy
            right value is inclusive and left is exclusive).
        (4) Can also be a sktime compatible ForecastingHorizon object.
        (5) If fh = None, then fold_strategy must be a sktime compatible cross validation
            object. In this case, fh is derived from this object.


    seasonal_period: list or int or str, default = None
        Seasonal period in timeseries data. If not provided the frequency of the data
        index is mapped to a seasonal period as follows:

        * B, C = 5
        * D = 7
        * W = 52
        * M, BM, CBM, MS, BMS, CBMS = 12
        * SM, SMS = 24
        * Q, BQ, QS, BQS = 4
        * A, Y, BA, BY, AS, YS, BAS, BYS = 1
        * H = 24
        * T, min = 60
        * S = 60

        Alternatively you can provide a custom `seasonal_period` by passing
        it as an integer or a string corresponding to the keys above (e.g.
        'W' for weekly data, 'M' for monthly data, etc.). You can also provide
        a list of such values to use in models that accept multiple seasonal values
        (currently TBATS). For models that don't accept multiple seasonal values, the
        first value of the list will be used as the seasonal period.


    point_alpha: Optional[float], default = None
        The alpha (quantile) value to use for the point predictions. By default
        this is set to None which uses sktime's predict() method to get the
        point prediction (the mean or the median of the forecast distribution).
        If this is set to a floating point value, then it switches to using the
        predict_quantiles() method to get the point prediction at the user
        specified quantile.
        Reference: https://robjhyndman.com/hyndsight/quantile-forecasts-in-r/

        NOTE:
        (1) Not all models support predict_quantiles(), hence, if a float
        value is provided, these models will be disabled.
        (2) Under some conditions, the user may want to only work with models
        that support prediction intervals. Utilizing note 1 to our advantage,
        the point_alpha argument can be set to 0.5 (or any float value depending
        on the quantile that the user wants to use for point predictions).
        This will disable models that do not support prediction intervals.


    coverage: Union[float, List[float]], default = 0.9
        The coverage to be used for prediction intervals (only applicable for
        models that support prediction intervals).

        If a float value is provides, it corresponds to the coverage needed
        (e.g. 0.9 means 90% coverage). This corresponds to lower and upper
        quantiles = 0.05 and 0.95 respectively.

        Alternately, if user wants to get the intervals at specific quantiles,
        a list of 2 values can be provided directly. e.g. coverage = [0.2. 0.9]
        will return the lower interval corresponding to a quantile of 0.2 and
        an upper interval corresponding to a quantile of 0.9.


    enforce_exogenous: bool, default = True
        When set to True and the data includes exogenous variables, only models
        that support exogenous variables are loaded in the environment.When
        set to False, all models are included and in this case, models that do
        not support exogenous variables will model the data as a univariate
        forecasting problem.


    n_jobs: int, default = -1
        The number of jobs to run in parallel (for functions that supports parallel
        processing) -1 means using all processors. To run all functions on single
        processor set n_jobs to None.


    use_gpu: bool or str, default = False
        Parameter not in use for now. Behavior may change in future.


    custom_pipeline: list of (str, transformer), dict or Pipeline, default = None
        Parameter not in use for now. Behavior may change in future.


    html: bool, default = True
        When set to False, prevents runtime display of monitor. This must be set to False
        when the environment does not support IPython. For example, command line terminal,
        Databricks Notebook, Spyder and other similar IDEs.


    session_id: int, default = None
        Controls the randomness of experiment. It is equivalent to 'random_state' in
        scikit-learn. When None, a pseudo random number is generated. This can be used
        for later reproducibility of the entire experiment.


    system_log: bool or str or logging.Logger, default = True
        Whether to save the system logging file (as logs.log). If the input
        is a string, use that as the path to the logging file. If the input
        already is a logger object, use that one instead.


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


    fig_kwargs: dict, default = {} (empty dict)
        The global setting for any plots. Pass these as key-value pairs.
        Example: fig_kwargs = {"height": 1000, "template": "simple_white"}

        Available keys are:

        hoverinfo: hoverinfo passed to Plotly figures. Can be any value supported
            by Plotly (e.g. "text" to display, "skip" or "none" to disable.).
            When not provided, hovering over certain plots may be disabled by
            PyCaret when the data exceeds a  certain number of points (determined
            by `big_data_threshold`).

        renderer: The renderer used to display the plotly figure. Can be any value
            supported by Plotly (e.g. "notebook", "png", "svg", etc.). Note that certain
            renderers (like "svg") may need additional libraries to be installed. Users
            will have to do this manually since they don't come preinstalled wit plotly.
            When not provided, plots use plotly's default render when data is below a
            certain number of points (determined by `big_data_threshold`) otherwise it
            switches to a static "png" renderer.

        template: The template to use for the plots. Can be any value supported by Plotly.
            If not provided, defaults to "ggplot2"

        width: The width of the plot in pixels. If not provided, defaults to None
            which lets Plotly decide the width.

        height: The height of the plot in pixels. If not provided, defaults to None
            which lets Plotly decide the height.

        rows: The number of rows to use for plots where this can be customized,
            e.g. `ccf`. If not provided, defaults to None which lets PyCaret decide
            based on number of subplots to be plotted.

        cols: The number of columns to use for plots where this can be customized,
            e.g. `ccf`. If not provided, defaults to 4

        big_data_threshold: The number of data points above which hovering over
            certain plots can be disabled and/or renderer switched to a static
            renderer. This is useful when the time series being modeled has a lot
            of data which can make notebooks slow to render. Also note that setting
            the `display_format` to a plotly-resampler figure ("plotly-dash" or
            "plotly-widget") can circumvent these problems by performing dynamic data
            aggregation.

        resampler_kwargs: The keyword arguments that are fed to configure the
            `plotly-resampler` visualizations (i.e., `display_format` "plotly-dash" or
            "plotly-widget") which downsampler will be used; how many datapoints are
            shown in the front-end. When the plotly-resampler figure is renderd via Dash
            (by setting the `display_format` to "plotly-dash"), one can also use the
            "show_dash" key within this dictionary to configure the show_dash method its
            args.

            example::

                fig_kwargs = {
                    ...,
                    "resampler_kwargs":  {
                        "default_n_shown_samples": 1000,
                        "show_dash": {"mode": "inline", "port": 9012}
                    }
                }


    Returns:
        Global variables that can be changed using the ``set_config`` function.

    """

    exp = _EXPERIMENT_CLASS()
    set_current_experiment(exp)
    return exp.setup(
        data=data,
        data_func=data_func,
        target=target,
        index=index,
        ignore_features=ignore_features,
        numeric_imputation_target=numeric_imputation_target,
        numeric_imputation_exogenous=numeric_imputation_exogenous,
        transform_target=transform_target,
        transform_exogenous=transform_exogenous,
        scale_target=scale_target,
        scale_exogenous=scale_exogenous,
        fold_strategy=fold_strategy,
        fold=fold,
        fh=fh,
        seasonal_period=seasonal_period,
        point_alpha=point_alpha,
        coverage=coverage,
        enforce_exogenous=enforce_exogenous,
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
        verbose=verbose,
        profile=profile,
        profile_kwargs=profile_kwargs,
        fig_kwargs=fig_kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def compare_models(
    include: Optional[List[Union[str, Any]]] = None,
    exclude: Optional[List[str]] = None,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    sort: str = "MASE",
    n_select: int = 1,
    budget_time: Optional[float] = None,
    turbo: bool = True,
    errors: str = "ignore",
    fit_kwargs: Optional[dict] = None,
    engine: Optional[Dict[str, str]] = None,
    verbose: bool = True,
    parallel: Optional[ParallelBackend] = None,
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


    sort: str, default = 'MASE'
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


    engine: Optional[Dict[str, str]] = None
        The engine to use for the models, e.g. for auto_arima, users can
        switch between "pmdarima" and "statsforecast" by specifying
        engine={"auto_arima": "statsforecast"}


    verbose: bool, default = True
        Score grid is not printed when verbose is set to False.


    parallel: pycaret.internal.parallel.parallel_backend.ParallelBackend, default = None
        A ParallelBackend instance. For example if you have a SparkSession ``session``,
        you can use ``FugueBackend(session)`` to make this function running using
        Spark. For more details, see
        :class:`~pycaret.parallel.fugue_backend.FugueBackend`


    Returns:
        Trained model or list of trained models, depending on the ``n_select`` param.


    Warnings
    --------
    - Changing turbo parameter to False may result in very high training times.

    - No models are logged in ``MLflow`` when ``cross_validation`` parameter is False.

    """

    return _CURRENT_EXPERIMENT.compare_models(
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
        engine=engine,
        verbose=verbose,
        parallel=parallel,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_allowed_engines(estimator: str) -> Optional[str]:
    """Get all the allowed engines for the specified model

    Parameters
    ----------
    estimator : str
        Identifier for the model for which the engines should be retrieved,
        e.g. "auto_arima"

    Returns
    -------
    Optional[str]
        The allowed engines for the model. If the model only supports the
        default engine, then it return `None`.
    """

    return _CURRENT_EXPERIMENT.get_allowed_engines(estimator=estimator)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_engine(estimator: str) -> Optional[str]:
    """Gets the model engine currently set in the experiment for the specified
    model.

    Parameters
    ----------
    estimator : str
        Identifier for the model for which the engine should be retrieved,
        e.g. "auto_arima"

    Returns
    -------
    Optional[str]
        The engine for the model. If the model only supports the default sktime
        engine, then it return `None`.
    """

    return _CURRENT_EXPERIMENT.get_engine(estimator=estimator)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def create_model(
    estimator: Union[str, Any],
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    fit_kwargs: Optional[dict] = None,
    engine: Optional[str] = None,
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

        NOTE: The available estimators depend on multiple factors such as what
        libraries have been installed and the setup of the experiment. As such,
        some of these may not be available for your experiment. To see the list
        of available models, please run `setup()` first, then `models()`.

        * 'naive' - Naive Forecaster
        * 'grand_means' - Grand Means Forecaster
        * 'snaive' - Seasonal Naive Forecaster (disabled when seasonal_period = 1)
        * 'polytrend' - Polynomial Trend Forecaster
        * 'arima' - ARIMA family of models (ARIMA, SARIMA, SARIMAX)
        * 'auto_arima' - Auto ARIMA
        * 'exp_smooth' - Exponential Smoothing
        * 'croston' - Croston Forecaster
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
        * 'catboost_cds_dt' - CatBoost w/ Cond. Deseasonalize & Detrending


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


    engine: Optional[str] = None
        The engine to use for the model, e.g. for auto_arima, users can
        switch between "pmdarima" and "statsforecast" by specifying
        engine="statsforecast".


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

    return _CURRENT_EXPERIMENT.create_model(
        estimator=estimator,
        fold=fold,
        round=round,
        cross_validation=cross_validation,
        fit_kwargs=fit_kwargs,
        engine=engine,
        verbose=verbose,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def tune_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    n_iter: int = 10,
    custom_grid: Optional[Union[Dict[str, list], Any]] = None,
    optimize: str = "MASE",
    custom_scorer=None,
    search_algorithm: Optional[str] = None,
    choose_better: bool = True,
    fit_kwargs: Optional[dict] = None,
    return_tuner: bool = False,
    verbose: bool = True,
    tuner_verbose: Union[int, bool] = True,
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


    optimize: str, default = 'MASE'
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

    return _CURRENT_EXPERIMENT.tune_model(
        estimator=estimator,
        fold=fold,
        round=round,
        n_iter=n_iter,
        custom_grid=custom_grid,
        optimize=optimize,
        custom_scorer=custom_scorer,
        search_algorithm=search_algorithm,
        choose_better=choose_better,
        fit_kwargs=fit_kwargs,
        return_tuner=return_tuner,
        verbose=verbose,
        tuner_verbose=tuner_verbose,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def blend_models(
    estimator_list: list,
    method: str = "mean",
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    choose_better: bool = False,
    optimize: str = "MASE",
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


    optimize: str, default = 'MASE'
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

    return _CURRENT_EXPERIMENT.blend_models(
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


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def plot_model(
    estimator: Optional[Any] = None,
    plot: Optional[str] = None,
    return_fig: bool = False,
    return_data: bool = False,
    verbose: bool = False,
    display_format: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
    save: Union[str, bool] = False,
) -> Optional[Tuple[str, list]]:

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
    >>> plot_model(plot="diff", data_kwargs={"order_list": [1, 2], "acf": True, "pacf": True})
    >>> plot_model(plot="diff", data_kwargs={"lags_list": [[1], [1, 12]], "acf": True, "pacf": True})
    >>> arima = create_model('arima')
    >>> plot_model(plot = 'ts')
    >>> plot_model(plot = 'decomp', data_kwargs = {'type' : 'multiplicative'})
    >>> plot_model(plot = 'decomp', data_kwargs = {'seasonal_period': 24})
    >>> plot_model(estimator = arima, plot = 'forecast', data_kwargs = {'fh' : 24})
    >>> tuned_arima = tune_model(arima)
    >>> plot_model([arima, tuned_arima], data_kwargs={"labels": ["Baseline", "Tuned"]})


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
        * 'decomp' - Classical Decomposition
        * 'decomp_stl' - STL Decomposition
        * 'diagnostics' - Diagnostics Plot
        * 'diff' - Difference Plot
        * 'periodogram' - Frequency Components (Periodogram)
        * 'fft' - Frequency Components (FFT)
        * 'ccf' - Cross Correlation (CCF)
        * 'forecast' - "Out-of-Sample" Forecast Plot
        * 'insample' - "In-Sample" Forecast Plot
        * 'residuals' - Residuals Plot


    return_fig: : bool, default = False
        When set to True, it returns the figure used for plotting.


    return_data: bool, default = False
        When set to True, it returns the data for plotting.
        If both return_fig and return_data is set to True, order of return
        is figure then data.


    verbose: bool, default = True
        Unused for now


    display_format: str, default = None
        To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
        Currently, not all plots are supported.


    data_kwargs: dict, default = None
        Dictionary of arguments passed to the data for plotting.

        Available keys are:

        nlags: The number of lags to use when plotting correlation plots, e.g.
            ACF, PACF, CCF. If not provided, default internally calculated
            values are used.

        seasonal_period: The seasonal period to use for decomposition plots.
            If not provided, the default internally detected seasonal period
            is used.

        type: The type of seasonal decomposition to perform. Options are:
            ["additive", "multiplicative"]

        order_list: The differencing orders to use for difference plots. e.g.
            [1, 2] will plot first and second order differences (corresponding
            to d = 1 and 2 in ARIMA models).

        lags_list: An alternate and more explicit alternate to "order_list"
            allowing users to specify the exact lags to plot. e.g.
            [1, [1, 12]] will plot first difference and a second plot with
            first difference (d = 1 in ARIMA) and seasonal 12th difference
            (D=1, s=12 in ARIMA models). Also note that "order_list" = [2]
            can be alternately specified as lags_list = [[1, 1]] i.e. successive
            differencing twice.

        acf: True/False
            When specified in difference plots and set to True, this will plot
            the ACF of the differenced data as well.

        pacf: True/False
            When specified in difference plots and set to True, this will plot
            the PACF of the differenced data as well.

        periodogram: True/False
            When specified in difference plots and set to True, this will plot
            the Periodogram of the differenced data as well.

        fft: True/False
            When specified in difference plots and set to True, this will plot
            the FFT of the differenced data as well.

        labels: When estimator(s) are provided, the corresponding labels to
            use for the plots. If not provided, the model class is used to
            derive the labels.

        include: When data contains exogenous variables, then only specific
            exogenous variables can be plotted using this key.
            e.g. include = ["col1", "col2"]

        exclude: When data contains exogenous variables, specific exogenous
            variables can be excluded from the plots using this key.
            e.g. exclude = ["col1", "col2"]

        alpha: The quantile value to use for point prediction. If not provided,
            then the value specified during setup is used.

        coverage: The coverage value to use for prediction intervals.  If not
            provided, then the value specified during setup is used.

        fh: The forecast horizon to use for forecasting. If not provided, then
            the one used during model training is used.

        X: When a model trained with exogenous variables has been finalized,
            user can provide the future values of the exogenous variables to
            make future target time series predictions using this key.

        plot_data_type: When plotting the data used for modeling, user may
            wish to see plots with the original data set provided, the imputed
            dataset (if imputation is set) or the transformed dataset (which
            includes any imputation and transformation set by the user). This
            keyword can be used to specify which data type to use.

            NOTE:
            (1) If no imputation is specified, then plotting the "imputed"
                data type will produce the same results as the "original" data type.
            (2) If no transforations are specified, then plotting the "transformed"
                data type will produce the same results as the "imputed" data type.

            Allowed values are (if not specified, defaults to the first one in the list):

            "ts": ["original", "imputed", "transformed"]
            "train_test_split": ["original", "imputed", "transformed"]
            "cv": ["original"]
            "acf": ["transformed", "imputed", "original"]
            "pacf": ["transformed", "imputed", "original"]
            "decomp": ["transformed", "imputed", "original"]
            "decomp_stl": ["transformed", "imputed", "original"]
            "diagnostics": ["transformed", "imputed", "original"]
            "diff": ["transformed", "imputed", "original"]
            "forecast": ["original", "imputed"]
            "insample": ["original", "imputed"]
            "residuals": ["original", "imputed"]
            "periodogram": ["transformed", "imputed", "original"]
            "fft": ["transformed", "imputed", "original"]
            "ccf": ["transformed", "imputed", "original"]

            Some plots (marked as True below) will also allow specifying
            multiple of data types at once.

            "ts": True
            "train_test_split": True
            "cv": False
            "acf": True
            "pacf": True
            "decomp": True
            "decomp_stl": True
            "diagnostics": True
            "diff": False
            "forecast": False
            "insample": False
            "residuals": False
            "periodogram": True
            "fft": True
            "ccf": False


    fig_kwargs: dict, default = {} (empty dict)
        The setting to be used for the plot. Overrides any global setting
        passed during setup. Pass these as key-value pairs. For available
        keys, refer to the `setup` documentation.


    save: string or bool, default = False
        When set to True, Plot is saved as a 'png' file in current working directory.
        When a path destination is given, Plot is saved as a 'png' file the given path to the directory of choice.


    Returns:
        Path to saved file and list containing figure and data, if any.

    """

    return _CURRENT_EXPERIMENT.plot_model(
        estimator=estimator,
        plot=plot,
        return_fig=return_fig,
        return_data=return_data,
        display_format=display_format,
        data_kwargs=data_kwargs,
        fig_kwargs=fig_kwargs,
        save=save,
    )


# not using check_if_global_is_not_none on purpose
def predict_model(
    estimator,
    fh=None,
    X=None,
    return_pred_int=False,
    alpha: Optional[float] = None,
    coverage: Union[float, List[float]] = 0.9,
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


    X: pd.DataFrame, default = None
        Exogenous Variables to be used for prediction.
        Before finalizing the estimator, X need not be passed even when the
        estimator is built using exogenous variables (since this is taken
        care of internally by using the exogenous variables from test split).
        When estimator has been finalized and estimator used exogenous
        variables, then X must be passed.


    return_pred_int: bool, default = False
        When set to True, it returns lower bound and upper bound
        prediction interval, in addition to the point prediction.


    alpha: Optional[float], default = None
        The alpha (quantile) value to use for the point predictions. Refer to
        the "point_alpha" description in the setup docstring for details.


    coverage: Union[float, List[float]], default = 0.9
        The coverage to be used for prediction intervals. Refer to the "coverage"
        description in the setup docstring for details.


    round: int, default = 4
        Number of decimal places to round predictions to.


    verbose: bool, default = True
        When set to False, holdout score grid is not printed.


    Returns:
        pandas.DataFrame


    """

    experiment = _CURRENT_EXPERIMENT
    if experiment is None:
        experiment = _EXPERIMENT_CLASS()

    return experiment.predict_model(
        estimator=estimator,
        fh=fh,
        X=X,
        return_pred_int=return_pred_int,
        alpha=alpha,
        coverage=coverage,
        round=round,
        verbose=verbose,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def finalize_model(
    estimator, fit_kwargs: Optional[dict] = None, model_only: bool = False
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
        Trained pipeline or model object fitted on complete dataset.


    """

    return _CURRENT_EXPERIMENT.finalize_model(
        estimator=estimator, fit_kwargs=fit_kwargs, model_only=model_only
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def deploy_model(model, model_name: str, authentication: dict, platform: str = "aws"):

    """
    This function deploys the transformation pipeline and trained model on cloud.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> data = get_data('airline')
    >>> from pycaret.time_series import *
    >>> exp_name = setup(data = data, fh = 12)
    >>> arima = create_model('arima')
    >>> deploy_model(
            model = arima, model_name = 'arima-for-deployment',
            platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'}
        )


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

    return _CURRENT_EXPERIMENT.deploy_model(
        model=model,
        model_name=model_name,
        authentication=authentication,
        platform=platform,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def save_model(model, model_name: str, model_only: bool = True, verbose: bool = True):

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

    return _CURRENT_EXPERIMENT.save_model(
        model=model, model_name=model_name, model_only=model_only, verbose=verbose
    )


# not using check_if_global_is_not_none on purpose
def load_model(
    model_name: str,
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

    experiment = _CURRENT_EXPERIMENT
    if experiment is None:
        experiment = _EXPERIMENT_CLASS()

    return experiment.load_model(
        model_name=model_name,
        platform=platform,
        authentication=authentication,
        verbose=verbose,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def pull(pop: bool = False) -> pd.DataFrame:
    """
    Returns last printed score grid. Use ``pull`` function after
    any training function to store the score grid in pandas.DataFrame.


    pop: bool, default = False
        If True, will pop (remove) the returned dataframe from the
        display container.


    Returns:
        pandas.DataFrame

    """
    return _CURRENT_EXPERIMENT.pull(pop=pop)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def models(
    type: Optional[str] = None, internal: bool = False, raise_errors: bool = True
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
    return _CURRENT_EXPERIMENT.models(
        type=type, internal=internal, raise_errors=raise_errors
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_metrics(
    reset: bool = False, include_custom: bool = True, raise_errors: bool = True
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

    return _CURRENT_EXPERIMENT.get_metrics(
        reset=reset, include_custom=include_custom, raise_errors=raise_errors
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def add_metric(
    id: str, name: str, score_func: type, greater_is_better: bool = True, **kwargs
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

    return _CURRENT_EXPERIMENT.add_metric(
        id=id,
        name=name,
        score_func=score_func,
        greater_is_better=greater_is_better,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def remove_metric(name_or_id: str):

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
    return _CURRENT_EXPERIMENT.remove_metric(name_or_id=name_or_id)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_logs(experiment_name: Optional[str] = None, save: bool = False) -> pd.DataFrame:

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

    return _CURRENT_EXPERIMENT.get_logs(experiment_name=experiment_name, save=save)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_config(variable: str):

    """
    This function retrieves the global variables created when initializing the
    ``setup`` function. Following variables are accessible:

    - X: Period/Index of X
    - y: Time Series as pd.Series
    - X_train: Period/Index of X_train
    - y_train: Time Series as pd.Series (Train set only)
    - X_test: Period/Index of X_test
    - y_test: Time Series as pd.Series  (Test set only)
    - fh: forecast horizon
    - enforce_pi: enforce prediction interval in models
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment
    - logging_param: log_experiment param
    - log_plots_param: log_plots param
    - USI: Unique session ID parameter
    - data_before_preprocess: data before preprocessing
    - gpu_param: use_gpu param configured through setup
    - fold_generator: CV splitter configured in fold_strategy
    - fold_param: fold params defined in the setup
    - seasonality_present: seasonality as detected in the setup
    - seasonality_period: seasonality_period as detected in the setup


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> airline = get_data('airline')
    >>> from pycaret.time_series import *
    >>> exp_name = setup(data = airline,  fh = 12)
    >>> X_train = get_config('X_train')


    Returns:
        Global variable


    """

    return _CURRENT_EXPERIMENT.get_config(variable=variable)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def set_config(variable: str, value):

    """
    This function resets the global variables. Following variables are
    accessible:

    - X: Period/Index of X
    - y: Time Series as pd.Series
    - X_train: Period/Index of X_train
    - y_train: Time Series as pd.Series (Train set only)
    - X_test: Period/Index of X_test
    - y_test: Time Series as pd.Series  (Test set only)
    - fh: forecast horizon
    - enforce_pi: enforce prediction interval in models
    - seed: random state set through session_id
    - prep_pipe: Transformation pipeline
    - n_jobs_param: n_jobs parameter used in model training
    - html_param: html_param configured through setup
    - master_model_container: model storage container
    - display_container: results display container
    - exp_name_log: Name of experiment
    - logging_param: log_experiment param
    - log_plots_param: log_plots param
    - USI: Unique session ID parameter
    - data_before_preprocess: data before preprocessing
    - gpu_param: use_gpu param configured through setup
    - fold_generator: CV splitter configured in fold_strategy
    - fold_param: fold params defined in the setup
    - seasonality_present: seasonality as detected in the setup
    - seasonality_period: seasonality_period as detected in the setup


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> airline = get_data('airline')
    >>> from pycaret.time_series import *
    >>> exp_name = setup(data = airline,  fh = 12)
    >>> set_config('seed', 123)


    Returns:
        None

    """

    return _CURRENT_EXPERIMENT.set_config(variable=variable, value=value)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def save_config(file_name: str):

    """
    This function save all global variables to a pickle file, allowing to
    later resume without rerunning the ``setup``.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> airline = get_data('airline')
    >>> from pycaret.time_series import *
    >>> exp_name = setup(data = airline,  fh = 12)
    >>> save_config('myvars.pkl')


    Returns:
        None

    """

    return _CURRENT_EXPERIMENT.save_config(file_name=file_name)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def load_config(file_name: str):

    """
    This function loads global variables from a pickle file into Python
    environment.


    Example
    -------
    >>> from pycaret.time_series import load_config
    >>> load_config('myvars.pkl')


    Returns:
        Global variables

    """

    return _CURRENT_EXPERIMENT.load_config(file_name=file_name)


def set_current_experiment(experiment: TSForecastingExperiment):
    """
    Set the current experiment to be used with the functional API.

    experiment: TSForecastingExperiment
        Experiment object to use.

    Returns:
        None
    """
    global _CURRENT_EXPERIMENT

    if not isinstance(experiment, TSForecastingExperiment):
        raise TypeError(
            f"experiment must be a PyCaret TSForecastingExperiment object, got {type(experiment)}."
        )
    _CURRENT_EXPERIMENT = experiment


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def check_stats(
    estimator: Optional[Any] = None,
    test: str = "all",
    alpha: float = 0.05,
    split: str = "all",
) -> pd.DataFrame:
    """This function is used to get summary statistics and run statistical
    tests on the original data or model residuals.

    Example
    --------
    >>> from pycaret.datasets import get_data
    >>> airline = get_data('airline')
    >>> from pycaret.time_series import *
    >>> exp_name = setup(data = airline,  fh = 12)
    >>> check_stats(test="summary")
    >>> check_stats(test="adf")
    >>> arima = create_model('arima')
    >>> check_stats(arima, test = 'white_noise')


    Parameters
    ----------
    estimator : sktime compatible object, optional
        Trained model object, by default None


    test : str, optional
        Name of the test to be performed, by default "all"

        Options are:

        * 'summary' - Summary Statistics
        * 'white_noise' - Ljung-Box Test for white noise
        * 'adf' - ADF test for difference stationarity
        * 'kpss' - KPSS test for trend stationarity
        * 'stationarity' - ADF and KPSS test
        * 'normality' - Shapiro Test for Normality
        * 'all' - All of the above tests


    alpha : float, optional
        Significance Level, by default 0.05


    split : str, optional
        The split of the original data to run the test on. Only applicable
        when test is run on the original data (not residuals), by default "all"

        Options are:

        * 'all' - Complete Dataset
        * 'train' - The Training Split of the dataset
        * 'test' - The Test Split of the dataset


    data_type : str, optional
        The data type to use for the statistical test, by default "transformed".

        User may wish to perform the tests on the original data set provided,
        the imputed dataset (if imputation is set) or the transformed dataset
        (which includes any imputation and transformation set by the user).
        This keyword can be used to specify which data type to use.

        Allowed values are: ["original", "imputed", "transformed"]

        NOTE:
        (1) If no imputation is specified, then testing on the "imputed"
            data type will produce the same results as the "original" data type.
        (2) If no transformations are specified, then testing the "transformed"
            data type will produce the same results as the "imputed" data type.
        (3) By default, tests are done on the "transformed" data since that
            is the data that is fed to the model during training.


    data_kwargs : Optional[Dict], optional
        Users can specify `lags list` or `order_list` to run the test for the
        data as well as for its lagged versions, by default None

        >>> check_stats(test="white_noise", data_kwargs={"order_list": [1, 2]})
        >>> check_stats(test="white_noise", data_kwargs={"lags_list": [1, [1, 12]]})


    Returns:
    --------
    pd.DataFrame
        Dataframe with the test results
    """
    return _CURRENT_EXPERIMENT.check_stats(
        estimator=estimator, test=test, alpha=alpha, split=split
    )
