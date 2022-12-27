"""
The purpose of this module is to serve as a central repository of time series models.
The `time_series` module will call `get_all_model_containers()`, which will return
instances of all classes in this module that have `TimeSeriesContainer` as a base
(but not `TimeSeriesContainer` itself). In order to add a new model, you only need
to create a new class that has `TimeSeriesContainer` as a base, set all of the
required parameters in the `__init__` and then call `super().__init__` to complete
the process. Refer to the existing classes for examples.
"""

import logging
import random
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd
from packaging import version
from sktime.forecasting.base import BaseForecaster  # type: ignore
from sktime.forecasting.compose import (  # type: ignore
    TransformedTargetForecaster,
    make_reduction,
)
from sktime.forecasting.trend import PolynomialTrendForecaster  # type: ignore
from sktime.transformations.series.detrend import (  # type: ignore
    ConditionalDeseasonalizer,
    Detrender,
)
from sktime.transformations.series.summarize import WindowSummarizer

import pycaret.containers.base_container
from pycaret.containers.models.base_model import (
    ModelContainer,
    leftover_parameters_to_categorical_distributions,
)
from pycaret.internal.distributions import (
    CategoricalDistribution,
    Distribution,
    IntUniformDistribution,
    UniformDistribution,
)
from pycaret.utils._dependencies import _check_soft_dependencies
from pycaret.utils.datetime import (
    coerce_datetime_to_period_index,
    coerce_period_to_datetime_index,
)
from pycaret.utils.generic import get_logger, np_list_arange, param_grid_to_lists
from pycaret.utils.time_series import TSModelTypes
from pycaret.utils.time_series.forecasting.models import _check_enforcements

# First one in the list is the default ----
ALL_ALLOWED_ENGINES: Dict[str, List[str]] = {
    "auto_arima": ["pmdarima", "statsforecast"],
    "lr_cds_dt": ["sklearn", "sklearnex"],
    "en_cds_dt": ["sklearn", "sklearnex"],
    "ridge_cds_dt": ["sklearn", "sklearnex"],
    "lasso_cds_dt": ["sklearn", "sklearnex"],
    "lar_cds_dt": ["sklearn"],
    "llar_cds_dt": ["sklearn"],
    "br_cds_dt": ["sklearn"],
    "huber_cds_dt": ["sklearn"],
    "par_cds_dt": ["sklearn"],
    "omp_cds_dt": ["sklearn"],
    "knn_cds_dt": ["sklearn", "sklearnex"],
    "dt_cds_dt": ["sklearn"],
    "rf_cds_dt": ["sklearn"],
    "et_cds_dt": ["sklearn"],
    "gbr_cds_dt": ["sklearn"],
    "ada_cds_dt": ["sklearn"],
    "xgboost_cds_dt": ["sklearn"],
    "lightgbm_cds_dt": ["sklearn"],
    "catboost_cds_dt": ["sklearn"],
    # "svm_cds_dt": ["sklearn", "sklearnex"],
}


def get_container_default_engines() -> Dict[str, str]:
    """Get the default engines from all models

    Returns
    -------
    Dict[str, str]
        Default engines for all containers. If unspecified, it is not included
        in the return dictionary.
    """
    default_engines = {}
    for id, all_engines in ALL_ALLOWED_ENGINES.items():
        default_engines[id] = all_engines[0]
    return default_engines


class TimeSeriesContainer(ModelContainer):
    """
    Base time series model container class, for easier definition of containers.
    Ensures consistent format before being turned into a dataframe row.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool, default = True
        Should the model be used with 'turbo = True' in compare_models().
    eq_function : type, default = None
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    args : dict, default = {} (empty dict)
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool, default = False
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list, default = {} (empty dict)
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution, default = {} (empty dict)
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict, default = {} (empty dict)
        The arguments to always pass to the tuner.
    is_gpu_enabled : bool, default = None
        If None, will try to automatically determine.
    is_boosting_supported : bool, default = None
        If None, will try to automatically determine.
    tunable : type, default = None
        If a special tunable model is used for tuning, type of
        that model, else None.

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool
        Should the model be used with 'turbo = True' in compare_models().
    eq_function : type
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    args : dict
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict
        The arguments to always pass to the tuner.
    is_gpu_enabled : bool
        If None, will try to automatically determine.
    is_boosting_supported : bool
        If None, will try to automatically determine.
    tunable : type
        If a special tunable model is used for tuning, type of
        that model, else None.

    """

    def __init__(
        self,
        id: str,
        name: str,
        class_def: type,
        is_turbo: bool = True,
        eq_function: Optional[type] = None,
        args: Dict[str, Any] = None,
        is_special: bool = False,
        tune_grid: Dict[str, list] = None,
        tune_distribution: Dict[str, Union[List[Any], Distribution]] = None,
        tune_args: Dict[str, Any] = None,
        is_gpu_enabled: Optional[bool] = None,
        tunable: Optional[type] = None,
    ) -> None:

        if not args:
            args = {}

        if not tune_grid:
            tune_grid = {}

        if not tune_distribution:
            tune_distribution = {}

        if not tune_args:
            tune_args = {}

        super().__init__(
            id=id,
            name=name,
            class_def=class_def,
            eq_function=eq_function,
            args=args,
            is_special=is_special,
        )
        self.is_turbo = is_turbo
        self.tune_grid = param_grid_to_lists(tune_grid)
        self.tune_distribution = tune_distribution
        self.tune_args = tune_args
        self.tunable = tunable

        self.is_boosting_supported = True
        self.is_soft_voting_supported = True

        if is_gpu_enabled is not None:
            self.is_gpu_enabled = is_gpu_enabled
        else:
            self.is_gpu_enabled = bool(self.get_package_name() == "cuml")

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        """
        Returns a dictionary of the model properties, to
        be turned into a pandas DataFrame row.

        Parameters
        ----------
        internal : bool, default = True
            If True, will return all properties. If False, will only
            return properties intended for the user to see.

        Returns
        -------
        dict of str : Any

        """
        d = [
            ("ID", self.id),
            ("Name", self.name),
            ("Reference", self.reference),
            ("Turbo", self.is_turbo),
        ]

        if internal:
            d += [
                ("Special", self.is_special),
                ("Class", self.class_def),
                ("Equality", self.eq_function),
                ("Args", self.args),
                ("Tune Grid", self.tune_grid),
                ("Tune Distributions", self.tune_distribution),
                ("Tune Args", self.tune_args),
                ("GPU Enabled", self.is_gpu_enabled),
                ("Tunable Class", self.tunable),
            ]

        return dict(d)

    @property
    def _set_args(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        return args

    @property
    def _set_tune_args(self) -> Dict[str, Any]:
        tune_args: Dict[str, Any] = {}
        return tune_args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid: Dict[str, List[Any]] = {}
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions: Dict[str, List[Any]] = {}
        return tune_distributions


#########################
# BASELINE MODELS ####
#########################


class NaiveContainer(TimeSeriesContainer):
    model_type = TSModelTypes.BASELINE

    def __init__(self, experiment) -> None:
        """
        For Naive Forecaster,
          - `sp` must always be 1
          - `strategy` can be either 'last' or 'drift' but not 'mean'
             'mean' is reserved for Grand Means Model
        `sp` is hard coded to 1 irrespective of the `sp` value or whether
        seasonality is detected or not.
        """
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.naive import NaiveForecaster  # type: ignore

        # Disable container if certain features are not supported but enforced ----
        dummy = NaiveForecaster()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is NaiveForecaster
            and x.sp == 1
            and (x.strategy == "last" or x.strategy == "drift")
        )

        super().__init__(
            id="naive",
            name="Naive Forecaster",
            class_def=NaiveForecaster,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
            eq_function=eq_function,
        )

    @property
    def _set_args(self) -> Dict[str, Any]:
        args = {"strategy": "last", "sp": 1}
        return args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "strategy": ["last", "drift"],
            "sp": [1],
        }
        return tune_grid


class GrandMeansContainer(TimeSeriesContainer):
    model_type = TSModelTypes.BASELINE

    def __init__(self, experiment) -> None:
        """
        For Grand Means Forecaster,
          - `sp` must always be 1
          - `strategy` must always be 'mean'
        `sp` is hard coded to 1 irrespective of the `sp` value or whether
        seasonality is detected or not.
        """
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.naive import NaiveForecaster  # type: ignore

        # Disable container if certain features are not supported but enforced ----
        dummy = NaiveForecaster()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is NaiveForecaster
            and x.sp == 1
            and (x.strategy == "mean")
        )

        super().__init__(
            id="grand_means",
            name="Grand Means Forecaster",
            class_def=NaiveForecaster,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
            eq_function=eq_function,
        )

    @property
    def _set_args(self) -> Dict[str, Any]:
        args = {"strategy": "mean", "sp": 1}
        return args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "strategy": ["mean"],
            "sp": [1],
        }
        return tune_grid


class SeasonalNaiveContainer(TimeSeriesContainer):
    model_type = TSModelTypes.BASELINE

    def __init__(self, experiment) -> None:
        """
        For Seasonal Naive Model,
          - `sp` must NOT be 1
          - `strategy` can be either 'last' or 'mean'
        If sp = 1, this model is disabled.
        If sp != 1, model is enabled even when seasonality is not detected.
        """
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.naive import NaiveForecaster  # type: ignore

        # Disable container if certain features are not supported but enforced ----
        dummy = NaiveForecaster()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        self.seasonality_present = experiment.seasonality_present
        self.sp = experiment.primary_sp_to_use

        if self.sp == 1:
            self.active = False
            return

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is NaiveForecaster and x.sp != 1

        super().__init__(
            id="snaive",
            name="Seasonal Naive Forecaster",
            class_def=NaiveForecaster,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
            eq_function=eq_function,
        )

    @property
    def _set_args(self) -> Dict[str, Any]:
        args = {"sp": self.sp}
        return args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "strategy": ["last", "mean"],
            "sp": [self.sp, 2 * self.sp],
            # Removing fh for now since it can be less than sp which causes an error
            # Will need to add checks for it later if we want to incorporate it
            "window_length": [None],  # , len(fh)]
        }
        return tune_grid


class PolyTrendContainer(TimeSeriesContainer):
    model_type = TSModelTypes.BASELINE

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.trend import PolynomialTrendForecaster  # type: ignore

        # Disable container if certain features are not supported but enforced ----
        dummy = PolynomialTrendForecaster()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="polytrend",
            name="Polynomial Trend Forecaster",
            class_def=PolynomialTrendForecaster,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
        )

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {"degree": [1, 2, 3, 4, 5], "with_intercept": [True, False]}
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "degree": IntUniformDistribution(lower=1, upper=10),
            "with_intercept": CategoricalDistribution(values=[True, False]),
        }
        return tune_distributions


######################################
# CLASSICAL STATISTICAL MODELS ####
######################################


class ArimaContainer(TimeSeriesContainer):
    model_type = TSModelTypes.CLASSICAL

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        random.seed(experiment.seed)
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.arima import ARIMA  # type: ignore

        # Disable container if certain features are not supported but enforced ----
        dummy = ARIMA()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        seasonality_present = experiment.seasonality_present
        self.sp = experiment.primary_sp_to_use

        # args = self._set_args
        # tune_args = self._set_tune_args
        # tune_grid = self._set_tune_grid
        # tune_distributions = self._set_tune_distributions

        args = {"seasonal_order": (0, 1, 0, self.sp)} if seasonality_present else {}
        tune_args = {}

        def return_order_related_params(
            n_samples: int,
            p_start: int,
            p_end: int,
            d_start: int,
            d_end: int,
            q_start: int,
            q_end: int,
            P_start: int,
            P_end: int,
            D_start: int,
            D_end: int,
            Q_start: int,
            Q_end: int,
            sp: int,
            seasonal_max_multiplier: int,
        ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int, int]]]:

            random.seed(experiment.seed)
            np.random.seed(experiment.seed)
            p_values = [random.randint(p_start, p_end) for _ in range(n_samples)]
            q_values = [random.randint(q_start, q_end) for _ in range(n_samples)]
            d_values = [random.randint(d_start, d_end) for _ in range(n_samples)]
            orders = list(zip(p_values, d_values, q_values))

            # SP values can be 0 (removed) or sp or 2 * sp.
            # 0 was removed --> gives the following error
            # "ValueError: Must include nonzero seasonal periodicity if including seasonal AR, MA, or differencing."
            sp_values_ = [
                sp * seasonal_multiplier
                for seasonal_multiplier in range(1, seasonal_max_multiplier + 1)
            ]
            P_values = [random.randint(P_start, P_end) for _ in range(n_samples)]
            Q_values = [random.randint(Q_start, Q_end) for _ in range(n_samples)]
            D_values = [random.randint(D_start, D_end) for _ in range(n_samples)]
            SP_values = [random.choice(sp_values_) for _ in range(n_samples)]
            seasonal_orders = list(zip(P_values, D_values, Q_values, SP_values))

            return orders, seasonal_orders

        # TODO: With larger values of p, q, we run into the following issues
        # Issue 1: Run Time
        # Issue 2: LinAlgError: LU decomposition error.
        #     - Comes from statsmodels
        #     - https://github.com/statsmodels/statsmodels/issues/5459
        #     - https://stackoverflow.com/questions/54136280/sarimax-python-np-linalg-linalg-linalgerror-lu-decomposition-error
        # Issue 3: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
        #     - Comes from sktime validation after prediction
        # Need to look into this further
        n_samples_grid = 2  # 2 for 'order', 2 for 'seasonal_order', 2 for intercept will give 8 combinations
        seasonal_max_multiplier = (
            1  # Use sp value directly (user can specify 0 if needed)
        )
        p_start = 0
        p_end = 1  # sp-1  # slow run times with higher values, maybe add turbo option
        d_start = 0
        d_end = 1
        q_start = 0
        q_end = 1  # sp-1  # slow run times with higher values, maybe add turbo option
        P_start = 0
        P_end = 1
        D_start = 0
        D_end = 1
        Q_start = 0
        Q_end = 1

        # Technically this is random as well but since there are so many hyperparameter options,
        # this seemed the most reasonable choice rather than manually listing values
        orders, seasonal_orders = return_order_related_params(
            n_samples=n_samples_grid,
            p_start=p_start,
            p_end=p_end,
            d_start=d_start,
            d_end=d_end,
            q_start=q_start,
            q_end=q_end,
            P_start=P_start,
            P_end=P_end,
            D_start=D_start,
            D_end=D_end,
            Q_start=Q_start,
            Q_end=Q_end,
            sp=self.sp,
            seasonal_max_multiplier=seasonal_max_multiplier,
        )
        tune_grid = {
            "order": orders,
            "seasonal_order": seasonal_orders,
            "with_intercept": [True, False],
        }

        n_samples_random = 100
        seasonal_max_multiplier = 2
        orders, seasonal_orders = return_order_related_params(
            n_samples=n_samples_random,
            p_start=p_start,
            p_end=p_end,
            d_start=d_start,
            d_end=d_end,
            q_start=q_start,
            q_end=q_end,
            P_start=P_start,
            P_end=P_end,
            D_start=D_start,
            D_end=D_end,
            Q_start=Q_start,
            Q_end=Q_end,
            sp=self.sp,
            seasonal_max_multiplier=seasonal_max_multiplier,
        )
        tune_distributions = {
            "order": CategoricalDistribution(values=orders),
            "seasonal_order": CategoricalDistribution(values=seasonal_orders),
            "with_intercept": CategoricalDistribution(values=[True, False]),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="arima",
            name="ARIMA",
            class_def=ARIMA,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
        )


class AutoArimaContainer(TimeSeriesContainer):
    model_type = TSModelTypes.CLASSICAL

    def __init__(self, experiment) -> None:
        self.seed = experiment.seed
        np.random.seed(self.seed)
        self.gpu_imported = False

        id = "auto_arima"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "pmdarima":
            from sktime.forecasting.arima import AutoARIMA
        elif self.engine == "statsforecast":
            _check_soft_dependencies("statsforecast", extra="models", severity="error")
            from sktime.forecasting.statsforecast import (
                StatsForecastAutoARIMA as AutoARIMA,
            )

        # Disable container if certain features are not supported but enforced ----
        dummy = AutoARIMA()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        self.seasonality_present = experiment.seasonality_present
        self.sp = experiment.primary_sp_to_use

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id=id,
            name="Auto ARIMA",
            class_def=AutoARIMA,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
        )

    @property
    def _set_args(self) -> Dict[str, Any]:
        # TODO: Check if there is a formal test for type of seasonality
        args = {"sp": self.sp} if self.seasonality_present else {}

        if self.engine == "pmdarima":
            # Add irrespective of whether seasonality is present or not
            args["random_state"] = self.seed
            args["suppress_warnings"] = True
        return args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        if self.seasonality_present:
            # Allow search of p and q till `sp` value
            tune_grid = {
                "max_p": [self.sp],
                "max_q": [self.sp],
            }
        else:
            tune_grid = {}
        tune_grid["max_order"] = [None]  # don't limit the order of params
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        if self.seasonality_present:
            tune_distributions = {
                # Auto ARIMA is slow, consider removing 2 * sp
                "sp": CategoricalDistribution(values=[self.sp, 2 * self.sp]),
            }
        else:
            tune_distributions = {}
        return tune_distributions


class ExponentialSmoothingContainer(TimeSeriesContainer):
    model_type = TSModelTypes.CLASSICAL

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.exp_smoothing import (
            ExponentialSmoothing,  # type: ignore
        )

        # Disable container if certain features are not supported but enforced ----
        dummy = ExponentialSmoothing()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        self.seasonality_present = experiment.seasonality_present
        self.sp = experiment.primary_sp_to_use
        self.strictly_positive = experiment.strictly_positive

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is ExponentialSmoothing

        super().__init__(
            id="exp_smooth",
            name="Exponential Smoothing",
            class_def=ExponentialSmoothing,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
            eq_function=eq_function,  # Added to differentiate between ExponentialSmoothing and Theta which are of same parent class
        )

    @property
    def _set_args(self) -> Dict[str, Any]:
        # TODO: Check if there is a formal test for type of seasonality
        if self.seasonality_present and self.strictly_positive:
            seasonal = "mul"
        elif self.seasonality_present and not self.strictly_positive:
            seasonal = "add"
        else:
            seasonal = None
        args = {"sp": self.sp, "seasonal": seasonal}
        # Add irrespective of whether seasonality is present or not
        args["trend"] = "add"
        return args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        if self.seasonality_present:
            tune_grid = {
                # TODO: Check if add and additive are doing the same thing
                "trend": ["add", "mul", "additive", "multiplicative", None]
                if self.strictly_positive
                else ["add", "additive", None],
                # "damped_trend": [True, False],
                "seasonal": ["add", "mul", "additive", "multiplicative"]
                if self.strictly_positive
                else ["add", "additive"],
                "use_boxcox": [True, False] if self.strictly_positive else [False],
                "sp": [self.sp],
            }
        else:
            tune_grid = {
                # TODO: Check if add and additive are doing the same thing
                "trend": ["add", "mul", "additive", "multiplicative", None]
                if self.strictly_positive
                else ["add", "additive", None],
                # "damped_trend": [True, False],
                "seasonal": [None],
                "use_boxcox": [True, False] if self.strictly_positive else [False],
                "sp": [None],
            }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        if self.seasonality_present:
            tune_distributions = {
                "trend": CategoricalDistribution(
                    values=["add", "mul", "additive", "multiplicative", None]
                    if self.strictly_positive
                    else ["add", "additive", None],
                ),
                # "damped_trend": [True, False],
                "seasonal": CategoricalDistribution(
                    values=["add", "mul", "additive", "multiplicative"]
                    if self.strictly_positive
                    else ["add", "additive"],
                ),
                # "initial_level": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_level has been set.
                # "initial_trend": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_trend has been set.
                # "initial_seasonal": UniformDistribution(lower=0, upper=1), # ValueError: initialization method is estimated but initial_seasonal has been set.
                "use_boxcox": CategoricalDistribution(
                    values=[True, False] if self.strictly_positive else [False]
                ),  # 'log', float
                "sp": CategoricalDistribution(values=[self.sp, 2 * self.sp]),
            }
        else:
            tune_distributions = {
                "trend": CategoricalDistribution(
                    values=["add", "mul", "additive", "multiplicative", None]
                    if self.strictly_positive
                    else ["add", "additive", None],
                ),
                # "damped_trend": [True, False],
                "seasonal": CategoricalDistribution(values=[None]),
                # "initial_level": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_level has been set.
                # "initial_trend": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_trend has been set.
                # "initial_seasonal": UniformDistribution(lower=0, upper=1), # ValueError: initialization method is estimated but initial_seasonal has been set.
                "use_boxcox": CategoricalDistribution(
                    values=[True, False] if self.strictly_positive else [False]
                ),  # 'log', float
                "sp": CategoricalDistribution(values=[None]),
            }
        return tune_distributions


class CrostonContainer(TimeSeriesContainer):
    """
    SKtime documentation:
    https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.forecasting.croston.Croston.html

    """

    model_type = TSModelTypes.CLASSICAL

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.croston import Croston  # type: ignore

        # Disable container if certain features are not supported but enforced ----
        dummy = Croston()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="croston",
            name="Croston",
            class_def=Croston,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            is_gpu_enabled=self.gpu_imported,
        )

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        # lack of research/evidence for suitable range here,
        # SKtime and R implementations are default 0.1
        smoothing_grid: List[float] = [0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        tune_grid = {"smoothing": smoothing_grid}
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "smoothing": UniformDistribution(lower=0.01, upper=1, log=True)
        }
        return tune_distributions


class ETSContainer(TimeSeriesContainer):
    model_type = TSModelTypes.CLASSICAL

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.ets import AutoETS  # type: ignore

        # Disable container if certain features are not supported but enforced ----
        dummy = AutoETS()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        self.seasonality_present = experiment.seasonality_present
        self.sp = experiment.primary_sp_to_use
        self.strictly_positive = experiment.strictly_positive

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ets",
            name="ETS",
            class_def=AutoETS,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
        )

    @property
    def _set_args(self) -> Dict[str, Any]:
        # TODO: Check if there is a formal test for type of seasonality
        if self.seasonality_present and self.strictly_positive:
            seasonal = "mul"
        elif self.seasonality_present and not self.strictly_positive:
            seasonal = "add"
        else:
            seasonal = None
        args = {"sp": self.sp, "seasonal": seasonal}
        # Add irrespective of whether seasonality is present or not
        args["trend"] = "add"
        return args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        if self.seasonality_present:
            tune_grid = {
                "error": ["add", "mul"] if self.strictly_positive else ["add"],
                "trend": ["add", "mul", None]
                if self.strictly_positive
                else ["add", None],
                # "damped_trend": [True, False],
                "seasonal": ["add", "mul"] if self.strictly_positive else ["add"],
                "sp": [self.sp],
            }
        else:
            tune_grid = {
                "error": ["add", "mul"] if self.strictly_positive else ["add"],
                "trend": ["add", "mul", None]
                if self.strictly_positive
                else ["add", None],
                # "damped_trend": [True, False],
                "seasonal": [None],
                "sp": [1],
            }
        return tune_grid


class ThetaContainer(TimeSeriesContainer):
    model_type = TSModelTypes.CLASSICAL

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.theta import ThetaForecaster  # type: ignore

        # Disable container if certain features are not supported but enforced ----
        dummy = ThetaForecaster()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        self.seasonality_present = experiment.seasonality_present
        self.sp = experiment.primary_sp_to_use
        self.strictly_positive = experiment.strictly_positive

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is ThetaForecaster

        super().__init__(
            id="theta",
            name="Theta Forecaster",
            class_def=ThetaForecaster,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
            eq_function=eq_function,  # Added to differentiate between ExponentialSmoothing and Theta which are of same parent class
        )

    @property
    def _set_args(self) -> Dict[str, Any]:
        # https://github.com/sktime/sktime/issues/940
        if self.strictly_positive:
            deseasonalize = True
        else:
            deseasonalize = False

        # sp is automatically ignored if deseasonalize = False
        args = {"sp": self.sp, "deseasonalize": deseasonalize}
        return args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        # TODO; Update after Bug is fixed in sktime
        # https://github.com/sktime/sktime/issues/692
        # ThetaForecaster does not work with "initial_level" different from None
        if self.seasonality_present:
            tune_grid = {
                # "initial_level": [0.1, 0.5, 0.9],
                "deseasonalize": [True] if self.strictly_positive else [False],
                "sp": [self.sp, 2 * self.sp],
            }
        else:
            tune_grid = {
                # "initial_level": [0.1, 0.5, 0.9],
                "deseasonalize": [False],
                "sp": [1],
            }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        # TODO; Update after Bug is fixed in sktime
        # https://github.com/sktime/sktime/issues/692
        # ThetaForecaster does not work with "initial_level" different from None
        if self.seasonality_present:
            tune_distributions = {
                # "initial_level": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_level has been set.
            }
        else:
            tune_distributions = {
                # "initial_level": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_level has been set.
            }
        return tune_distributions


class TBATSContainer(TimeSeriesContainer):
    model_type = TSModelTypes.CLASSICAL

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.tbats import TBATS

        # Disable container if certain features are not supported but enforced ----
        dummy = TBATS()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        self.sp = experiment.all_sps_to_use

        self.seasonality_present = experiment.seasonality_present

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="tbats",
            name="TBATS",
            class_def=TBATS,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
            is_turbo=False,
        )

    @property
    def _set_args(self) -> dict:
        args = (
            {
                "sp": self.sp,
                "use_box_cox": True,
                "use_arma_errors": True,
                "show_warnings": False,
            }
            if self.seasonality_present
            else {}
        )
        return args

    @property
    def _set_tune_grid(self) -> dict:
        tune_grid = {
            "use_damped_trend": [True, False],
            "use_trend": [True, False],
            "sp": [self.sp],
        }
        return tune_grid


class BATSContainer(TimeSeriesContainer):
    model_type = TSModelTypes.CLASSICAL

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from sktime.forecasting.bats import BATS  # type: ignore

        # Disable container if certain features are not supported but enforced ----
        dummy = BATS()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        self.sp = experiment.primary_sp_to_use
        self.seasonality_present = experiment.seasonality_present

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="bats",
            name="BATS",
            class_def=BATS,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
            is_turbo=False,
        )

    @property
    def _set_args(self) -> dict:
        args = (
            {
                "sp": self.sp,
                "use_box_cox": True,
                "use_arma_errors": True,
                "show_warnings": False,
            }
            if self.seasonality_present
            else {}
        )
        return args

    @property
    def _set_tune_grid(self) -> dict:
        tune_grid = {
            "use_damped_trend": [True, False],
            "use_trend": [True, False],
            "sp": [self.sp],
        }
        return tune_grid


class ProphetContainer(TimeSeriesContainer):
    model_type = TSModelTypes.LINEAR

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        if not _check_soft_dependencies("prophet", extra=None, severity="warning"):
            self.active = False
            return

        from sktime.forecasting.fbprophet import Prophet

        # Disable container if certain features are not supported but enforced ----
        dummy = Prophet()
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        # Disable Prophet if Index is not of allowed type (e.g. if it is RangeIndex)
        allowed_index_types = [pd.PeriodIndex, pd.DatetimeIndex]
        index_type = experiment.index_type
        self.active = True if index_type in allowed_index_types else False
        if not self.active:
            return

        self.sp = experiment.primary_sp_to_use
        self.seasonality_present = experiment.seasonality_present

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="prophet",
            name="Prophet",
            class_def=ProphetPeriodPatched,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
            is_turbo=False,
        )

    @property
    def _set_args(self) -> dict:
        return {}

    @property
    def _set_tune_args(self) -> dict:
        return {}

    @property
    def _set_tune_grid(self) -> dict:
        return {"growth": ["linear"]}  # param_grid must not be empty

    @property
    def _set_tune_distributions(self) -> dict:
        return {
            # Based on https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
            "seasonality_mode": CategoricalDistribution(["additive", "multiplicative"]),
            "changepoint_prior_scale": UniformDistribution(
                lower=0.001, upper=0.5, log=True
            ),
            "seasonality_prior_scale": UniformDistribution(
                lower=0.01, upper=10, log=True
            ),
            "holidays_prior_scale": UniformDistribution(lower=0.01, upper=10, log=True),
        }


#################################
# REGRESSION BASED MODELS ####
#################################


class CdsDtContainer(TimeSeriesContainer):
    """Abstract container for sktime  reduced regression forecaster with
    conditional deseasonalizing and detrending.
    """

    active = False
    model_type = None

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        self.seed = experiment.seed
        self.fe_target_rr = experiment.fe_target_rr
        np.random.seed(self.seed)

        # Import the right regressor
        self.gpu_imported = False
        self.gpu_param = experiment.gpu_param
        self.n_jobs_param = experiment.n_jobs_param

        self._set_engine_related_vars(
            id=self.id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        regressor_class = self.return_regressor_class()  # e.g. LinearRegression
        regressor_args = self._set_regressor_args
        if regressor_class is not None:
            self.regressor = regressor_class(**regressor_args)
        else:
            self.regressor = None

        if self.regressor is None:
            self.active = False
            return

        # Disable container if certain features are not supported but enforced ----
        dummy = BaseCdsDtForecaster(regressor=self.regressor)
        self.active = _check_enforcements(forecaster=dummy, experiment=experiment)
        if not self.active:
            return

        # Set the model hyperparameters
        self.sp = experiment.primary_sp_to_use

        self.strictly_positive = experiment.strictly_positive

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDtForecaster
            and type(x.regressor) is regressor_class
        )

        super().__init__(
            id=self.id,
            name=self.name,
            class_def=BaseCdsDtForecaster,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
            eq_function=eq_function,
        )

    @property
    @abstractmethod
    def id(self) -> str:
        """Model ID

        Returns
        -------
        str
            The model id that is used to reference this model
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model Name

        Returns
        -------
        str
            The detailed name of the model used for display purposes
        """
        pass

    @abstractmethod
    def return_regressor_class(self):
        """Returns the Class of the regressor used in the forecaster"""
        pass

    @property
    def _set_regressor_args(self) -> Dict[str, Any]:
        regressor_class = self.return_regressor_class()
        regressor_args: Dict[str, Any] = {}
        if regressor_class is not None:
            regressor = regressor_class()
            if hasattr(regressor, "n_jobs"):
                regressor_args["n_jobs"] = self.n_jobs_param
            if hasattr(regressor, "random_state"):
                regressor_args["random_state"] = self.seed
            if hasattr(regressor, "seed"):
                regressor_args["seed"] = self.seed
        return regressor_args

    @property
    def _set_args(self) -> Dict[str, Any]:
        # Set sp based on seasonal period and window length to 'sp'
        args: Dict[str, Any] = {
            "regressor": self.regressor,
            "sp": self.sp,
            "window_length": self.sp,
            "fe_target_rr": self.fe_target_rr,
        }
        return args


# ===============#
# LINEAR MODELS #
# ===============#


class LinearCdsDtContainer(CdsDtContainer):
    id = "lr_cds_dt"
    name = "Linear w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def return_regressor_class(self):

        if self.engine == "sklearn":
            from sklearn.linear_model import LinearRegression
        elif self.engine == "sklearnex":
            _check_soft_dependencies("sklearnex", extra=None, severity="error")
            from sklearnex.linear_model import LinearRegression

        if self.gpu_param == "force":
            from cuml.linear_model import LinearRegression  # type: ignore

            self.logger.info("Imported cuml.linear_model.LinearRegression")
            self.gpu_imported = True
        elif self.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.linear_model import LinearRegression  # type: ignore

                self.logger.info("Imported cuml.linear_model.LinearRegression")
                self.gpu_imported = True

        return LinearRegression

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
        }
        return tune_distributions


class ElasticNetCdsDtContainer(CdsDtContainer):
    id = "en_cds_dt"
    name = "Elastic Net w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def return_regressor_class(self):

        if self.engine == "sklearn":
            from sklearn.linear_model import ElasticNet
        elif self.engine == "sklearnex":
            _check_soft_dependencies("sklearnex", extra=None, severity="error")
            from sklearnex.linear_model import ElasticNet

        if self.gpu_param == "force":
            from cuml.linear_model import ElasticNet  # type: ignore

            self.logger.info("Imported cuml.linear_model.ElasticNet")
            self.gpu_imported = True
        elif self.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.linear_model import ElasticNet  # type: ignore

                self.logger.info("Imported cuml.linear_model.ElasticNet")
                self.gpu_imported = True

        return ElasticNet

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__alpha": [0.01, 0.1, 1, 10],
            "regressor__l1_ratio": [0.01, 0.1, 0.5, 1.0],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__alpha": UniformDistribution(0, 1),
            "regressor__l1_ratio": UniformDistribution(0.01, 0.9999999999),
        }
        return tune_distributions


class RidgeCdsDtContainer(CdsDtContainer):
    id = "ridge_cds_dt"
    name = "Ridge w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def return_regressor_class(self):

        if self.engine == "sklearn":
            from sklearn.linear_model import Ridge
        elif self.engine == "sklearnex":
            _check_soft_dependencies("sklearnex", extra=None, severity="error")
            from sklearnex.linear_model import Ridge

        if self.gpu_param == "force":
            from cuml.linear_model import Ridge  # type: ignore

            self.logger.info("Imported cuml.linear_model.Ridge")
            self.gpu_imported = True
        elif self.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.linear_model import Ridge  # type: ignore

                self.logger.info("Imported cuml.linear_model.Ridge")
                self.gpu_imported = True

        return Ridge

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__alpha": [0.01, 0.1, 1, 10],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__alpha": UniformDistribution(0.001, 10),
            "regressor__fit_intercept": CategoricalDistribution(values=[True, False]),
            "regressor__normalize": CategoricalDistribution(values=[True, False]),
        }
        return tune_distributions


class LassoCdsDtContainer(CdsDtContainer):
    id = "lasso_cds_dt"
    name = "Lasso w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def return_regressor_class(self):

        if self.engine == "sklearn":
            from sklearn.linear_model import Lasso
        elif self.engine == "sklearnex":
            _check_soft_dependencies("sklearnex", extra=None, severity="error")
            from sklearnex.linear_model import Lasso

        if self.gpu_param == "force":
            from cuml.linear_model import Lasso  # type: ignore

            self.logger.info("Imported cuml.linear_model.Lasso")
            self.gpu_imported = True
        elif self.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.linear_model import Lasso  # type: ignore

                self.logger.info("Imported cuml.linear_model.Lasso")
                self.gpu_imported = True

        return Lasso

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__alpha": [0.01, 0.1, 1, 10],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__alpha": UniformDistribution(0.001, 10),
            "regressor__fit_intercept": CategoricalDistribution(values=[True, False]),
            "regressor__normalize": CategoricalDistribution(values=[True, False]),
        }
        return tune_distributions


class LarsCdsDtContainer(CdsDtContainer):
    id = "lar_cds_dt"
    name = "Least Angular Regressor w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def return_regressor_class(self):
        from sklearn.linear_model import Lars

        return Lars

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__eps": [0.0001, 0.001, 0.01, 0.1],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__eps": UniformDistribution(0.00001, 0.1, log=True),
            "regressor__fit_intercept": CategoricalDistribution(values=[True, False]),
            "regressor__normalize": CategoricalDistribution(values=[True, False]),
        }
        return tune_distributions


class LassoLarsCdsDtContainer(CdsDtContainer):
    id = "llar_cds_dt"
    name = "Lasso Least Angular Regressor w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def return_regressor_class(self):
        from sklearn.linear_model import LassoLars

        return LassoLars

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__alpha": [0.01, 0.1, 1, 10],
            "regressor__eps": [0.0001, 0.001, 0.01, 0.1],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__alpha": UniformDistribution(0.0000001, 1, log=True),
            "regressor__eps": UniformDistribution(0.00001, 0.1, log=True),
            "regressor__fit_intercept": CategoricalDistribution(values=[True, False]),
            "regressor__normalize": CategoricalDistribution(values=[True, False]),
        }
        return tune_distributions


class BayesianRidgeCdsDtContainer(CdsDtContainer):
    id = "br_cds_dt"
    name = "Bayesian Ridge w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def return_regressor_class(self):
        from sklearn.linear_model import BayesianRidge

        return BayesianRidge

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__alpha_1": [0.01, 0.1],
            "regressor__alpha_2": [0.01, 0.1],
            "regressor__lambda_1": [0.01, 0.1],
            "regressor__lambda_2": [0.01, 0.1],
            "regressor__compute_score": [True, False],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__alpha_1": UniformDistribution(
                0.0000000001, 0.9999999999, log=True
            ),
            "regressor__alpha_2": UniformDistribution(
                0.0000000001, 0.9999999999, log=True
            ),
            "regressor__lambda_1": UniformDistribution(
                0.0000000001, 0.9999999999, log=True
            ),
            "regressor__lambda_2": UniformDistribution(
                0.0000000001, 0.9999999999, log=True
            ),
        }
        return tune_distributions


class HuberCdsDtContainer(CdsDtContainer):
    id = "huber_cds_dt"
    name = "Huber w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def return_regressor_class(self):
        from sklearn.linear_model import HuberRegressor

        return HuberRegressor

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__epsilon": [1, 1.5, 1.9],
            "regressor__alpha": [
                0.0000001,
                0.000001,
                0.00001,
                0.0001,
                0.001,
                0.01,
                0.1,
                0.9,
            ],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__epsilon": UniformDistribution(1, 2),
            "regressor__alpha": UniformDistribution(0.0000000001, 0.9999999999),
        }
        return tune_distributions


class PassiveAggressiveCdsDtContainer(CdsDtContainer):
    id = "par_cds_dt"
    name = "Passive Aggressive w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def return_regressor_class(self):
        from sklearn.linear_model import PassiveAggressiveRegressor

        return PassiveAggressiveRegressor

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__epsilon": [0.1, 0.5, 0.9],
            "regressor__C": [0, 5, 10],
            "regressor__fit_intercept": [True, False],
            "regressor__loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            "regressor__shuffle": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__C": UniformDistribution(0, 10),
            "regressor__epsilon": UniformDistribution(0.0000000001, 0.9999999999),
        }
        return tune_distributions


class OrthogonalMatchingPursuitCdsDtContainer(CdsDtContainer):
    id = "omp_cds_dt"
    name = "Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.LINEAR

    def __init__(self, experiment) -> None:
        if experiment.X_train is None:
            self.num_features = 0
        else:
            self.num_features = len(experiment.X_train.columns)
        super().__init__(experiment=experiment)

    def return_regressor_class(self):
        from sklearn.linear_model import OrthogonalMatchingPursuit

        return OrthogonalMatchingPursuit

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_nonzero_coefs": np_list_arange(
                1, self.num_features + 1, 1, inclusive=True
            ),
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
        }
        return tune_distributions


# =======================#
# NEIGHBORS BASED MODELS #
# =======================#


class KNeighborsCdsDtContainer(CdsDtContainer):
    id = "knn_cds_dt"
    name = "K Neighbors w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.NEIGHBORS

    def __init__(self, experiment) -> None:
        if experiment.X_train is None:
            self.num_features = 0
        else:
            self.num_features = len(experiment.X_train.columns)
        super().__init__(experiment=experiment)

    def return_regressor_class(self):
        if self.engine == "sklearn":
            from sklearn.neighbors import KNeighborsRegressor
        elif self.engine == "sklearnex":
            _check_soft_dependencies("sklearnex", extra=None, severity="error")
            from sklearnex.neighbors import KNeighborsRegressor

        if self.gpu_param == "force":
            from cuml.neighbors import KNeighborsRegressor  # type: ignore

            self.logger.info("Imported cuml.neighbors.KNeighborsRegressor")
            self.gpu_imported = True
        elif self.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.neighbors import KNeighborsRegressor  # type: ignore

                self.logger.info("Imported cuml.neighbors.KNeighborsRegressor")
                self.gpu_imported = True

        return KNeighborsRegressor

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {}

        # common
        tune_grid["sp"] = [self.sp]
        tune_grid["deseasonal_model"] = ["additive"]
        tune_grid["degree"] = [1]
        tune_grid["window_length"] = [10]
        tune_grid["regressor__n_neighbors"] = range(1, 51, 10)
        tune_grid["regressor__weights"] = ["uniform"]
        tune_grid["regressor__metric"] = ["minkowski", "euclidean", "manhattan"]

        if not self.gpu_imported:
            tune_grid["regressor__weights"] += ["distance"]
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {}
        tune_distributions["regressor__n_neighbors"] = IntUniformDistribution(1, 51)
        return tune_distributions


# ==================#
# TREE BASED MODELS #
# ==================#


class DecisionTreeCdsDtContainer(CdsDtContainer):
    id = "dt_cds_dt"
    name = "Decision Tree w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.TREE

    def return_regressor_class(self):
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__max_depth": np_list_arange(1, 10, 10, inclusive=True),
            "regressor__max_features": [1.0, "sqrt", "log2"],
            "regressor__min_impurity_decrease": [0.1, 0.5],
            "regressor__min_samples_leaf": [2, 6],
            "regressor__min_samples_split": [2, 10],
            "regressor__criterion": ["mse", "mae", "friedman_mse"],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__max_depth": IntUniformDistribution(lower=1, upper=10),
            "regressor__min_impurity_decrease": UniformDistribution(
                lower=0.000000001, upper=0.5, log=True
            ),
            # "regressor__max_features": UniformDistribution(0.4, 1.0),  # TODO: Adding this eventually samples outside this range - strange!
            "regressor__min_samples_leaf": IntUniformDistribution(2, 6),
            "regressor__min_samples_split": IntUniformDistribution(2, 10),
        }
        return tune_distributions


class RandomForestCdsDtContainer(CdsDtContainer):
    id = "rf_cds_dt"
    name = "Random Forest w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.TREE

    def return_regressor_class(self):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_estimators": np_list_arange(10, 300, 150, inclusive=True),
            "regressor__max_depth": np_list_arange(1, 10, 10, inclusive=True),
            "regressor__min_impurity_decrease": [0.1, 0.5],
            "regressor__max_features": [1.0, "sqrt", "log2"],
            "regressor__bootstrap": [True, False],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__n_estimators": IntUniformDistribution(lower=10, upper=300),
            "regressor__max_depth": IntUniformDistribution(lower=1, upper=10),
            "regressor__min_impurity_decrease": UniformDistribution(lower=0, upper=0.5),
            "regressor__max_features": CategoricalDistribution(
                values=[1.0, "sqrt", "log2"]
            ),
            "regressor__bootstrap": CategoricalDistribution(values=[True, False]),
        }
        return tune_distributions


class ExtraTreesCdsDtContainer(CdsDtContainer):
    id = "et_cds_dt"
    name = "Extra Trees w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.TREE

    def return_regressor_class(self):
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_estimators": np_list_arange(10, 300, 150, inclusive=True),
            # "regressor__criterion": ["mse", "mae"],  # Too many combinations
            "regressor__max_depth": np_list_arange(1, 10, 10, inclusive=True),
            "regressor__min_impurity_decrease": [0.1, 0.5],
            "regressor__max_features": [1.0, "sqrt", "log2"],
            "regressor__bootstrap": [True, False],
            # "regressor__min_samples_split": [2, 5, 7, 9, 10],  # Too many combinations
            # "regressor__min_samples_leaf": [2, 3, 4, 5, 6],  # Too many combinations
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__n_estimators": IntUniformDistribution(lower=10, upper=300),
            "regressor__criterion": CategoricalDistribution(values=["mse", "mae"]),
            "regressor__max_depth": IntUniformDistribution(lower=1, upper=11),
            "regressor__min_impurity_decrease": UniformDistribution(
                0.000000001, 0.5, log=True
            ),
            "regressor__max_features": CategoricalDistribution(
                values=[0.4, 1.0, "sqrt", "log2"]
            ),
            "regressor__bootstrap": CategoricalDistribution(values=[True, False]),
            "regressor__min_samples_split": IntUniformDistribution(lower=2, upper=10),
            "regressor__min_samples_leaf": IntUniformDistribution(lower=1, upper=5),
        }
        return tune_distributions


# ========================#
# GRADIENT BOOSTED MODELS #
# ========================#


class GradientBoostingCdsDtContainer(CdsDtContainer):
    id = "gbr_cds_dt"
    name = "Gradient Boosting w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.TREE

    def return_regressor_class(self):
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_estimators": np_list_arange(10, 300, 150, inclusive=True),
            "regressor__learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "regressor__subsample": [0.5, 1],
            "regressor__max_depth": np_list_arange(1, 10, 10, inclusive=True),
            "regressor__min_impurity_decrease": [0.1, 0.5],
            "regressor__max_features": [1.0, "sqrt", "log2"],
            # "regressor__min_samples_split": [2, 5, 7, 9, 10],  # Too many combinations
            # "regressor__min_samples_leaf": [2, 3, 4, 5, 6],  # Too many combinations
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__n_estimators": IntUniformDistribution(10, 300),
            "regressor__learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            # "regressor__subsample": UniformDistribution(0.2, 1),  # TODO: Adding this eventually samples outside this range - strange!
            "regressor__min_samples_split": IntUniformDistribution(2, 10),
            "regressor__min_samples_leaf": IntUniformDistribution(1, 5),
            "regressor__max_depth": IntUniformDistribution(1, 11),
            # "regressor__max_features": UniformDistribution(0.4, 1.0),  # TODO: Adding this eventually samples outside this range - strange!
            "regressor__min_impurity_decrease": UniformDistribution(
                0.000000001, 0.5, log=True
            ),
        }
        return tune_distributions


class AdaBoostCdsDtContainer(CdsDtContainer):
    id = "ada_cds_dt"
    name = "AdaBoost w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.TREE

    def return_regressor_class(self):
        from sklearn.ensemble import AdaBoostRegressor

        return AdaBoostRegressor

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_estimators": np_list_arange(10, 300, 150, inclusive=True),
            "regressor__learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "regressor__loss": ["linear", "square", "exponential"],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__n_estimators": IntUniformDistribution(10, 300),
            "regressor__learning_rate": UniformDistribution(0.000001, 0.5, log=True),
        }
        return tune_distributions


class XGBCdsDtContainer(CdsDtContainer):
    id = "xgboost_cds_dt"
    name = "Extreme Gradient Boosting w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.TREE

    def return_regressor_class(self):
        if _check_soft_dependencies("xgboost", extra="models", severity="warning"):
            import xgboost
        else:
            self.active = False
            return

        if version.parse(xgboost.__version__) < version.parse("1.1.0"):
            self.logger.warning(
                f"Wrong xgboost version. Expected xgboost>=1.1.0, got xgboost=={xgboost.__version__}"
            )
            self.active = False
            return

        from xgboost import XGBRegressor

        return XGBRegressor

    @property
    def _set_regressor_args(self) -> Dict[str, Any]:
        regressor_args = super()._set_regressor_args
        regressor_args["verbosity"] = 0
        regressor_args["booster"] = "gbtree"
        regressor_args["tree_method"] = "gpu_hist" if self.gpu_param else "auto"
        return regressor_args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_estimators": np_list_arange(10, 300, 150, inclusive=True),
            "regressor__learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "regressor__max_depth": np_list_arange(1, 10, 10, inclusive=True),
            "regressor__subsample": [0.5, 1],
            "regressor__colsample_bytree": [0.5, 1],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "regressor__n_estimators": IntUniformDistribution(10, 300),
            "regressor__max_depth": IntUniformDistribution(1, 10),
            # "regressor__subsample": UniformDistribution(0.2, 1),  # TODO: Adding this eventually samples outside this range - strange!
            # "regressor__colsample_bytree": UniformDistribution(0.5, 1), # TODO: Adding this eventually samples outside this range - strange!
            "regressor__min_child_weight": IntUniformDistribution(1, 4),
            "regressor__reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "regressor__reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            "regressor__scale_pos_weight": UniformDistribution(1, 50),
        }
        return tune_distributions


class LGBMCdsDtContainer(CdsDtContainer):
    id = "lightgbm_cds_dt"
    name = "Light Gradient Boosting w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.TREE

    def return_regressor_class(self):
        from lightgbm import LGBMRegressor
        from lightgbm.basic import LightGBMError

        self.is_gpu_enabled = False
        if self.gpu_param:
            try:
                lgb = LGBMRegressor(device="gpu")
                lgb.fit(np.zeros((2, 2)), [0, 1])
                self.is_gpu_enabled = True
                del lgb
            except LightGBMError:
                self.is_gpu_enabled = False
                if self.gpu_param == "force":
                    raise RuntimeError(
                        "LightGBM GPU mode not available. Consult https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html."
                    )

        return LGBMRegressor

    @property
    def _set_regressor_args(self) -> Dict[str, Any]:
        regressor_args = super()._set_regressor_args
        if self.is_gpu_enabled:
            regressor_args["device"] = "gpu"
        return regressor_args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            # [LightGBM] [Warning] Accuracy may be bad since you didn't explicitly set num_leaves OR 2^max_depth > num_leaves. (num_leaves=31).
            "regressor__num_leaves": [2, 8],  # 31 is default
            "regressor__n_estimators": np_list_arange(10, 300, 150, inclusive=True),
            "regressor__learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "regressor__max_depth": np_list_arange(1, 10, 10, inclusive=True),
            "regressor__subsample": [0.5, 1],
            "regressor__colsample_bytree": [0.5, 1],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            "regressor__num_leaves": IntUniformDistribution(2, 256),
            "regressor__n_estimators": IntUniformDistribution(10, 300),
            "regressor__learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "regressor__max_depth": IntUniformDistribution(1, 10),
            # "regressor__subsample": UniformDistribution(0, 1),  # TODO: Potentially lightgbm.basic.LightGBMError: Check failed: (num_data) > (0)
            # "regressor__colsample_bytree": UniformDistribution(0, 1),  # TODO: Potentially lightgbm.basic.LightGBMError: Check failed: (num_data) > (0)
            # "regressor__min_split_gain": UniformDistribution(0, 1),  # TODO: lightgbm.basic.LightGBMError: Check failed: (num_data) > (0)
            "regressor__reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "regressor__reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            # # "regressor__feature_fraction": UniformDistribution(0.4, 1),  # TODO: Adding this eventually samples outside this range - strange!
            # # "regressor__bagging_fraction": UniformDistribution(0.4, 1),  # TODO: Adding this eventually samples outside this range - strange!
            "regressor__bagging_freq": IntUniformDistribution(0, 7),
            "regressor__min_child_samples": IntUniformDistribution(1, 100),
        }
        return tune_distributions


class CatBoostCdsDtContainer(CdsDtContainer):
    id = "catboost_cds_dt"
    name = "CatBoost Regressor w/ Cond. Deseasonalize & Detrending"
    active = True  # set back to True as the parent has False
    model_type = TSModelTypes.TREE

    def __init__(self, experiment) -> None:
        # suppress output
        logging.getLogger("catboost").setLevel(logging.ERROR)

        self.use_gpu = experiment.gpu_param == "force" or (
            experiment.gpu_param and len(experiment.y_train) >= 50000
        )

        super().__init__(experiment=experiment)

    def return_regressor_class(self):
        if _check_soft_dependencies("catboost", extra="models", severity="warning"):
            import catboost
        else:
            self.active = False
            return

        if version.parse(catboost.__version__) < version.parse("0.23.2"):
            self.logger.warning(
                f"Wrong catboost version. Expected catboost>=0.23.2, got catboost=={catboost.__version__}"
            )
            self.active = False
            return

        from catboost import CatBoostRegressor

        return CatBoostRegressor

    @property
    def _set_regressor_args(self) -> Dict[str, Any]:
        regressor_args = super()._set_regressor_args
        regressor_args["verbose"] = False
        regressor_args["thread_count"] = self.n_jobs_param
        regressor_args["task_type"] = "GPU" if self.use_gpu else "CPU"
        regressor_args["border_count"] = 32 if self.use_gpu else 254
        return regressor_args

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        tune_grid = {
            "sp": [self.sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__eta": [
                0.00001,
                0.01,
                0.5,
            ],  # too low values leads to learning rate errors (Learning rate should be non-zero)
            "regressor__depth": [1, 6, 12],
            "regressor__n_estimators": np_list_arange(10, 100, 100, inclusive=True),
            "regressor__random_strength": np_list_arange(
                0.01, 0.8, 0.8, inclusive=True
            ),
            "regressor__l2_leaf_reg": [1, 30, 200],
        }
        if self.use_gpu:
            tune_grid["regressor__depth"] = [1, 5, 9]
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[self.sp, 2 * self.sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
                if self.strictly_positive
                else ["additive"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=self.sp, upper=2 * self.sp),
            # # TODO: Including any of these regressor parameters results in error
            # TypeError: Parameter value is not iterable or distribution (key='sp', value=CategoricalDistribution(values=[12, 24]))
            # "regressor__eta": UniformDistribution(0.000001, 0.5, log=True),
            # "regressor__depth": IntUniformDistribution(1, 11),
            # "regressor__n_estimators": IntUniformDistribution(10, 300),
            # "regressor__random_strength": UniformDistribution(0, 0.8),
            # "regressor__l2_leaf_reg": IntUniformDistribution(1, 200, log=True),
        }
        if self.use_gpu:
            tune_distributions["regressor__depth"] = IntUniformDistribution(1, 8)
        return tune_distributions


# ===================================#
# TODO: MODELS TO BE SEPARATED LATER #
# ===================================#


class BaseCdsDtForecaster(BaseForecaster):
    # https://github.com/sktime/sktime/blob/v0.8.0/extension_templates/forecasting.py
    model_type = None

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator use the exogenous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce-index-type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,
    }

    def __init__(
        self,
        regressor: Any,
        sp: int = 1,
        deseasonal_model: str = "additive",
        degree: int = 1,
        window_length: int = 10,
        fe_target_rr: Optional[list] = None,
    ):
        """Base Class for time series using scikit models which includes
        Conditional Deseasonalizing and Detrending

        Parameters
        ----------
        regressor : Any
            The regressor to be used for the reduced regression model
        sp : int, optional
            Seasonality period used to deseasonalize, by default 1
        deseasonal_model : str, optional
            model used to deseasonalize, 'multiplicative' or 'additive', by default 'additive'
        degree : int, optional
            degree of detrender, by default 1
        window_length : int, optional
            Window Length used for the Reduced Forecaster, by default 10.
            If fe_target_rr is provided, window_length is ignored.
        fe_target_rr : Optional[list], optional
            Custom transformations used to extract features from the target (useful
            for extracting lagged features), by default None which takes the lags
            based on a window_length parameter. If provided, window_length is ignored.
        """
        self.regressor = regressor
        self.sp = sp
        self.deseasonal_model = deseasonal_model
        self.degree = degree
        self.window_length = window_length

        if fe_target_rr is None:
            # All target lags as features.
            # NOTE: Previously, this forecaster class used the `window_length` argument
            # in make_reduction. Now we have moved to using the `transformers` argument.
            # The order of columns matter for some models like tree based models
            # Hence we start with the furthest away lag and end with the most recent lag.
            # This behavior matches the behavior of the `window_length`` argument in
            # make_reduction which is used in this forecaster class.
            kwargs = {
                "lag_feature": {"lag": list(np.arange(self.window_length, 0, -1))}
            }
            self.fe_target_rr = [WindowSummarizer(**kwargs, n_jobs=1)]
        else:
            self.fe_target_rr = fe_target_rr

        super(BaseCdsDtForecaster, self).__init__()

    def _fit(self, y, X=None, fh=None):
        self._forecaster = TransformedTargetForecaster(
            [
                (
                    "conditional_deseasonalise",
                    ConditionalDeseasonalizer(model=self.deseasonal_model, sp=self.sp),
                ),
                (
                    "detrend",
                    Detrender(forecaster=PolynomialTrendForecaster(degree=self.degree)),
                ),
                (
                    "forecast",
                    make_reduction(
                        estimator=self.regressor,
                        scitype="tabular-regressor",
                        transformers=self.fe_target_rr,
                        window_length=None,
                        strategy="recursive",
                        pooling="global",
                    ),
                ),
            ]
        )
        self._forecaster.fit(y=y, X=X, fh=fh)
        self._cutoff = self._forecaster.cutoff

        # this should happen last
        self._is_fitted = True

        return self

    def _predict(self, fh=None, X=None):
        self.check_is_fitted()
        y = self._forecaster.predict(fh=fh, X=X)

        return y


if _check_soft_dependencies("prophet", extra=None, severity="warning"):
    from sktime.forecasting.fbprophet import Prophet  # type: ignore

    class ProphetPeriodPatched(Prophet):
        def fit(self, y, X=None, fh=None, **fit_params):
            # sktime Prophet only supports DatetimeIndex
            # Hence coerce the index if it is not DatetimeIndex
            y = coerce_period_to_datetime_index(y)
            X = coerce_period_to_datetime_index(X)
            return super().fit(y, X=X, fh=fh, **fit_params)

        @staticmethod
        def _get_orig_freq(X):
            # Store original frequency setting for later ----
            orig_freq = None
            if isinstance(X, (pd.DataFrame, pd.Series)):
                orig_freq = X.index.freq
            return orig_freq

        @staticmethod
        def _coerce_datetime_to_period_index(preds, orig_freq):
            try:
                # preds has freq=None. Hence passing using original_freq for conversion.
                preds = coerce_datetime_to_period_index(data=preds, freq=orig_freq)
            except Exception as exception:
                warnings.warn(
                    "Exception occurred in ProphetPeriodPatched predict method "
                    "during conversion from DatetimeIndex to PeriodIndex: \n"
                    f"{exception}"
                )

            return preds

        def predict(self, fh=None, X=None):
            """Forecast time series at future horizon.
            Parameters
            ----------
            fh : int, list, np.array or ForecastingHorizon
                Forecasting horizon
            X : pd.DataFrame, required
                Exogenous time series
            Returns
            -------
            y_pred : pd.Series
                Point predictions
            """

            # Store original frequency setting for later ----
            orig_freq = self._get_orig_freq(X)

            # TODO: Disable Prophet when Index is of any type other than DatetimeIndex or PeriodIndex
            # In that case, pycaret will always pass PeriodIndex from outside
            # since Datetime index are converted to PeriodIndex in pycaret
            # Ref: https://github.com/sktime/sktime/blob/v0.10.0/sktime/forecasting/base/_fh.py#L524
            # But sktime Prophet only supports DatetimeIndex
            # Hence coerce the index internally if it is not DatetimeIndex
            X = coerce_period_to_datetime_index(X)

            y = super().predict(fh=fh, X=X)

            # sktime Prophet returns back DatetimeIndex
            # Convert back to PeriodIndex for pycaret
            y = self._coerce_datetime_to_period_index(preds=y, orig_freq=orig_freq)

            return y

        def predict_quantiles(self, fh, X=None, alpha=None):
            """Compute/return prediction quantiles for a forecast.

            private _predict_quantiles containing the core logic,
                called from predict_quantiles and possibly predict_interval

            State required:
                Requires state to be "fitted".

            Accesses in self:
                Fitted model attributes ending in "_"
                self.cutoff

            Parameters
            ----------
            fh : guaranteed to be ForecastingHorizon
                The forecasting horizon with the steps ahead to to predict.
            X : optional (default=None)
                guaranteed to be of a type in self.get_tag("X_inner_mtype")
                Exogeneous time series to predict from.
            alpha : list of float (guaranteed not None and floats in [0,1] interval)
                A list of probabilities at which quantile forecasts are computed.

            Returns
            -------
            pred_quantiles : pd.DataFrame
                Column has multi-index: first level is variable name from y in fit,
                    second level being the quantile forecasts for each alpha.
                    Quantile forecasts are calculated for each a in alpha.
                Row index is fh. Entries are quantile forecasts, for var in col index,
                    at quantile probability in second-level col index, for each row index.
            """

            # Store original frequency setting for later ----
            orig_freq = self._get_orig_freq(X)

            # TODO: Disable Prophet when Index is of any type other than DatetimeIndex or PeriodIndex
            # In that case, pycaret will always pass PeriodIndex from outside
            # since Datetime index are converted to PeriodIndex in pycaret
            # Ref: https://github.com/sktime/sktime/blob/v0.10.0/sktime/forecasting/base/_fh.py#L524
            # But sktime Prophet only supports DatetimeIndex
            # Hence coerce the index internally if it is not DatetimeIndex
            X = coerce_period_to_datetime_index(X)

            preds = super().predict_quantiles(fh=fh, X=X, alpha=alpha)

            # sktime Prophet returns back DatetimeIndex
            # Convert back to PeriodIndex for pycaret
            preds = self._coerce_datetime_to_period_index(
                preds=preds, orig_freq=orig_freq
            )

            return preds


class EnsembleTimeSeriesContainer(TimeSeriesContainer):
    model_type = "ensemble"

    def __init__(self, experiment) -> None:
        self.logger = get_logger()
        np.random.seed(experiment.seed)
        self.gpu_imported = False

        from pycaret.internal.ensemble import _EnsembleForecasterWithVoting

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # if not self.gpu_imported:
        #     args["n_jobs"] = experiment.n_jobs_param

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ensemble_forecaster",
            name="EnsembleForecaster",
            class_def=_EnsembleForecasterWithVoting,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=self.gpu_imported,
        )


def get_all_model_containers(
    experiment, raise_errors: bool = True
) -> Dict[str, TimeSeriesContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), experiment, TimeSeriesContainer, raise_errors
    )
