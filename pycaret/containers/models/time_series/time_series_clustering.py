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


class TimeSeriesClusteringContainer(ModelContainer):
    """
    Tiem series clustering model container class.

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

class TimeSeriesKMeansContainer(TimeSeriesClusteringContainer):
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

class TimeSeriesKMeansDBAContainer(TimeSeriesClusteringContainer):
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


class TimeSeriesKMedoidsContainer(TimeSeriesClusteringContainer):
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


class TimeSeriesKShapesContainer(TimeSeriesClusteringContainer):
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


class TimeSeriesKernelKMeansContainer(TimeSeriesClusteringContainer):
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


class TimeSeriesLloydsContainer(TimeSeriesClusteringContainer):
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
