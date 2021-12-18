"""
The purpose of this module is to serve as a central repository of time series models.
The `time_series` module will call `get_all_model_containers()`, which will return
instances of all classes in this module that have `TimeSeriesContainer` as a base
(but not `TimeSeriesContainer` itself). In order to add a new model, you only need
to create a new class that has `TimeSeriesContainer` as a base, set all of the
required parameters in the `__init__` and then call `super().__init__` to complete
the process. Refer to the existing classes for examples.
"""

from typing import Union, Dict, List, Tuple, Any, Optional
from abc import abstractmethod
import random
import numpy as np  # type: ignore
import pandas as pd
import logging

from sktime.forecasting.base import BaseForecaster  # type: ignore
from sktime.forecasting.compose import make_reduction, TransformedTargetForecaster  # type: ignore
from sktime.forecasting.trend import PolynomialTrendForecaster  # type: ignore
from sktime.transformations.series.detrend import ConditionalDeseasonalizer, Detrender  # type: ignore
from sktime.forecasting.base._sktime import DEFAULT_ALPHA  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from pycaret.containers.models.base_model import (
    ModelContainer,
    leftover_parameters_to_categorical_distributions,
)
from pycaret.internal.utils import (
    param_grid_to_lists,
    get_logger,
    np_list_arange,
)
from pycaret.internal.distributions import (
    Distribution,
    UniformDistribution,
    IntUniformDistribution,
    CategoricalDistribution,
)
from pycaret.internal.utils import TSModelTypes
import pycaret.containers.base_container


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
    args : dict, default = {}
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool, default = False
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list, default = {}
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution, default = {}
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict, default = {}
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

    def disable_pred_int_enforcement(self, forecaster, enforce_pi: bool) -> bool:
        """Checks to see if prediction interbal should be enforced. If it should
        but the forecaster does not support it, the container will be disabled

        Parameters
        ----------
        forecaster : `sktime` compatible forecaster
            forecaster to check for prediction interval capability.
            Can be a dummy object of the forecasting class
        enforce_pi : bool
            Should prediction interval be enforced?

        Returns
        -------
        bool
            True if user wants to enforce prediction interval and forecaster
            supports it. False otherwise.
        """
        if enforce_pi and not forecaster.get_tag("capability:pred_int"):
            return False
        return True


#########################
#### BASELINE MODELS ####
#########################


class NaiveContainer(TimeSeriesContainer):
    model_type = TSModelTypes.BASELINE

    def __init__(self, globals_dict: dict) -> None:
        """
        For Naive Forecaster,
          - `sp` must always be 1
          - `strategy` can be either 'last' or 'drift' but not 'mean'
             'mean' is reserved for Grand Means Model
        `sp` is hard coded to 1 irrespective of the `sp` value or whether
        seasonality is detected or not.
        """
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from sktime.forecasting.naive import NaiveForecaster  # type: ignore

        dummy = NaiveForecaster()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
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

    def __init__(self, globals_dict: dict) -> None:
        """
        For Grand Means Forecaster,
          - `sp` must always be 1
          - `strategy` must always be 'mean'
        `sp` is hard coded to 1 irrespective of the `sp` value or whether
        seasonality is detected or not.
        """
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from sktime.forecasting.naive import NaiveForecaster  # type: ignore

        dummy = NaiveForecaster()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
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

    def __init__(self, globals_dict: dict) -> None:
        """
        For Seasonal Naive Model,
          - `sp` must NOT be 1
          - `strategy` can be either 'last' or 'mean'
        If sp = 1, this model is disabled.
        If sp != 1, model is enabled even when seasonality is not detected.
        """
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from sktime.forecasting.naive import NaiveForecaster  # type: ignore

        dummy = NaiveForecaster()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        self.seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        self.sp = sp if sp is not None else 1

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

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from sktime.forecasting.trend import PolynomialTrendForecaster  # type: ignore

        dummy = PolynomialTrendForecaster()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
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
#### CLASSICAL STATISTICAL MODELS ####
######################################


class ArimaContainer(TimeSeriesContainer):
    model_type = TSModelTypes.CLASSICAL

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        random.seed(globals_dict["seed"])
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from sktime.forecasting.arima import ARIMA  # type: ignore

        dummy = ARIMA()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1

        # args = self._set_args
        # tune_args = self._set_tune_args
        # tune_grid = self._set_tune_grid
        # tune_distributions = self._set_tune_distributions

        args = {"seasonal_order": (0, 1, 0, sp)} if seasonality_present else {}
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

            random.seed(globals_dict["seed"])
            np.random.seed(globals_dict["seed"])
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
            sp=sp,
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
            sp=sp,
            seasonal_max_multiplier=seasonal_max_multiplier,
        )
        tune_distributions = {
            "order": CategoricalDistribution(values=orders),
            "seasonal_order": CategoricalDistribution(values=seasonal_orders),
            "with_intercept": CategoricalDistribution(values=[True, False]),
        }

        if not self.gpu_imported:
            args["n_jobs"] = globals_dict["n_jobs_param"]

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

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        self.seed = globals_dict["seed"]
        np.random.seed(self.seed)
        self.gpu_imported = False

        from sktime.forecasting.arima import AutoARIMA  # type: ignore

        dummy = AutoARIMA()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        self.seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        self.sp = sp if sp is not None else 1

        args = self._set_args
        tune_args = self._set_tune_args
        tune_grid = self._set_tune_grid
        tune_distributions = self._set_tune_distributions
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="auto_arima",
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

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from sktime.forecasting.exp_smoothing import ExponentialSmoothing  # type: ignore

        dummy = ExponentialSmoothing()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        self.seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        self.sp = sp if sp is not None else 1

        self.strictly_positive = globals_dict.get("strictly_positive")

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

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from sktime.forecasting.croston import Croston  # type: ignore

        dummy = Croston()
        # check if pi is enforced.
        self.active:bool = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"])

        # if not, make the model unavailiable
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
            is_gpu_enabled=self.gpu_imported
        )

    @property
    def _set_tune_grid(self) -> Dict[str, List[Any]]:
        # lack of research/evidence for suitable range here,
        # SKtime and R implementations are default 0.1
        smoothing_grid: List[float] = [0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        tune_grid = {"smoothing" : smoothing_grid}
        return tune_grid

    @property
    def _set_tune_distributions(self) -> Dict[str, List[Any]]:
        tune_distributions = {"smoothing": UniformDistribution(
                lower=0.01, upper=1, log=True
            )}
        return tune_distributions



class ETSContainer(TimeSeriesContainer):
    model_type = TSModelTypes.CLASSICAL

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from sktime.forecasting.ets import AutoETS  # type: ignore

        dummy = AutoETS()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        self.seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        self.sp = sp if sp is not None else 1

        self.strictly_positive = globals_dict.get("strictly_positive")

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

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from sktime.forecasting.theta import ThetaForecaster  # type: ignore

        dummy = ThetaForecaster()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        self.seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        self.sp = sp if sp is not None else 1

        self.strictly_positive = globals_dict.get("strictly_positive")

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
        # https://github.com/alan-turing-institute/sktime/issues/940
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
        # https://github.com/alan-turing-institute/sktime/issues/692
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
        # https://github.com/alan-turing-institute/sktime/issues/692
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

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        try:
            from sktime.forecasting.tbats import TBATS  # type: ignore
        except ImportError:
            logger.warning("Couldn't import sktime.forecasting.bats")
            self.active = False
            return

        dummy = TBATS()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        sp = globals_dict.get("seasonal_period")
        self.sp = sp if sp is not None else 1

        self.seasonality_present = globals_dict.get("seasonality_present")

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

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        try:
            from sktime.forecasting.bats import BATS  # type: ignore
        except ImportError:
            logger.warning("Couldn't import sktime.forecasting.bats")
            self.active = False
            return

        dummy = BATS()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        sp = globals_dict.get("seasonal_period")
        self.sp = sp if sp is not None else 1

        self.seasonality_present = globals_dict.get("seasonality_present")

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

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        if not ProphetPeriodPatched:
            logger.warning("Couldn't import sktime.forecasting.fbprophet")
            self.active = False
            return

        from sktime.forecasting.fbprophet import Prophet

        dummy = Prophet()
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        sp = globals_dict.get("seasonal_period")
        self.sp = sp if sp is not None else 1

        self.seasonality_present = globals_dict.get("seasonality_present")
        self.freq = globals_dict.get("freq")

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
#### REGRESSION BASED MODELS ####
#################################


class CdsDtContainer(TimeSeriesContainer):
    """Abstract container for sktime  reduced regression forecaster with
    conditional deseasonalizing and detrending.
    """

    active = False
    model_type = None

    def __init__(self, globals_dict: dict) -> None:
        self.logger = get_logger()
        self.seed = globals_dict["seed"]
        np.random.seed(self.seed)

        # Import the right regressor
        self.gpu_imported = False
        self.gpu_param = globals_dict["gpu_param"]
        self.n_jobs_param = globals_dict["n_jobs_param"]

        regressor_class = self.return_regressor_class()  # e.g. LinearRegression
        regressor_args = self._set_regressor_args
        if regressor_class is not None:
            self.regressor = regressor_class(**regressor_args)
        else:
            self.regressor = None

        if self.regressor is None:
            self.active = False
            return

        dummy = BaseCdsDtForecaster(regressor=self.regressor)
        self.active = self.disable_pred_int_enforcement(
            forecaster=dummy, enforce_pi=globals_dict["enforce_pi"]
        )
        if not self.active:
            return

        # Set the model hyperparameters
        sp = globals_dict.get("seasonal_period")
        self.sp = sp if sp is not None else 1

        self.strictly_positive = globals_dict.get("strictly_positive")

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
        from sklearn.linear_model import LinearRegression

        if self.gpu_param == "force":
            from cuml.linear_model import LinearRegression  # type: ignore

            self.logger.info("Imported cuml.linear_model.LinearRegression")
            self.gpu_imported = True
        elif self.gpu_param:
            try:
                from cuml.linear_model import LinearRegression  # type: ignore

                self.logger.info("Imported cuml.linear_model.LinearRegression")
                self.gpu_imported = True
            except ImportError:
                self.logger.warning(
                    "Couldn't import cuml.linear_model.LinearRegression"
                )
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
        from sklearn.linear_model import ElasticNet

        if self.gpu_param == "force":
            from cuml.linear_model import ElasticNet  # type: ignore

            self.logger.info("Imported cuml.linear_model.ElasticNet")
            self.gpu_imported = True
        elif self.gpu_param:
            try:
                from cuml.linear_model import ElasticNet  # type: ignore

                self.logger.info("Imported cuml.linear_model.ElasticNet")
                self.gpu_imported = True
            except ImportError:
                self.logger.warning("Couldn't import cuml.linear_model.ElasticNet")
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
        from sklearn.linear_model import Ridge

        if self.gpu_param == "force":
            from cuml.linear_model import Ridge  # type: ignore

            self.logger.info("Imported cuml.linear_model.Ridge")
            self.gpu_imported = True
        elif self.gpu_param:
            try:
                from cuml.linear_model import Ridge  # type: ignore

                self.logger.info("Imported cuml.linear_model.Ridge")
                self.gpu_imported = True
            except ImportError:
                self.logger.warning("Couldn't import cuml.linear_model.Ridge")
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
        from sklearn.linear_model import Lasso

        if self.gpu_param == "force":
            from cuml.linear_model import Lasso  # type: ignore

            self.logger.info("Imported cuml.linear_model.Lasso")
            self.gpu_imported = True
        elif self.gpu_param:
            try:
                from cuml.linear_model import Lasso  # type: ignore

                self.logger.info("Imported cuml.linear_model.Lasso")
                self.gpu_imported = True
            except ImportError:
                self.logger.warning("Couldn't import cuml.linear_model.Lasso")
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

    def __init__(self, globals_dict: dict) -> None:
        self.num_features = len(globals_dict["X_train"].columns)
        super().__init__(globals_dict=globals_dict)

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

    def __init__(self, globals_dict: dict) -> None:
        self.num_features = len(globals_dict["X_train"].columns)
        super().__init__(globals_dict=globals_dict)

    def return_regressor_class(self):
        from sklearn.neighbors import KNeighborsRegressor

        if self.gpu_param == "force":
            from cuml.neighbors import KNeighborsRegressor  # type: ignore

            self.logger.info("Imported cuml.neighbors.KNeighborsRegressor")
            self.gpu_imported = True
        elif self.gpu_param:
            try:
                from cuml.neighbors import KNeighborsRegressor  # type: ignore

                self.logger.info("Imported cuml.neighbors.KNeighborsRegressor")
                self.gpu_imported = True
            except ImportError:
                self.logger.warning(
                    "Couldn't import cuml.neighbors.KNeighborsRegressor"
                )
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
        try:
            import xgboost
        except ImportError:
            self.logger.warning("Couldn't import xgboost.XGBRegressor")
            self.active = False
            return

        xgboost_version = tuple([int(x) for x in xgboost.__version__.split(".")])
        if xgboost_version < (1, 1, 0):
            self.logger.warning(
                f"Wrong xgboost version. Expected xgboost>=1.1.0, got xgboost=={xgboost_version}"
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
                        f"LightGBM GPU mode not available. Consult https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html."
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

    def __init__(self, globals_dict: dict) -> None:
        # suppress output
        logging.getLogger("catboost").setLevel(logging.ERROR)

        self.use_gpu = globals_dict["gpu_param"] == "force" or (
            globals_dict["gpu_param"] and len(globals_dict["X_train"]) >= 50000
        )

        super().__init__(globals_dict=globals_dict)

    def return_regressor_class(self):
        try:
            import catboost
        except ImportError:
            self.logger.warning("Couldn't import catboost.CatBoostRegressor")
            self.active = False
            return

        catboost_version = tuple([int(x) for x in catboost.__version__.split(".")])
        if catboost_version < (0, 23, 2):
            self.logger.warning(
                f"Wrong catboost version. Expected catboost>=0.23.2, got catboost=={catboost_version}"
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
    # https://github.com/alan-turing-institute/sktime/blob/v0.8.0/extension_templates/forecasting.py
    model_type = None

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "univariate-only": True,  # does estimator use the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce-index-type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,
    }

    def __init__(
        self, regressor, sp=1, deseasonal_model="additive", degree=1, window_length=10
    ):
        """Base Class for time series using scikit models which includes
        Conditional Deseasonalizing and Detrending

        Parameters
        ----------
        regressor : [type]
            [description]
        sp : int, optional
            Seasonality period used to deseasonalize, by default 1
        deseasonal_model : str, optional
            model used to deseasonalize, 'multiplicative' or 'additive', by default 'additive'
        degree : int, optional
            degree of detrender, by default 1
        window_length : int, optional
            Window Length used for the Reduced Forecaster, by default 10
        """
        self.regressor = regressor
        self.sp = sp
        self.deseasonal_model = deseasonal_model
        self.degree = degree
        self.window_length = window_length

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
                        window_length=self.window_length,
                        strategy="recursive",
                    ),
                ),
            ]
        )
        self._forecaster.fit(y=y, X=X, fh=fh)
        self._cutoff = self._forecaster.cutoff

        # this should happen last
        self._is_fitted = True

        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        # check_is_fitted(self)

        self.check_is_fitted()
        y = self._forecaster.predict(
            fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha
        )

        return y


try:
    from sktime.forecasting.fbprophet import Prophet  # type: ignore
    from sktime.forecasting.base._base import DEFAULT_ALPHA

    class ProphetPeriodPatched(Prophet):
        def fit(self, y, X=None, fh=None, **fit_params):
            if isinstance(y, (pd.Series, pd.DataFrame)):
                if isinstance(y.index, pd.PeriodIndex):
                    y.index = y.index.to_timestamp(freq=y.index.freq)

            return super().fit(y, X=X, fh=fh, **fit_params)

        def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
            y = super().predict(
                fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha
            )
            try:
                y.index = y.index.to_period(freq=y.index.freq)
            except Exception:
                pass
            return y


except ImportError:
    Prophet = None
    ProphetPeriodPatched = None


class EnsembleTimeSeriesContainer(TimeSeriesContainer):
    model_type = "ensemble"

    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        self.gpu_imported = False

        from pycaret.internal.ensemble import _EnsembleForecasterWithVoting

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # if not self.gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

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
    globals_dict: dict, raise_errors: bool = True
) -> Dict[str, TimeSeriesContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), globals_dict, TimeSeriesContainer, raise_errors
    )


