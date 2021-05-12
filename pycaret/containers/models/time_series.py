# Module: containers.models.time_series
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of time series models. The `time_series` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `TimeSeriesContainer`
# as a base (but not `TimeSeriesContainer` itself). In order to add a new model, you only need to create a new class that has
# `TimeSeriesContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

import logging
from typing import Union, Dict, List, Tuple, Any, Optional

import numpy as np  # type: ignore
import random

from sktime.forecasting.base._sktime import _SktimeForecaster  # type: ignore
from sktime.forecasting.compose import ReducedForecaster, TransformedTargetForecaster  # type: ignore
from sktime.forecasting.trend import PolynomialTrendForecaster  # type: ignore
from sktime.transformations.series.detrend import ConditionalDeseasonalizer, Deseasonalizer, Detrender  # type: ignore
from sktime.forecasting.base._sktime import DEFAULT_ALPHA  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from pycaret.containers.models.base_model import (
    ModelContainer,
    leftover_parameters_to_categorical_distributions,
)
from pycaret.internal.utils import (
    param_grid_to_lists,
    get_logger,
    get_class_name,
    np_list_arange,
)
from pycaret.internal.distributions import (
    Distribution,
    UniformDistribution,
    IntUniformDistribution,
    CategoricalDistribution,
)
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


#########################
#### BASELINE MODELS ####
#########################


class NaiveContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.naive import NaiveForecaster  # type: ignore

        seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1

        args = {"sp": sp} if seasonality_present else {}
        tune_args: Dict[str, Any] = {}

        # fh = globals_dict["fh"]
        if seasonality_present:
            tune_grid = {
                "strategy": ["last", "mean", "drift"],
                "sp": [sp, 2 * sp],
                # Removing fh for now since it can be less than sp which causes an error
                # Will need to add checks for it later if we want to incorporate it
                "window_length": [None],  # , len(fh)]
            }
            tune_distributions: Dict[str, List[Any]] = {}
        else:
            tune_grid = {
                "strategy": ["last", "mean", "drift"],
                "sp": [1],
                # Removing fh for now since it can be less than sp which causes an error
                # Will need to add checks for it later if we want to incorporate it
                "window_length": [None],  # , len(fh)]
            }
            tune_distributions: Dict[str, List[Any]] = {}

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="naive",
            name="Naive",
            class_def=NaiveForecaster,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
        )


class PolyTrendContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.trend import PolynomialTrendForecaster  # type: ignore

        args = {}
        tune_args = {}
        tune_grid = {"degree": [1, 2, 3, 4, 5], "with_intercept": [True, False]}
        tune_distributions = {
            "degree": IntUniformDistribution(lower=1, upper=10),
            "with_intercept": CategoricalDistribution(values=[True, False]),
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="poly_trend",
            name="PolyTrend",
            class_def=PolynomialTrendForecaster,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
        )


######################################
#### CLASSICAL STATISTICAL MODELS ####
######################################


class ArimaContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        random.seed(globals_dict["seed"])
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.arima import ARIMA  # type: ignore

        seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1

        args = {"seasonal_order": (0, 0, 0, sp)} if seasonality_present else {}
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

        if not gpu_imported:
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
            is_gpu_enabled=gpu_imported,
        )


class ExponentialSmoothingContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.exp_smoothing import ExponentialSmoothing  # type: ignore

        seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1

        args = {"sp": sp, "seasonal": "add"} if seasonality_present else {}
        tune_args = {}

        # tune_grid = {"fit_intercept": [True, False], "normalize": [True, False]}
        tune_grid = {
            "trend": [
                "add",
                "mul",
                "additive",
                "multiplicative",
                None,
            ],  # TODO: Check if add and additive are doing the same thing
            # "damped_trend": [True, False],
            "seasonal": ["add", "mul", "additive", "multiplicative", None],
            "use_boxcox": [True, False],
            "sp": [sp],
        }
        tune_distributions = {
            "trend": CategoricalDistribution(
                values=["add", "mul", "additive", "multiplicative", None]
            ),
            # "damped_trend": [True, False],
            "seasonal": CategoricalDistribution(
                values=["add", "mul", "additive", "multiplicative", None]
            ),
            # "initial_level": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_level has been set.
            # "initial_trend": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_trend has been set.
            # "initial_seasonal": UniformDistribution(lower=0, upper=1), # ValueError: initialization method is estimated but initial_seasonal has been set.
            "use_boxcox": CategoricalDistribution(values=[True, False]),  # 'log', float
            "sp": CategoricalDistribution(values=[None, sp, 2 * sp]),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is ExponentialSmoothing

        super().__init__(
            id="exp_smooth",
            name="ExponentialSmoothing",
            class_def=ExponentialSmoothing,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,  # Added to differentiate between ExponentialSmoothing and Theta which are of same parent class
        )


class AutoETSContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.ets import AutoETS  # type: ignore

        seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1

        args = {"sp": sp, "seasonal": "add"} if seasonality_present else {}
        tune_args = {}

        tune_grid = {
            "error": ["add", "mul"],
            "trend": ["add", "mul", None],
            # "damped_trend": [True, False],
            "seasonal": ["add", "mul", None],
            "sp": [sp],
        }
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="auto_ets",
            name="AutoETS",
            class_def=AutoETS,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
        )


class ThetaContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.theta import ThetaForecaster  # type: ignore

        seasonality_present = globals_dict.get("seasonality_present")
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1

        args = {"sp": sp, "deseasonalize": True} if seasonality_present else {}
        tune_args = {}

        # TODO; Update after Bug is fixed in sktime
        # https://github.com/alan-turing-institute/sktime/issues/692
        # ThetaForecaster does not work with "initial_level" different from None
        tune_grid = {
            # "initial_level": [0.1, 0.5, 0.9],
            "deseasonalize": [True, False],
            "sp": [1, sp, 2 * sp],
        }
        tune_distributions = {
            # "initial_level": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_level has been set.
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is ThetaForecaster

        super().__init__(
            id="theta",
            name="Theta",
            class_def=ThetaForecaster,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,  # Added to differentiate between ExponentialSmoothing and Theta which are of same parent class
        )


class TBATSContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.tbats import TBATS  # type: ignore

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
            is_gpu_enabled=gpu_imported,
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
    def _set_tune_args(self) -> dict:
        return {}

    @property
    def _set_tune_grid(self) -> dict:
        tune_grid = {
            "use_damped_trend": [True, False],
            "use_trend": [True, False],
            "sp": [self.sp],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> dict:
        return {}


class BATSContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.bats import BATS  # type: ignore

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
            is_gpu_enabled=gpu_imported,
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
    def _set_tune_args(self) -> dict:
        return {}

    @property
    def _set_tune_grid(self) -> dict:
        tune_grid = {
            "use_damped_trend": [True, False],
            "use_trend": [True, False],
            "sp": [self.sp],
        }
        return tune_grid

    @property
    def _set_tune_distributions(self) -> dict:
        return {}


#################################
#### REGRESSION BASED MODELS ####
#################################

# ===============#
# LINEAR MODELS #
# ===============#


class LinearCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import LinearRegression

        # TODO add GPU support
        gpu_imported = False

        if globals_dict["gpu_param"] == "force":
            from cuml.linear_model import LinearRegression  # type: ignore

            logger.info("Imported cuml.linear_model.LinearRegression")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml.linear_model import LinearRegression  # type: ignore

                logger.info("Imported cuml.linear_model.LinearRegression")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.linear_model.LinearRegression")

        regressor_args = {}
        if not gpu_imported:
            regressor_args["n_jobs"] = globals_dict["n_jobs_param"]
        regressor = LinearRegression(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt and type(x.regressor) is LinearRegression
        )

        super().__init__(
            id="lr_cds_dt",
            name="Linear w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class ElasticNetCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import ElasticNet

        # TODO add GPU support
        gpu_imported = False

        if globals_dict["gpu_param"] == "force":
            from cuml.linear_model import ElasticNet  # type: ignore

            logger.info("Imported cuml.linear_model.ElasticNet")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml.linear_model import ElasticNet  # type: ignore

                logger.info("Imported cuml.linear_model.ElasticNet")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.linear_model.ElasticNet")

        regressor_args = {}
        if not gpu_imported:
            regressor_args["random_state"] = globals_dict["seed"]
        regressor = ElasticNet(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__alpha": [0.01, 0.1, 1, 10],
            "regressor__l1_ratio": [0.01, 0.1, 0.5, 1.0],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__alpha": UniformDistribution(0, 1),
            "regressor__l1_ratio": UniformDistribution(0.01, 0.9999999999),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is BaseCdsDt and type(x.regressor) is ElasticNet

        super().__init__(
            id="en_cds_dt",
            name="Elastic Net w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class RidgeCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import Ridge

        # TODO add GPU support
        gpu_imported = False

        if globals_dict["gpu_param"] == "force":
            from cuml.linear_model import Ridge  # type: ignore

            logger.info("Imported cuml.linear_model.Ridge")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml.linear_model import Ridge  # type: ignore

                logger.info("Imported cuml.linear_model.Ridge")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.linear_model.Ridge")

        regressor_args = {}
        if not gpu_imported:
            regressor_args["random_state"] = globals_dict["seed"]
        regressor = Ridge(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__alpha": [0.01, 0.1, 1, 10],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__alpha": UniformDistribution(0.001, 10),
            "regressor__fit_intercept": CategoricalDistribution(values=[True, False]),
            "regressor__normalize": CategoricalDistribution(values=[True, False]),
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is BaseCdsDt and type(x.regressor) is Ridge

        super().__init__(
            id="ridge_cds_dt",
            name="Ridge w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class LassoCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import Lasso

        # TODO add GPU support
        gpu_imported = False

        if globals_dict["gpu_param"] == "force":
            from cuml.linear_model import Lasso  # type: ignore

            logger.info("Imported cuml.linear_model.Lasso")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml.linear_model import Lasso  # type: ignore

                logger.info("Imported cuml.linear_model.Lasso")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.linear_model.Lasso")

        regressor_args = {}
        if not gpu_imported:
            regressor_args["random_state"] = globals_dict["seed"]
        regressor = Lasso(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__alpha": [0.01, 0.1, 1, 10],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__alpha": UniformDistribution(0.001, 10),
            "regressor__fit_intercept": CategoricalDistribution(values=[True, False]),
            "regressor__normalize": CategoricalDistribution(values=[True, False]),
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is BaseCdsDt and type(x.regressor) is Lasso

        super().__init__(
            id="lasso_cds_dt",
            name="Lasso w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class LarsCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import Lars

        # TODO add GPU support
        gpu_imported = False

        regressor_args = {"random_state": globals_dict["seed"]}
        regressor = Lars(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__eps": [0.0001, 0.001, 0.01, 0.1],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__eps": UniformDistribution(0.00001, 0.1, log=True),
            "regressor__fit_intercept": CategoricalDistribution(values=[True, False]),
            "regressor__normalize": CategoricalDistribution(values=[True, False]),
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is BaseCdsDt and type(x.regressor) is Lars

        super().__init__(
            id="lar_cds_dt",
            name="Least Angular Regressor w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class LassoLarsCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import LassoLars

        # TODO add GPU support
        gpu_imported = False

        regressor_args = {"random_state": globals_dict["seed"]}
        regressor = LassoLars(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__alpha": [0.01, 0.1, 1, 10],
            "regressor__eps": [0.0001, 0.001, 0.01, 0.1],
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__alpha": UniformDistribution(0.0000001, 1, log=True),
            "regressor__eps": UniformDistribution(0.00001, 0.1, log=True),
            "regressor__fit_intercept": CategoricalDistribution(values=[True, False]),
            "regressor__normalize": CategoricalDistribution(values=[True, False]),
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = lambda x: type(x) is BaseCdsDt and type(x.regressor) is LassoLars

        super().__init__(
            id="llar_cds_dt",
            name="Lasso Least Angular Regressor w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class BayesianRidgeCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import BayesianRidge

        # TODO add GPU support
        gpu_imported = False

        regressor_args = {}
        regressor = BayesianRidge(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
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
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
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

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt and type(x.regressor) is BayesianRidge
        )

        super().__init__(
            id="br_cds_dt",
            name="Bayesian Ridge w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class HuberCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import HuberRegressor

        # TODO add GPU support
        gpu_imported = False

        regressor_args = {}
        regressor = HuberRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
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
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__epsilon": UniformDistribution(1, 2),
            "regressor__alpha": UniformDistribution(0.0000000001, 0.9999999999),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt and type(x.regressor) is HuberRegressor
        )

        super().__init__(
            id="huber_cds_dt",
            name="Huber w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class PassiveAggressiveCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import PassiveAggressiveRegressor

        # TODO add GPU support
        gpu_imported = False

        regressor_args = {"random_state": globals_dict["seed"]}
        regressor = PassiveAggressiveRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__epsilon": [0.1, 0.5, 0.9],
            "regressor__C": [0, 5, 10],
            "regressor__fit_intercept": [True, False],
            "regressor__loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            "regressor__shuffle": [True, False],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__C": UniformDistribution(0, 10),
            "regressor__epsilon": UniformDistribution(0.0000000001, 0.9999999999),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt
            and type(x.regressor) is PassiveAggressiveRegressor
        )

        super().__init__(
            id="par_cds_dt",
            name="Passive Aggressive w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class OrthogonalMatchingPursuitCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.linear_model import OrthogonalMatchingPursuit

        # TODO add GPU support
        gpu_imported = False

        regressor_args = {}
        regressor = OrthogonalMatchingPursuit(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_nonzero_coefs": np_list_arange(
                1, len(globals_dict["X_train"].columns) + 1, 1, inclusive=True
            ),
            "regressor__fit_intercept": [True, False],
            "regressor__normalize": [True, False],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt
            and type(x.regressor) is OrthogonalMatchingPursuit
        )

        super().__init__(
            id="omp_cds_dt",
            name="Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


# =======================#
# NEIGHBORS BASED MODELS #
# =======================#


class KNeighborsCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.neighbors import KNeighborsRegressor

        # TODO add GPU support
        gpu_imported = False

        if globals_dict["gpu_param"] == "force":
            from cuml.neighbors import KNeighborsRegressor  # type: ignore

            logger.info("Imported cuml.neighbors.KNeighborsRegressor")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml.neighbors import KNeighborsRegressor  # type: ignore

                logger.info("Imported cuml.neighbors.KNeighborsRegressor")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.neighbors.KNeighborsRegressor")

        regressor_args = {}
        if not gpu_imported:
            regressor_args["n_jobs"] = globals_dict["n_jobs_param"]
        regressor = KNeighborsRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1

        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["sp"] = [sp]
        tune_grid["deseasonal_model"] = ["additive"]
        tune_grid["degree"] = [1]
        tune_grid["window_length"] = [10]
        tune_grid["regressor__n_neighbors"] = range(1, 51, 10)
        tune_grid["regressor__weights"] = ["uniform"]
        tune_grid["regressor__metric"] = ["minkowski", "euclidean", "manhattan"]

        if not gpu_imported:
            tune_grid["regressor__weights"] += ["distance"]

        tune_distributions["regressor__n_neighbors"] = IntUniformDistribution(1, 51)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt and type(x.regressor) is KNeighborsRegressor
        )

        super().__init__(
            id="knn_cds_dt",
            name="K Neighbors w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


# ===================#
# TREE BASED MODELS #
# ===================#


class DecisionTreeCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.tree import DecisionTreeRegressor

        # TODO add GPU support

        gpu_imported = False

        regressor_args = {"random_state": globals_dict["seed"]}
        regressor = DecisionTreeRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
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
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__max_depth": IntUniformDistribution(lower=1, upper=10),
            "regressor__min_impurity_decrease": UniformDistribution(
                lower=0.000000001, upper=0.5, log=True
            ),
            # "regressor__max_features": UniformDistribution(0.4, 1.0),  # TODO: Adding this eventually samples outside this range - strange!
            "regressor__min_samples_leaf": IntUniformDistribution(2, 6),
            "regressor__min_samples_split": IntUniformDistribution(2, 10),
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt
            and type(x.regressor) is DecisionTreeRegressor
        )

        super().__init__(
            id="dt_cds_dt",
            name="Decision Tree w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class RandomForestCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.ensemble import RandomForestRegressor

        # TODO add GPU support

        gpu_imported = False

        regressor_args = (
            {
                "random_state": globals_dict["seed"],
                "n_jobs": globals_dict["n_jobs_param"],
            }
            if not gpu_imported
            else {"seed": globals_dict["seed"]}
        )
        regressor = RandomForestRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_estimators": np_list_arange(10, 300, 150, inclusive=True),
            "regressor__max_depth": np_list_arange(1, 10, 10, inclusive=True),
            "regressor__min_impurity_decrease": [0.1, 0.5],
            "regressor__max_features": [1.0, "sqrt", "log2"],
            "regressor__bootstrap": [True, False],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__n_estimators": IntUniformDistribution(lower=10, upper=300),
            "regressor__max_depth": IntUniformDistribution(lower=1, upper=10),
            "regressor__min_impurity_decrease": UniformDistribution(lower=0, upper=0.5),
            "regressor__max_features": CategoricalDistribution(
                values=[1.0, "sqrt", "log2"]
            ),
            "regressor__bootstrap": CategoricalDistribution(values=[True, False]),
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt
            and type(x.regressor) is RandomForestRegressor
        )

        super().__init__(
            id="rf_cds_dt",
            name="Random Forest w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class ExtraTreesCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.ensemble import ExtraTreesRegressor

        # TODO add GPU support

        gpu_imported = False

        regressor_args = (
            {
                "random_state": globals_dict["seed"],
                "n_jobs": globals_dict["n_jobs_param"],
            }
            if not gpu_imported
            else {"seed": globals_dict["seed"]}
        )
        regressor = ExtraTreesRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
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
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
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

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt and type(x.regressor) is ExtraTreesRegressor
        )

        super().__init__(
            id="et_cds_dt",
            name="Extra Trees w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


# =========================#
# GRADIENT BOOSTED MODELS #
# =========================#


class GradientBoostingCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.ensemble import GradientBoostingRegressor

        # TODO add GPU support

        gpu_imported = False

        regressor_args = {"random_state": globals_dict["seed"]}
        regressor = GradientBoostingRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
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
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
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

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt
            and type(x.regressor) is GradientBoostingRegressor
        )

        super().__init__(
            id="gbr_cds_dt",
            name="Gradient Boosting w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class AdaBoostCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from sklearn.ensemble import AdaBoostRegressor

        # TODO add GPU support

        gpu_imported = False

        regressor_args = {"random_state": globals_dict["seed"]}
        regressor = AdaBoostRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_estimators": np_list_arange(10, 300, 150, inclusive=True),
            "regressor__learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "regressor__loss": ["linear", "square", "exponential"],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
            "regressor__n_estimators": IntUniformDistribution(10, 300),
            "regressor__learning_rate": UniformDistribution(0.000001, 0.5, log=True),
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt and type(x.regressor) is AdaBoostRegressor
        )

        super().__init__(
            id="ada_cds_dt",
            name="AdaBoost w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class XGBCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        try:
            import xgboost
        except ImportError:
            logger.warning("Couldn't import xgboost.XGBRegressor")
            self.active = False
            return

        xgboost_version = tuple([int(x) for x in xgboost.__version__.split(".")])
        if xgboost_version < (1, 1, 0):
            logger.warning(
                f"Wrong xgboost version. Expected xgboost>=1.1.0, got xgboost=={xgboost_version}"
            )
            self.active = False
            return

        from xgboost import XGBRegressor

        # TODO add GPU support

        gpu_imported = False

        regressor_args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
            "verbosity": 0,
            "booster": "gbtree",
            "tree_method": "gpu_hist" if globals_dict["gpu_param"] else "auto",
        }
        regressor = XGBRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
            "deseasonal_model": ["additive"],
            "degree": [1],
            "window_length": [10],
            "regressor__n_estimators": np_list_arange(10, 300, 150, inclusive=True),
            "regressor__learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "regressor__max_depth": np_list_arange(1, 10, 10, inclusive=True),
            "regressor__subsample": [0.5, 1],
            "regressor__colsample_bytree": [0.5, 1],
        }
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
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

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt and type(x.regressor) is XGBRegressor
        )

        super().__init__(
            id="xgboost_cds_dt",
            name="Extreme Gradient Boosting w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class LGBMCdsDtContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from lightgbm import LGBMRegressor
        from lightgbm.basic import LightGBMError

        # TODO add GPU support

        gpu_imported = False

        regressor_args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
        }
        regressor = LGBMRegressor(**regressor_args)

        args = {"regressor": regressor}
        tune_args = {}
        sp = globals_dict.get("seasonal_period")
        sp = sp if sp is not None else 1
        tune_grid = {
            "sp": [sp],
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
        tune_distributions = {
            "sp": CategoricalDistribution(
                values=[sp, 2 * sp]
            ),  # TODO: 'None' errors out here
            "deseasonal_model": CategoricalDistribution(
                values=["additive", "multiplicative"]
            ),
            "degree": IntUniformDistribution(lower=1, upper=10),
            "window_length": IntUniformDistribution(lower=sp, upper=2 * sp),
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

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        eq_function = (
            lambda x: type(x) is BaseCdsDt and type(x.regressor) is LGBMRegressor
        )

        is_gpu_enabled = False
        if globals_dict["gpu_param"]:
            try:
                lgb = LGBMRegressor(device="gpu")
                lgb.fit(np.zeros((2, 2)), [0, 1])
                is_gpu_enabled = True
                del lgb
            except LightGBMError:
                is_gpu_enabled = False
                if globals_dict["gpu_param"] == "force":
                    raise RuntimeError(
                        f"LightGBM GPU mode not available. Consult https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html."
                    )

        if is_gpu_enabled:
            args["device"] = "gpu"

        super().__init__(
            id="lightgbm_cds_dt",
            name="Light Gradient Boosting w/ Cond. Deseasonalize & Detrending",
            class_def=BaseCdsDt,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            eq_function=eq_function,
        )


class BaseCdsDt(_SktimeForecaster):
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

    def fit(self, y, X=None, fh=None):
        self.forecaster_ = TransformedTargetForecaster(
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
                    ReducedForecaster(
                        regressor=self.regressor,
                        scitype="regressor",
                        window_length=self.window_length,
                        strategy="recursive",
                    ),
                ),
            ]
        )
        self.forecaster_.fit(y=y, X=X, fh=fh)
        return self

    # def predict(self, X=None):
    #     return self.forecaster.predict(X=X)

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        check_is_fitted(self)
        return self.forecaster_.predict(
            fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha
        )


class EnsembleTimeSeriesContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from pycaret.internal.ensemble import _EnsembleForecasterWithVoting

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # if not gpu_imported:
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
            is_gpu_enabled=gpu_imported,
        )


def get_all_model_containers(
    globals_dict: dict, raise_errors: bool = True
) -> Dict[str, TimeSeriesContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), globals_dict, TimeSeriesContainer, raise_errors
    )
