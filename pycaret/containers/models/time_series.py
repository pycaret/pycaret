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
from sktime.transformations.series.detrend import Deseasonalizer, Detrender  # type: ignore
from sktime.forecasting.base._sktime import DEFAULT_ALPHA  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore

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
    CategoricalDistribution
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
        tune_distribution: Dict[str, Distribution] = None,
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


class NaiveContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.naive import NaiveForecaster  # type: ignore

        args = {}
        tune_args = {}
        sp = globals_dict["seasonal_parameter"]
        # fh = globals_dict["fh"]

        tune_grid = {
            "strategy": ['last', 'mean', 'drift'],
            "sp": [1, sp, 2*sp],
            # Removing fh for now since it can be less than sp which causes an error
            # Will need to add checks for it later if we want to incorporate it
            "window_length": [None] # , len(fh)]
        }
        tune_distributions = {
        }

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

        from sktime.forecasting.trend import PolynomialTrendForecaster

        args = {}
        tune_args = {}
        tune_grid = {
            "degree": [1,2,3,4,5],
            "with_intercept": [True, False]
        }
        tune_distributions = {
            "degree": IntUniformDistribution(lower=1, upper=10),
            "with_intercept": CategoricalDistribution(values=[True, False])
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


class ArimaContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        random.seed(globals_dict["seed"])
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.arima import ARIMA

        args = {}
        tune_args = {}
        sp = globals_dict["seasonal_parameter"]

        def return_order_related_params(
            n_samples: int,
            p_start: int, p_end: int,
            d_start: int, d_end: int,
            q_start: int, q_end: int,
            P_start: int, P_end: int,
            D_start: int, D_end: int,
            Q_start: int, Q_end: int,
            sp: int, seasonal_max_multiplier: int
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
            sp_values_ = [sp * seasonal_multiplier for seasonal_multiplier in range(1, seasonal_max_multiplier+1)]
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
        seasonal_max_multiplier = 1 # Use sp value directly (user can specify 0 if needed)
        p_start=0
        p_end=2 # sp-1  # slow run times with higher values, maybe add turbo option
        d_start=0
        d_end=1
        q_start=0
        q_end=2 # sp-1  # slow run times with higher values, maybe add turbo option
        P_start=0
        P_end=2
        D_start=0
        D_end=1
        Q_start=0
        Q_end=2

        # Technically this is random as well but since there are so many hyperparameter options,
        # this seemed the most reasonable choice rather than manually listing values
        orders, seasonal_orders = return_order_related_params(
            n_samples=n_samples_grid,
            p_start=p_start, p_end=p_end,
            d_start=d_start, d_end=d_end,
            q_start=q_start, q_end=q_end,
            P_start=P_start, P_end=P_end,
            D_start=D_start, D_end=D_end,
            Q_start=Q_start, Q_end=Q_end,
            sp=sp, seasonal_max_multiplier=seasonal_max_multiplier
        )
        tune_grid = {
            "order": orders,
            "seasonal_order": seasonal_orders,
            "with_intercept": [True, False]
        }

        n_samples_random = 100
        seasonal_max_multiplier = 2
        orders, seasonal_orders = return_order_related_params(
            n_samples=n_samples_random,
            p_start=p_start, p_end=p_end,
            d_start=d_start, d_end=d_end,
            q_start=q_start, q_end=q_end,
            P_start=P_start, P_end=P_end,
            D_start=D_start, D_end=D_end,
            Q_start=Q_start, Q_end=Q_end,
            sp=sp, seasonal_max_multiplier=seasonal_max_multiplier
        )
        tune_distributions = {
            "order": CategoricalDistribution(values=orders),
            "seasonal_order": CategoricalDistribution(values=seasonal_orders),
            "with_intercept": CategoricalDistribution(values=[True, False])
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

        from sktime.forecasting.exp_smoothing import ExponentialSmoothing

        args = {}
        tune_args = {}
        sp = globals_dict["seasonal_parameter"]

        # tune_grid = {"fit_intercept": [True, False], "normalize": [True, False]}
        tune_grid = {
            "trend": ["add", "mul", "additive", "multiplicative", None],   # TODO: Check if add and additive are doing the same thing
            # "damped_trend": [True, False],
            "seasonal": ["add", "mul", "additive", "multiplicative", None],
            "use_boxcox": [True, False],
            "sp": [sp]
        }
        tune_distributions = {
            "trend": CategoricalDistribution(values=["add", "mul", "additive", "multiplicative", None]),
            # "damped_trend": [True, False],
            "seasonal": CategoricalDistribution(values=["add", "mul", "additive", "multiplicative", None]),
            # "initial_level": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_level has been set.
            # "initial_trend": UniformDistribution(lower=0, upper=1),  # ValueError: initialization method is estimated but initial_trend has been set.
            # "initial_seasonal": UniformDistribution(lower=0, upper=1), # ValueError: initialization method is estimated but initial_seasonal has been set.
            "use_boxcox": CategoricalDistribution(values=[True, False]),  # 'log', float
            "sp": CategoricalDistribution(values=[None, sp, 2*sp])
        }

        # if not gpu_imported:
        #     args["n_jobs"] = globals_dict["n_jobs_param"]

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
            eq_function=eq_function  # Added to differentiate between ExponentialSmoothing and Theta which are of same parent class
        )


# # TODO: Does not work
# class AutoETSContainer(TimeSeriesContainer):
#     def __init__(self, globals_dict: dict) -> None:
#         logger = get_logger()
#         np.random.seed(globals_dict["seed"])
#         gpu_imported = False

#         from sktime.forecasting.ets import AutoETS

#         args = {}
#         tune_args = {}
#         # tune_grid = {"fit_intercept": [True, False], "normalize": [True, False]}
#         tune_grid = {}
#         tune_distributions = {}

#         # if not gpu_imported:
#         #     args["n_jobs"] = globals_dict["n_jobs_param"]

#         leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

#         super().__init__(
#             id="auto_ets",
#             name="AutoETS",
#             class_def=AutoETS,
#             args=args,
#             tune_grid=tune_grid,
#             tune_distribution=tune_distributions,
#             tune_args=tune_args,
#             is_gpu_enabled=gpu_imported
#         )


class ThetaContainer(TimeSeriesContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sktime.forecasting.theta import ThetaForecaster

        args = {}
        tune_args = {}
        sp = globals_dict["seasonal_parameter"]
        # TODO; Update after Bug is fixed in sktime
        # https://github.com/alan-turing-institute/sktime/issues/692
        # ThetaForecaster does not work with "initial_level" different from None
        tune_grid = {
            # "initial_level": [0.1, 0.5, 0.9],
            "deseasonalize": [True, False],
            "sp": [1, sp, 2*sp]
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
            eq_function=eq_function  # Added to differentiate between ExponentialSmoothing and Theta which are of same parent class
        )


# TODO: Does not work with blending of models

# pycaret\time_series.py:2140: in _fit_and_score
#     forecaster.fit(y_train, X_train, **fit_params)
# pycaret\internal\ensemble.py:88: in fit
#     self._fit_forecasters(forecasters, y, X, fh)
# ..\..\..\..\AppData\Roaming\Python\Python37\site-packages\sktime\forecasting\base\_meta.py:65: in _fit_forecasters
#     for forecaster in forecasters
# C:\ProgramData\Anaconda3\envs\pycaret_dev\lib\site-packages\joblib\parallel.py:1054: in __call__
#     self.retrieve()
# C:\ProgramData\Anaconda3\envs\pycaret_dev\lib\site-packages\joblib\parallel.py:933: in retrieve
#     self._output.extend(job.get(timeout=self.timeout))
# C:\ProgramData\Anaconda3\envs\pycaret_dev\lib\site-packages\joblib\_parallel_backends.py:542: in wrap_future_result
#     return future.result(timeout=timeout)
# C:\ProgramData\Anaconda3\envs\pycaret_dev\lib\concurrent\futures\_base.py:435: in result
#     return self.__get_result()
# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

# self = <Future at 0x23485de8108 state=finished raised BrokenProcessPool>

#     def __get_result(self):
#         if self._exception:
# >           raise self._exception
# E           joblib.externals.loky.process_executor.BrokenProcessPool: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable.

# class RandomForestDTSContainer(TimeSeriesContainer):
#     def __init__(self, globals_dict: dict) -> None:
#         logger = get_logger()
#         np.random.seed(globals_dict["seed"])
#         gpu_imported = False

#         args = {}
#         tune_args = {}
#         # tune_grid = {"fit_intercept": [True, False], "normalize": [True, False]}
#         tune_grid = {}
#         tune_distributions = {}

#         # if not gpu_imported:
#         #     args["n_jobs"] = globals_dict["n_jobs_param"]

#         leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

#         super().__init__(
#             id="rf_dts",
#             name="RandomForestDTS",
#             class_def=RandomForestDTS,
#             args=args,
#             tune_grid=tune_grid,
#             tune_distribution=tune_distributions,
#             tune_args=tune_args,
#             is_gpu_enabled=gpu_imported
#         )



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


class BaseDTS(_SktimeForecaster):
    def __init__(self, regressor, sp=1, model='additive', degree=1, window_length=10):
        """Base Class for time series using scikit models which includes
        Deseasonalizing and Detrending

        Parameters
        ----------
        regressor : [type]
            [description]
        sp : int, optional
            Seasonality period used to deseasonalize, by default 1
        model : str, optional
            model used to deseasonalize, 'multiplicative' or 'additive', by default 'additive'
        degree : int, optional
            degree of detrender, by default 1
        window_length : int, optional
            Window Length used for the Reduced Forecaster, by default 10
        """
        self.regressor = regressor
        self.sp = sp
        self.model = model
        self.degree = degree
        self.window_length = window_length

    def fit(self, y, X=None, fh=None):
        self.forecaster = TransformedTargetForecaster(
            [
                (
                    "deseasonalise",
                    Deseasonalizer(model=self.model, sp=self.sp)
                ),
                (
                    "detrend",
                    Detrender(
                        forecaster=PolynomialTrendForecaster(degree=self.degree)
                    )
                ),
                (
                    "forecast",
                    ReducedForecaster(
                        regressor=self.regressor,
                        scitype='regressor',
                        window_length=self.window_length,
                        strategy="recursive"
                    ),
                ),
            ]
        )
        self.forecaster.fit(y=y, X=X, fh=fh)
        return self

    # def predict(self, X=None):
    #     return self.forecaster.predict(X=X)

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        return self.forecaster.predict(fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha)


class RandomForestDTS(BaseDTS):
    def __init__(self, sp=1, model='additive', degree=1, window_length=10):
        regressor = RandomForestRegressor()
        super(RandomForestDTS, self).__init__(
            regressor=regressor,
            sp=sp,
            model=model,
            degree=degree,
            window_length=window_length
        )
