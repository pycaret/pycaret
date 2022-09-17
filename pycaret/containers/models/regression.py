# Module: containers.models.regression
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of regression models. The `regression` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `RegressionContainer`
# as a base (but not `RegressionContainer` itself). In order to add a new model, you only need to create a new class that has
# `RegressionContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from packaging import version

import pycaret.containers.base_container
from pycaret.containers.models.base_model import (
    ModelContainer,
    leftover_parameters_to_categorical_distributions,
)
from pycaret.internal.distributions import (
    Distribution,
    IntUniformDistribution,
    UniformDistribution,
)
from pycaret.utils._dependencies import _check_soft_dependencies
from pycaret.utils.generic import get_logger, np_list_arange, param_grid_to_lists

# First one in the list is the default ----
ALL_ALLOWED_ENGINES: Dict[str, List[str]] = {
    "lr": ["sklearn", "sklearnex"],
    "lasso": ["sklearn", "sklearnex"],
    "ridge": ["sklearn", "sklearnex"],
    "en": ["sklearn", "sklearnex"],
    "knn": ["sklearn", "sklearnex"],
    "svm": ["sklearn", "sklearnex"],
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


class RegressorContainer(ModelContainer):
    """
    Base regression model container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.

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
    shap : bool or str, default = False
        If False, SHAP is not supported. Otherwise, one of 'type1', 'type2' to determine SHAP type.
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
    shap : bool or str
        If False, SHAP is not supported. Otherwise, one of 'type1', 'type2' to determine SHAP type.
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
        shap: Union[bool, str] = False,
        is_gpu_enabled: Optional[bool] = None,
        tunable: Optional[type] = None,
    ) -> None:

        self.shap = shap
        if not (isinstance(shap, bool) or shap in ["type1", "type2"]):
            raise ValueError("shap must be either bool or 'type1', 'type2'.")

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
                ("SHAP", self.shap),
                ("GPU Enabled", self.is_gpu_enabled),
                ("Tunable Class", self.tunable),
            ]

        return dict(d)


class LinearRegressionContainer(RegressorContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "lr"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "sklearn":
            from sklearn.linear_model import LinearRegression
        elif self.engine == "sklearnex":
            if _check_soft_dependencies("sklearnex", extra=None, severity="warning"):
                from sklearnex.linear_model import LinearRegression
            else:
                from sklearn.linear_model import LinearRegression

        if experiment.gpu_param == "force":
            from cuml.linear_model import LinearRegression

            logger.info("Imported cuml.linear_model.LinearRegression")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.linear_model import LinearRegression

                logger.info("Imported cuml.linear_model.LinearRegression")
                gpu_imported = True

        args = {}
        tune_args = {}
        tune_grid = {"fit_intercept": [True, False], "normalize": [True, False]}
        tune_distributions = {}

        if not gpu_imported:
            args["n_jobs"] = experiment.n_jobs_param

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id=id,
            name="Linear Regression",
            class_def=LinearRegression,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class LassoRegressionContainer(RegressorContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "lasso"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "sklearn":
            from sklearn.linear_model import Lasso
        elif self.engine == "sklearnex":
            if _check_soft_dependencies("sklearnex", extra=None, severity="warning"):
                from sklearnex.linear_model import Lasso
            else:
                from sklearn.linear_model import Lasso

        if experiment.gpu_param == "force":
            from cuml.linear_model import Lasso

            logger.info("Imported cuml.linear_model.Lasso")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.linear_model import Lasso

                logger.info("Imported cuml.linear_model.Lasso")
                gpu_imported = True

        args = {}
        tune_args = {}
        tune_grid = {
            "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
        tune_distributions = {"alpha": UniformDistribution(0.001, 10)}

        if not gpu_imported:
            args["random_state"] = experiment.seed

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id=id,
            name="Lasso Regression",
            class_def=Lasso,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class RidgeRegressionContainer(RegressorContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "ridge"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "sklearn":
            from sklearn.linear_model import Ridge
        elif self.engine == "sklearnex":
            if _check_soft_dependencies("sklearnex", extra=None, severity="warning"):
                from sklearnex.linear_model import Ridge
            else:
                from sklearn.linear_model import Ridge

        if experiment.gpu_param == "force":
            from cuml.linear_model import Ridge

            logger.info("Imported cuml.linear_model.Ridge")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.linear_model import Ridge

                logger.info("Imported cuml.linear_model.Ridge")
                gpu_imported = True

        args = {}
        tune_args = {}
        tune_grid = {
            "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
        tune_distributions = {"alpha": UniformDistribution(0.001, 10)}

        if not gpu_imported:
            args["random_state"] = experiment.seed

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id=id,
            name="Ridge Regression",
            class_def=Ridge,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class ElasticNetContainer(RegressorContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "en"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "sklearn":
            from sklearn.linear_model import ElasticNet
        elif self.engine == "sklearnex":
            if _check_soft_dependencies("sklearnex", extra=None, severity="warning"):
                from sklearnex.linear_model import ElasticNet
            else:
                from sklearn.linear_model import ElasticNet

        if experiment.gpu_param == "force":
            from cuml.linear_model import ElasticNet

            logger.info("Imported cuml.linear_model.ElasticNet")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.linear_model import ElasticNet

                logger.info("Imported cuml.linear_model.ElasticNet")
                gpu_imported = True

        args = {}
        tune_args = {}
        tune_grid = {
            "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
            "l1_ratio": np_list_arange(0.01, 1, 0.001, inclusive=False),
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
        tune_distributions = {
            "alpha": UniformDistribution(0, 1),
            "l1_ratio": UniformDistribution(0.01, 0.9999999999),
        }

        if not gpu_imported:
            args["random_state"] = experiment.seed

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id=id,
            name="Elastic Net",
            class_def=ElasticNet,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class LarsContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.linear_model import Lars

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "eps": [
                0.00001,
                0.0001,
                0.001,
                0.01,
                0.05,
                0.0005,
                0.005,
                0.00005,
                0.02,
                0.007,
                0.1,
            ],
        }
        tune_distributions = {
            "eps": UniformDistribution(0.00001, 0.1),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="lar",
            name="Least Angle Regression",
            class_def=Lars,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class LassoLarsContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.linear_model import LassoLars

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
            ],
            "eps": [
                0.00001,
                0.0001,
                0.001,
                0.01,
                0.05,
                0.0005,
                0.005,
                0.00005,
                0.02,
                0.007,
                0.1,
            ],
        }
        tune_distributions = {
            "eps": UniformDistribution(0.00001, 0.1),
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="llar",
            name="Lasso Least Angle Regression",
            class_def=LassoLars,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class OrthogonalMatchingPursuitContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.linear_model import OrthogonalMatchingPursuit

        args = {}
        tune_args = {}
        tune_grid = {
            "n_nonzero_coefs": range(1, len(experiment.X_train.columns) + 1),
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
        tune_distributions = {
            "n_nonzero_coefs": IntUniformDistribution(
                1, len(experiment.X_train.columns)
            )
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="omp",
            name="Orthogonal Matching Pursuit",
            class_def=OrthogonalMatchingPursuit,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class BayesianRidgeContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.linear_model import BayesianRidge

        args = {}
        tune_args = {}
        tune_grid = {
            "alpha_1": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
            ],
            "alpha_2": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
            ],
            "lambda_1": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
            ],
            "lambda_2": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
            ],
            "compute_score": [True, False],
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
        tune_distributions = {
            "alpha_1": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "alpha_2": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "lambda_1": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "lambda_2": UniformDistribution(0.0000000001, 0.9999999999, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="br",
            name="Bayesian Ridge",
            class_def=BayesianRidge,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class AutomaticRelevanceDeterminationContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.linear_model import ARDRegression

        args = {"n_iter": 1000}
        tune_args = {}
        tune_grid = {
            "alpha_1": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
            ],
            "alpha_2": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
            ],
            "lambda_1": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
            ],
            "lambda_2": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
            ],
            "threshold_lambda": [
                5000,
                10000,
                15000,
                20000,
                25000,
                30000,
                35000,
                40000,
                45000,
                50000,
                55000,
                60000,
            ],
            "compute_score": [True, False],
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
        tune_distributions = {
            "alpha_1": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "alpha_2": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "lambda_1": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "lambda_2": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "threshold_lambda": IntUniformDistribution(1000, 100000),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ard",
            name="Automatic Relevance Determination",
            class_def=ARDRegression,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_turbo=False,
            shap=False,
        )


class PassiveAggressiveRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.linear_model import PassiveAggressiveRegressor

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "C": np_list_arange(0, 10, 0.001, inclusive=True),
            "fit_intercept": [True, False],
            "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            "epsilon": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "shuffle": [True, False],
        }
        tune_distributions = {
            "C": UniformDistribution(0, 10),
            "epsilon": UniformDistribution(0.0000000001, 0.9999999999),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="par",
            name="Passive Aggressive Regressor",
            class_def=PassiveAggressiveRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class RANSACRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.linear_model import RANSACRegressor

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "min_samples": np_list_arange(0, 1, 0.05, inclusive=True),
            "max_trials": np_list_arange(1, 20, 1, inclusive=True),
            "max_skips": np_list_arange(1, 20, 1, inclusive=True),
            "stop_n_inliers": np_list_arange(1, 25, 1, inclusive=True),
            "stop_probability": np_list_arange(0, 1, 0.01, inclusive=True),
            "loss": ["absolute_loss", "squared_loss"],
        }
        tune_distributions = {
            "min_samples": UniformDistribution(0, 1),
            "max_trials": IntUniformDistribution(1, 20),
            "max_skips": IntUniformDistribution(1, 20),
            "stop_n_inliers": IntUniformDistribution(1, 25),
            "stop_probability": UniformDistribution(0, 1),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ransac",
            name="Random Sample Consensus",
            class_def=RANSACRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_turbo=False,
            shap=False,
        )


class TheilSenRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.linear_model import TheilSenRegressor

        args = {
            "n_jobs": experiment.n_jobs_param,
            "random_state": experiment.seed,
            "max_iter": 1000,
        }
        tune_args = {}
        tune_grid = {
            "fit_intercept": [True, False],
        }
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="tr",
            name="TheilSen Regressor",
            class_def=TheilSenRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_turbo=False,
            shap=False,
        )


class HuberRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.linear_model import HuberRegressor

        args = {}
        tune_args = {}
        tune_grid = {
            "epsilon": [1, 1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
            ],
            "fit_intercept": [True, False],
        }
        tune_distributions = {
            "epsilon": UniformDistribution(1, 2),
            "alpha": UniformDistribution(0.0000000001, 0.9999999999),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="huber",
            name="Huber Regressor",
            class_def=HuberRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class KernelRidgeContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.kernel_ridge import KernelRidge

        args = {}
        tune_args = {}
        tune_grid = {
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
            ],
        }
        tune_distributions = {
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="kr",
            name="Kernel Ridge",
            class_def=KernelRidge,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_turbo=False,
            shap=False,
        )


class SVRContainer(RegressorContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "svm"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "sklearn":
            from sklearn.svm import SVR
        elif self.engine == "sklearnex":
            if _check_soft_dependencies("sklearnex", extra=None, severity="warning"):
                from sklearnex.svm import SVR
            else:
                from sklearn.svm import SVR

        if experiment.gpu_param == "force":
            from cuml.svm import SVR

            logger.info("Imported cuml.svm.SVR")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.svm import SVR

                logger.info("Imported cuml.svm.SVR")
                gpu_imported = True

        args = {}
        tune_args = {}
        tune_grid = {
            "C": np_list_arange(0, 10, 0.001, inclusive=True),
            "epsilon": [1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
        }
        tune_distributions = {
            "epsilon": UniformDistribution(1, 2),
            "C": UniformDistribution(0, 10),
        }

        if not gpu_imported:
            tune_grid["shrinking"] = [True, False]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id=id,
            name="Support Vector Regression",
            class_def=SVR,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            is_turbo=False,
            shap=False,
        )


class KNeighborsRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "knn"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "sklearn":
            from sklearn.neighbors import KNeighborsRegressor
        elif self.engine == "sklearnex":
            if _check_soft_dependencies("sklearnex", extra=None, severity="warning"):
                from sklearnex.neighbors import KNeighborsRegressor
            else:
                from sklearn.neighbors import KNeighborsRegressor

        if experiment.gpu_param == "force":
            from cuml.neighbors import KNeighborsRegressor

            logger.info("Imported cuml.neighbors.KNeighborsRegressor")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.neighbors import KNeighborsRegressor

                logger.info("Imported cuml.neighbors.KNeighborsRegressor")
                gpu_imported = True

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["n_neighbors"] = range(1, 51)
        tune_grid["weights"] = ["uniform"]
        tune_grid["metric"] = ["minkowski", "euclidean", "manhattan"]

        if not gpu_imported:
            args["n_jobs"] = experiment.n_jobs_param
            tune_grid["weights"] += ["distance"]

        tune_distributions["n_neighbors"] = IntUniformDistribution(1, 51)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id=id,
            name="K Neighbors Regressor",
            class_def=KNeighborsRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class DecisionTreeRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.tree import DecisionTreeRegressor

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "max_depth": np_list_arange(1, 16, 1, inclusive=True),
            "max_features": [1.0, "sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4, 5, 6],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "criterion": ["mse", "mae", "friedman_mse"],
        }
        tune_distributions = {
            "max_depth": IntUniformDistribution(1, 16),
            "max_features": UniformDistribution(0.4, 1),
            "min_samples_leaf": IntUniformDistribution(2, 6),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="dt",
            name="Decision Tree Regressor",
            class_def=DecisionTreeRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
        )


class RandomForestRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        from sklearn.ensemble import RandomForestRegressor

        if experiment.gpu_param == "force":
            import cuml.ensemble

            logger.info("Imported cuml.ensemble")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                import cuml.ensemble

                logger.info("Imported cuml.ensemble")
                gpu_imported = True

        if gpu_imported:
            RandomForestRegressor = (
                pycaret.internal.cuml_wrappers.get_random_forest_regressor()
            )

        if not gpu_imported:
            args = {
                "random_state": experiment.seed,
                "n_jobs": experiment.n_jobs_param,
            }
        else:
            import cuml

            if version.parse(cuml.__version__) >= version.parse("0.19"):
                args = {"random_state": experiment.seed}
            else:
                args = {"seed": experiment.seed}

        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "max_depth": IntUniformDistribution(1, 11),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
            "max_features": UniformDistribution(0.4, 1),
        }

        if gpu_imported:
            tune_grid["split_criterion"] = [2, 3]
        else:
            tune_grid["criterion"] = ["mse", "mae"]
            tune_grid["min_samples_split"] = [2, 5, 7, 9, 10]
            tune_grid["min_samples_leaf"] = [2, 3, 4, 5, 6]
            tune_distributions["min_samples_split"] = IntUniformDistribution(2, 10)
            tune_distributions["min_samples_leaf"] = IntUniformDistribution(2, 6)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="rf",
            name="Random Forest Regressor",
            class_def=RandomForestRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap="type2",
        )


class ExtraTreesRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.ensemble import ExtraTreesRegressor

        args = {
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
        }
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "criterion": ["mse", "mae"],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_samples_leaf": [2, 3, 4, 5, 6],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "max_depth": IntUniformDistribution(1, 11),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_samples_leaf": IntUniformDistribution(1, 5),
            "max_features": UniformDistribution(0.4, 1),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="et",
            name="Extra Trees Regressor",
            class_def=ExtraTreesRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
        )


class AdaBoostRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.ensemble import AdaBoostRegressor

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "loss": ["linear", "square", "exponential"],
        }
        tune_distributions = {
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "n_estimators": IntUniformDistribution(10, 300),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ada",
            name="AdaBoost Regressor",
            class_def=AdaBoostRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class GradientBoostingRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.ensemble import GradientBoostingRegressor

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "subsample": np_list_arange(0.2, 1, 0.05, inclusive=True),
            "min_samples_split": [2, 4, 5, 7, 9, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "subsample": UniformDistribution(0.2, 1),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_samples_leaf": IntUniformDistribution(1, 5),
            "max_depth": IntUniformDistribution(1, 11),
            "max_features": UniformDistribution(0.4, 1),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="gbr",
            name="Gradient Boosting Regressor",
            class_def=GradientBoostingRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class MLPRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)

        from sklearn.neural_network import MLPRegressor

        from pycaret.internal.tunable import TunableMLPRegressor

        args = {"random_state": experiment.seed, "max_iter": 500}
        tune_args = {}
        tune_grid = {
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
            ],
            "hidden_layer_size_0": [50, 100],
            "hidden_layer_size_1": [0, 50, 100],
            "hidden_layer_size_2": [0, 50, 100],
            "activation": ["tanh", "identity", "logistic", "relu"],
        }
        tune_distributions = {
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "hidden_layer_size_0": IntUniformDistribution(50, 100),
            "hidden_layer_size_1": IntUniformDistribution(0, 100),
            "hidden_layer_size_2": IntUniformDistribution(0, 100),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="mlp",
            name="MLP Regressor",
            class_def=MLPRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_turbo=False,
            shap=False,
            tunable=TunableMLPRegressor,
        )


class XGBRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        if _check_soft_dependencies("xgboost", extra="models", severity="warning"):
            import xgboost
        else:
            self.active = False
            return

        if version.parse(xgboost.__version__) < version.parse("1.1.0"):
            logger.warning(
                f"Wrong xgboost version. Expected xgboost>=1.1.0, got xgboost=={xgboost.__version__}"
            )
            self.active = False
            return

        from xgboost import XGBRegressor

        args = {
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
            "verbosity": 0,
            "booster": "gbtree",
            "tree_method": "gpu_hist" if experiment.gpu_param else "auto",
        }
        tune_args = {}
        tune_grid = {
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "subsample": [0.2, 0.3, 0.5, 0.7, 0.9, 1],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "min_child_weight": [1, 2, 3, 4],
            "reg_alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "scale_pos_weight": np_list_arange(0, 50, 0.1, inclusive=True),
        }
        tune_distributions = {
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "n_estimators": IntUniformDistribution(10, 300),
            "subsample": UniformDistribution(0.2, 1),
            "max_depth": IntUniformDistribution(1, 11),
            "colsample_bytree": UniformDistribution(0.5, 1),
            "min_child_weight": IntUniformDistribution(1, 4),
            "reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            "scale_pos_weight": UniformDistribution(1, 50),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="xgboost",
            name="Extreme Gradient Boosting",
            class_def=XGBRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
            is_gpu_enabled=bool(experiment.gpu_param),
        )


class LGBMRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from lightgbm import LGBMRegressor
        from lightgbm.basic import LightGBMError

        args = {
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
        }
        tune_args = {}
        tune_grid = {
            "num_leaves": [
                2,
                4,
                6,
                8,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                150,
                200,
                256,
            ],
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "min_split_gain": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "feature_fraction": np_list_arange(0.4, 1, 0.1, inclusive=True),
            "bagging_fraction": np_list_arange(0.4, 1, 0.1, inclusive=True),
            "bagging_freq": [0, 1, 2, 3, 4, 5, 6, 7],
            "min_child_samples": np_list_arange(1, 100, 5, inclusive=True),
        }
        tune_distributions = {
            "num_leaves": IntUniformDistribution(2, 256),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "n_estimators": IntUniformDistribution(10, 300),
            "min_split_gain": UniformDistribution(0, 1),
            "reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            "feature_fraction": UniformDistribution(0.4, 1),
            "bagging_fraction": UniformDistribution(0.4, 1),
            "bagging_freq": IntUniformDistribution(0, 7),
            "min_child_samples": IntUniformDistribution(1, 100),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        is_gpu_enabled = False
        if experiment.gpu_param:
            try:
                lgb = LGBMRegressor(device="gpu")
                lgb.fit(np.zeros((2, 2)), [0, 1])
                is_gpu_enabled = "gpu"
                del lgb
            except Exception:
                try:
                    lgb = LGBMRegressor(device="cuda")
                    lgb.fit(np.zeros((2, 2)), [0, 1])
                    is_gpu_enabled = "cuda"
                    del lgb
                except LightGBMError:
                    is_gpu_enabled = False
                    if experiment.gpu_param == "force":
                        raise RuntimeError(
                            "LightGBM GPU mode not available. Consult https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html."
                        )

        if is_gpu_enabled == "gpu":
            args["device"] = "gpu"
        elif is_gpu_enabled == "cuda":
            args["device"] = "cuda"

        super().__init__(
            id="lightgbm",
            name="Light Gradient Boosting Machine",
            class_def=LGBMRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
            is_gpu_enabled=is_gpu_enabled,
        )


class CatBoostRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        if _check_soft_dependencies("catboost", extra="models", severity="warning"):
            import catboost
        else:
            self.active = False
            return

        if version.parse(catboost.__version__) < version.parse("0.23.2"):
            logger.warning(
                f"Wrong catboost version. Expected catboost>=0.23.2, got catboost=={catboost.__version__}"
            )
            self.active = False
            return

        from catboost import CatBoostRegressor

        # suppress output
        logging.getLogger("catboost").setLevel(logging.ERROR)

        use_gpu = experiment.gpu_param == "force" or (
            experiment.gpu_param and len(experiment.X_train) >= 50000
        )

        args = {
            "random_state": experiment.seed,
            "verbose": False,
            "thread_count": experiment.n_jobs_param,
            "task_type": "GPU" if use_gpu else "CPU",
            "border_count": 32 if use_gpu else 254,
        }
        tune_args = {}
        tune_grid = {
            "eta": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "depth": list(range(1, 12)),
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "random_strength": np_list_arange(0, 0.8, 0.1, inclusive=True),
            "l2_leaf_reg": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 200],
        }
        tune_distributions = {
            "eta": UniformDistribution(0.000001, 0.5, log=True),
            "depth": IntUniformDistribution(1, 11),
            "n_estimators": IntUniformDistribution(10, 300),
            "random_strength": UniformDistribution(0, 0.8),
            "l2_leaf_reg": IntUniformDistribution(1, 200, log=True),
        }

        if use_gpu:
            tune_grid["depth"] = list(range(1, 9))
            tune_distributions["depth"] = (IntUniformDistribution(1, 8),)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="catboost",
            name="CatBoost Regressor",
            class_def=CatBoostRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
            is_gpu_enabled=use_gpu,
        )


class DummyRegressorContainer(RegressorContainer):
    def __init__(self, experiment) -> None:
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.dummy import DummyRegressor

        args = {"strategy": "mean"}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="dummy",
            name="Dummy Regressor",
            class_def=DummyRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class BaggingRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.ensemble import BaggingRegressor

        args = {
            "random_state": experiment.seed,
            "n_jobs": 1 if experiment.gpu_param else None,
        }
        tune_args = {}
        tune_grid = {
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
            "max_features": np_list_arange(0.4, 1, 0.1, inclusive=True),
            "max_samples": np_list_arange(0.4, 1, 0.1, inclusive=True),
        }
        tune_distributions = {
            "max_features": UniformDistribution(0.4, 1),
            "max_samples": UniformDistribution(0.4, 1),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="Bagging",
            name="Bagging Regressor",
            class_def=BaggingRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=False,
        )


class StackingRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.ensemble import StackingRegressor

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="Stacking",
            name="Stacking Regressor",
            class_def=StackingRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=False,
        )


class VotingRegressorContainer(RegressorContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.ensemble import VotingRegressor

        from pycaret.internal.tunable import TunableVotingRegressor

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # VotingRegressor is a special case. Its weights can be tuned, but we do not know how many of them will be there
        # before it is initiated. Therefore, code to handle it will be added directly to tune_model().

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="Voting",
            name="Voting Regressor",
            class_def=VotingRegressor,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=False,
            tunable=TunableVotingRegressor,
        )


def get_all_model_containers(
    experiment: Any, raise_errors: bool = True
) -> Dict[str, RegressorContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), experiment, RegressorContainer, raise_errors
    )
