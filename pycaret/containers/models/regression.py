# Module: containers.models.regression
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of regression models. The `regression` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `RegressionContainer`
# as a base (but not `RegressionContainer` itself). In order to add a new model, you only need to create a new class that has
# `RegressionContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

import logging
from typing import Union, Dict, Any, Optional
from pycaret.containers.models.base_model import ModelContainer
from pycaret.internal.utils import param_grid_to_lists, get_logger, get_class_name
from pycaret.internal.distributions import *
import pycaret.containers.base_container
import numpy as np


def _leftover_parameters_to_categorical_distributions(
    tune_grid: dict, tune_distributions: dict
) -> None:
    for k, v in tune_grid.items():
        if not k in tune_distributions:
            tune_distributions[k] = CategoricalDistribution(v)


class RegressionContainer(ModelContainer):
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
    shap : bool or str, default = False
        If False, SHAP is not supported. Otherwise, one of 'type1', 'type2' to determine SHAP type.
    is_gpu_enabled : bool, default = None
        If None, will try to automatically determine.
    is_boosting_supported : bool, default = None
        If None, will try to automatically determine.

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
    """

    def __init__(
        self,
        id: str,
        name: str,
        class_def: type,
        is_turbo: bool = True,
        args: Dict[str, Any] = None,
        is_special: bool = False,
        tune_grid: Dict[str, list] = None,
        tune_distribution: Dict[str, Distrubution] = None,
        tune_args: Dict[str, Any] = None,
        shap: Union[bool, str] = False,
        is_gpu_enabled: Optional[bool] = None,
        is_boosting_supported: Optional[bool] = None,
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

        super().__init__(id, name, class_def, args, is_special)
        self.is_turbo = is_turbo
        self.tune_grid = param_grid_to_lists(tune_grid)
        self.tune_distribution = tune_distribution
        self.tune_args = tune_args

        try:
            model_instance = class_def()

            self.is_boosting_supported = bool(
                hasattr(model_instance, "class_weights")
                or hasattr(model_instance, "predict_proba")
            )

            self.is_soft_voting_supported = bool(
                hasattr(model_instance, "predict_proba")
            )

            del model_instance
        except:
            self.is_boosting_supported = False
            self.is_soft_voting_supported = False
        finally:
            if is_boosting_supported is not None:
                self.is_boosting_supported = is_boosting_supported

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
                ("Args", self.args),
                ("Tune Grid", self.tune_grid),
                ("Tune Distributions", self.tune_distribution),
                ("Tune Args", self.tune_args),
                ("SHAP", self.shap),
                ("GPU Enabled", self.is_gpu_enabled),
                ("Boosting Supported", self.is_boosting_supported),
            ]

        return dict(d)


class LinearRegressionContainer(RegressionContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.linear_model import LinearRegression

        # if globals_dict["gpu_param"] == "force":
        #     from cuml.linear_model import LogisticRegression

        #     logger.info("Imported cuml.linear_model.LogisticRegression")
        #     gpu_imported = True
        # elif globals_dict["gpu_param"]:
        #     try:
        #         from cuml.linear_model import LogisticRegression

        #         logger.info("Imported cuml.linear_model.LogisticRegression")
        #         gpu_imported = True
        #     except ImportError:
        #         logger.warning("Couldn't import cuml.linear_model.LogisticRegression")

        args = {"n_jobs": globals_dict["n_jobs_param"]}
        tune_args = {}
        tune_grid = {"fit_intercept": [True, False], "normalize": [True, False]}
        tune_distributions = {}

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="lr",
            name="Linear Regression",
            class_def=LinearRegression,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )

class BayesianRidgeContainer(RegressionContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.linear_model import BayesianRidge

        # if globals_dict["gpu_param"] == "force":
        #     from cuml.linear_model import LogisticRegression

        #     logger.info("Imported cuml.linear_model.LogisticRegression")
        #     gpu_imported = True
        # elif globals_dict["gpu_param"]:
        #     try:
        #         from cuml.linear_model import LogisticRegression

        #         logger.info("Imported cuml.linear_model.LogisticRegression")
        #         gpu_imported = True
        #     except ImportError:
        #         logger.warning("Couldn't import cuml.linear_model.LogisticRegression")

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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


def get_all_model_containers(
    globals_dict: dict, raise_errors: bool = True
) -> Dict[str, RegressionContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), globals_dict, RegressionContainer, raise_errors
    )

