# Module: containers.models.classification
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of classification models. The `classification` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `ClassifierContainer`
# as a base (but not `ClassifierContainer` itself). In order to add a new model, you only need to create a new class that has
# `ClassifierContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

import logging
import pycaret.internal.cuml_wrappers
from typing import Union, Dict, Any, Optional
from pycaret.containers.models.base_model import (
    ModelContainer,
    leftover_parameters_to_categorical_distributions,
)
from pycaret.internal.cuml_wrappers import get_svc_classifier
from pycaret.internal.utils import (
    param_grid_to_lists,
    get_logger,
    get_class_name,
    np_list_arange,
)
from pycaret.internal.distributions import *
import pycaret.containers.base_container
import numpy as np


class ClassifierContainer(ModelContainer):
    """
    Base classification model container class, for easier definition of containers. Ensures consistent format
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
    is_soft_voting_supported : bool, default = None
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
    is_soft_voting_supported : bool
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
        is_boosting_supported: Optional[bool] = None,
        is_soft_voting_supported: Optional[bool] = None,
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
            if is_soft_voting_supported is not None:
                self.is_soft_voting_supported = is_soft_voting_supported

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
                ("Boosting Supported", self.is_boosting_supported),
                ("Soft Voting", self.is_soft_voting_supported),
                ("Tunable Class", self.tunable),
            ]

        return dict(d)


class LogisticRegressionClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.linear_model import LogisticRegression

        if globals_dict["gpu_param"] == "force":
            from cuml.linear_model import LogisticRegression

            logger.info("Imported cuml.linear_model.LogisticRegression")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml.linear_model import LogisticRegression

                logger.info("Imported cuml.linear_model.LogisticRegression")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.linear_model.LogisticRegression")

        args = {"max_iter": 1000}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["penalty"] = ["l2", "none"]
        tune_grid["C"] = np_list_arange(0, 10, 0.001, inclusive=True)

        if gpu_imported:
            tune_grid["penalty"] += ["l1"]
        else:
            args["random_state"] = globals_dict["seed"]

            tune_grid["class_weight"] = ["balanced", {}]

        tune_distributions["C"] = UniformDistribution(0, 10)
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="lr",
            name="Logistic Regression",
            class_def=LogisticRegression,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class KNeighborsClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.neighbors import KNeighborsClassifier

        if globals_dict["gpu_param"] == "force":
            from cuml.neighbors import KNeighborsClassifier

            logger.info("Imported cuml.neighbors.KNeighborsClassifier")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml.neighbors import KNeighborsClassifier

                logger.info("Imported cuml.neighbors.KNeighborsClassifier")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.neighbors.KNeighborsClassifier")

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["n_neighbors"] = range(1, 51)
        tune_grid["weights"] = ["uniform"]
        tune_grid["metric"] = ["minkowski", "euclidean", "manhattan"]

        if not gpu_imported:
            args["n_jobs"] = globals_dict["n_jobs_param"]
            tune_grid["weights"] += ["distance"]

        tune_distributions["n_neighbors"] = IntUniformDistribution(1, 51)
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="knn",
            name="K Neighbors Classifier",
            class_def=KNeighborsClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class GaussianNBClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.naive_bayes import GaussianNB

        args = {}
        tune_args = {}
        tune_grid = {
            "var_smoothing": [
                0.000000001,
                0.000000002,
                0.000000005,
                0.000000008,
                0.000000009,
                0.0000001,
                0.0000002,
                0.0000003,
                0.0000005,
                0.0000007,
                0.0000009,
                0.00001,
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.007,
                0.009,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.1,
                1,
            ]
        }
        tune_distributions = {
            "var_smoothing": UniformDistribution(0.000000001, 1, log=True)
        }

        super().__init__(
            id="nb",
            name="Naive Bayes",
            class_def=GaussianNB,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class DecisionTreeClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.tree import DecisionTreeClassifier

        args = {"random_state": globals_dict["seed"]}
        tune_args = {}
        tune_grid = {
            "max_depth": np_list_arange(1, 16, 1, inclusive=True),
            "max_features": [1.0, "sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4, 5, 6],
            "min_samples_split": [2, 5, 7, 9, 10],
            "criterion": ["gini", "entropy"],
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
            name="Decision Tree Classifier",
            class_def=DecisionTreeClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
        )


class SGDClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.linear_model import SGDClassifier

        if globals_dict["gpu_param"] == "force":
            from cuml import MBSGDClassifier as SGDClassifier

            logger.info("Imported cuml.MBSGDClassifier")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml import MBSGDClassifier as SGDClassifier

                logger.info("Imported cuml.MBSGDClassifier")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.MBSGDClassifier")

        args = {"tol": 0.001, "loss": "hinge", "penalty": "l2", "eta0": 0.001}
        tune_args = {}
        tune_grid = {
            "penalty": ["elasticnet", "l2", "l1"],
            "l1_ratio": np_list_arange(0.0000000001, 1, 0.01, inclusive=False),
            "alpha": [
                0.0000001,
                0.000001,
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
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "fit_intercept": [True, False],
            "learning_rate": ["constant", "invscaling", "adaptive", "optimal"],
            "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        tune_distributions = {
            "l1_ratio": UniformDistribution(0.0000000001, 0.9999999999),
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "eta0": UniformDistribution(0.001, 0.5, log=True),
        }

        if gpu_imported:
            tune_grid["learning_rate"].remove("optimal")
            batch_size = [
                (512, 50000),
                (256, 25000),
                (128, 10000),
                (64, 5000),
                (32, 1000),
                (16, 0),
            ]
            for arg, x_len in batch_size:
                if len(globals_dict["X_train"]) >= x_len:
                    args["batch_size"] = arg
                    break
        else:
            args["random_state"] = globals_dict["seed"]
            args["n_jobs"] = globals_dict["n_jobs_param"]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="svm",
            name="SVM - Linear Kernel",
            class_def=SGDClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class SVCClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.svm import SVC

        if globals_dict["gpu_param"] == "force":
            from cuml.svm import SVC

            logger.info("Imported cuml.svm.SVC")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml.svm import SVC

                logger.info("Imported cuml.svm.SVC")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.svm.SVC")

        args = {
            "gamma": "auto",
            "C": 1.0,
            "probability": True,
            "kernel": "rbf",
            "random_state": globals_dict["seed"],
        }
        tune_args = {}
        tune_grid = {
            "C": np_list_arange(0, 50, 0.01, inclusive=True),
            "class_weight": ["balanced", {}],
        }
        tune_distributions = {
            "C": UniformDistribution(0, 50),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        if gpu_imported:
            SVC = get_svc_classifier()

        super().__init__(
            id="rbfsvm",
            name="SVM - Radial Kernel",
            class_def=SVC,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_turbo=False,
        )


class GaussianProcessClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.gaussian_process import GaussianProcessClassifier

        args = {
            "copy_X_train": False,
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
        }
        tune_args = {}
        tune_grid = {
            "max_iter_predict": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,]
        }
        tune_distributions = {"max_iter_predict": IntUniformDistribution(100, 1000)}

        super().__init__(
            id="gpc",
            name="Gaussian Process Classifier",
            class_def=GaussianProcessClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_turbo=False,
        )


class MLPClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.neural_network import MLPClassifier
        from pycaret.internal.tunable import TunableMLPClassifier

        args = {"random_state": globals_dict["seed"], "max_iter": 500}
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
            name="MLP Classifier",
            class_def=MLPClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_turbo=False,
            tunable=TunableMLPClassifier,
        )


class RidgeClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.linear_model import RidgeClassifier

        if globals_dict["gpu_param"] == "force":
            import cuml.linear_model

            logger.info("Imported cuml.linear_model")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                import cuml.linear_model

                logger.info("Imported cuml.linear_model")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.linear_model")

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        if gpu_imported:
            RidgeClassifier = pycaret.internal.cuml_wrappers.get_ridge_classifier()
        else:
            args = {"random_state": globals_dict["seed"]}

        tune_grid = {
            "normalize": [True, False],
        }

        tune_grid["alpha"] = np_list_arange(0.01, 10, 0.01, inclusive=False)
        tune_grid["fit_intercept"] = [True, False]
        tune_distributions["alpha"] = UniformDistribution(0.001, 10)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ridge",
            name="Ridge Classifier",
            class_def=RidgeClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_gpu_enabled=gpu_imported,
        )
        if gpu_imported:
            self.reference = get_class_name(cuml.linear_model.Ridge)


class RandomForestClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.ensemble import RandomForestClassifier

        if globals_dict["gpu_param"] == "force":
            import cuml.ensemble

            logger.info("Imported cuml.ensemble")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                import cuml.ensemble

                logger.info("Imported cuml.ensemble")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.ensemble")

        if gpu_imported:
            RandomForestClassifier = (
                pycaret.internal.cuml_wrappers.get_random_forest_classifier()
            )

        args = (
            {
                "random_state": globals_dict["seed"],
                "n_jobs": globals_dict["n_jobs_param"],
            }
            if not gpu_imported
            else {"seed": globals_dict["seed"]}
        )
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
            tune_grid["split_criterion"] = [0, 1]
        else:
            tune_grid["criterion"] = ["gini", "entropy"]
            tune_grid["class_weight"] = ["balanced", "balanced_subsample", {}]
            tune_grid["min_samples_split"] = [2, 5, 7, 9, 10]
            tune_grid["min_samples_leaf"] = [2, 3, 4, 5, 6]
            tune_distributions["min_samples_split"] = IntUniformDistribution(2, 10)
            tune_distributions["min_samples_leaf"] = IntUniformDistribution(2, 6)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="rf",
            name="Random Forest Classifier",
            class_def=RandomForestClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
            is_gpu_enabled=gpu_imported,
        )
        if gpu_imported:
            self.reference = get_class_name(cuml.ensemble.RandomForestClassifier)


class QuadraticDiscriminantAnalysisContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        args = {}
        tune_args = {}
        tune_grid = {"reg_param": np_list_arange(0, 1, 0.01, inclusive=True)}
        tune_distributions = {"reg_param": UniformDistribution(0, 1)}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="qda",
            name="Quadratic Discriminant Analysis",
            class_def=QuadraticDiscriminantAnalysis,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class AdaBoostClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.ensemble import AdaBoostClassifier

        args = {"random_state": globals_dict["seed"]}
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
            "algorithm": ["SAMME", "SAMME.R"],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ada",
            name="Ada Boost Classifier",
            class_def=AdaBoostClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class GradientBoostingClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.ensemble import GradientBoostingClassifier

        args = {"random_state": globals_dict["seed"]}
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
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
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
            "max_features": UniformDistribution(0.4, 1),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="gbc",
            name="Gradient Boosting Classifier",
            class_def=GradientBoostingClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class LinearDiscriminantAnalysisContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        args = {}
        tune_args = {}
        tune_grid = {
            "solver": ["lsqr", "eigen"],
            "shrinkage": [
                "empirical",
                "auto",
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
            ],
        }
        tune_distributions = {
            "shrinkage": UniformDistribution(0.0001, 1, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="lda",
            name="Linear Discriminant Analysis",
            class_def=LinearDiscriminantAnalysis,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class ExtraTreesClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.ensemble import ExtraTreesClassifier

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
        }
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "criterion": ["gini", "entropy"],
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
            "class_weight": ["balanced", "balanced_subsample", {}],
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
            name="Extra Trees Classifier",
            class_def=ExtraTreesClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
        )


class XGBClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from xgboost import XGBClassifier

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
            "verbosity": 0,
            "booster": "gbtree",
            "tree_method": "gpu_hist" if globals_dict["gpu_param"] else "auto",
        }
        tune_args = {}
        tune_grid = {
            "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
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
            class_def=XGBClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
            is_gpu_enabled=bool(globals_dict["gpu_param"]),
        )


class LGBMClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from lightgbm import LGBMClassifier
        from lightgbm.basic import LightGBMError

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
        }
        tune_args = {}
        tune_grid = {
            "num_leaves": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
            "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
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
            "bagging_freq": [1, 2, 3, 4, 5, 6, 7],
            "min_child_samples": np_list_arange(5, 100, 5, inclusive=True),
        }
        tune_distributions = {
            "num_leaves": IntUniformDistribution(10, 200),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "n_estimators": IntUniformDistribution(10, 300),
            "min_split_gain": UniformDistribution(0, 1),
            "reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            "min_data_in_leaf": IntUniformDistribution(10, 10000),
            "feature_fraction": UniformDistribution(0.4, 1),
            "bagging_fraction": UniformDistribution(0.4, 1),
            "bagging_freq": IntUniformDistribution(1, 7),
            "min_child_samples": IntUniformDistribution(5, 100),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        is_gpu_enabled = False
        if globals_dict["gpu_param"]:
            try:
                lgb = LGBMClassifier(device="gpu")
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
            id="lightgbm",
            name="Light Gradient Boosting Machine",
            class_def=LGBMClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
            is_gpu_enabled=is_gpu_enabled,
        )


class CatBoostClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from catboost import CatBoostClassifier

        # suppress output
        logging.getLogger("catboost").setLevel(logging.ERROR)

        use_gpu = globals_dict["gpu_param"] == "force" or (
            globals_dict["gpu_param"] and len(globals_dict["X_train"]) >= 50000
        )

        args = {
            "random_state": globals_dict["seed"],
            "verbose": False,
            "thread_count": globals_dict["n_jobs_param"],
            "task_type": "GPU" if use_gpu else "CPU",
            "border_count": 32 if use_gpu else 254,
        }
        tune_args = {}
        tune_grid = {
            "depth": list(range(1, 12)),
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "random_strength": np_list_arange(0, 0.8, 0.1, inclusive=True),
            "l2_leaf_reg": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 200],
        }
        tune_distributions = {
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
            name="CatBoost Classifier",
            class_def=CatBoostClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
            is_gpu_enabled=use_gpu,
        )


class BaggingClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.ensemble import BaggingClassifier

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": 1 if globals_dict["gpu_param"] else None,
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
            name="Bagging Classifier",
            class_def=BaggingClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=False,
        )


class StackingClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.ensemble import StackingClassifier

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="Stacking",
            name="Stacking Classifier",
            class_def=StackingClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=False,
        )


class VotingClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.ensemble import VotingClassifier
        from pycaret.internal.tunable import TunableVotingClassifier

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # VotingClassifier is a special case. Its weights can be tuned, but we do not know how many of them will be there
        # before it is initiated. Therefore, code to handle it will be added directly to tune_model().

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="Voting",
            name="Voting Classifier",
            class_def=VotingClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=False,
            tunable=TunableVotingClassifier,
        )


class CalibratedClassifierCVContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.calibration import CalibratedClassifierCV

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="CalibratedCV",
            name="Calibrated Classifier CV",
            class_def=CalibratedClassifierCV,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=False,
        )


def get_all_model_containers(
    globals_dict: dict, raise_errors: bool = True
) -> Dict[str, ClassifierContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), globals_dict, ClassifierContainer, raise_errors
    )
