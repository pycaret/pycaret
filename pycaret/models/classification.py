import inspect
import logging
from pycaret import internal
from typing import Any, Union, Dict
from pycaret.models.base_model import ModelContainer
from pycaret.internal.utils import param_grid_to_lists, get_logger
from pycaret.internal.distributions import *
import numpy as np


def _leftover_parameters_to_categorical_distributions(
    tune_grid: dict, tune_distributions: dict
) -> None:
    for k, v in tune_grid.items():
        if not k in tune_distributions:
            tune_distributions[k] = CategoricalDistribution(v)


class ClassifierContainer(ModelContainer):
    """
    Base classification model container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.
    """

    def __init__(
        self,
        id: str,
        name: str,
        class_def: type,
        is_turbo: bool = True,
        args: dict = {},
        is_special: bool = False,
        tune_grid: dict = {},
        tune_distribution: dict = {},
        tune_args: dict = {},
        shap: Union[bool, str] = False,
        is_gpu_enabled: bool = None,
        is_boosting_supported: bool = None,
        is_soft_voting_supported: bool = None,
    ) -> None:

        self.shap = shap
        if not (isinstance(shap, bool) or shap in ["type1", "type2"]):
            raise ValueError("shap must be either bool or 'type1', 'type2'.")

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
            if is_soft_voting_supported is not None:
                self.is_soft_voting_supported = is_soft_voting_supported

        if is_gpu_enabled is not None:
            self.is_gpu_enabled = is_gpu_enabled
        else:
            self.is_gpu_enabled = bool(self.get_package_name() != "sklearn")

    def get_dict(self, internal: bool = True) -> dict:
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
                ("Soft Voting", self.is_soft_voting_supported),
            ]

        return dict(d)

    def get_keys(self, internal: bool = True) -> list:
        return list(self.get_dict(internal).keys())


class LogisticRegressionClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.linear_model import LogisticRegression

        if globals_dict["gpu_param"] == "Force":
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

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["penalty"] = ["l2", "none"]
        tune_grid["C"] = (np.arange(0, 10, 0.001),)

        if gpu_imported:
            tune_grid["penalty"] += ["l1"]
        else:
            args["random_state"] = globals_dict["seed"]

            tune_grid["class_weight"] = ["balanced", {}]

        tune_distributions["C"] = UniformDistribution(0, 10)
        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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

        if globals_dict["gpu_param"] == "Force":
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
        tune_grid["n_neighbors"] = (range(1, 51),)
        tune_grid["weights"] = ["uniform"]
        tune_grid["metric"] = ["minkowski", "euclidean", "manhattan"]

        if gpu_imported:
            pass
        else:
            args["n_jobs"] = globals_dict["n_jobs_param"]

            tune_grid["weights"] += ["distance"]

        tune_distributions["n_neighbors"] = IntUniformDistribution(1, 51)
        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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
            "max_depth": list(range(1, int(len(globals_dict["X"].columns) + 1 * 0.85))),
            "max_features": list(range(1, len(globals_dict["X"].columns) + 1)),
            "min_samples_leaf": [2, 3, 4, 5, 6],
            "criterion": ["gini", "entropy"],
        }
        tune_distributions = {
            "max_depth": IntUniformDistribution(
                1, int(len(globals_dict["X"].columns) * 0.85)
            ),
            "max_features": IntUniformDistribution(1, len(globals_dict["X"].columns)),
            "min_samples_leaf": IntUniformDistribution(2, 6),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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

        if globals_dict["gpu_param"] == "Force":
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

        args = {"tol": 0.001, "loss": "hinge"}
        tune_args = {}
        tune_grid = {
            "penalty": ["elasticnet", "l2", "l1"],
            "l1_ratio": np.arange(0.0000000001, 0.9999999999, 0.01),
            "alpha": [0.0001, 0.001, 0.01, 0.0002, 0.002, 0.02, 0.0005, 0.005, 0.05,],
            "fit_intercept": [True, False],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        tune_distributions = {
            "l1_ratio": UniformDistribution(0, 1),
            "alpha": UniformDistribution(0.0000000001, 0.9999999999),
            "eta0": UniformDistribution(0.001, 0.5),
        }

        # common
        tune_grid["n_neighbors"] = (range(1, 51),)
        tune_grid["weights"] = ["uniform"]
        tune_grid["metric"] = ["minkowski", "euclidean", "manhattan"]

        if gpu_imported:
            tune_grid["learning_rate"] += ["optimal"]
        else:
            args["random_state"] = globals_dict["seed"]
            args["n_jobs"] = globals_dict["n_jobs_param"]

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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

        # known limitation - cuML SVC only supports binary problems
        if globals_dict["y"].value_counts().count() <= 2:
            if globals_dict["gpu_param"] == "Force":
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
            "C": 1,
            "probability": True,
            "kernel": "rbf",
            "random_state": globals_dict["seed"],
        }
        tune_args = {}
        tune_grid = {
            "C": np.arange(0, 50, 0.01),
            "class_weight": ["balanced", {}],
        }
        tune_distributions = {
            "C": UniformDistribution(0, 50),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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

        args = {"random_state": globals_dict["seed"], "max_iter": 500}
        tune_args = {}
        tune_grid = {
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": np.arange(0, 1, 0.0001),
            "hidden_layer_sizes": [
                (50, 50, 50),
                (50, 100, 50),
                (100,),
                (100, 50, 100),
                (100, 100, 100),
            ],
            "activation": ["tanh", "identity", "logistic", "relu"],
        }
        tune_distributions = {
            "alpha": UniformDistribution(0, 1),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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
        )


class RidgeClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = False

        from sklearn.linear_model import RidgeClassifier

        if globals_dict["gpu_param"] == "Force":
            from cuml import MBSGDClassifier as RidgeClassifier

            logger.info("Imported cuml.MBSGDClassifier")
            gpu_imported = True
        elif globals_dict["gpu_param"]:
            try:
                from cuml import MBSGDClassifier as RidgeClassifier

                logger.info("Imported cuml.MBSGDClassifier")
                gpu_imported = True
            except ImportError:
                logger.warning("Couldn't import cuml.MBSGDClassifier")

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        if gpu_imported:
            args = {
                "tol": 0.001,
                "loss": "squared_loss",
                "penalty": "l2",
            }
            tune_grid = {
                "learning_rate": ["constant", "invscaling", "adaptive"],
                "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            }
            tune_distributions = {
                "eta0": UniformDistribution(0.001, 0.5),
            }
        else:
            tune_grid = {
                "normalize": [True, False],
            }
            args = {"random_state": globals_dict["seed"]}

        tune_grid["alpha"] = np.arange(0.001, 0.999, 0.001)
        tune_grid["fit_intercept"] = [True, False]
        tune_distributions["alpha"] = (UniformDistribution(0.0000000001, 0.9999999999),)

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ridge",
            name="Ridge Classifier",
            class_def=RidgeClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class RandomForestClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.ensemble import RandomForestClassifier

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
        }
        tune_args = {}
        tune_grid = {
            "n_estimators": [int(x) for x in np.linspace(10, 1000, num=100)],
            "criterion": ["gini", "entropy"],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, "auto", "log2"],
            "bootstrap": [True, False],
            "ccp_alpha": np.arange(0.0, 0.01, 0.001),
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 1000),
            "max_depth": IntUniformDistribution(10, 110),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_samples_leaf": IntUniformDistribution(1, 5),
            "ccp_alpha": UniformDistribution(0, 0.01),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="rf",
            name="Random Forest Classifier",
            class_def=RandomForestClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
        )


class QuadraticDiscriminantAnalysisContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        args = {}
        tune_args = {}
        tune_grid = {"reg_param": np.arange(0, 1, 0.01)}
        tune_distributions = {"reg_param": UniformDistribution(0, 1)}

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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
            "n_estimators": np.arange(10, 200, 5),
            "learning_rate": np.arange(0, 1, 0.01),
            "algorithm": ["SAMME", "SAMME.R"],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 200),
            "learning_rate": UniformDistribution(0, 1),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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
            "n_estimators": np.arange(10, 200, 5),
            "learning_rate": np.arange(0, 1, 0.01),
            "subsample": np.arange(0.1, 1, 0.05),
            "min_samples_split": [2, 4, 5, 7, 9, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
            "max_features": ["auto", "sqrt", "log2"],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 200),
            "learning_rate": UniformDistribution(0, 1),
            "subsample": UniformDistribution(0.1, 1),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_samples_leaf": IntUniformDistribution(1, 5),
            "max_depth": IntUniformDistribution(10, 110),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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
            "shrinkage": UniformDistribution(0.000000001, 1, log=True),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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
            "n_estimators": np.arange(10, 200, 5),
            "criterion": ["gini", "entropy"],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 200),
            "max_depth": IntUniformDistribution(10, 110),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_samples_leaf": IntUniformDistribution(1, 5),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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
            "learning_rate": np.arange(0, 1, 0.01),
            "n_estimators": np.arange(10, 100, 20)
            if globals_dict["y"].value_counts().count() > 2
            else [10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,],
            "subsample": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
            "max_depth": [int(x) for x in np.linspace(1, 11, num=11)],
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "min_child_weight": [1, 2, 3, 4],
        }
        tune_distributions = {
            "learning_rate": UniformDistribution(0, 1),
            "n_estimators": IntUniformDistribution(10, 100)
            if globals_dict["y"].value_counts().count() > 2
            else IntUniformDistribution(10, 1000, log=True),
            "subsample": UniformDistribution(0.1, 1),
            "max_depth": IntUniformDistribution(1, 11),
            "colsample_bytree": UniformDistribution(0.5, 1),
            "min_child_weight": IntUniformDistribution(1, 4),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
        }
        tune_args = {}
        tune_grid = {
            "num_leaves": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
            "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "n_estimators": [10, 30, 50, 70, 90, 100, 120, 150, 170, 200],
            "min_split_gain": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "reg_lambda": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
        tune_distributions = {
            "num_leaves": IntUniformDistribution(10, 200),
            "max_depth": IntUniformDistribution(10, 110),
            "learning_rate": UniformDistribution(0.1, 1),
            "n_estimators": IntUniformDistribution(10, 200),
            "min_split_gain": UniformDistribution(0, 1),
            "reg_alpha": UniformDistribution(0.1, 1),
            "reg_lambda": UniformDistribution(0.1, 1),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="lightgbm",
            name="Light Gradient Boosting Machine",
            class_def=LGBMClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
        )


class CatBoostClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from catboost import CatBoostClassifier

        # suppress output
        logging.getLogger("catboost").setLevel(logging.ERROR)

        use_gpu = globals_dict["gpu_param"] == "Force" or (
            globals_dict["gpu_param"] and len(globals_dict["X"]) >= 50000
        )

        args = {
            "random_state": globals_dict["seed"],
            "verbose": False,
            "thread_count": globals_dict["n_jobs_param"],
            "task_type": "GPU" if use_gpu else "CPU",
        }
        tune_args = {}
        tune_grid = {
            "depth": [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
            "iterations": [250, 100, 500, 1000],
            "learning_rate": [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
            "l2_leaf_reg": [3, 1, 5, 10, 100],
            "border_count": [32, 5, 10, 20, 50, 100, 200],
        }
        tune_distributions = {
            "depth": IntUniformDistribution(1, 10),
            "iterations": IntUniformDistribution(250, 1000, log=True),
            "learning_rate": UniformDistribution(0.0001, 0.3, log=True),
            "l2_leaf_reg": IntUniformDistribution(1, 100, log=True),
            "border_count": IntUniformDistribution(5, 200, log=True),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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
            "n_jobs": globals_dict["gpu_n_jobs_param"],
        }
        tune_args = {}
        tune_grid = {
            "n_estimators": np.arange(10, 300, 10),
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
        }

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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
        )


class OneVsRestClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.multiclass import OneVsRestClassifier

        args = {"n_jobs": globals_dict["gpu_n_jobs_param"]}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="OneVsRest",
            name="One Vs Rest Classifier",
            class_def=OneVsRestClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=False,
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

        _leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

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


def get_all_model_containers(globals_dict: dict) -> Dict[str, ClassifierContainer]:
    model_container_classes = [
        obj
        for name, obj in globals().items()
        if inspect.isclass(obj) and ClassifierContainer in obj.__bases__
    ]

    model_containers = [obj(globals_dict) for obj in model_container_classes]

    return {container.id: container for container in model_containers}

