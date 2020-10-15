# Module: internal.tune_sklearn_patches
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# Provides methods returning monkey patched tune-sklearn classes to allow for Pipeline support.
from pycaret.internal.logging import get_logger
import numpy as np


def numpy_types_to_python(o):
    try:
        if isinstance(o, np.generic) and not isinstance(o, np.ndarray):
            return o.item()
    except:
        pass
    return o


def get_tune_trainable():
    import tune_sklearn._trainable
    import ray
    from sklearn.base import clone

    class _Trainable(tune_sklearn._trainable._Trainable):
        def _setup(self, config):
            """Sets up Trainable attributes during initialization.

            Also sets up parameters for the sklearn estimator passed in.

            Args:
                config (dict): contains necessary parameters to complete the `fit`
                    routine for the estimator. Also includes parameters for early
                    stopping if it is set to true.

            """
            self.estimator_list = clone(config.pop("estimator_list"))
            self.early_stopping = config.pop("early_stopping")
            X_id = config.pop("X_id")
            self.X = ray.get(X_id)
            try:
                config.pop("early_stop_type")
            except:
                pass

            y_id = config.pop("y_id")
            self.y = ray.get(y_id)
            self.groups = config.pop("groups")
            self.fit_params = config.pop("fit_params")
            self.scoring = config.pop("scoring")
            self.max_iters = config.pop("max_iters")
            self.cv = config.pop("cv")
            self.return_train_score = config.pop("return_train_score")
            self.n_jobs = config.pop("n_jobs")

            self.estimator_config = {
                k: numpy_types_to_python(v) for k, v in config.items()
            }
            self.train_accuracy = None
            self.test_accuracy = None
            self.saved_models = []  # XGBoost specific

            if self.early_stopping:
                assert self._can_early_start()
                n_splits = self.cv.get_n_splits(self.X, self.y)
                self.fold_scores = np.empty(n_splits, dtype=dict)
                self.fold_train_scores = np.empty(n_splits, dtype=dict)
                if not self._can_partial_fit() and self._can_warm_start_iter():
                    # max_iter here is different than the max_iters the user sets.
                    # max_iter is to make sklearn only fit for one epoch,
                    # while max_iters (which the user can set) is the usual max
                    # number of calls to _trainable.
                    self.estimator_config["actual_estimator__warm_start"] = True
                    self.estimator_config["actual_estimator__max_iter"] = 1

                if not self._can_partial_fit() and self._can_warm_start_ensemble():
                    # Each additional call on a warm start ensemble only trains
                    # new estimators added to the ensemble. We start with 0
                    # and add an estimator before each call to fit in _train(),
                    # training the ensemble incrementally.
                    self.estimator_config["actual_estimator__warm_start"] = True
                    self.estimator_config["actual_estimator__n_estimators"] = 0

                for i in range(n_splits):
                    self.estimator_list[i].set_params(**self.estimator_config)

                if self._is_xgb():
                    self.saved_models = [None for _ in range(n_splits)]
            else:
                self.main_estimator.set_params(**self.estimator_config)

        def _is_xgb(self):
            from xgboost.sklearn import XGBModel

            return isinstance(self.main_estimator.steps[-1][1], XGBModel)

        def _early_stopping_partial_fit(self, i, estimator, X_train, y_train):
            """Handles early stopping on estimators that support `partial_fit`.

            """
            try:
                estimator.partial_fit(X_train, y_train, classes=np.unique(self.y))
            except:
                estimator.partial_fit(X_train, y_train)

        def _can_warm_start_ensemble(self):
            estimator = self.main_estimator.steps[-1][1]
            from sklearn.ensemble import BaseEnsemble

            is_ensemble_subclass = issubclass(type(estimator), BaseEnsemble)

            return (
                hasattr(estimator, "warm_start")
                and hasattr(estimator, "n_estimators")
                and is_ensemble_subclass
            )

        def _can_warm_start_iter(self):
            estimator = self.main_estimator.steps[-1][1]
            from sklearn.tree import BaseDecisionTree

            is_not_tree_subclass = not issubclass(type(estimator), BaseDecisionTree)
            is_not_ensemble_subclass = not self._can_warm_start_ensemble()

            return (
                hasattr(estimator, "warm_start")
                and hasattr(estimator, "max_iter")
                and is_not_ensemble_subclass
                and is_not_tree_subclass
            )

        def _early_stopping_ensemble(self, i, estimator, X_train, y_train):
            """Handles early stopping on ensemble estimators.

            """
            # User will not be able to fine tune the n_estimators
            # parameter using ensemble early stopping
            updated_n_estimators = (
                estimator.get_params()["actual_estimator__n_estimators"] + 1
            )
            estimator.set_params(
                **{"actual_estimator__n_estimators": updated_n_estimators}
            )
            estimator.fit(X_train, y_train)

        def _early_stopping_xgb(self, i, estimator, X_train, y_train):
            """Handles early stopping on XGBoost estimators.

            """
            estimator.fit(
                X_train, y_train, actual_estimator__xgb_model=self.saved_models[i]
            )
            self.saved_models[i] = estimator.steps[-1][1].get_booster()

        def _can_partial_fit(self):
            from tune_sklearn.utils import check_partial_fit

            return (
                not self._is_xgb()
                and not self._can_warm_start_ensemble()
                and check_partial_fit(self.main_estimator)
            )

    return _Trainable


def get_early_stop():
    from tune_sklearn._detect_xgboost import is_xgboost_model
    from tune_sklearn.utils import (
        check_warm_start_iter,
        check_warm_start_ensemble,
        check_partial_fit,
        _check_multimetric_scoring,
    )

    def _can_early_stop(self):

        """Helper method to determine if it is possible to do early stopping.

        Only sklearn estimators with `partial_fit` or `warm_start` can be early
        stopped. warm_start works by picking up training from the previous
        call to `fit`.

        Returns:
            bool: if the estimator can early stop

        """

        estimator = self.estimator.steps[-1][1]

        can_partial_fit = check_partial_fit(estimator)
        can_warm_start = check_warm_start_iter(estimator)
        can_warm_start_ensemble = check_warm_start_ensemble(estimator)
        is_gbm = is_xgboost_model(estimator)

        return can_partial_fit or can_warm_start or can_warm_start_ensemble or is_gbm

    return _can_early_stop


def _clean_config_dict(self, config):
    """Helper to remove keys from the ``config`` dictionary returned from
    ``tune.run``.

    Args:
        config (:obj:`dict`): Dictionary of all hyperparameter
            configurations and extra output from ``tune.run``., Keys for
            hyperparameters are the hyperparameter variable names
            and the values are the numeric values set to those variables.

    Returns:
        config (:obj:`dict`): Dictionary of all hyperparameter
            configurations without the output from ``tune.run``., Keys for
            hyperparameters are the hyperparameter variable names
            and the values are the numeric values set to those variables.
    """
    for key in [
        "estimator_list",
        "early_stopping",
        "X_id",
        "y_id",
        "groups",
        "cv",
        "fit_params",
        "scoring",
        "max_iters",
        "return_train_score",
        "n_jobs",
    ]:
        config.pop(key, None)
    return {k: numpy_types_to_python(v) for k, v in config.items()}


def get_tune_sklearn_tunesearchcv():
    import tune_sklearn.tune_search

    tune_sklearn.tune_search.TuneBaseSearchCV._can_early_stop = get_early_stop()
    tune_sklearn.tune_search.TuneBaseSearchCV._clean_config_dict = _clean_config_dict

    tune_sklearn.tune_search._Trainable = get_tune_trainable()
    return tune_sklearn.tune_search.TuneSearchCV


def get_tune_sklearn_tunegridsearchcv():
    import tune_sklearn.tune_gridsearch

    tune_sklearn.tune_gridsearch.TuneBaseSearchCV._can_early_stop = get_early_stop()
    tune_sklearn.tune_search.TuneBaseSearchCV._clean_config_dict = _clean_config_dict

    tune_sklearn.tune_gridsearch._Trainable = get_tune_trainable()
    return tune_sklearn.tune_gridsearch.TuneGridSearchCV
