from copy import deepcopy
from typing import Any, Optional, Union

import pandas as pd
from pycaret.internal.logging import get_logger
from pycaret.internal.Display import Display
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline


def is_sklearn_pipeline(object):
    from sklearn.pipeline import Pipeline

    return isinstance(object, Pipeline)


def is_sklearn_cv_generator(object):
    return not isinstance(object, str) and hasattr(object, "split")


def is_fitted(estimator) -> bool:
    try:
        check_is_fitted(estimator)
        return True
    except:
        return False


class fit_if_not_fitted(object):
    """
    Context which fits an estimator if it's not fitted.
    """

    def __init__(
        self,
        estimator,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups=None,
        **fit_kwargs,
    ):
        logger = get_logger()
        self.estimator = deepcopy(estimator)
        if not is_fitted(self.estimator):
            try:
                self.estimator._carry_over_final_estimator_fit_vars()
            except:
                pass
            if not is_fitted(self.estimator):
                logger.info(f"fit_if_not_fitted: {estimator} is not fitted, fitting")
                try:
                    self.estimator.fit(X_train, y_train, groups=groups, **fit_kwargs)
                except:
                    self.estimator.fit(X_train, y_train, **fit_kwargs)

    def __enter__(self):
        return self.estimator

    def __exit__(self, type, value, traceback):
        return


def supports_partial_fit(estimator, params: dict = None) -> bool:
    # special case for MLP
    from sklearn.neural_network import MLPClassifier

    if isinstance(estimator, MLPClassifier):
        try:
            if (
                params and "solver" in params and "lbfgs" in list(params["solver"])
            ) or estimator.solver == "lbfgs":
                return False
        except:
            return False

    if isinstance(estimator, Pipeline):
        return hasattr(estimator.steps[-1][1], "partial_fit")

    return hasattr(estimator, "partial_fit")
