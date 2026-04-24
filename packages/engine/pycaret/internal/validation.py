from copy import deepcopy

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from pycaret.internal.logging import get_logger


def is_sklearn_pipeline(object):
    from sklearn.pipeline import Pipeline

    return isinstance(object, Pipeline)


def is_sklearn_cv_generator(object):
    return not isinstance(object, str) and hasattr(object, "split")


def is_fitted(estimator) -> bool:
    try:
        check_is_fitted(estimator)
        return True
    except Exception:
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
            if not is_fitted(self.estimator):
                logger.info(f"fit_if_not_fitted: {estimator} is not fitted, fitting")
                try:
                    self.estimator.fit(X_train, y_train, groups=groups, **fit_kwargs)
                except Exception:
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
        except Exception:
            return False

    if isinstance(estimator, Pipeline):
        return hasattr(estimator.steps[-1][1], "partial_fit")

    return hasattr(estimator, "partial_fit")
