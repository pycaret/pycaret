from typing import Any, Optional, Union
from pycaret.internal.logging import get_logger
from pycaret.internal.Display import Display


def is_sklearn_pipeline(object):
    from sklearn.pipeline import Pipeline

    return isinstance(object, Pipeline)


def is_sklearn_cv_generator(object):
    return not isinstance(object, str) and hasattr(object, "split")

