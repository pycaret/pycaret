from yellowbrick.utils.helpers import get_model_name as get_model_name_original

from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator


def is_estimator(model):
    return callable(getattr(model, "fit", None))


def get_model_name(model):
    return get_model_name_original(get_estimator_from_meta_estimator(model))
