from sklearn.metrics._scorer import _PredictScorer, get_scorer  # type: ignore

from pycaret.utils.generic import check_metric

version_ = "3.0.0.rc8"
nightly_version_ = "3.0.0"

__version__ = version_


def version():
    return version_


def nightly_version():
    return nightly_version_
