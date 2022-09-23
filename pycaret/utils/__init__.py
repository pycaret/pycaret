from sklearn.metrics._scorer import _PredictScorer, get_scorer  # type: ignore

version_ = "3.0.0.rc4"
nightly_version_ = "3.0.0"

__version__ = version_


def version():
    return version_


def nightly_version():
    return nightly_version_
