"""Model-registry equality tests — checks that every registered model's
equality predicate uniquely matches itself.

Rewritten for PyCaret 4.0 OOP API. The 3.x pattern of
``exp = ClassificationExperiment(); exp.setup(...)`` becomes
``exp = ClassificationExperiment(target=...).fit(...)`` — target is a
constructor parameter, not a setup kwarg.
"""

from unittest.mock import patch

import numba
import pycaret.datasets
import pytest
from numba.core.dispatcher import Dispatcher
from pycaret.tasks import (
    AnomalyExperiment,
    ClassificationExperiment,
    ClusteringExperiment,
    RegressionExperiment,
    TimeSeriesExperiment,
)


@pytest.fixture
def disable_numba():
    """Force numba to use the original Python functions.

    Required because numba-compiled code in pyod (anomaly) can misbehave
    under pytest; calling the underlying py_func via a Dispatcher patch is
    the documented workaround.
    """
    old = numba.config.DISABLE_JIT
    numba.config.DISABLE_JIT = True

    def pyfunc_call(self, *args, **kwargs):
        return self.py_func(*args, **kwargs)

    with patch.object(Dispatcher, "__call__", pyfunc_call):
        yield
    numba.config.DISABLE_JIT = old


def _unwrap_pipeline(obj):
    """Post session-24: supervised ``create_model`` returns a full Pipeline
    (preprocessor + trained model). The equality predicates in the model
    registry check ``isinstance(model, <class>)`` against the bare model
    class — so we pull the last step out before handing to the predicate.

    Post session-40 (phase 5b): TS ``create_model`` returns a sktime
    ``ForecastingPipeline`` whose final step is a
    ``TransformedTargetForecaster`` whose final step is the actual model.
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    try:
        from sktime.forecasting.compose import ForecastingPipeline
    except ImportError:  # pragma: no cover — sktime is a hard dep
        ForecastingPipeline = None

    if ForecastingPipeline is not None and isinstance(obj, ForecastingPipeline):
        # ForecastingPipeline → TransformedTargetForecaster → model
        inner = obj.steps[-1][1]
        if hasattr(inner, "steps"):
            return inner.steps[-1][1]
        return inner

    if isinstance(obj, SkPipeline):
        return obj.steps[-1][1]
    return obj


def check_exp(exp, **kwargs):
    """For every registered non-special model, train and assert its equality
    predicate uniquely matches itself among all other containers."""
    model_definitions = exp.models(internal=True).to_dict("index")
    for id, model_definition in model_definitions.items():
        if model_definition["Special"]:
            continue
        model = _unwrap_pipeline(exp.create_model(id, **kwargs).pipeline)
        for id_2, model_definition_2 in model_definitions.items():
            if id_2 == id:
                assert model_definition_2["Equality"](model)
            else:
                assert not model_definition_2["Equality"](model)


def test_model_equality_classification():
    df = pycaret.datasets.get_data("juice")
    exp = ClassificationExperiment(target="Purchase").fit(df)
    check_exp(exp, cross_validation=False)


def test_model_equality_regression():
    df = pycaret.datasets.get_data("boston")
    exp = RegressionExperiment(target="medv").fit(df)
    check_exp(exp, cross_validation=False)


def test_model_equality_time_series():
    df = pycaret.datasets.get_data("airline")
    exp = TimeSeriesExperiment(fh=12).fit(df)
    check_exp(exp, cross_validation=False)


def test_model_equality_clustering():
    df = pycaret.datasets.get_data("jewellery")
    exp = ClusteringExperiment().fit(df)
    check_exp(exp)


def test_model_equality_anomaly(disable_numba):
    df = pycaret.datasets.get_data("anomaly")
    exp = AnomalyExperiment().fit(df)
    check_exp(exp)


if __name__ == "__main__":
    test_model_equality_classification()
    test_model_equality_regression()
    test_model_equality_time_series()
    test_model_equality_clustering()
    test_model_equality_anomaly()
