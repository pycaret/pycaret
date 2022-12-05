from unittest.mock import patch

import numba
import pytest
from numba.core.dispatcher import Dispatcher

import pycaret.datasets
from pycaret.anomaly import AnomalyExperiment
from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.regression import RegressionExperiment
from pycaret.time_series import TSForecastingExperiment


@pytest.fixture
def disable_numba():
    """
    Forces numba to use the original python functions.

    This is required as numba code in pyod (anomaly) seems to not work
    correctly leading to exceptions if ran from within pytest.
    """
    old = numba.config.DISABLE_JIT
    # This will not affect already compiled functions...
    numba.config.DISABLE_JIT = True

    # ...which is why we force the Numba dispatcher to simply
    # call the underlying python function for already compiled
    # ones
    def pyfunc_call(self, *args, **kwargs):
        return self.py_func(*args, **kwargs)

    with patch.object(Dispatcher, "__call__", pyfunc_call):
        yield
    numba.config.DISABLE_JIT = old


def check_exp(exp, **kwargs):
    model_definitions = exp.models(internal=True).to_dict("index")
    for id, model_definition in model_definitions.items():
        if model_definition["Special"]:
            continue
        print(id)
        model = exp.create_model(id, **kwargs)
        for id_2, model_definition_2 in model_definitions.items():
            print(f"{id_2}.eq_function({id})")
            if id_2 == id:
                assert model_definition_2["Equality"](model)
            else:
                assert not model_definition_2["Equality"](model)


def test_model_equality_classification():
    exp = ClassificationExperiment()
    exp.setup(
        pycaret.datasets.get_data("juice"),
        target="Purchase",
    )
    check_exp(exp, cross_validation=False)


def test_model_equality_regression():
    exp = RegressionExperiment()
    exp.setup(
        pycaret.datasets.get_data("boston"),
        target="medv",
    )
    check_exp(exp, cross_validation=False)


def test_model_equality_time_series():
    exp = TSForecastingExperiment()
    exp.setup(
        pycaret.datasets.get_data("airline"),
        fh=12,
    )
    check_exp(exp, cross_validation=False)


def test_model_equality_clustering():
    exp = ClusteringExperiment()
    exp.setup(
        pycaret.datasets.get_data("jewellery"),
    )
    check_exp(exp)


def test_model_equality_anomaly(disable_numba):
    exp = AnomalyExperiment()
    exp.setup(
        pycaret.datasets.get_data("anomaly"),
    )
    check_exp(exp)


if __name__ == "__main__":
    test_model_equality_classification()
    test_model_equality_regression()
    test_model_equality_time_series()
    test_model_equality_clustering()
    test_model_equality_anomaly()
