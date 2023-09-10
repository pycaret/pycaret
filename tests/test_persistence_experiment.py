import os

import joblib
import pytest

import pycaret.anomaly
import pycaret.classification
import pycaret.clustering
import pycaret.datasets
import pycaret.regression
import pycaret.time_series
from pycaret.anomaly import AnomalyExperiment
from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.regression import RegressionExperiment
from pycaret.time_series import TSForecastingExperiment


def check_experiment_equality(exp, new_exp):
    for key, value in exp.variables.items():
        if key == "memory":
            continue
        # Reset memory as it will never be equal
        if key == "pipeline":
            value.memory = None
            new_exp.variables[key].memory = None
        try:
            assert value == new_exp.variables[key]
        except Exception:
            # For numpy arrays
            assert joblib.hash(value) == joblib.hash(new_exp.variables[key])


@pytest.mark.parametrize("preprocess_data", (True, False))
def test_anomaly_persistence(tmpdir, preprocess_data):
    data = pycaret.datasets.get_data("anomaly")
    exp = AnomalyExperiment()
    exp.setup(
        data,
        normalize=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_experiment(exp_path)

    new_exp = AnomalyExperiment.load_experiment(
        exp_path,
        data=data if preprocess_data else exp.data,
        preprocess_data=preprocess_data,
    )
    assert "normalize" in new_exp.pipeline.named_steps
    check_experiment_equality(exp, new_exp)


@pytest.mark.parametrize("preprocess_data", (True, False))
def test_clustering_persistence(tmpdir, preprocess_data):
    data = pycaret.datasets.get_data("jewellery")
    exp = ClusteringExperiment()
    exp.setup(
        data,
        normalize=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_experiment(exp_path)

    new_exp = ClusteringExperiment.load_experiment(
        exp_path,
        data=data if preprocess_data else exp.data,
        preprocess_data=preprocess_data,
    )
    assert "normalize" in new_exp.pipeline.named_steps
    check_experiment_equality(exp, new_exp)


@pytest.mark.parametrize("preprocess_data", (True, False))
def test_classification_persistence(tmpdir, preprocess_data):
    data = pycaret.datasets.get_data("juice")
    exp = ClassificationExperiment()
    exp.setup(
        data,
        target="Purchase",
        normalize=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_experiment(exp_path)

    new_exp = ClassificationExperiment.load_experiment(
        exp_path,
        data=data if preprocess_data else exp.data,
        preprocess_data=preprocess_data,
    )
    assert "normalize" in new_exp.pipeline.named_steps
    check_experiment_equality(exp, new_exp)


@pytest.mark.parametrize("preprocess_data", (True, False))
def test_regression_persistence(tmpdir, preprocess_data):
    data = pycaret.datasets.get_data("boston")
    exp = RegressionExperiment()
    exp.setup(
        data,
        target="medv",
        normalize=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_experiment(exp_path)

    new_exp = RegressionExperiment.load_experiment(
        exp_path,
        data=data if preprocess_data else exp.data,
        preprocess_data=preprocess_data,
    )
    assert "normalize" in new_exp.pipeline.named_steps
    check_experiment_equality(exp, new_exp)


@pytest.mark.parametrize("preprocess_data", (True, False))
def test_time_series_persistence(tmpdir, load_pos_and_neg_data, preprocess_data):
    data = load_pos_and_neg_data
    exp = TSForecastingExperiment()
    exp.setup(
        data,
        transform_target="sqrt",
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_experiment(exp_path)

    new_exp = TSForecastingExperiment.load_experiment(
        exp_path,
        data=data if preprocess_data else exp.data,
        preprocess_data=preprocess_data,
    )
    # check experiment equality is not working for TS due to sktime,
    # so we simply compare results
    model = exp.create_model("ets")
    results = exp.pull()
    preds = exp.predict_model(model)

    new_model = new_exp.create_model("ets")
    new_results = new_exp.pull()
    new_preds = new_exp.predict_model(new_model)

    assert preds.equals(new_preds)
    assert results.equals(new_results)
