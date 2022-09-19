import os

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


def test_anomaly_persistence(tmpdir):
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
    exp.save_config(exp_path)

    new_exp = AnomalyExperiment()
    new_exp.setup(
        data,
        html=False,
        normalize=False,
        session_id=123,
        n_jobs=1,
    )
    assert "normalize" not in new_exp.pipeline.named_steps
    new_exp.load_config(exp_path)
    assert "normalize" in new_exp.pipeline.named_steps


def test_clustering_persistence(tmpdir):
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
    exp.save_config(exp_path)

    new_exp = ClusteringExperiment()
    new_exp.setup(
        data,
        normalize=False,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    assert "normalize" not in new_exp.pipeline.named_steps
    new_exp.load_config(exp_path)
    assert "normalize" in new_exp.pipeline.named_steps


def test_classification_persistence(tmpdir):
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
    exp.save_config(exp_path)

    new_exp = ClassificationExperiment()
    new_exp.setup(
        data,
        target="Purchase",
        normalize=False,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    assert "normalize" not in new_exp.pipeline.named_steps
    new_exp.load_config(exp_path)
    assert "normalize" in new_exp.pipeline.named_steps


def test_regression_persistence(tmpdir):
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
    exp.save_config(exp_path)

    new_exp = RegressionExperiment()
    new_exp.setup(
        data,
        target="medv",
        normalize=False,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    assert "normalize" not in new_exp.pipeline.named_steps
    new_exp.load_config(exp_path)
    assert "normalize" in new_exp.pipeline.named_steps


def test_time_series_persistence(tmpdir, load_pos_and_neg_data):
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
    exp.save_config(exp_path)

    new_exp = TSForecastingExperiment()
    new_exp.setup(
        data,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    assert (
        "transformer_target"
        not in new_exp.pipeline.named_steps["forecaster"].named_steps
    )
    new_exp.load_config(exp_path)
    assert (
        "transformer_target" in new_exp.pipeline.named_steps["forecaster"].named_steps
    )
