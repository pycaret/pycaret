import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest

import pycaret.anomaly
import pycaret.classification
import pycaret.clustering
import pycaret.datasets
import pycaret.regression
import pycaret.time_series


def test_anomaly_persistence(tmpdir):
    cls = pycaret.anomaly.AnomalyExperiment
    data = pycaret.datasets.get_data("anomaly")
    exp = cls().setup(
        data,
        normalize=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_config(exp_path)
    new_exp = cls().setup(
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
    cls = pycaret.clustering.ClusteringExperiment
    data = pycaret.datasets.get_data("jewellery")
    exp = cls().setup(
        data,
        normalize=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_config(exp_path)
    new_exp = cls().setup(
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
    cls = pycaret.classification.ClassificationExperiment
    data = pycaret.datasets.get_data("juice")
    exp = cls().setup(
        data,
        target="Purchase",
        normalize=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_config(exp_path)
    new_exp = cls().setup(
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
    cls = pycaret.regression.RegressionExperiment
    data = pycaret.datasets.get_data("boston")
    exp = cls().setup(
        data,
        target="medv",
        normalize=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_config(exp_path)
    new_exp = cls().setup(
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
    cls = pycaret.time_series.TSForecastingExperiment
    data = load_pos_and_neg_data
    exp = cls().setup(
        data,
        transform_target="sqrt",
        html=False,
        session_id=123,
        n_jobs=1,
    )
    exp_path = os.path.join(tmpdir, "exp.pkl")
    exp.save_config(exp_path)
    new_exp = cls().setup(
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
