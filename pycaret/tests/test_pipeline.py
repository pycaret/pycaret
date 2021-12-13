# coding: utf-8

"""
Package: PyCaret
Author: Mavs
Description: Unit tests for pipeline.py

"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import pycaret.regression
import pycaret.classification
from pycaret.datasets import get_data


@pytest.fixture
def pipeline():
    """Get a pipeline from atom with/without final estimator."""
    dataset = get_data("juice", verbose=False)
    pc = pycaret.classification.setup(
        data=dataset,
        polynomial_features=True,
        polynomial_degree=2,
        # remove_multicollinearity=True,
        # multicollinearity_threshold=0.8,
        # fix_imbalance=True,
        # normalize=True,
        # pca=True,
        # pca_components=8,
        verbose=False,
    )

    return pc.pipeline


def test_fit(pipeline):
    """Assert that the pipeline can be fitted normally."""
    data = get_data("juice", verbose=False)
    assert pipeline.fit(data.iloc[:, :-1], data.iloc[:, -1])


def test_transforms_only_y():
    """Assert that the pipeline can transform the target column only."""
    data = get_data("bank", verbose=False)
    pc = pycaret.classification.setup(
        data=data,
        preprocess=False,
        custom_pipeline=("label_encoder", LabelEncoder()),
    )
    y = pc.pipeline.fit_transform(y=data.iloc[:, -1])
    assert isinstance(y, pd.Series)


def test_transform(pipeline):
    """Assert that the pipeline uses transform normally."""
    data = get_data("juice", verbose=False)
    pipeline.fit(data.iloc[:, :-1], data.iloc[:, -1])
    assert isinstance(pipeline.transform(data.iloc[:, :-1]), pd.DataFrame)
    assert isinstance(pipeline.transform(data.iloc[:, :-1], data.iloc[:, -1]), tuple)


def test_fit_transform(pipeline):
    """Assert that the pipeline can be fit-transformed normally."""
    data = get_data("juice", verbose=False)
    pipeline.steps.append(("test", "passthrough"))
    X, y = pipeline.fit_transform(data.iloc[:, :-1], data.iloc[:, -1])
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)


def test_transform_imbalancer(pipeline):
    """Assert that the pipeline ignores FixImbalancer during predicting."""
    data = get_data("juice", verbose=False)
    pipeline.fit(data.iloc[:, :-1], data.iloc[:, -1])
    assert len(pipeline.transform(data.iloc[:, :-1])) == len(data)
