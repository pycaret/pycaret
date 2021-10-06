import pandas as pd
import numpy as np
import datetime

import pytest
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.estimator_checks import check_estimator
import pycaret.datasets
import pycaret.internal.preprocess
from pycaret.internal.preprocess import TransformedTargetClassifier


@pytest.fixture
def get_data():
    return pycaret.datasets.get_data("juice")


def test_date_features(get_data):
    """Assert that features are extracted from date columns."""
    X = get_data.copy()
    X["date"] = pd.date_range()
    pc = pycaret.classification.setup(get_data, date_features=["date"])

    X = pc._internal_pipeline.fit_transform()
    assert "date_day" in pc.data