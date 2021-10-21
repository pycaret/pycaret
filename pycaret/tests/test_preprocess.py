import pytest
import numpy as np
import pandas as pd

import pycaret.datasets
import pycaret.regression
import pycaret.classification
from pycaret.internal.preprocess import TransformedTargetClassifier


def test_select_target_by_index():
    """Assert that the target can be selected by its column index."""
    juice = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(juice, target=2)
    assert pc.target_param == "WeekofPurchase"


def test_preprocess_is_False():
    """Assert that preprocessing is skipped when preprocess=False."""
    juice = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(juice, preprocess=False)
    X, _ = pc._internal_pipeline.fit_transform(pc.X, pc.y)
    assert X["Purchase"].dtype.kind not in "ifu"  # No encoding of categorical columns


def test_encode_target():
    """Assert that the target column is automatically encoded."""
    telescope = pycaret.datasets.get_data("telescope")
    pc = pycaret.classification.setup(telescope)
    _, y = pc._internal_pipeline.fit_transform(pc.X, pc.y)
    assert y.dtype.kind in "ifu"


def test_date_features():
    """Assert that features are extracted from date features."""
    juice = pycaret.datasets.get_data("juice")
    juice["date"] = pd.date_range(start="1/1/2018", periods=len(juice))
    pc = pycaret.classification.setup(juice, target=-2, date_features=["date"])
    X, _ = pc._internal_pipeline.fit_transform(pc.X, pc.y)
    assert all([f"date_{attr}" in X for attr in ("day", "month", "year")])


@pytest.mark.parametrize("imputation_method", ["zero", "mean", "median"])
def test_simple_numeric_imputation(imputation_method):
    """Assert that missing values are imputed."""
    juice = pycaret.datasets.get_data("juice")
    juice.loc[100, "WeekofPurchase"] = np.nan
    pc = pycaret.classification.setup(
        data=juice,
        imputation_type="simple",
        numeric_iterative_imputer=imputation_method,
    )
    X, _ = pc._internal_pipeline.fit_transform(pc.X, pc.y)
    assert X.isna().sum().sum() == 0


@pytest.mark.parametrize("imputation_method", ["constant", "mode"])
def test_simple_categorical_imputation(imputation_method):
    """Assert that missing values are imputed."""
    juice = pycaret.datasets.get_data("juice")
    juice.loc[100, "Purchase"] = np.nan
    pc = pycaret.classification.setup(
        data=juice,
        imputation_type="simple",
        categorical_imputation=imputation_method,
    )
    X, _ = pc._internal_pipeline.fit_transform(pc.X, pc.y)
    assert X.isna().sum().sum() == 0


@pytest.mark.parametrize("embedding_method", ["bow", "tf-idf"])
def test_text_embedding(embedding_method):
    """Assert that text columns are embedded."""
    spx = pycaret.datasets.get_data("spx")
    pc = pycaret.regression.setup(
        data=spx.iloc[:50, :],  # Less rows for faster processing
        text_features=["text"],
        text_features_method=embedding_method,
    )
    X, _ = pc._internal_pipeline.fit_transform(pc.X, pc.y)
    assert X.shape[1] > 50  # Text column is now embedding


def test_ordinal_features():
    """Assert that ordinal features are encoded correctly."""
    employee = pycaret.datasets.get_data("employee")
    print(employee["salary"].unique())
    pc = pycaret.regression.setup(
        data=employee,
        ordinal_features={"salary": ["low", "medium", "high"]},
    )
    X, _ = pc._internal_pipeline.fit_transform(pc.X, pc.y)
    assert list(X["salary"].unique()) == [0.0, 1.0, 2.0]
