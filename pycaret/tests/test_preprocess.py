# coding: utf-8

"""
Package: PyCaret
Author: Mavs
Description: Unit tests for pipeline.py

"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN

import pycaret.datasets
import pycaret.regression
import pycaret.classification


def test_select_target_by_index():
    """Assert that the target can be selected by its column index."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(data, target=2)
    assert pc.target_param == "WeekofPurchase"


def test_preprocess_is_False():
    """Assert that preprocessing is skipped when preprocess=False."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(data, preprocess=False)
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X["Purchase"].dtype.kind not in "ifu"  # No encoding of categorical columns


def test_ignore_features():
    """Assert that features can be ignored in preprocessing."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(data, ignore_features=["Purchase"])
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert "Purchase" not in X


def test_encode_target():
    """Assert that the target column is automatically encoded."""
    data = pycaret.datasets.get_data("telescope")
    pc = pycaret.classification.setup(data)
    _, y = pc.pipeline.transform(pc.X, pc.y)
    assert y.dtype.kind in "ifu"


def test_date_features():
    """Assert that features are extracted from date features."""
    data = pycaret.datasets.get_data("juice")
    data["date"] = pd.date_range(start="1/1/2018", periods=len(data))
    pc = pycaret.classification.setup(data, target=-2, date_features=["date"])
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert all([f"date_{attr}" in X for attr in ("day", "month", "year")])


@pytest.mark.parametrize(
    "imputation_method", [0, "drop", "mean", "median", "mode", "knn"]
)
def test_simple_numeric_imputation(imputation_method):
    """Assert that missing values are imputed."""
    data = pycaret.datasets.get_data("juice")
    data.loc[100, "WeekofPurchase"] = np.nan
    pc = pycaret.classification.setup(
        data=data,
        imputation_type="simple",
        numeric_iterative_imputer=imputation_method,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.isna().sum().sum() == 0


@pytest.mark.parametrize("imputation_method", ["drop", "missing", "mode"])
def test_simple_categorical_imputation(imputation_method):
    """Assert that missing values are imputed."""
    data = pycaret.datasets.get_data("juice")
    data.loc[100, "Purchase"] = np.nan
    pc = pycaret.classification.setup(
        data=data,
        imputation_type="simple",
        categorical_imputation=imputation_method,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.isna().sum().sum() == 0


def test_iterative_imputer():
    """Test iterative imputer"""
    data = pycaret.datasets.get_data("juice")
    categories = {}
    for i, col in enumerate(data.columns):
        # leave two columns and target filled
        if col in ("STORE", "PriceCH", "DiscMM"):
            continue
        if col in ("Purchase", "Store7"):
            categories[col] = set(data[col].unique())
        data.loc[data.sample(frac=0.1, random_state=i).index, col] = pd.np.nan
    for imputer in ["catboost", "lightgbm", "rf", "lr"]:
        data = data.copy()
        pc = pycaret.classification.setup(
            data=data,
            imputation_type="iterative",
            numeric_iterative_imputer=imputer,
            categorical_iterative_imputer=imputer,
        )
        transformer = pc.pipeline.named_steps["iterative_imputer"]
        df = transformer.transform(data, data["STORE"])[0]
        assert not df.isnull().values.any()
        assert all(categories[col] == set(df[col].unique()) for col in categories)
        df = transformer.transform(data, data["STORE"])[0]
        assert not df.isnull().values.any()
        assert all(categories[col] == set(df[col].unique()) for col in categories)


@pytest.mark.parametrize("embedding_method", ["bow", "tf-idf"])
def test_text_embedding(embedding_method):
    """Assert that text columns are embedded."""
    data = pycaret.datasets.get_data("spx")
    pc = pycaret.regression.setup(
        data=data.iloc[:50, :],  # Less rows for faster processing
        text_features=["text"],
        text_features_method=embedding_method,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.shape[1] > 50  # Text column is now embedding


def test_ordinal_features():
    """Assert that ordinal features are encoded correctly."""
    data = pycaret.datasets.get_data("employee")
    pc = pycaret.classification.setup(
        data=data,
        imputation_type=None,
        ordinal_features={"salary": ["low", "medium", "high"]},
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    mapping = pc.pipeline.steps[0][1].transformer.mapping
    assert mapping[0]["mapping"]["low"] == 0
    assert mapping[0]["mapping"]["medium"] == 1
    assert mapping[0]["mapping"]["high"] == 2


def test_categorical_features():
    """Assert that categorical features are encoded correctly."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(data)
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert list(sorted(X["Purchase"].unique())) == [0.0, 1.0]


@pytest.mark.parametrize("transformation_method", ["yeo-johnson", "quantile"])
def test_transformation(transformation_method):
    """Assert that features can be transformed to a gaussian distribution."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        transformation=True,
        transformation_method=transformation_method,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert list(X["Purchase"].unique()) != [0.0, 1.0]


@pytest.mark.parametrize("normalize_method", ["zscore", "minmax", "maxabs", "robust"])
def test_transformation(normalize_method):
    """Assert that features can be normalized."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        normalize=True,
        normalize_method=normalize_method,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X["WeekofPurchase"].max() < 5


def test_low_variance_threshold():
    """Assert that features with low variance are dropped."""
    data = pycaret.datasets.get_data("juice")
    data["feature"] = 1  # Minimal variance
    pc = pycaret.classification.setup(
        data=data,
        target="STORE",
        low_variance_threshold=1.0,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert "feature" not in X


def test_remove_multicollinearity():
    """Assert that one of two collinear features are dropped."""
    data = pycaret.datasets.get_data("juice")
    data["Id 2"] = list(range(len(data)))  # Correlated with Id
    pc = pycaret.classification.setup(
        data=data,
        target="STORE",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.9999,
    )

    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert "Id" in X and "Id 2" not in X


def test_bin_numeric_features():
    """Assert that numeric features can be binned."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(data=data, bin_numeric_features=["Id"])
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X["Id"].nunique() == 5


@pytest.mark.parametrize("outliers_method", ["iforest", "ee", "lof"])
def test_remove_outliers(outliers_method):
    """Assert that outliers can be removed."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        remove_outliers=True,
        outliers_method=outliers_method,
        outliers_threshold=0.2,
    )
    assert pc.pipeline.steps[3][0] == "remove_outliers"


def test_polynomial_features():
    """Assert that polynomial features can be created."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        polynomial_features=True,
        polynomial_degree=2,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.shape[1] > data.shape[1]  # Extra features were created


@pytest.mark.parametrize("fix_imbalance_method", [None, ADASYN()])
def test_fix_imbalance(fix_imbalance_method):
    """Assert that the classes can be balanced."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        fix_imbalance=True,
        fix_imbalance_method=fix_imbalance_method,
    )
    assert pc.pipeline.steps[3][0] == "balance"  # Rows are over-sampled


@pytest.mark.parametrize("pca_method", ["linear", "kernel", "incremental"])
def test_pca(pca_method):
    """Assert that pca can be applied."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        pca=True,
        pca_method=pca_method,
        pca_components=10,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.shape[1] == 10


def test_keep_features():
    """Assert that features are not dropped through preprocess."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        keep_features=["Id"],
        pca=True,
        pca_components=8,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert "Id" in X


@pytest.mark.parametrize("fs_method", ["univariate", "classic", "sequential"])
def test_feature_selection(fs_method):
    """Assert that feature selection can be applied."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        feature_selection=True,
        feature_selection_method=fs_method,
        feature_selection_estimator="rf",
        n_features_to_select=12,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.shape[1] == 12


def test_custom_pipeline_is_list():
    """Assert that a custom pipeline can be provided."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        custom_pipeline=[("pca", PCA(n_components=5))],
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.shape[1] == 5


def test_custom_pipeline_is_pipeline():
    """Assert that a custom pipeline can be provided."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        custom_pipeline=Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=5))]
        ),
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.shape[1] == 5
