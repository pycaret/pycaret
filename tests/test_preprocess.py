# coding: utf-8

"""
Package: PyCaret
Author: Mavs
Description: Unit tests for pipeline.py

"""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import ADASYN
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pycaret.classification
import pycaret.datasets
import pycaret.regression


def test_select_target_by_index():
    """Assert that the target can be selected by its column index."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(data, target=2)
    assert pc.target_param == "WeekofPurchase"


def test_select_target_by_str():
    """Assert that the target can be selected by its column name."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(data, target="WeekofPurchase")
    assert pc.target_param == "WeekofPurchase"


def test_nans_in_target_column():
    """Assert that the target can be selected by its column name."""
    data = pycaret.datasets.get_data("juice")
    data.loc[3, "WeekofPurchase"] = np.nan
    with pytest.raises(ValueError, match=r".*missing values found.*"):
        pycaret.classification.setup(data, target="WeekofPurchase")


def test_select_target_by_sequence():
    """Assert that the target can be a sequence."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(data, target=[1] * len(data))
    assert pc.target_param == "target"


def test_input_is_array():
    """Assert that the input can be a numpy array."""
    pc = pycaret.classification.setup(np.eye(4), target=[1, 0, 0, 1])
    assert isinstance(pc.dataset, pd.DataFrame)
    assert pc.target_param == "target"


def test_input_is_sparse():
    """Assert that the input can be a scipy sparse matrix."""
    pc = pycaret.classification.setup(
        data=csr_matrix((300, 4)),
        target=[1, 0, 1] * 100,
        preprocess=False,
    )
    assert isinstance(pc.dataset, pd.DataFrame)
    assert pc.target_param == "target"


def test_assign_index_is_false():
    """Assert that the index is reset when index=False."""
    data = pycaret.datasets.get_data("juice")
    data.index = list(range(100, len(data) + 100))
    pc = pycaret.classification.setup(data, index=False)
    assert pc.dataset.index[0] == 0


def test_assign_index_is_true():
    """Assert that the index remains unchanged when index=True."""
    data = pycaret.datasets.get_data("juice")
    data.index = list(range(100, len(data) + 100))
    pc = pycaret.classification.setup(
        data=data,
        index=True,
        data_split_shuffle=False,
        data_split_stratify=False,
    )
    assert pc.dataset.index[0] == 100


@pytest.mark.parametrize("index", [1, "WeekofPurchase", list(range(2, 1072))])
def test_assign_index(index):
    """Assert that the index can be assigned."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        index=index,
        data_split_shuffle=False,
        data_split_stratify=False,
        preprocess=False,
    )
    assert pc.dataset.index[0] != 0


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


def test_weird_chars_in_column_names():
    """Assert that weird characters from column names are dropped."""
    data = pycaret.datasets.get_data("parkinsons")
    data.columns = ["[col"] + list(data.columns[1:])
    assert "[" in data.columns[0]

    pc = pycaret.regression.setup(data)
    assert pc.dataset_transformed.columns[0] == "col"


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


def test_custom_date_features():
    """Assert that features are extracted from date features."""
    data = pycaret.datasets.get_data("juice")
    data["date"] = pd.date_range(start="1/1/2018", periods=len(data))
    pc = pycaret.classification.setup(
        data,
        target=-2,
        date_features=["date"],
        create_date_columns=["quarter"],
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert "date_quarter" in X and "day" not in X


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


@pytest.mark.parametrize("dtypes_to_select", ("mixed", "num_only", "cat_only"))
@pytest.mark.parametrize("imputer", ("catboost", "lightgbm", "rf", "lr"))
def test_iterative_imputer(dtypes_to_select, imputer):
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

    if dtypes_to_select == "num_only":
        data_subset = data.select_dtypes(include="float")
        categories = {}
    elif dtypes_to_select == "cat_only":
        data_subset = data.select_dtypes(exclude="float")
    else:
        data_subset = data
    data_subset["STORE"] = data["STORE"]

    data_subset = data_subset.copy()
    pc = pycaret.classification.setup(
        data=data_subset,
        target="STORE",
        imputation_type="iterative",
        numeric_iterative_imputer=imputer,
        categorical_iterative_imputer=imputer,
    )
    transformer = pc.pipeline.named_steps["iterative_imputer"]
    df = transformer.transform(data_subset, data_subset["STORE"])[0]
    assert not df.isnull().values.any()
    assert all(categories[col] == set(df[col].unique()) for col in categories)
    df = transformer.transform(data_subset, data_subset["STORE"])[0]
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


def test_encoding_ordinal_features():
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


def test_encoding_grouping_rare_categories():
    """Assert that rare categories are grouped before encoding."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(data, rare_to_value=0.5)
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert "rare" in pc.pipeline.steps[-1][1].transformer.mapping[0]["mapping"]


def test_encoding_categorical_features():
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
def test_normalize(normalize_method):
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
        low_variance_threshold=0,
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert "feature" not in X


def test_feature_grouping():
    """Assert that feature groups are replaced for stats."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        target="STORE",
        group_features=[list(data.columns[:2]), list(data.columns[3:5])],
        group_names=["gr1", "gr2"],
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert "Id" not in X
    assert "mean(gr1)" in X and "median(gr2)" in X


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
        low_variance_threshold=None,
        remove_outliers=True,
        outliers_method=outliers_method,
        outliers_threshold=0.2,
    )
    assert pc.pipeline.steps[-1][0] == "remove_outliers"


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


@pytest.mark.parametrize(
    "fix_imbalance_method", ["smote", "nearmiss", "SMOTEENN", ADASYN()]
)
def test_fix_imbalance(fix_imbalance_method):
    """Assert that the classes can be balanced."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        low_variance_threshold=None,
        fix_imbalance=True,
        fix_imbalance_method=fix_imbalance_method,
    )
    assert pc.pipeline.steps[-1][0] == "balance"  # Rows are sampled


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
    """Assert that a custom pipeline can be provided as list."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        custom_pipeline=[("pca", PCA(n_components=5))],
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.shape[1] == 5


def test_custom_pipeline_is_pipeline():
    """Assert that a custom pipeline can be provided as a Pipeline object."""
    data = pycaret.datasets.get_data("juice")
    pc = pycaret.classification.setup(
        data=data,
        custom_pipeline=Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=5))]
        ),
    )
    X, _ = pc.pipeline.transform(pc.X, pc.y)
    assert X.shape[1] == 5


@pytest.mark.parametrize("pos", [-1, 0, 1])
def test_custom_pipeline_positions(pos):
    """Assert that a custom pipeline can be provided at a specific position."""
    data = pycaret.datasets.get_data("cancer")
    pc = pycaret.classification.setup(
        data=data,
        remove_outliers=True,
        remove_multicollinearity=True,
        custom_pipeline=[("scaler", StandardScaler())],
        custom_pipeline_position=pos,
    )
    assert pc.pipeline.steps[pos][0] == "scaler"
