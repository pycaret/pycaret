import pandas as pd
import numpy as np
import datetime
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
import pytest


def test_sklearn_pipeline_simple_imputer():
    """
    Test if the simple imputer in pycaret works with sklearn's pipeline
    """

    # Load an example dataset and set the features and target
    data = pycaret.datasets.get_data("juice")
    target = "Purchase"
    features = data.columns.tolist()
    features.remove(target)
    data_features = data[features]
    data_target = data[target]

    # Set the numeric and categorical features
    categorical_features = data_features.select_dtypes(
        include=["category", "object"]
    ).columns
    numeric_features = [x for x in features if x not in categorical_features]

    # Initiate a pycaret simple imputer
    simple_imputer = pycaret.internal.preprocess.Simple_Imputer(
        numeric_strategy="mean",
        categorical_strategy="most frequent",
        time_strategy="most frequent",
        target=target,
    )

    # Apply the simple imputer to both the categorical and numeric features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", simple_imputer),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ],
        verbose=True,
    )

    # Numeric features don't require to be encoded for a ML model to work
    numeric_transformer = Pipeline(steps=[("imputer", simple_imputer)], verbose=True)

    # Obtain the full preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Append classifier to preprocessing pipeline. Now we have a full prediction pipeline.
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier()),
        ],
        verbose=True,
    )

    # Test if the full pipeline works with sklearn's randomized search
    param_dist = {"classifier__n_estimators": stats.randint(10, 20)}
    search = RandomizedSearchCV(clf, param_distributions=param_dist)
    search.fit(data_features, data_target)

    # Check if the best parameter falls within the defined range
    assert 10 <= search.best_params_["classifier__n_estimators"] <= 20


def test_simple_imputer():
    """
    Test if the simple imputer imputes correctly for various data types
    """

    # Load an example dataset and set the features and target
    test_length = 6
    data = pycaret.datasets.get_data("juice")[0:test_length]
    target = "Purchase"

    # Add columns for testing additional data types
    data["time"] = pd.date_range(
        datetime.datetime(2020, 12, 1), periods=test_length, freq="d"
    )
    data.loc[3:7, "time"] = pd.to_datetime(datetime.datetime(2020, 12, 30))
    data["time_delta_day"] = datetime.timedelta(days=10)
    data["time_delta_hour"] = datetime.timedelta(hours=10)
    data["missing_num_col"] = 100
    data["missing_num_col"] = data["missing_num_col"].astype("int32")

    # Make the values of first row missing
    data.loc[0, :] = np.nan

    # Initiate a pycaret simple imputer
    simple_imputer = pycaret.internal.preprocess.Simple_Imputer(
        numeric_strategy="mean",
        categorical_strategy="most frequent",
        time_strategy="mean",
        target=target,
    )
    result = simple_imputer.fit_transform(data)

    # Check if the target is not imputed
    assert result[target].isnull().sum() == 1

    # Check if all the features are imputed
    result.drop(columns=[target], inplace=True)
    assert result.isnull().sum().sum() == 0

    # Check if the missing values are imputed to the correct values
    assert result.loc[0, "time"] == pd.to_datetime(datetime.datetime(2020, 12, 19))
    assert result.loc[0, "time_delta_day"] == datetime.timedelta(days=10)
    assert result.loc[0, "time_delta_hour"] == datetime.timedelta(hours=10)
    assert result.loc[0, "missing_num_col"] == 100


@pytest.mark.skip(
    reason="sktime 0.7.0 needs sklearn 0.24.0 which causes this to fail. Re-enable when preprocessing is reworked."
)
def test_complete_sklearn_pipeline():
    """
    Test if the pycaret's pipeline works with sklearn's pipeline
    """

    # Load an example dataset and set the features and target
    data = pycaret.datasets.get_data("juice")
    target = "Purchase"
    features = data.columns.tolist()
    features.remove(target)
    data_features = data[features]
    data_target = data[target]

    # Initiate a pycaret pipeline
    pycaret_preprocessor = pycaret.internal.preprocess.Preprocess_Path_One_Sklearn(
        train_data=data,
        target_variable=target,
        display_types=False,
        apply_pca=True,
        pca_variance_retained_or_number_of_components=5,
        pca_method="incremental",
    )
    transformed_data = pycaret_preprocessor.fit_transform(
        X=data_features, y=data_target
    )

    assert isinstance(transformed_data, pd.DataFrame)

    # Append classifier to preprocessing pipeline. Now we have a full prediction pipeline.
    clf = Pipeline(
        steps=[
            ("preprocessor", pycaret_preprocessor),
            ("classifier", RandomForestClassifier()),
        ],
        verbose=True,
    )

    # Test if the full pipeline works with sklearn's randomized search
    param_dist = {"classifier__n_estimators": stats.randint(10, 20)}
    search = RandomizedSearchCV(clf, param_distributions=param_dist)
    search.fit(data_features, data_target)

    # Check if the best parameter falls within the defined range
    assert 10 <= search.best_params_["classifier__n_estimators"] <= 20


def test_target_transformer():
    # Load an example dataset and set the features and target
    data = pycaret.datasets.get_data("juice")
    target = "Purchase"
    features = data.columns.tolist()
    features.remove(target)
    data_features = data[features]
    data_target = data[target]

    # Set the numeric and categorical features
    categorical_features = data_features.select_dtypes(
        include=["category", "object"]
    ).columns
    numeric_features = [x for x in features if x not in categorical_features]

    # Initiate a pycaret simple imputer
    simple_imputer = pycaret.internal.preprocess.Simple_Imputer(
        numeric_strategy="mean",
        categorical_strategy="most frequent",
        time_strategy="most frequent",
        target=target,
    )

    # Apply the simple imputer to both the categorical and numeric features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", simple_imputer),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ],
        verbose=True,
    )

    # Numeric features don't require to be encoded for a ML model to work
    numeric_transformer = Pipeline(steps=[("imputer", simple_imputer)], verbose=True)

    # Obtain the full preprocessing pipeline
    preprocessor_X = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    trans_target_classifier = TransformedTargetClassifier(
        classifier=RandomForestClassifier(), transformer=LabelEncoder()
    )

    clf = Pipeline(
        steps=[
            ("preprocessor_x", preprocessor_X),
            ("trans_target_classifier", trans_target_classifier),
        ],
        verbose=True,
    )

    # Make sure the complete pipeline works with sklearn's randomized search
    param_dist = {
        "trans_target_classifier__classifier__n_estimators": stats.randint(10, 20)
    }
    search = RandomizedSearchCV(clf, param_distributions=param_dist)
    search.fit(data_features, data_target)
    predictions = search.best_estimator_.predict(data_features)

    # Check if the predictions are reasonably accurate, since we should be able to overfit the train set
    assert accuracy_score(predictions, data_target) > 0.95

    # Check if the encoded target is correct
    clf.fit(data_features, data_target)
    assert (
        clf.named_steps["trans_target_classifier"].transformer_.transform(data_target)
        == LabelEncoder().fit_transform(data_target)
    ).all()

    # Check if TransformedTargetClassifier is sklearn-compatible
    check_estimator(TransformedTargetClassifier())  # This should pass


def test_auto_infer_label():
    # loading dataset
    data = pycaret.datasets.get_data("juice")
    data.loc[:, 'test_target'] = np.random.randint(5, 8, data.shape[0])
    data.loc[:, 'test_target'] = data.loc[:, 'test_target'].astype(np.int64)  # should not encode
    target = 'test_target'

    # init setup
    _ = pycaret.classification.setup(
        data,
        target=target,
        log_experiment=True,
        silent=True,
        html=False,
        session_id=123,
        n_jobs=1
    )

    with pytest.raises(AttributeError):
        _ = pycaret.classification.get_config('prep_pipe').named_steps["dtypes"].replacement


def test():
    # loading dataset
    data = pycaret.datasets.get_data("juice")
    target = "Purchase"

    # preprocess all in one
    pipe = pycaret.internal.preprocess.Preprocess_Path_One(
        train_data=data, target_variable=target, display_types=False
    )
    X = pipe.fit_transform(data)
    assert isinstance(X, pd.core.frame.DataFrame)


if __name__ == "__main__":
    test()
