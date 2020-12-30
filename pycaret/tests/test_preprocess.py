import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pycaret.datasets
import pycaret.internal.preprocess


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
    categorical_features = data_features.select_dtypes(include=['category', 'object']).columns
    numeric_features = [x for x in features if x not in categorical_features]

    # Initiate a pycaret simple imputer
    simple_imputer = pycaret.internal.preprocess.Simple_Imputer(
        numeric_strategy='mean',
        categorical_strategy='most frequent',
        target=target)

    # Apply the simple imputer to both the categorical and numeric features
    categorical_transformer = Pipeline(steps=[('imputer', simple_imputer),
                                              ('encoder', OneHotEncoder(handle_unknown='ignore'))],
                                       verbose=True)

    # Numeric features don't require to be encoded for a ML model to work
    numeric_transformer = Pipeline(steps=[('imputer', simple_imputer)], verbose=True)

    # Obtain the full preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline. Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', xgb.XGBClassifier())],
                   verbose=True)

    # Test if the full pipeline works with sklearn's randomized search
    param_dist = {'classifier__n_estimators': stats.randint(10, 20)}
    search = RandomizedSearchCV(clf, param_distributions=param_dist)
    search.fit(data_features, data_target)

    # Check if the best parameter falls within the defined range
    assert 10 <= search.best_params_['classifier__n_estimators'] <= 20


def test_simple_imputer():
    """
    Test if the simple imputer imputes correctly for various data types
    """

    # Load an example dataset and set the features and target
    data = pycaret.datasets.get_data("juice")
    target = "Purchase"

    # Add columns for testing additional data types
    data["time"] = pd.to_datetime('now')
    data["time_delta"] = datetime.timedelta(days=10)
    data["missing_num_col"] = 100
    data["missing_num_col"] = data["missing_num_col"].astype("int32")

    # Make the values of first row missing
    data.loc[0, :] = np.nan

    # Initiate a pycaret simple imputer
    simple_imputer = pycaret.internal.preprocess.Simple_Imputer(
        numeric_strategy='mean',
        categorical_strategy='most frequent',
        target=target)
    result = simple_imputer.fit_transform(data)

    # Check if the target is not imputed
    assert result[target].isnull().sum() == 1

    # Check if all the features are imputed
    result.drop(columns=[target], inplace=True)
    assert result.isnull().sum().sum() == 0

    # Check if the missing values are imputed to the correct values
    assert result.loc[0, "time"] == result.loc[1, "time"]
    assert result.loc[0, "time_delta"] == datetime.timedelta(days=10)
    assert result.loc[0, "missing_num_col"] == 100


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
