import pandas as pd

import pycaret.classification
import pycaret.datasets
import pycaret.regression


def test_classification_predict_model():
    # loading classification dataset
    data = pycaret.datasets.get_data("juice")
    assert isinstance(data, pd.DataFrame)

    training_data = data.sample(frac=0.90)
    unseen_data = data.drop(training_data.index)

    # init setup
    pycaret.classification.setup(
        data,
        target="Purchase",
        ignore_features=["WeekofPurchase"],
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    lr_model = pycaret.classification.create_model("lr")
    predictions = pycaret.classification.predict_model(lr_model, data=unseen_data)
    # Check that columns of raw data are contained in columns of returned dataframe
    assert all(item in predictions.columns for item in unseen_data.columns)


def test_regression_predict_model():
    # loading classification dataset
    data = pycaret.datasets.get_data("boston")
    assert isinstance(data, pd.DataFrame)

    training_data = data.sample(frac=0.90)
    unseen_data = data.drop(training_data.index)

    # init setup
    pycaret.regression.setup(
        data,
        target="medv",
        ignore_features=["crim", "zn"],
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    lr_model = pycaret.regression.create_model("lr")
    predictions = pycaret.regression.predict_model(lr_model, data=unseen_data)
    # Check that columns of raw data are contained in columns of returned dataframe
    assert all(item in predictions.columns for item in unseen_data.columns)
