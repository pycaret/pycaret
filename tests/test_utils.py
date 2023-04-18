import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import sklearn.model_selection
import sklearn.preprocessing

import pycaret.classification
import pycaret.datasets
import pycaret.regression
import pycaret.utils
from pycaret.utils.constants import LABEL_COLUMN
from pycaret.utils.generic import check_metric


def test_utils():
    # version
    version = pycaret.utils.version()
    assert isinstance(version, str)
    version = pycaret.utils.__version__
    assert isinstance(version, str)

    # preparation(classification)
    data = pycaret.datasets.get_data("juice")
    train, test = sklearn.model_selection.train_test_split(
        data, train_size=0.8, random_state=1
    )
    clf1 = pycaret.classification.setup(
        train,
        target="Purchase",
        html=False,
        session_id=123,
        n_jobs=1,
    )
    model = pycaret.classification.create_model("lightgbm")
    final_model = pycaret.classification.finalize_model(model)
    result = pycaret.classification.predict_model(
        final_model, data=test.drop("Purchase", axis=1), encoded_labels=True
    )
    actual = clf1.pipeline.transform(y=test["Purchase"])
    prediction = result[LABEL_COLUMN]

    # provisional support
    actual = actual.dropna(axis=0, how="any")
    actual = actual.reset_index()
    actual = actual["Purchase"].astype(np.int64)
    prediction = prediction.dropna(axis=0, how="any")
    prediction = prediction.reset_index()
    prediction = prediction[LABEL_COLUMN].astype(np.int64)

    # check metric(classification)
    accuracy = check_metric(actual, prediction, "Accuracy")
    assert isinstance(accuracy, float)
    assert accuracy >= 0
    assert accuracy <= 1
    recall = check_metric(actual, prediction, "Recall")
    assert isinstance(recall, float)
    assert recall >= 0
    assert recall <= 1
    precision = check_metric(actual, prediction, "Precision")
    assert isinstance(precision, float)
    assert precision >= 0
    assert precision <= 1
    f1 = check_metric(actual, prediction, "F1")
    assert isinstance(f1, float)
    assert f1 >= 0
    assert f1 <= 1
    kappa = check_metric(actual, prediction, "Kappa")
    assert isinstance(kappa, float)
    assert kappa >= -1
    assert kappa <= 1
    auc = check_metric(actual, prediction, "AUC")
    assert isinstance(auc, float)
    assert auc >= 0
    assert auc <= 1
    mcc = check_metric(actual, prediction, "MCC")
    assert isinstance(mcc, float)
    assert mcc >= -1
    assert mcc <= 1

    # preparation(regression)
    data = pycaret.datasets.get_data("boston")
    train, test = sklearn.model_selection.train_test_split(
        data, train_size=0.8, random_state=1
    )
    pycaret.regression.setup(
        data,
        target="medv",
        html=False,
        session_id=123,
        n_jobs=1,
    )
    model = pycaret.regression.create_model("lightgbm")
    final_model = pycaret.regression.finalize_model(model)
    result = pycaret.regression.predict_model(
        final_model, data=test.drop("medv", axis=1)
    )
    actual = test["medv"]
    prediction = result[LABEL_COLUMN]

    # provisional support
    actual = actual.dropna(axis=0, how="any")
    actual = actual.reset_index()
    actual = actual.drop("index", axis=1)
    prediction = prediction.dropna(axis=0, how="any")
    prediction = prediction.reset_index()
    prediction = prediction.drop("index", axis=1)

    # check metric(regression)
    mae = check_metric(actual, prediction, "MAE")
    assert isinstance(mae, float)
    assert mae >= 0
    mse = check_metric(actual, prediction, "MSE")
    assert isinstance(mse, float)
    assert mse >= 0
    rmse = check_metric(actual, prediction, "RMSE")
    assert isinstance(rmse, float)
    assert rmse >= 0
    r2 = check_metric(actual, prediction, "R2")
    assert isinstance(r2, float)
    assert r2 <= 1
    rmsle = check_metric(actual, prediction, "RMSLE")
    assert isinstance(rmsle, float)
    assert rmsle >= 0
    mape = check_metric(actual, prediction, "MAPE")
    assert isinstance(mape, float)
    assert mape >= 0

    # Ensure metric is rounded to 2 decimals
    mape = check_metric(actual, prediction, "MAPE", 2)
    npt.assert_almost_equal(mape, 0.05, decimal=2)

    # Ensure metric is rounded to default value
    mape = check_metric(actual, prediction, "MAPE")
    npt.assert_almost_equal(mape, 0.045, decimal=2)

    # preparation (timeseries)
    data = pycaret.datasets.get_data("airline", verbose=False)
    train, test = sklearn.model_selection.train_test_split(
        data, train_size=0.8, random_state=1, shuffle=False
    )

    prediction = pd.Series([100] * len(test), index=test.index)
    actual = test

    # check metric(timeseries)
    smape = check_metric(actual, prediction, "SMAPE")
    assert isinstance(smape, float)
    assert smape >= 0
    mape = check_metric(actual, prediction, "MAPE")
    assert isinstance(mape, float)
    assert mape >= 0
    # mase = pycaret.utils.check_metric(test, prediction, "MASE", train=train)
    # assert isinstance(mase, float)
    # assert mase >= 0
    mae = check_metric(actual, prediction, "MAE")
    assert isinstance(mae, float)
    assert mae >= 0
    rmse = check_metric(actual, prediction, "RMSE")
    assert isinstance(rmse, float)
    assert rmse >= 0

    # Ensure metric is rounded to 2 decimals
    smape = check_metric(actual, prediction, "SMAPE", 2)
    npt.assert_almost_equal(smape, 1.24, decimal=2)

    # Ensure metric is rounded to default value
    smape = check_metric(actual, prediction, "SMAPE")
    npt.assert_almost_equal(smape, 1.2448, decimal=4)

    # Metric does not exist
    with pytest.raises(ValueError, match="Couldn't find metric"):
        check_metric(actual, prediction, "INEXISTENTMETRIC")

    assert 1 == 1


if __name__ == "__main__":
    test_utils()
