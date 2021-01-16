import os, sys

sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pytest
import pycaret.utils
import pycaret.classification
import pycaret.datasets
import pycaret.regression
import sklearn.model_selection
import sklearn.preprocessing


def test():
    # version
    version = pycaret.utils.version()
    assert isinstance(version, str)
    nightly_version = pycaret.utils.nightly_version()
    assert isinstance(nightly_version, str)
    version = pycaret.utils.__version__
    assert isinstance(version, str)

    # preparation(classification)
    data = pycaret.datasets.get_data("juice")
    target = "Purchase"
    le = sklearn.preprocessing.LabelEncoder()
    le = le.fit(data[target])
    data[target] = le.transform(data[target])
    train, test = sklearn.model_selection.train_test_split(
        data, train_size=0.8, random_state=1
    )
    clf1 = pycaret.classification.setup(
        train, target=target, silent=True, html=False, session_id=123, n_jobs=1,
    )
    model = pycaret.classification.create_model("lightgbm")
    data_unseen = test.drop(columns=target)
    final_model = pycaret.classification.finalize_model(model)
    result = pycaret.classification.predict_model(final_model, data=data_unseen)
    actual = test[target]
    prediction = result["Label"]

    # provisional support
    actual = actual.dropna(axis=0, how="any")
    actual = actual.reset_index()
    actual = actual["Purchase"].astype(np.int64)
    prediction = prediction.dropna(axis=0, how="any")
    prediction = prediction.reset_index()
    prediction = prediction["Label"].astype(np.int64)

    # check metric(classification)
    accuracy = pycaret.utils.check_metric(actual, prediction, "Accuracy")
    assert isinstance(accuracy, float)
    assert accuracy >= 0
    assert accuracy <= 1
    recall = pycaret.utils.check_metric(actual, prediction, "Recall")
    assert isinstance(recall, float)
    assert recall >= 0
    assert recall <= 1
    precision = pycaret.utils.check_metric(actual, prediction, "Precision")
    assert isinstance(precision, float)
    assert precision >= 0
    assert precision <= 1
    f1 = pycaret.utils.check_metric(actual, prediction, "F1")
    assert isinstance(f1, float)
    assert f1 >= 0
    assert f1 <= 1
    kappa = pycaret.utils.check_metric(actual, prediction, "Kappa")
    assert isinstance(kappa, float)
    assert kappa >= -1
    assert kappa <= 1
    auc = pycaret.utils.check_metric(actual, prediction, "AUC")
    assert isinstance(auc, float)
    assert auc >= 0
    assert auc <= 1
    mcc = pycaret.utils.check_metric(actual, prediction, "MCC")
    assert isinstance(mcc, float)
    assert mcc >= -1
    assert mcc <= 1

    # preparation(regression)
    data = pycaret.datasets.get_data("boston")
    target = "medv"
    train, test = sklearn.model_selection.train_test_split(
        data, train_size=0.8, random_state=1
    )
    reg1 = pycaret.regression.setup(
        data, target="medv", silent=True, html=False, session_id=123, n_jobs=1,
    )
    model = pycaret.regression.create_model("lightgbm")
    data_unseen = test.drop(columns=target)
    final_model = pycaret.regression.finalize_model(model)
    result = pycaret.regression.predict_model(final_model, data=data_unseen)
    actual = test[target]
    prediction = result["Label"]

    # provisional support
    actual = actual.dropna(axis=0, how="any")
    actual = actual.reset_index()
    actual = actual.drop("index", axis=1)
    prediction = prediction.dropna(axis=0, how="any")
    prediction = prediction.reset_index()
    prediction = prediction.drop("index", axis=1)

    # check metric(regression)
    mae = pycaret.utils.check_metric(actual, prediction, "MAE")
    assert isinstance(mae, float)
    assert mae >= 0
    mse = pycaret.utils.check_metric(actual, prediction, "MSE")
    assert isinstance(mse, float)
    assert mse >= 0
    rmse = pycaret.utils.check_metric(actual, prediction, "RMSE")
    assert isinstance(rmse, float)
    assert rmse >= 0
    r2 = pycaret.utils.check_metric(actual, prediction, "R2")
    assert isinstance(r2, float)
    assert r2 <= 1
    rmsle = pycaret.utils.check_metric(actual, prediction, "RMSLE")
    assert isinstance(rmsle, float)
    assert rmsle >= 0
    mape = pycaret.utils.check_metric(actual, prediction, "MAPE")
    assert isinstance(mape, float)
    assert mape >= 0

    # Ensure metric is rounded to 2 decimals
    mape = pycaret.utils.check_metric(actual, prediction, "MAPE", 2)
    assert mape == 0.05

    # Ensure metric is rounded to default value
    mape = pycaret.utils.check_metric(actual, prediction, "MAPE")
    assert mape == 0.0469

    # Metric does not exist
    with pytest.raises(ValueError, match="Couldn't find metric"):
        pycaret.utils.check_metric(actual, prediction, "INEXISTENTMETRIC")

    assert 1 == 1


if __name__ == "__main__":
    test()
