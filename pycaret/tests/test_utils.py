import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pytest
import pycaret.classification
import pycaret.datasets
import sklearn.model_selection
import sklearn.preprocessing


def test():
    # version
    pycaret.utils.version()
    pycaret.utils.__version__()

    # preparation(classification)
    data = pycaret.datasets.get_data("juice")
    target = "Purchase"
    le = sklearn.preprocessing.LabelEncoder()
    le = le.fit(data[target])
    data[target] = le.transform(data[target])
    train, test = sklearn.model_selection.train_test_split(data, train_size=0.8, random_state=1)
    clf1 = pycaret.classification.setup(train, target=target, silent=True, html=False, session_id=123)
    model = pycaret.classification.create_model("lightgbm")
    data_unseen = test.drop(columns=target)
    result = pycaret.classification.predict_model(model, data=data_unseen)
    actual = test[target]
    prediction = result["Label"]

    # check metric(classification)
    pycaret.utils.check_metric(actual, prediction, "Accuracy")
    pycaret.utils.check_metric(actual, prediction, "Recall")
    pycaret.utils.check_metric(actual, prediction, "Precision")
    pycaret.utils.check_metric(actual, prediction, "F1")
    pycaret.utils.check_metric(actual, prediction, "Kappa")
    pycaret.utils.check_metric(actual, prediction, "AUC")
    pycaret.utils.check_metric(actual, prediction, "MCC")

    # preparation(regression)
    data = pycaret.datasets.get_data("boston")
    target = "medv"
    train, test = sklearn.model_selection.train_test_split(data, train_size=0.8, random_state=1)
    reg1 = pycaret.regression.setup(data, target="medv", silent=True, html=False, session_id=123)
    model = pycaret.regression.create_model("lightgbm")
    data_unseen = test.drop(columns=target)
    result = pycaret.regression.predict_model(model, data=data_unseen)
    actual = test[target].reset_index()
    prediction = result["Label"].reset_index()

    # check metric(regression)
    pycaret.utils.check_metric(actual, prediction, "MAE")
    pycaret.utils.check_metric(actual, prediction, "MSE")
    pycaret.utils.check_metric(actual, prediction, "RMSE")
    pycaret.utils.check_metric(actual, prediction, "R2")
    pycaret.utils.check_metric(actual, prediction, "RMSLE")
    pycaret.utils.check_metric(actual, prediction, "MAPE")

    # enable colab
    pycaret.utils.enable_colab()

    assert 1 == 1

if __name__ == "__main__":
    test()
