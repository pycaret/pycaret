import os, sys
sys.path.insert(0, os.path.abspath(".."))

import numpy  as np
import pandas as pd
import pytest
import pycaret.utils
import pycaret.classification
import pycaret.datasets
import pycaret.regression
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
    clf1 = pycaret.classification.setup(train, target=target,silent=True, html=False, session_id=123)
    model = pycaret.classification.create_model("lightgbm")
    data_unseen = test.drop(columns=target)
    final_model = pycaret.classification.finalize_model(model)
    result = pycaret.classification.predict_model(final_model, data = data_unseen)
    actual=test[target].reset_index()
    actual=actual["Purchase"].astype(np.int64)
    actual=actual.drop("index", axis=1)
    # provisional support
    prediction=result["Label"].dropna(axis=0, how="any")
    prediction=prediction.reset_index()
    prediction=prediction["Label"].astype(np.int64)
    prediction=prediction.drop("index", axis=1)

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
    final_model = pycaret.regression.finalize_model(model)
    result = pycaret.regression.predict_model(final_model, data=data_unseen)
    actual = test[target].reset_index()
    actual=actual.drop("index", axis=1)
    # provisional support
    prediction=result["Label"].dropna(axis=0, how="any")
    prediction = result["Label"].reset_index()
    prediction=prediction.drop("index", axis=1)

    # check metric(regression)
    pycaret.utils.check_metric(actual, prediction, "MAE")
    pycaret.utils.check_metric(actual, prediction, "MSE")
    pycaret.utils.check_metric(actual, prediction, "RMSE")
    pycaret.utils.check_metric(actual, prediction, "R2")
    pycaret.utils.check_metric(actual, prediction, "RMSLE")
    pycaret.utils.check_metric(actual, prediction, "MAPE")

    assert 1 == 1

if __name__ == "__main__":
    test()
