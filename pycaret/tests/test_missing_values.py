import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets


def test():

    # loading dataset
    data = pycaret.datasets.get_data("hepatitis")
    assert data.isnull().sum().sum() > 0

    # numeric imputation = mean, categorical_imputation = constant
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        numeric_imputation="mean",
        categorical_imputation="constant",
        n_jobs=1,
    )

    X_train = pycaret.classification.get_config("X_train")
    assert X_train.isnull().sum().sum() == 0

    X_test = pycaret.classification.get_config("X_test")
    assert X_test.isnull().sum().sum() == 0

    y_train = pycaret.classification.get_config("y_train")
    assert y_train.isnull().sum().sum() == 0

    y_test = pycaret.classification.get_config("y_test")
    assert y_test.isnull().sum().sum() == 0

    lr = pycaret.classification.create_model("lr")
    pycaret.classification.predict_model(lr, data=data)

    # numeric imputation = median, categorical_imputation = mode
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        numeric_imputation="median",
        categorical_imputation="mode",
        n_jobs=1,
    )

    X_train = pycaret.classification.get_config("X_train")
    assert X_train.isnull().sum().sum() == 0

    X_test = pycaret.classification.get_config("X_test")
    assert X_test.isnull().sum().sum() == 0

    y_train = pycaret.classification.get_config("y_train")
    assert y_train.isnull().sum().sum() == 0

    y_test = pycaret.classification.get_config("y_test")
    assert y_test.isnull().sum().sum() == 0

    lr = pycaret.classification.create_model("lr")
    pycaret.classification.predict_model(lr, data=data)

    # numeric imputation = zero
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        numeric_imputation="zero",
        n_jobs=1,
    )

    X_train = pycaret.classification.get_config("X_train")
    assert X_train.isnull().sum().sum() == 0

    X_test = pycaret.classification.get_config("X_test")
    assert X_test.isnull().sum().sum() == 0

    y_train = pycaret.classification.get_config("y_train")
    assert y_train.isnull().sum().sum() == 0

    y_test = pycaret.classification.get_config("y_test")
    assert y_test.isnull().sum().sum() == 0

    lr = pycaret.classification.create_model("lr")
    pycaret.classification.predict_model(lr, data=data)


if __name__ == "__main__":
    test()
