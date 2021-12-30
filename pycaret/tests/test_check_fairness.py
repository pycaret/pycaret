import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import pytest
import pycaret.classification
import pycaret.regression
import pycaret.datasets


def test_check_fairness_classification():

    # loading dataset
    data = pycaret.datasets.get_data("income")

    # initialize setup
    clf1 = pycaret.classification.setup(
        data,
        target="income >50K",
        silent=True,
        html=False,
        n_jobs=1,
    )

    # train model
    lightgbm = pycaret.classification.create_model("lightgbm", fold = 3)

    # check fairness
    lightgbm_fairness = pycaret.classification.check_fairness(lightgbm, ['sex'])
    assert isinstance(lightgbm_fairness, pd.DataFrame)


def test_check_fairness_regression():

    # loading dataset
    data = pycaret.datasets.get_data("boston")

    # initialize setup
    reg1 = pycaret.regression.setup(
        data,
        target="medv",
        silent=True,
        html=False,
        n_jobs=1,
    )

    # train model
    lightgbm = pycaret.regression.create_model("lightgbm", fold = 3)

    # check fairness
    lightgbm_fairness = pycaret.regression.check_fairness(lightgbm, ['chas'])
    assert isinstance(lightgbm_fairness, pd.DataFrame)

if __name__ == "__main__":
    test_check_fairness_classification()
    test_check_fairness_regression()