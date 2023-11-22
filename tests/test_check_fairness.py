import pandas as pd

import pycaret.classification
import pycaret.datasets
import pycaret.regression


def test_check_fairness_binary_classification():
    # loading dataset
    data = pycaret.datasets.get_data("income")

    # initialize setup
    pycaret.classification.setup(
        data,
        target="income >50K",
        html=False,
        n_jobs=1,
    )

    # train model
    lightgbm = pycaret.classification.create_model("lightgbm", fold=3)

    # check fairness
    lightgbm_fairness = pycaret.classification.check_fairness(lightgbm, ["sex"])
    assert isinstance(lightgbm_fairness, pd.DataFrame)


def test_check_fairness_multiclass_classification():
    # loading dataset
    data = pycaret.datasets.get_data("iris")

    # initialize setup
    pycaret.classification.setup(
        data,
        target="species",
        html=False,
        n_jobs=1,
        train_size=0.8,
    )

    # train model
    lightgbm = pycaret.classification.create_model("lightgbm", cross_validation=False)

    # check fairness
    lightgbm_fairness = pycaret.classification.check_fairness(
        lightgbm, ["sepal_length"]
    )
    assert isinstance(lightgbm_fairness, pd.DataFrame)


def test_check_fairness_regression():
    # loading dataset
    data = pycaret.datasets.get_data("boston")

    # initialize setup
    pycaret.regression.setup(
        data,
        target="medv",
        html=False,
        n_jobs=1,
    )

    # train model
    lightgbm = pycaret.regression.create_model("lightgbm", fold=3)

    # check fairness
    lightgbm_fairness = pycaret.regression.check_fairness(lightgbm, ["chas"])
    assert isinstance(lightgbm_fairness, pd.DataFrame)


if __name__ == "__main__":
    test_check_fairness_binary_classification()
    test_check_fairness_multiclass_classification()
    test_check_fairness_regression()
