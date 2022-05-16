import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.regression
import pycaret.datasets


ALGORITHMS_LIST_CLASSIFICATION = {
    "knn": [],
    "svm": []
}

ALGORITHMS_LIST_REGRESSION = [
    "lr"
    "knn",
    "svm",
    "lasso",
    "ridge"
]


def test():

    juice_dataframe = pycaret.datasets.get_data("juice")
    boston_dataframe = pycaret.datasets.get_data("boston")

    # init setup
    clf1 = pycaret.classification.setup(
        juice_dataframe,
        target="Purchase",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        silent=True,
        html=False,
        session_id=123,
        n_jobs=1
    )

    # check classification algorithms
    for classification_algo in ALGORITHMS_LIST_CLASSIFICATION:
        algo = create_model(classification_algo)
        parent_library = algo.__module__
        assert parent_library.startswith("sklearn") == True

    reg1 = pycaret.regression.setup(
        boston_dataframe,
        target="medv",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        silent=True,
        log_experiment=True,
        html=False,
        session_id=124,
        n_jobs=1,
        experiment_name=uuid.uuid4().hex
    )

    # check regression algorithms
    for classification_algo in ALGORITHMS_LIST_REGRESSION:
        algo = create_model(classification_algo)
        parent_library = algo.__module__
        assert parent_library.startswith("sklearn") == True

    # TEST INTEL EXTENSION FOR SKLEARN

    clf1 = pycaret.classification.setup(
        juice_dataframe,
        target="Purchase",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        silent=True,
        html=False,
        session_id=123,
        n_jobs=1,
        use_intelex=True
    )

    # check classification algorithms
    for classification_algo in ALGORITHMS_LIST_CLASSIFICATION:
        algo = create_model(classification_algo)
        parent_library = algo.__module__
        assert parent_library.startswith("daal4py") == True

    reg1 = pycaret.regression.setup(
        boston_dataframe,
        target="medv",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        silent=True,
        log_experiment=True,
        html=False,
        session_id=124,
        n_jobs=1,
        experiment_name=uuid.uuid4().hex,
        use_intelex=True
    )

    # check regression algorithms
    for classification_algo in ALGORITHMS_LIST_REGRESSION:
        algo = create_model(classification_algo)
        parent_library = algo.__module__
        assert parent_library.startswith("daal4py") == True
