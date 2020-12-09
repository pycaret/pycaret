import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets


def test():

    # loading dataset
    data = pycaret.datasets.get_data("cancer")

    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        remove_outliers=True,
        outliers_threshold=0.05,
        n_jobs=1,
    )
    assert (
        pd.concat(
            [
                pycaret.classification.get_config("X_train"),
                pycaret.classification.get_config("X_test"),
            ]
        ).shape[0]
        < data.shape[0]
    )


if __name__ == "__main__":
    test()
