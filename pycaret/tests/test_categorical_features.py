import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets


def test():

    # loading dataset
    data = pycaret.datasets.get_data("blood")
    print(data)

    # categorical_feature = Recency
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        categorical_features=["Recency"],
        n_jobs=1,
    )
    # X_train = pycaret.classification.get_config('X_train')
    # assert len([x for x in X_train.columns if 'Recency' in x]) == data.Recency.nunique()


if __name__ == "__main__":
    test()
