import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets


def test():

    # loading dataset
    data = pycaret.datasets.get_data("blood")
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        create_clusters=True,
        cluster_iter=10,
        n_jobs=1,
    )
    assert len(pycaret.classification.get_config("X").columns) > len(data.columns)


if __name__ == "__main__":
    test()
