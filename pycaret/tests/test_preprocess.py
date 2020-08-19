import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.datasets
import pycaret.preprocess

def test():
    # loading dataset
    data = pycaret.datasets.get_data("juice")
    target = "Purchase"

    # preprocess all in one
    X = pycaret.preprocess.Preprocess_Path_One(train_data=data, target_variable=target, display_types=False)
    assert isinstance(X, pd.core.frame.DataFrame)

    # preprocess all in one unsupervised
    X = pycaret.preprocess.Preprocess_Path_Two(train_data=data, display_types=False)
    assert isinstance(X, pd.core.frame.DataFrame)

    assert 1 == 1
    
if __name__ == "__main__":
    test()
