import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.datasets
import pycaret.preprocess


def test():
    index = pd.read_csv("datasets/index.csv")

    for _, row in index.iterrows():

        # loading dataset
        data = pycaret.datasets.get_data(row['Dataset'])

        if row['Target Variable 1']!='None':
            X = pycaret.preprocess.Preprocess_Path_One(train_data=data, target_variable=row['Target Variable 1'], display_types=False)
            assert isinstance(X, pd.core.frame.DataFrame)
        if row['Target Variable 2']!='None':
            X = pycaret.preprocess.Preprocess_Path_One(train_data=data, target_variable=row['Target Variable 2'], display_types=False)
            assert isinstance(X, pd.core.frame.DataFrame)

        # preprocess all in one unsupervised
        X = pycaret.preprocess.Preprocess_Path_Two(train_data=data, display_types=False)
        assert isinstance(X, pd.core.frame.DataFrame)

    assert 1 == 1


if __name__ == "__main__":
    test()
