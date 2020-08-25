import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets

def test():
    
    # loading dataset
    data = pycaret.datasets.get_data('blood')

    # numeric imputation = mean, categorical_imputation = constant
    clf1 = pycaret.classification.setup(data, target = 'Class', silent = True, html = False, categorical_features = ['Recency'])
    X_train = pycaret.classification.get_config('X_train')
    assert X_train.shape[1] == data.shape[1] - 1 - 1 + data.Recency.nunique()

if __name__ == "__main__":
    test()