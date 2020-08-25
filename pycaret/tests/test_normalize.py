import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import pytest
import pycaret.classification
import pycaret.datasets

def test():
    
    # loading dataset
    data = pycaret.datasets.get_data('blood')

    # normalize_method = 'minmax'
    clf1 = pycaret.classification.setup(data, target = 'Class', silent = True, html = False, normalize = True, normalize_method = 'minmax')
    assert np.array(pycaret.classification.get_config('X_train')).min() == 0
    assert np.array(pycaret.classification.get_config('X_train')).max() == 1

    # normalize_method = 'maxabs'
    clf1 = pycaret.classification.setup(data, target = 'Class', silent = True, html = False, normalize = True, normalize_method = 'maxabs')
    assert np.array(pycaret.classification.get_config('X_train')).min() == 0
    assert np.array(pycaret.classification.get_config('X_train')).max() == 1

    # normalize_method = 'zscore'
    clf1 = pycaret.classification.setup(data, target = 'Class', silent = True, html = False, normalize = True, normalize_method = 'zscore')
    assert np.array(pycaret.classification.get_config('X_train')).min() > -10
    assert np.array(pycaret.classification.get_config('X_train')).max() < 10

    # normalize_method = 'robust'
    clf1 = pycaret.classification.setup(data, target = 'Class', silent = True, html = False, normalize = True, normalize_method = 'robust')
    # to create assert later on

if __name__ == "__main__":
    test()