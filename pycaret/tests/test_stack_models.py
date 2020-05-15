import os, sys
sys.path.insert(0, os.path.abspath(".."))

#stack_models_test
import pytest
import pycaret.regression
import pycaret.datasets

def test_stack_models():
    data = pycaret.datasets.get_data('boston')
    data = data.head(50)
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, html=False, session_id=123)
    estimator_list = pycaret.regression.compare_models(blacklist = ['catboost', 'tr'], n_select=3, verbose=False) #select top 3
    stacker = pycaret.regression.stack_models(estimator_list=estimator_list, choose_better=False, verbose=False)
    assert type(stacker) == lists