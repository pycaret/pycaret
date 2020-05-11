import os, sys
sys.path.insert(0, os.path.abspath(".."))

#create_stacknet_test
import pytest
import pycaret.regression
import pycaret.datasets

def test_create_stacknet():
    data = pycaret.datasets.get_data('boston')
    data = data.head(50)
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, html=False, session_id=123)
    estimator_list = pycaret.regression.compare_models(blacklist = ['catboost', 'tr'], n_select=6, verbose=False) #select top 3
    layer1 = estimator_list[:3]
    layer2 = estimator_list[3:]
    stacker = pycaret.regression.create_stacknet(estimator_list=[layer1,layer2], improve_only=False, verbose=False)
    assert type(stacker) == list