import os, sys
sys.path.insert(0, os.path.abspath(".."))

#tune_model_test
import pytest
import pycaret.regression
import pycaret.datasets

def test_model_tuning_r2():
    data = pycaret.datasets.get_data('boston')
    data = data.head(50)
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, verbose=False, html=False, session_id=123)
    model = pycaret.regression.tune_model(pycaret.regression.create_model('rf', verbose = False), verbose=False, optimize = 'R2')
    assert hasattr(model, 'predict')

def test_model_tuning_mae():
    data = pycaret.datasets.get_data('boston')
    data = data.head(50)
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, verbose=False, html=False, session_id=123)
    model = pycaret.regression.tune_model(pycaret.regression.create_model('ada', verbose = False), verbose=False, optimize = 'MAE')
    assert hasattr(model, 'predict')

def test_model_tuning_mse():
    data = pycaret.datasets.get_data('boston')
    data = data.head(50)
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, verbose=False, html=False, session_id=123)
    model = pycaret.regression.tune_model(pycaret.regression.create_model('lr', verbose = False), verbose=False, optimize = 'MSE')
    assert hasattr(model, 'predict')


