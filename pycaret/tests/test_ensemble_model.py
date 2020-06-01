import os, sys
sys.path.insert(0, os.path.abspath(".."))

#ensemble_model_test
import pytest
import pycaret.regression
import pycaret.datasets

#Bagging
def test_model_bagging():
    data = pycaret.datasets.get_data('boston')
    data = data.head(50)
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, verbose=False, html=False, session_id=123)
    model = pycaret.regression.ensemble_model(pycaret.regression.create_model('dt', verbose = False), verbose=False, method = 'Bagging')
    assert hasattr(model, 'predict')

#Boosting
def test_model_boosting():
    data = pycaret.datasets.get_data('boston')
    data = data.head(50)
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, verbose=False, html=False, session_id=123)
    model = pycaret.regression.ensemble_model(pycaret.regression.create_model('et', verbose = False), verbose=False, method = 'Boosting')
    assert hasattr(model, 'predict')