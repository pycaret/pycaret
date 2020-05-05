import os, sys
sys.path.insert(0, os.path.abspath(".."))

#tune_model_test
import pytest
import pycaret.regression
import pycaret.datasets

available_regressors = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 
                            'mlp', 'xgboost', 'lightgbm', 'catboost']

def test_tune_model():
    data = pycaret.datasets.get_data('boston')
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, verbose=False, html=False, session_id=123)
    tuned_models = []
    for i in available_regressors:
        c = pycaret.regression.tune_model(estimator = pycaret.regression.create_model(i, verbose=False), verbose = False)
        tuned_models.append(c)
    assert len(tuned_models) == len(available_regressors)