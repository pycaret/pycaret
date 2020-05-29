import os, sys
sys.path.insert(0, os.path.abspath(".."))

#create_model_test
import pytest
import pycaret.regression
import pycaret.datasets

available_regressors = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 
                            'mlp', 'xgboost', 'lightgbm'] #excluded catboost to speedup training


available_classifiers = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 
                            'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']


def test_create_model_reg():
    data = pycaret.datasets.get_data('boston')
    data = data.head(50)
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, verbose=False, html=False, session_id=123)
    trained_models = []
    for i in available_regressors:
        c = pycaret.regression.create_model(i, verbose=False)
        trained_models.append(c)
    assert len(trained_models) == len(available_regressors)

def test_create_model_clf():
    data = pycaret.datasets.get_data('juice')
    data = data.head(100)
    clf1 = pycaret.classification.setup(data, target='Purchase',silent=True, verbose=False, html=False, session_id=786)
    trained_models = []
    for i in available_classifiers:
        c = pycaret.classification.create_model(i, verbose=False)
        trained_models.append(c)
    assert len(trained_models) == len(available_classifiers)