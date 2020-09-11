import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets

def test():
    # loading dataset
    data = pycaret.datasets.get_data('juice')
    assert isinstance(data, pd.core.frame.DataFrame)

    # init setup
    clf1 = pycaret.classification.setup(data, target='Purchase', log_experiment=True, silent=True, html=False, session_id=123)
    
    models = [pycaret.classification.create_model('lr', fold=2), pycaret.classification.create_model('xgboost', fold=2), pycaret.classification.create_model('lightgbm', fold=2), pycaret.classification.create_model('catboost', fold=2)]

    models.append(pycaret.classification.stack_models(models))
    models.append(pycaret.classification.ensemble_model(models[0]))

    for model in models:
        print(f"Testing model {model}")
        pycaret.classification.tune_model(model, fold=2, search_library='scikit-learn', search_algorithm='random', early_stopping=False)
        pycaret.classification.tune_model(model, fold=2, search_library='scikit-optimize', search_algorithm='bayesian', early_stopping=False)
        pycaret.classification.tune_model(model, fold=2, search_library='scikit-optimize', search_algorithm='bayesian', early_stopping=False)
        pycaret.classification.tune_model(model, fold=2, search_library='optuna', search_algorithm='tpe', early_stopping=False)
        pycaret.classification.tune_model(model, fold=2, search_library='tune-sklearn', search_algorithm='random', early_stopping=False)
 
    # test early stopping (enabled by default)
    model = pycaret.classification.create_model('svm', fold=2)

    pycaret.classification.tune_model(model, fold=2, search_library='optuna', search_algorithm='tpe')
    pycaret.classification.tune_model(model, fold=2, search_library='tune-sklearn', search_algorithm='hyperopt')
    pycaret.classification.tune_model(model, fold=2, search_library='tune-sklearn', search_algorithm='bayesian')
    # bohb is broken in current ray[tune] release
    #pycaret.classification.tune_model(model, fold=2, search_library='tune-sklearn', search_algorithm='bohb')

    assert 1 == 1
    
if __name__ == "__main__":
    test()
