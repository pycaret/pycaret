import os, sys
sys.path.insert(0, os.path.abspath(".."))

#compare_models_test
import pytest
import pycaret.regression
import pycaret.datasets

def test():
    # loading dataset
    data = pycaret.datasets.get_data('boston')

    # init setup
    reg1 = pycaret.regression.setup(data, target='medv',silent=True, html=False, session_id=123)

    # compare models
    top5 = pycaret.regression.compare_models(n_select = 5)

    # tune model
    tuned_top5 = [pycaret.regression.tune_model(i) for i in top5]

    # ensemble model
    bagged_top5 = [pycaret.regression.ensemble_model(i) for i in tuned_top5]

    # blend models
    blender = pycaret.regression.blend_models()

    # stack models
    stacker = pycaret.regression.stack_models(estimator_list = top5[1:], meta_model = top5[0])

    # automl
    best = pycaret.regression.automl()
    best_holdout = pycaret.regression.automl(use_holdout = True)
    
    # get config
    X_train = pycaret.regression.get_config('X_train')
    X_test = pycaret.regression.get_config('X_test')
    y_train = pycaret.regression.get_config('y_train')
    y_test = pycaret.regression.get_config('y_test')

    assert 1 == 1
    
if __name__ == "__main__":
    test()
