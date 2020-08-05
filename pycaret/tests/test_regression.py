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
    top3 = pycaret.regression.compare_models(n_select = 3, blacklist = ['catboost'])

    # tune model
    tuned_top3 = [pycaret.regression.tune_model(i, n_iter=3) for i in top3]

    # ensemble model
    bagged_top3 = [pycaret.regression.ensemble_model(i) for i in tuned_top3]

    # blend models
    blender = pycaret.regression.blend_models(top3)

    # stack models
    stacker = pycaret.regression.stack_models(estimator_list = top3[1:], meta_model = top3[0])

    # compare created models
    pycaret.regression.compare_models(estimator_list=top3+tuned_top3+bagged_top3+[blender])

    # get config
    X_train = pycaret.regression.get_config('X_train')
    X_test = pycaret.regression.get_config('X_test')
    y_train = pycaret.regression.get_config('y_train')
    y_test = pycaret.regression.get_config('y_test')

    assert 1 == 1
    
if __name__ == "__main__":
    test()
