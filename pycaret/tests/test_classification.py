import os, sys
sys.path.insert(0, os.path.abspath(".."))

#compare_models_test
import pytest
import pycaret.classification
import pycaret.datasets

def test():
    # loading dataset
    data = pycaret.datasets.get_data('juice')

    # init setup
    clf1 = pycaret.classification.setup(data, target='Purchase',silent=True, html=False, session_id=123)

    # compare models
    top5 = pycaret.classification.compare_models(n_select = 5, blacklist=['catboost'])

    # tune model
    tuned_top5 = [pycaret.classification.tune_model(i) for i in top5]

    # ensemble model
    bagged_top5 = [pycaret.classification.ensemble_model(i) for i in tuned_top5]

    # blend models
    blender = pycaret.classification.blend_models()

    # stack models
    stacker = pycaret.classification.stack_models(estimator_list = top5[1:], meta_model = top5[0])

    # automl
    best = pycaret.classification.automl()
    best_holdout = pycaret.classification.automl(use_holdout = True)
    
    # get config
    X_train = pycaret.classification.get_config('X_train')
    X_test = pycaret.classification.get_config('X_test')
    y_train = pycaret.classification.get_config('y_train')
    y_test = pycaret.classification.get_config('y_test')

if __name__ == "__main__":
    test()
