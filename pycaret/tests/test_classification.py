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
    top3 = pycaret.classification.compare_models(n_select = 3, blacklist=['catboost'])

    # tune model
    tuned_top3 = [pycaret.classification.tune_model(i) for i in top3]

    # ensemble model
    bagged_top3 = [pycaret.classification.ensemble_model(i) for i in tuned_top3]

    # blend models
    blender = pycaret.classification.blend_models(top3)

    # stack models
    stacker = pycaret.classification.stack_models(estimator_list = top3)

    # select best model
    best = pycaret.classification.automl(optimize = 'MCC')
    
    # hold out predictions
    predict_holdout = pycaret.classification.predict_model(best)

    # predictions on new dataset
    predict_holdout = pycaret.classification.predict_model(best, data=data)

    # calibrate model
    calibrated_best = pycaret.classification.calibrate_model(best)

    # finalize model
    final_best = pycaret.classification.finalize_model(best)

    # save model
    pycaret.classification.save_model(best, 'best_model_23122019')
 
    # load model
    saved_best = pycaret.classification.load_model('best_model_23122019')
    
    # returns table of models
    all_models = pycaret.classification.models()
    
    # get config
    X_train = pycaret.classification.get_config('X_train')
    X_test = pycaret.classification.get_config('X_test')
    y_train = pycaret.classification.get_config('y_train')
    y_test = pycaret.classification.get_config('y_test')

    # set config
    pycaret.classification.set_config('seed', 123) 
    
    assert 1 == 1
    
if __name__ == "__main__":
    test()
