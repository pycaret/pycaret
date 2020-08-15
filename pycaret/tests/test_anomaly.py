import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pytest
import pycaret.anomaly
import pycaret.datasets

def test():
    # loading dataset
    data = pycaret.datasets.get_data('anomaly')

    # init setup
    ano1 = pycaret.anomaly.setup(data, normalize=True, silent=True, html=False, session_id=123)

    # create model
    iforest = pycaret.anomaly.create_model('iforest')
    knn = pycaret.anomaly.create_model('knn')

    # assign model
    iforest_results = pycaret.anomaly.assign_model(iforest)
    knn_results = pycaret.anomaly.assign_model(knn)
    
    # predict model
    iforest_predictions = pycaret.anomaly.predict_model(model=iforest, data=data)
    knn_predictions = pycaret.anomaly.predict_model(model=knn, data=data)

    # get config
    X = pycaret.anomaly.get_config('X')
    seed = pycaret.anomaly.get_config('seed')
    
    # set config
    pycaret.anomaly.set_config('seed', 123) 
    
    # save model
    pycaret.anomaly.save_model(knn, 'knn_model_23122019')
    
    # load model
    saved_knn = pycaret.anomaly.load_model('knn_model_23122019')
    
    # returns table of models
    all_models = pycaret.anomaly.models()

    assert 1 == 1

if __name__ == "__main__":
    test()
