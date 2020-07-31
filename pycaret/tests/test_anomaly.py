import os, sys
sys.path.insert(0, os.path.abspath(".."))

#compare_models_test
import pytest
import pycaret.anomaly
import pycaret.datasets

def test():
    # loading dataset
    data = pycaret.datasets.get_data('boston')

    # init setup
    ano1 = pycaret.anomaly.setup(data, normalize = True, silent=True, html=False, session_id=123)

    # create model
    iforest = pycaret.anomaly.create_model('iforest')
    knn = pycaret.anomaly.create_model('knn')

    # assign model
    iforest_results = pycaret.anomaly.assign_model(iforest)
    knn_results = pycaret.anomaly.assign_model(knn)

    # tune model
    tuned_model = pycaret.anomaly.tune_model('iforest', supervised_target = 'medv', estimator = 'xgboost')

    # get config
    X = pycaret.anomaly.get_config('X')
    seed = pycaret.anomaly.get_config('seed')

if __name__ == "__main__":
    test()
