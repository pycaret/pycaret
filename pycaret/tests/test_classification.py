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
    clf1 = pycaret.classification.setup(data, target='Purchase',silent=True, feature_selection_method='boruta',
                                        html=False, session_id=123)
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

    # test webservice
    import importlib
    import json
    fapi = importlib.util.find_spec("fastapi")
    pyd = importlib.util.find_spec("pydantic")
    best = pycaret.classification.create_model('lr')
    if (fapi is not None) and (pyd is not None):
        # test service without token
        app = pycaret.classification.create_webservice(best, 'test_model', api_key=False)
        from fastapi.testclient import TestClient
        client = TestClient(app['Not_exist'])
        test_sample = json.loads(data.convert_dtypes().drop(columns=['Purchase']).loc[[0]].to_json(orient='records'))[0]
        response = client.post("predict/{}".format('test_model'),
                               json=test_sample)
        assert response.status_code == 200
        assert response.json()['prediction'] == '0'
        assert response.json()['input_data'] == test_sample
        # second test
        test_sample = json.loads(data.convert_dtypes().drop(columns=['Purchase']).loc[[3]].to_json(orient='records'))[0]
        response = client.post("predict/{}".format('test_model'),
                               json=test_sample)
        assert response.status_code == 200
        assert response.json()['prediction'] == '1'
        assert response.json()['input_data'] == test_sample
        # # test service with token
        key, app = pycaret.classification.create_webservice(best, 'test_model', api_key=True).popitem()
        client = TestClient(app)
        response = client.post("predict/{}".format('test_model'),
                               headers={"token": "WRONG_TOKEN"},
                               json=test_sample)
        assert response.status_code == 401
        response = client.post("predict/{}".format('test_model'),
                               headers={"token": key},
                               json=test_sample
                               )
        assert response.status_code == 200

    # set config
    pycaret.classification.set_config('seed', 123)

    assert 1 == 1
    
if __name__ == "__main__":
    test()
