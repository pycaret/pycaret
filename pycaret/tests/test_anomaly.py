import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.anomaly
import pycaret.datasets


def test():
    # loading dataset
    data = pycaret.datasets.get_data("anomaly")
    assert isinstance(data, pd.core.frame.DataFrame)

    # init setup
    ano1 = pycaret.anomaly.setup(
        data,
        normalize=True,
        log_experiment=True,
        silent=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # create model
    iforest = pycaret.anomaly.create_model("iforest")
    knn = pycaret.anomaly.create_model("knn")

    # assign model
    iforest_results = pycaret.anomaly.assign_model(iforest)
    knn_results = pycaret.anomaly.assign_model(knn)
    assert isinstance(iforest_results, pd.core.frame.DataFrame)
    assert isinstance(knn_results, pd.core.frame.DataFrame)

    # predict model
    iforest_predictions = pycaret.anomaly.predict_model(model=iforest, data=data)
    knn_predictions = pycaret.anomaly.predict_model(model=knn, data=data)
    assert isinstance(iforest_predictions, pd.core.frame.DataFrame)
    assert isinstance(knn_predictions, pd.core.frame.DataFrame)

    # get config
    X = pycaret.anomaly.get_config("X")
    seed = pycaret.anomaly.get_config("seed")
    assert isinstance(X, pd.core.frame.DataFrame)
    assert isinstance(seed, int)

    # set config
    pycaret.anomaly.set_config("seed", 124)
    seed = pycaret.anomaly.get_config("seed")
    assert seed == 124

    # save model
    pycaret.anomaly.save_model(knn, "knn_model_23122019")

    # load model
    saved_knn = pycaret.anomaly.load_model("knn_model_23122019")

    # returns table of models
    all_models = pycaret.anomaly.models()
    assert isinstance(all_models, pd.core.frame.DataFrame)

    assert 1 == 1


if __name__ == "__main__":
    test()
