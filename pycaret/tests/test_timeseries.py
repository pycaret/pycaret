import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pytest
import pandas as pd
import numpy as np
import pycaret.timeseries as ts
import pycaret.datasets as pcd


def test_timeseries():

    data = pcd.get_data('bike')
    data = data.tail(50)
    assert isinstance(data, pd.core.frame.DataFrame)

    # Test set up
    ts1 = ts.setup(data, target='cnt', silent=True, session_id=123, log_experiment=True)
    assert isinstance(ts1, tuple)
    assert isinstance(ts1[0], pd.core.frame.DataFrame)
    assert isinstance(ts1[1], pd.core.series.Series)
    assert isinstance(ts1[2], pd.core.frame.DataFrame)
    assert isinstance(ts1[3], pd.core.frame.DataFrame)
    assert isinstance(ts1[4], pd.core.series.Series)
    assert isinstance(ts1[5], pd.core.series.Series)
    assert isinstance(ts1[6], int)
    assert isinstance(ts1[9], list)
    assert isinstance(ts1[10], list)
    assert isinstance(ts1[11], list)
    assert isinstance(ts1[12], list)
    assert isinstance(ts1[13], str)
    assert isinstance(ts1[14], bool)
    assert isinstance(ts1[15], bool)
    assert isinstance(ts1[16], str)

    # Test create model
    estimators = ['sem', 'holt', 'auto_arima']
    random_estimator = np.random.choice(estimators)

    model, model_results = ts.create_model(estimator=random_estimator)

    assert model.__class__, "Model does not has __class__ attribute"
    assert model_results.shape[0] > 0, "Results data from model is null"

    # Test auto select
    auto_model, auto_model_results = ts.auto_select(splits=5, metric='rmse', verbose=True)

    assert auto_model.__class__, "Model does not has __class__ attribute"
    assert auto_model_results.shape[0] > 0, "Results data from auto_model is null"

    # Test forecast
    steps = int(np.random.choice(list(range(6, 12))))
    pred = ts.forecast(auto_model, steps=steps, plot=False)

    assert len(pred) == steps, f"len(pred)={len(pred)}!={steps}=steps"

    # save model
    ts.save_model(auto_model, 'auto_model_23122019')

    # load model
    ts.load_model('auto_model_23122019')
    
if __name__ == "__main__":
    test_timeseries()
