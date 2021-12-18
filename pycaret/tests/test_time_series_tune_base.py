"""Module to test time_series "tune_model" BASE functionality
"""

import pytest
import numpy as np
import pandas as pd

from pycaret.internal.pycaret_experiment import TimeSeriesExperiment

from .time_series_test_utils import _ALL_METRICS


##########################
#### Tests Start Here ####
##########################

def test_tune_custom_grid_and_choose_better(load_pos_and_neg_data):
    """Tests
    (1) passing a custom grid to tune_model, and
    (2) choose_better=True
    """

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        verbose=False,
        session_id=42,
    )

    model = exp.create_model("naive")

    # Custom Grid
    only_strategy = "mean"
    custom_grid = {"strategy": [only_strategy]}

    # By default choose_better = True
    tuned_model1 = exp.tune_model(model, custom_grid=custom_grid)

    # Choose Better = False
    tuned_model2 = exp.tune_model(model, custom_grid=custom_grid, choose_better=False)

    # Same strategy should be chosen since choose_better = True by default
    assert tuned_model1.strategy == model.strategy
    # should pick only value in custom grid
    assert tuned_model2.strategy == only_strategy
    # tuned model does improve score (verified manually), and choose_better
    # set to False. So pick worse value itself.
    assert tuned_model2.strategy != model.strategy


def test_tune_model_custom_folds(load_pos_and_neg_data):
    """test custom folds in tune_model"""
    exp = TimeSeriesExperiment()
    setup_fold = 3
    exp.setup(
        data=load_pos_and_neg_data,
        fold=setup_fold,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    #######################################
    ## Test Tune Model with custom folds ##
    #######################################
    model = exp.create_model("naive")
    _ = exp.tune_model(model)
    metrics1 = exp.pull()

    custom_fold = 5
    _ = exp.tune_model(model, fold=5)
    metrics2 = exp.pull()

    assert len(metrics1) == setup_fold + 2  # + 2 for Mean and SD
    assert len(metrics2) == custom_fold + 2  # + 2 for Mean and SD


@pytest.mark.parametrize("metric", _ALL_METRICS)
def test_tune_model_alternate_metric(load_pos_and_neg_data, metric):
    """tests model selection using non default metric"""
    exp = TimeSeriesExperiment()
    fh = 12
    fold = 2

    exp.setup(data=load_pos_and_neg_data, fold=fold, fh=fh, fold_strategy="sliding")

    model_obj = exp.create_model("naive")

    tuned_model_obj = exp.tune_model(model_obj, optimize=metric)
    y_pred = exp.predict_model(tuned_model_obj)
    assert isinstance(y_pred, pd.Series)

    expected_period_index = load_pos_and_neg_data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)


def test_tune_model_raises(load_pos_and_neg_data):
    """Tests conditions that raise an error due to lack of data"""

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        verbose=False,
        session_id=42,
    )

    model = exp.create_model("naive")
    with pytest.raises(ValueError) as errmsg:
        search_algorithm = "wrong_algorithm"
        _ = exp.tune_model(model, search_algorithm=search_algorithm)

    exceptionmsg = errmsg.value.args[0]

    assert (
        exceptionmsg
        == f"`search_algorithm` must be one of 'None, random, grid'. You passed '{search_algorithm}'."
    )
