"""Module to test time_series functionality
"""
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest
from time_series_test_utils import _return_model_names

from pycaret.time_series import TSForecastingExperiment

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.tuning_grid,
]

##############################
# Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

_model_names = _return_model_names()


############################
# Functions End Here ####
############################


##########################
# Tests Start Here ####
##########################


@pytest.mark.parametrize("model", _model_names)
def test_tune_model_grid(model, load_pos_and_neg_data):
    exp = TSForecastingExperiment()
    fh = 12
    fold = 2
    data = load_pos_and_neg_data

    exp.setup(data=data, fold=fold, fh=fh, fold_strategy="sliding")

    model_obj = exp.create_model(model)
    tuned_model_obj = exp.tune_model(model_obj, search_algorithm="grid")
    y_pred = exp.predict_model(tuned_model_obj)
    assert isinstance(y_pred, pd.DataFrame)

    expected_period_index = data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)
