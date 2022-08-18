"""Module to test setting of engines in time series
"""
from sktime.forecasting.arima import AutoARIMA as PmdAutoARIMA
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA

from pycaret.time_series import TSForecastingExperiment

##############################
#### Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.


############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


def test_engines_setup_global_args(load_pos_and_neg_data):
    """Tests the setting of engines using global arguments in setup."""

    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fold=2,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
        engines={"auto_arima": "statsforecast"},
    )

    #### Default Model Engine ----
    assert exp.get_engine("auto_arima") == "statsforecast"
    model = exp.create_model("auto_arima", cross_validation=False)
    assert isinstance(model, StatsForecastAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "statsforecast"


def test_engines_global_methods(load_pos_and_neg_data):
    """Tests the setting of engines using methods like set_engine (global changes)."""

    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fold=2,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
        engines={"auto_arima": "statsforecast"},
    )
    assert exp.get_engine("auto_arima") == "statsforecast"

    #### Globally reset engine ----
    exp._set_engine("auto_arima", "pmdarima")
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.create_model("auto_arima", cross_validation=False)
    assert isinstance(model, PmdAutoARIMA)


def test_create_model_engines_local_args(load_pos_and_neg_data):
    """Tests the setting of engines for create_model using local args."""

    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fold=2,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    #### Default Model Engine ----
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.create_model("auto_arima", cross_validation=False)
    assert isinstance(model, PmdAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "pmdarima"

    #### Override model engine locally ----
    model = exp.create_model(
        "auto_arima", engine="statsforecast", cross_validation=False
    )
    assert isinstance(model, StatsForecastAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.create_model("auto_arima")
    assert isinstance(model, PmdAutoARIMA)


def test_compare_models_engines_local_args(load_pos_and_neg_data):
    """Tests the setting of engines for compare_models using local args."""

    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fold=2,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    #### Default Model Engine ----
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.compare_models(include=["auto_arima"])
    assert isinstance(model, PmdAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "pmdarima"

    #### Override model engine locally ----
    model = exp.compare_models(
        include=["auto_arima"], engines={"auto_arima": "statsforecast"}
    )
    assert isinstance(model, StatsForecastAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.compare_models(include=["auto_arima"])
    assert isinstance(model, PmdAutoARIMA)
