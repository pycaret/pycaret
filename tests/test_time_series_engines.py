"""Module to test setting of engines in time series
"""

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sktime.forecasting.arima import AutoARIMA as PmdAutoARIMA
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA

from pycaret.time_series import TSForecastingExperiment

##############################
# Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.


############################
# Functions End Here ####
############################


##########################
# Tests Start Here ####
##########################


def test_engines_setup_global_args(load_pos_and_neg_data):
    """Tests the setting of engines using global arguments in setup.
    We test for both statistical models and regression models.
    """

    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fold=2,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
        engine={"auto_arima": "statsforecast", "lr_cds_dt": "sklearnex"},
    )

    # Default Model Engine ----
    # A. Statistical Models
    assert exp.get_engine("auto_arima") == "statsforecast"
    model = exp.create_model("auto_arima", cross_validation=False)
    assert isinstance(model, StatsForecastAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "statsforecast"

    # B. Regression Models
    assert exp.get_engine("lr_cds_dt") == "sklearnex"
    model = exp.create_model("lr_cds_dt", cross_validation=False)
    parent_library = model.regressor.__module__
    assert parent_library.startswith("sklearnex") or parent_library.startswith(
        "daal4py"
    )
    # Original engine should remain the same
    assert exp.get_engine("lr_cds_dt") == "sklearnex"


def test_engines_global_methods(load_pos_and_neg_data):
    """Tests the setting of engines using methods like set_engine (global changes).
    We test for both statistical models and regression models.
    """

    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fold=2,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
        engine={"auto_arima": "statsforecast", "lr_cds_dt": "sklearnex"},
    )

    # Globally reset engine ----
    # A. Statistical Models
    assert exp.get_engine("auto_arima") == "statsforecast"
    exp._set_engine("auto_arima", "pmdarima")
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.create_model("auto_arima", cross_validation=False)
    assert isinstance(model, PmdAutoARIMA)

    # B. Regression Models
    assert exp.get_engine("lr_cds_dt") == "sklearnex"
    exp._set_engine("lr_cds_dt", "sklearn")
    assert exp.get_engine("lr_cds_dt") == "sklearn"
    model = exp.create_model("lr_cds_dt", cross_validation=False)
    assert isinstance(model.regressor, SklearnLinearRegression)


def test_create_model_engines_local_args(load_pos_and_neg_data):
    """Tests the setting of engines for create_model using local args.
    We test for both statistical models and regression models.
    """

    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fold=2,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    # Default Model Engine ----
    # A. Statistical Models
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.create_model("auto_arima", cross_validation=False)
    assert isinstance(model, PmdAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "pmdarima"

    # B. Regression Models
    assert exp.get_engine("lr_cds_dt") == "sklearn"
    model = exp.create_model("lr_cds_dt", cross_validation=False)
    assert isinstance(model.regressor, SklearnLinearRegression)
    # Original engine should remain the same
    assert exp.get_engine("lr_cds_dt") == "sklearn"

    # Override model engine locally ----
    # A. Statistical Models
    model = exp.create_model(
        "auto_arima", engine="statsforecast", cross_validation=False
    )
    assert isinstance(model, StatsForecastAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.create_model("auto_arima")
    assert isinstance(model, PmdAutoARIMA)

    # B. Regression Models
    model = exp.create_model("lr_cds_dt", engine="sklearnex", cross_validation=False)
    parent_library = model.regressor.__module__
    assert parent_library.startswith("sklearnex") or parent_library.startswith(
        "daal4py"
    )
    # Original engine should remain the same
    assert exp.get_engine("lr_cds_dt") == "sklearn"
    model = exp.create_model("lr_cds_dt")
    assert isinstance(model.regressor, SklearnLinearRegression)


def test_compare_models_engines_local_args(load_pos_and_neg_data):
    """Tests the setting of engines for compare_models using local args.
    We test for both statistical models and regression models.
    """

    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fold=2,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    # Default Model Engine ----
    # A. Statistical Models
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.compare_models(include=["auto_arima"])
    assert isinstance(model, PmdAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "pmdarima"

    # B. Regression Models
    assert exp.get_engine("lr_cds_dt") == "sklearn"
    model = exp.compare_models(include=["lr_cds_dt"])
    assert isinstance(model.regressor, SklearnLinearRegression)
    # Original engine should remain the same
    assert exp.get_engine("lr_cds_dt") == "sklearn"

    # Override model engine locally ----
    # A. Statistical Models
    model = exp.compare_models(
        include=["auto_arima"], engine={"auto_arima": "statsforecast"}
    )
    assert isinstance(model, StatsForecastAutoARIMA)
    # Original engine should remain the same
    assert exp.get_engine("auto_arima") == "pmdarima"
    model = exp.compare_models(include=["auto_arima"])
    assert isinstance(model, PmdAutoARIMA)

    # B. Regression Models
    model = exp.compare_models(include=["lr_cds_dt"], engine={"lr_cds_dt": "sklearnex"})
    parent_library = model.regressor.__module__
    assert parent_library.startswith("sklearnex") or parent_library.startswith(
        "daal4py"
    )
    # Original engine should remain the same
    assert exp.get_engine("lr_cds_dt") == "sklearn"
    model = exp.compare_models(include=["lr_cds_dt"])
    assert isinstance(model.regressor, SklearnLinearRegression)
