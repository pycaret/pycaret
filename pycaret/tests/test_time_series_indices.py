import pandas as pd
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment


def create_models(exp: TSForecastingExperiment, prophet: bool = True):
    """Function to create a few trial models that support both univariate and
    multivariate forecasting.

    Parameters
    ----------
    exp : TSForecastingExperiment
        The Time Series experiment object
    prophet : bool, optional
        Should Prophet model be created, by default True
    """

    model = exp.create_model("arima")
    exp.predict_model(model)
    exp.plot_model(model, system=False)

    if "prophet" in exp.models().index:
        model = exp.create_model("prophet")
        exp.predict_model(model)
        exp.plot_model(model, system=False)


def test_time_series_indices():
    """Checks working with various types of indices with both univariate and multivariate datasets"""

    ####################
    #### Univariate ####
    ####################

    #### With Period Index ----
    exp = TSForecastingExperiment()
    data = get_data("airline")
    exp.setup(data=data, fh=12, fold=2, session_id=42)
    create_models(exp)

    #### With Datetime Index ----
    exp = TSForecastingExperiment()
    data = get_data("airline")
    data = data.to_timestamp()
    exp.setup(data=data, fh=12, fold=2, session_id=42)
    create_models(exp)

    #### With Int Index ----
    exp = TSForecastingExperiment()
    data = get_data("airline")
    data.reset_index(drop=True, inplace=True)
    exp.setup(data=data, fh=12, fold=2, seasonal_period=12, session_id=42)
    create_models(exp)

    #######################
    #### Multivariate  ####
    #######################

    # TODO: Find a better source of data ----
    data = pd.read_csv(
        "https://raw.githubusercontent.com/ngupta23/DS6373_TimeSeries/2b40f0071c3b7ec6a05dc0106f64e041f8cbaaef/Projects/gdp_prediction/data/economic_indicators_all_ex_3mo_china_inc_treas3mo.csv"
    )
    data["date"] = pd.to_datetime(data["date"].str.replace("\s", "-"))

    #### With Datetime Index Column ----
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        target="gdp_change",
        index="date",
        fh=2,
        fold=2,
        session_id=42,
    )
    create_models(exp)

    #### With Datetime Index ----
    data_temp = data.copy()
    data_temp.set_index("date", inplace=True)
    exp = TSForecastingExperiment()
    exp.setup(data=data_temp, target="gdp_change", fh=2, fold=2, session_id=42)
    create_models(exp)

    #### With Period Index ----
    data_temp = data.copy()
    data_temp.set_index("date", inplace=True)
    data_temp.index = data_temp.index.to_period()
    exp = TSForecastingExperiment()
    exp.setup(data=data_temp, target="gdp_change", fh=2, fold=2, session_id=42)
    create_models(exp)

    #### With Int Index ----
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        target="gdp_change",
        ignore_features=["date"],
        seasonal_period=4,
        fh=2,
        fold=2,
        session_id=42,
    )
    create_models(exp)
