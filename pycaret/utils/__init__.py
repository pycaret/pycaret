import pandas as pd
import numpy as np
from typing import Optional
import functools

version_ = "3.0.0"
nightly_version_ = "3.0.0"

__version__ = version_


def version():
    return version_


def nightly_version():
    return nightly_version_


def check_metric(
    actual: pd.Series,
    prediction: pd.Series,
    metric: str,
    round: int = 4,
    train: Optional[pd.Series] = None,
):

    """
    Function to evaluate classification, regression and timeseries metrics.


    actual : pandas.Series
        Actual values of the target variable.


    prediction : pandas.Series
        Predicted values of the target variable.


    train: pandas.Series
        Train values of the target variable.


    metric : str
        Metric to use.


    round: integer, default = 4
        Number of decimal places the metrics will be rounded to.


    Returns:
        float

    """

    # general dependencies
    import pycaret.containers.metrics.classification
    import pycaret.containers.metrics.regression
    import pycaret.containers.metrics.time_series

    globals_dict = {"y": prediction}
    metric_containers = {
        **pycaret.containers.metrics.classification.get_all_metric_containers(
            globals_dict
        ),
        **pycaret.containers.metrics.regression.get_all_metric_containers(globals_dict),
        **pycaret.containers.metrics.time_series.get_all_metric_containers(
            globals_dict
        ),
    }
    metrics = {
        v.name: functools.partial(v.score_func, **(v.args or {}))
        for k, v in metric_containers.items()
    }

    if isinstance(train, pd.Series):
        input_params = [actual, prediction, train]
    else:
        input_params = [actual, prediction]

    # metric calculation starts here

    if metric in metrics:
        try:
            result = metrics[metric](*input_params)
        except:
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            actual = le.fit_transform(actual)
            prediction = le.transform(prediction)
            result = metrics[metric](actual, prediction)
        result = np.around(result, round)
        return float(result)
    else:
        raise ValueError(
            f"Couldn't find metric '{metric}' Possible metrics are: {', '.join(metrics.keys())}."
        )


def enable_colab():
    from IPython.display import display, HTML, clear_output, update_display

    """
    Function to render plotly visuals in colab.
    """

    def configure_plotly_browser_state():

        import IPython

        display(
            IPython.core.display.HTML(
                """
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
                },
              });
            </script>
            """
            )
        )

    import IPython

    IPython.get_ipython().events.register(
        "pre_run_cell", configure_plotly_browser_state
    )
    print("Colab mode enabled.")


def get_system_logs():

    """
    Read and print 'logs.log' file from current active directory
    """

    with open("logs.log", "r") as file:
        lines = file.read().splitlines()

    for line in lines:
        if not line:
            continue

        columns = [col.strip() for col in line.split(":") if col]
        print(columns)


def coerce_period_to_datetime_index(
    data: Union[pd.Series, pd.DataFrame, Any], freq: Optional[str] = None
) -> Union[pd.Series, pd.DataFrame, Any]:
    """Converts a dataframe or series index from PeriodIndex to DatetimeIndex

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame]
        The data with a PeriodIndex that needs to be converted to DatetimeIndex
    freq : Optional[str], optional
        The frequency to be used to convert the index, by default None which
        uses data.index.freq to perform the conversion

    Returns
    -------
    Union[pd.Series, pd.DataFrame, Any]
        The data with DatetimeIndex. Note: If input is not of type pd.Series or
        pd.DataFrame, then the data is simply returned back as is without change.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if freq is None:
            freq = data.index.freq

        if isinstance(data.index, pd.PeriodIndex):
            data.index = data.index.to_timestamp(freq=freq)

            #### Corner Case Handling ----
            # When data.index is of type Q-DEC with only 2 points, frequency
            # is not set after conversion (details below). However, this
            # works OK if there are more than 2 data points.
            # Before Conversion: PeriodIndex(['2018Q2', '2018Q3'], dtype='period[Q-DEC]', freq='Q-DEC')
            # After Conversion: DatetimeIndex(['2018-06-30', '2018-09-30'], dtype='datetime64[ns]', freq=None)
            # Hence, setting it manually if the frequency is not set after conversion.
            if data.index.freq is None:
                data.index.freq = original_freq

    return data


def coerce_datetime_to_period_index(
    data: Union[pd.Series, pd.DataFrame, Any], freq: Optional[str] = None
) -> Union[pd.Series, pd.DataFrame, Any]:
    """Converts a dataframe or series index from DatetimeIndex to PeriodIndex

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame]
        The data with a DatetimeIndex that needs to be converted to PeriodIndex
    freq : Optional[str], optional
        The frequency to be used to convert the index, by default None which
        uses data.index.freq to perform the conversion

    Returns
    -------
    Union[pd.Series, pd.DataFrame, Any]
        The data with PeriodIndex. Note: If input is not of type pd.Series or
        pd.DataFrame, then the data is simply returned back as is without change.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if freq is None:
            freq = data.index.freq

        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.to_period(freq=freq)

    return data
