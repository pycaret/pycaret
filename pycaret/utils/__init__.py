import functools
from typing import Dict, Optional, Union, Any

import functools
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics._scorer import _PredictScorer, get_scorer  # type: ignore

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


def _get_metrics_dict(
    metrics_dict: Dict[str, Union[str, _PredictScorer]]
) -> Dict[str, _PredictScorer]:
    """Returns a metrics dictionary in which all values are callables
    of type _PredictScorer

    Parameters
    ----------
    metrics_dict : A metrics dictionary in which some values can be strings.
        If the value is a string, the corresponding callable metric is returned
        e.g. Dictionary Value of 'neg_mean_absolute_error' will return
        make_scorer(mean_absolute_error, greater_is_better=False)
    """
    return_metrics_dict = {}
    for k, v in metrics_dict.items():
        if isinstance(v, str):
            return_metrics_dict[k] = get_scorer(v)
        else:
            return_metrics_dict[k] = v
    return return_metrics_dict


def enable_colab():
    from IPython.display import HTML, clear_output, display, update_display

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


def _coerce_empty_dataframe_to_none(
    data: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Returns None if the data is an empty dataframe or None,
    else return the dataframe as is.

    Parameters
    ----------
    data : Optional[pd.DataFrame]
        Dataframe to be checked or None

    Returns
    -------
    Optional[pd.DataFrame]
        Returned Dataframe OR None (if dataframe is empty or None)
    """
    if isinstance(data, pd.DataFrame) and data.empty:
        return None
    else:
        return data


def _resolve_dict_keys(dict_: Dict[str, Any], key: str, defaults: Dict[str, Any]) -> Any:
    """Returns the value of "key" from `dict`. If key is not present, then the
    value is picked from the `defaults` dictionary. Note that `defaults` must
    contain the `key` else this will give an error.

    Parameters
    ----------
    dict : Dict[str, Any]
        The dictionary from which the "key"'s value must be obtained
    key : str
        The "key" whose value must be obtained
    defaults : Dict[str, Any]
        The dictionary containing the default value of the "key"

    Returns
    -------
    Any
        The value of the "key"

    Raises
    ------
    KeyError
        If the `defaults` dictionary does not contain the `key`
    """
    if key not in defaults:
        raise KeyError(f"Key '{key}' not present in Defaults dictionary.")
    return dict_.get(key, defaults[key])
