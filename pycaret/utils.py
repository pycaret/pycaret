# Module: Utility
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT

import pandas as pd

version_ = "2.2.3"
nightly_version_ = "2.2.3"

__version__ = version_


def version():
    return version_


def nightly_version():
    return nightly_version_


def check_metric(actual: pd.Series, prediction: pd.Series, metric: str, round: int = 4):

    """
    Function to evaluate classification and regression metrics.


    actual : pandas.Series
        Actual values of the target variable.


    prediction : pandas.Series
        Predicted values of the target variable.


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

    globals_dict = {"y": prediction}
    metric_containers = {
        **pycaret.containers.metrics.classification.get_all_metric_containers(
            globals_dict
        ),
        **pycaret.containers.metrics.regression.get_all_metric_containers(globals_dict),
    }
    metrics = {v.name: v.score_func for k, v in metric_containers.items()}

    # metric calculation starts here

    if metric in metrics:
        try:
            result = metrics[metric](actual, prediction)
        except:
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            actual = le.fit_transform(actual)
            prediction = le.transform(prediction)
            result = metrics[metric](actual, prediction)
        result = result.round(round)
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
