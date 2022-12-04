"""Module to benchmark auto detection of time series seasonal period
"""
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment


def test_benchmark_sp_to_use_using_auto():
    """Benchmark auto detection of seasonal periods. Any future changes must
    beat this benchmark."""

    properties = get_data("index", folder="time_series/seasonal", verbose=False)
    properties[["index", "s"]]
    detected_sp = []

    exp = TSForecastingExperiment()
    for _, (index, _) in enumerate(properties[["index", "s"]].values):
        y = get_data(index, folder="time_series/seasonal", verbose=False)
        exp.setup(data=y, session_id=index)
        detected_sp.append(exp.primary_sp_to_use)
    properties["detected_sp"] = detected_sp
    properties["equal"] = properties["s"] == properties["detected_sp"]
    properties["multiple"] = properties["detected_sp"] % properties["s"] == 0

    per_correct = len(properties.query("equal == True")) / len(properties)
    per_correct_multiple = len(properties.query("multiple == True")) / len(properties)

    # Current benchmark to beat ----
    assert per_correct > 0.9211  # 0.8076
    assert per_correct_multiple > 0.9307  # 0.8173
