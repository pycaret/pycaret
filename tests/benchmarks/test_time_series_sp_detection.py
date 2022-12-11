"""Module to benchmark auto detection of time series seasonal period
"""
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment


def test_benchmark_sp_to_use_using_auto():
    """Benchmark auto detection of seasonal periods. Any future changes must
    beat this benchmark."""

    properties = get_data("index", folder="time_series/seasonal", verbose=False)
    properties[["index", "s"]]
    candidate_sps = []

    sig_sps = []
    all_sps = []
    primary_sp = []

    sig_sps_no_harmonics = []
    all_sp_no_harmonics = []
    primary_sp_no_harmonics = []

    exp = TSForecastingExperiment()
    for _, (index, _) in enumerate(properties[["index", "s"]].values):
        y = get_data(index, folder="time_series/seasonal", verbose=False)
        exp.setup(data=y, session_id=index)
        candidate_sps.append(exp.candidate_sps)

        sig_sps.append(exp.significant_sps)
        all_sps.append(exp.all_sps_to_use)
        primary_sp.append(exp.primary_sp_to_use)

        sig_sps_no_harmonics.append(exp.significant_sps_no_harmonics)
        all_sp_no_harmonics.append(
            exp.significant_sps_no_harmonics[0 : len(exp.all_sps_to_use)]
        )
        primary_sp_no_harmonics.append(exp.significant_sps_no_harmonics[0])

    properties["candidate_sps"] = candidate_sps

    # 1.0 Test with harmonics included ----

    properties["sig_sps"] = sig_sps
    properties["all_sps"] = all_sps
    properties["primary_sp"] = primary_sp

    properties["equal"] = properties["s"] == properties["primary_sp"]
    properties["multiple"] = properties["primary_sp"] % properties["s"] == 0

    per_correct = len(properties.query("equal == True")) / len(properties)
    per_correct_multiple = len(properties.query("multiple == True")) / len(properties)

    # Current benchmark to beat ----
    assert per_correct > 0.9211
    assert per_correct_multiple > 0.9307

    # 2.0 Test with harmonics excluded ----

    properties["sig_sps_no_harmonics"] = sig_sps_no_harmonics
    properties["all_sp_no_harmonics"] = all_sp_no_harmonics
    properties["primary_sp_no_harmonics"] = primary_sp_no_harmonics

    properties["equal_no_harmonics"] = (
        properties["s"] == properties["primary_sp_no_harmonics"]
    )
    properties["multiple_no_harmonics"] = (
        properties["primary_sp_no_harmonics"] % properties["s"] == 0
    )

    per_correct_no_harmonics = len(
        properties.query("equal_no_harmonics == True")
    ) / len(properties)
    per_correct_multiple_no_harmonics = len(
        properties.query("multiple_no_harmonics == True")
    ) / len(properties)

    # Current benchmark to beat ----
    assert per_correct_no_harmonics > 0.1192
    assert per_correct_multiple_no_harmonics > 0.8153
