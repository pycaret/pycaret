# tests/test_time_series_eda.py
import numpy as np
import pandas as pd

# Use a headless backend so CI doesn't need a display
import matplotlib
matplotlib.use("Agg")

from pycaret.time_series import eda_ts_plot


def test_eda_ts_plot_series():
    """Smoke test for Series input; no plots should be shown."""
    s = pd.Series(np.random.randn(40))
    res = eda_ts_plot(s, show=False)
    assert isinstance(res, dict)
    assert "adf_pvalue" in res
    assert "adf_stationary" in res


def test_eda_ts_plot_dataframe():
    """Smoke test for DataFrame + target input."""
    df = pd.DataFrame({"sales": np.random.randn(40)})
    res = eda_ts_plot(df, target="sales", show=False)
    assert "length" in res and res["length"] == 40
    assert "adf_pvalue" in res


def test_eda_ts_plot_with_period_path():
    """Ensure the decomposition path runs when period is provided."""
    t = np.arange(0, 60)
    s = pd.Series(2 * np.sin(t / 6) + np.random.randn(60) * 0.1)
    res = eda_ts_plot(s, show=False, period=12)
    assert res["length"] == 60
