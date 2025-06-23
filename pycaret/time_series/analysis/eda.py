import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def eda_report(data: pd.DataFrame, date_col: str, target_col: str):
    """
    Generates EDA visualizations for time series forecasting.

    Parameters:
    - data (pd.DataFrame): Time series dataframe
    - date_col (str): Name of the datetime column
    - target_col (str): Name of the target column to forecast
    """

    data = data.copy()
    data[date_col] = data[date_col].dt.to_timestamp()

    data.set_index(date_col, inplace=True)

    ts = data[target_col]

    # Line plot
    plt.figure(figsize=(10, 4))
    plt.plot(ts, label='Time Series')
    plt.title('Line Plot')
    plt.legend()
    plt.show()

    # Rolling mean and std
    plt.figure(figsize=(10, 4))
    plt.plot(ts, label='Original')
    plt.plot(ts.rolling(12).mean(), label='Rolling Mean')
    plt.plot(ts.rolling(12).std(), label='Rolling Std')
    plt.legend()
    plt.title('Rolling Statistics')
    plt.show()

    # ADF Test
    result = adfuller(ts)
    print("\n ADF Test Results:")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")

    # Decomposition
    decomposition = seasonal_decompose(ts, model='additive', period=12)
    decomposition.plot()
    plt.show()

    # ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(ts.dropna(), ax=axes[0])
    plot_pacf(ts.dropna(), ax=axes[1])
    plt.suptitle("ACF and PACF")
    plt.show()
