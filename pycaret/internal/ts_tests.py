from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_trend(
    data: np.ndarray,
    trend_order: int = 1,
    **kwargs
):
    coeffs = np.polyfit(range(len(data)), data, trend_order)
    slope = coeffs[-2]
    return slope


def trend_and_seasonality_test(
    ts: Union[pd.DataFrame, pd.Series],
    seasonal_periods=(3, 5, 7, 12, 14, 30, 31),
    plots=False,
    **kwargs
):
    """

    Parameters
    ----------
    ts : Time series data
    seasonal_periods : Seasonal periods to test for seasonality
    plots : When set to True, returns the trend and seasonal plot for the given time series

    Returns
    -------
    Test result dict

    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Iterate through all the seasonal periods and best the seasonal period
    seasonal_values = {i: np.max(seasonal_decompose(ts, period=i).seasonal) for i in seasonal_periods}
    df = pd.DataFrame(index=seasonal_values.keys(), data=seasonal_values.values(), columns=['values'])

    is_seasonal = np.any(df[['values']] > 1)
    seasonal_period = df.idxmax()[0] if is_seasonal else None

    decompose_result = seasonal_decompose(ts, period=seasonal_period)
    t_values = decompose_result.trend.fillna(method='bfill').fillna(method='ffill')

    trend = find_trend(t_values)
    if plots:
        f, ax = plt.subplots(nrows=2, ncols=1)

        # trend plot
        ax[0].plot(ts.index, t_values)
        ax[0].plot(ts.index, trend * np.arange(1, len(t_values) + 1))
        ax[0].legend(['Trend from stl', 'Linear trend'])
        ax[0].set_title('Trend plot', size=12)

        # seasonal plot
        df.plot.bar(ax=ax[1], rot=0)
        plt.hlines(1, -1, len(df.index), linestyles='--', colors='black', alpha=0.5)
        ax[1].legend(['Significance level', 'Seasonal value'])
        ax[1].set_title('Seasonal plot', size=12)

        f.tight_layout()

    if abs(trend) >= 1:
        trend_nature = f"Strong - {'increasing' if trend > 0 else 'decreasing'} trend"
    elif 0.1 < abs(trend) < 1:
        trend_nature = f"Weak - {'increasing' if trend > 0 else 'decreasing'} trend"
    else:
        trend_nature = 'Trend absent'
    result = {
        'Trend': trend_nature,
        'Trend Slope': trend,
        'Seasonal': 'Seasonal in nature' if is_seasonal else 'Seasonality absent',
        'Seasonal Period': seasonal_period,
    }
    return result


def statistic_tests(
    ts: Union[pd.DataFrame, pd.Series],
    plots=False,
    **kwargs
):
    from scipy.stats import shapiro, norm

    normality_pvalue = shapiro(ts.values.squeeze())[1]

    if normality_pvalue > 0.05:
        normality = 'Gaussian'
    else:
        normality = 'Not gaussian'

    distinct_counts = dict(ts.iloc[:, 0].value_counts(normalize=True))
    mean = ts.mean()[0]
    std = ts.std()[0]
    result = {
        'Mean': mean,
        'Median': ts.median()[0],
        'Standard Deviation': std,
        'Variance': ts.var()[0],
        'Kurtosis': ts.kurt()[0],
        'Skewness': ts.skew()[0],
        'Count of distinct values': len(distinct_counts),
        '_Data distribution': {
            key: f'{round(val * 100)}%' for key, val in
            zip(list(distinct_counts.keys())[:3], list(distinct_counts.values())[:3])
        },
        'Length': len(ts),
        'Shapiro normality test': normality,
        '_Normality pvalue': round(normality_pvalue, 3)
    }
    if plots:
        sns.distplot(ts, fit=norm)
        plt.show()
    return result


def correlation_tests(
    ts: Union[pd.DataFrame, pd.Series],
    plots=False,
    **kwargs
):
    from statsmodels.tsa.stattools import acf, pacf

    acf_x, ci = acf(ts, alpha=0.05, nlags=20)
    lower = ci[:, 0] - acf_x
    upper = ci[:, 1] - acf_x
    acf_x = np.round(((acf_x >= upper) + (acf_x <= lower)) * acf_x, 2)
    acf_map = {val: idx for idx, val in enumerate(acf_x)}
    acf_values = list(filter(lambda x: abs(x) > 0, acf_x))
    acf_lags = list(map(lambda x: acf_map[x], acf_values))

    pacf_x, ci = pacf(ts, alpha=0.05, nlags=20)
    lower = ci[:, 0] - pacf_x
    upper = ci[:, 1] - pacf_x
    pacf_x = np.round(((pacf_x >= upper) + (pacf_x <= lower)) * pacf_x, 2)
    pacf_map = {val: idx for idx, val in enumerate(pacf_x)}
    pacf_values = list(filter(lambda x: abs(x) > 0, pacf_x))
    pacf_lags = list(map(lambda x: pacf_map[x], pacf_values))

    if plots:
        f, ax = plt.subplots(nrows=2, ncols=1)

        from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
        plot_acf(ts, lags=20, ax=ax[0])
        plot_pacf(ts, lags=20, ax=ax[1])
        f.tight_layout()
        plt.show()
    result = {
        'Significant ACF lags': acf_lags,
        'Significant PACF lags': pacf_lags,
        'Highest ACF lag': acf_map[np.max(acf_values[1:])],
        'Highest PACF lag': pacf_map[np.max(pacf_values[1:])],
        '_ACF values': acf_values,
        '_PACF values': pacf_values,
    }

    return result


def stationarity_tests(
    ts: Union[pd.DataFrame, pd.Series],
    cutoff=0.05,
    **kwargs
):
    from statsmodels.tsa.stattools import adfuller
    dftest = adfuller(ts, autolag='AIC', maxlag=20)
    pvalue = dftest[1]
    result = {
        'ADF stationarity': 'Difference Stationary' if pvalue < cutoff else 'Not stationary. Try differencing.',
        '_ADF pvalue': pvalue
    }

    from statsmodels.tsa.api import kpss
    statistic, p_value, n_lags, critical_values = kpss(ts, regression='ct', nlags='auto')

    result.update({
        'KPSS stationarity': "Not stationary. Try removing trend." if statistic > critical_values[
            "5%"] else 'Trend Stationary',
        '_KPSS pvalue': p_value
    })

    if result['KPSS stationarity'] == 'Trend Stationary' and result['ADF stationarity'] == 'Difference Stationary':
        result['Stationarity'] = 'Stationary'
    elif result['KPSS stationarity'] != 'Trend Stationary' and result['ADF stationarity'] != 'Difference Stationary':
        result['Stationarity'] = 'Not stationary. Try GARCH models.'
    else:
        result['Stationarity'] = 'Partial stationarity'
    return result


def ts_tests(
    ts: pd.DataFrame,
    show_hidden_keys: bool = False,
    **kwargs
):
    """
    This method performs all statistical tests for the given time series dataset. This is further help in modelling
    and most modelling inference

    Returns
    -------
    Dataframe with the results for the tests

    """
    test_results = {
        'Temporal': trend_and_seasonality_test(ts, **kwargs),
        'Correlation': correlation_tests(ts, **kwargs),
        'Statistics': statistic_tests(ts, **kwargs),
        'Stationarity': stationarity_tests(ts, **kwargs)
    }
    reform = {(outerKey, innerKey): values for outerKey, innerDict in test_results.items() for innerKey, values in
              innerDict.items() if (show_hidden_keys or innerKey[0] != '_')}

    df = pd.DataFrame(index=reform.keys(), data=reform.values(), columns=['Result'])
    return df


if __name__ == '__main__':
    from pycaret.datasets import get_data

    data = get_data('airline', verbose=False).to_frame()
    print(list(ts_tests(data, show_hidden_keys=True).index))
