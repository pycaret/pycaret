import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def eda_ts_plot(
    data: pd.DataFrame | pd.Series,
    target: str | None = None,
    show: bool = True,
    acf_lags: int = 30,
    period: int | None = None,
    plotly: bool = False,
):
    """
    Exploratory Data Analysis (EDA) for univariate time series.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The input series or dataframe.
    target : str, optional
        Column name if data is a DataFrame.
    show : bool, default=True
        Whether to display plots.
    acf_lags : int, default=30
        Number of lags for ACF plot.
    period : int, optional
        Seasonal period for decomposition.
    plotly : bool, default=False
        Use Plotly for interactive visualization.

    Returns
    -------
    dict
        Summary statistics + ADF results.
    """
    # ---- Handle input type ----
    if isinstance(data, pd.DataFrame):
        if target is None:
            raise ValueError("Please specify `target` when passing a DataFrame.")
        if target not in data.columns:
            raise KeyError(f"Target '{target}' not found in DataFrame.")
        series = data[target]
    elif isinstance(data, pd.Series):
        series = data
    else:
        raise TypeError("Input must be a pandas Series or DataFrame.")

    # ---- Basic summary ----
    summary = {
        "length": len(series),
        "missing_values": series.isna().sum(),
        "mean": float(series.mean()),
        "min": float(series.min()),
        "max": float(series.max()),
    }

    # ---- ADF Test ----
    try:
        adf_res = adfuller(series.dropna(), autolag="AIC")
        summary["adf_statistic"] = adf_res[0]
        summary["adf_pvalue"] = adf_res[1]
        summary["adf_stationary"] = bool(adf_res[1] < 0.05)
        print(
            f"[EDA] ADF p-value = {adf_res[1]:.4f} → "
            f"{'Stationary' if adf_res[1] < 0.05 else 'Non-Stationary'}"
        )
    except Exception as e:
        summary["adf_error"] = str(e)
        print(f"[EDA] ADF test skipped: {e}")

    # ---- Visualization ----
    if show:
        if plotly:
            # 1️⃣ Interactive line chart
            import plotly.express as px
            fig = px.line(series, title="Time Series Plot (Plotly)")
            fig.show()

            # 2️⃣ Interactive ACF via Plotly
            from statsmodels.tsa.stattools import acf
            import plotly.graph_objects as go

            vals = series.dropna()
            acf_vals = acf(vals, nlags=acf_lags)
            fig_acf = go.Figure(
                go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color="skyblue")
            )
            fig_acf.update_layout(title="Autocorrelation (ACF)", xaxis_title="Lag", yaxis_title="ACF")
            fig_acf.show()

            # 3️⃣ Seasonal decomposition (matplotlib backend)
            if period is not None and len(series.dropna()) >= 2 * period:
                decomposition = seasonal_decompose(series.dropna(), period=period, model="additive")
                decomposition.plot()
                plt.suptitle(f"Seasonal Decomposition (period={period})", fontsize=12)
                plt.tight_layout()
                plt.show()
            else:
                print("[EDA] skipping decomposition (provide `period` to enable)")
        else:
            # 1️⃣ Static line plot
            plt.figure(figsize=(10, 4))
            plt.plot(series, marker="o", linestyle="-", color="tab:blue")
            plt.title("Time Series Plot")
            plt.xlabel("Index / Time")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # 2️⃣ Static ACF
            plt.figure(figsize=(8, 4))
            plot_acf(series.dropna(), lags=acf_lags)
            plt.title("Autocorrelation (ACF)")
            plt.tight_layout()
            plt.show()

            # 3️⃣ Seasonal decomposition
            if period is not None and len(series.dropna()) >= 2 * period:
                decomposition = seasonal_decompose(series.dropna(), period=period, model="additive")
                decomposition.plot()
                plt.suptitle(f"Seasonal Decomposition (period={period})", fontsize=12)
                plt.tight_layout()
                plt.show()
            else:
                print("[EDA] skipping decomposition (provide `period` to enable)")

    return summary
