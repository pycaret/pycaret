from typing import Optional, Any, Union, Dict, Tuple, List

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.seasonal import seasonal_decompose, STL


from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)

from pycaret.utils import _resolve_dict_keys
from pycaret.utils.time_series import get_diffs
from pycaret.internal.plots.utils.time_series import (
    time_series_subplot,
    corr_subplot,
    dist_subplot,
    qq_subplot,
    frequency_components_subplot,
    plot_original_with_overlays,
    _update_fig_dimensions,
    _get_subplot_rows_cols,
    _resolve_hoverinfo,
    PlotReturnType,
)

__author__ = ["satya-pattnaik", "ngupta23"]


def _get_plot(
    plot: str,
    fig_defaults: Dict[str, Any],
    data: Optional[pd.DataFrame] = None,
    data_label: Optional[str] = None,
    X: Optional[List[pd.DataFrame]] = None,
    X_labels: Optional[List[str]] = None,
    cv: Optional[Union[ExpandingWindowSplitter, SlidingWindowSplitter]] = None,
    model_results: Optional[List[pd.DataFrame]] = None,
    model_names: Optional[List[str]] = None,
    return_pred_int: bool = False,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """TODO: Fill
    data = DataFrame for Target Variable (each column can represent one version
    of the target series, e.g. Original, Transformed, Imputed)
    data_label = Name of the target series
    """
    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    big_data_threshold = _resolve_dict_keys(
        dict_=fig_kwargs, key="big_data_threshold", defaults=fig_defaults
    )
    hoverinfo = _resolve_dict_keys(
        dict_=fig_kwargs, key="hoverinfo", defaults=fig_defaults
    )
    hoverinfo = _resolve_hoverinfo(
        hoverinfo=hoverinfo,
        threshold=big_data_threshold,
        data=data,
        X=X,
    )

    if plot in ["ts", "train_test_split"]:
        fig, plot_data = plot_series(
            y=data,
            y_label=data_label,
            fig_defaults=fig_defaults,
            X=X,
            X_labels=X_labels,
            hoverinfo=hoverinfo,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "cv":
        fig, plot_data = plot_cv(
            data=data,
            cv=cv,
            fig_defaults=fig_defaults,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )

    elif plot in ["decomp", "decomp_stl"]:
        fig, plot_data = plot_time_series_decomposition(
            y=data,
            y_label=data_label,
            fig_defaults=fig_defaults,
            model_name=model_names,
            plot=plot,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot in ["acf", "pacf"]:
        fig, plot_data = plot_xacf(
            y=data,
            y_label=data_label,
            plot=plot,
            fig_defaults=fig_defaults,
            model_name=model_names,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "diagnostics":
        fig, plot_data = plot_diagnostics(
            y=data,
            y_label=data_label,
            fig_defaults=fig_defaults,
            model_name=model_names,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot in ["forecast", "insample", "residuals"]:
        if return_pred_int:
            fig, plot_data = plot_predictions_with_confidence(
                data=data,
                predictions=model_results,
                fig_defaults=fig_defaults,
                model_names=model_names,
                data_kwargs=data_kwargs,
                fig_kwargs=fig_kwargs,
            )
        else:
            fig, plot_data = plot_model_results(
                original_data=data,
                model_results=model_results,
                plot=plot,
                model_names=model_names,
                fig_defaults=fig_defaults,
                hoverinfo=hoverinfo,
                fig_kwargs=fig_kwargs,
            )
    elif plot == "diff":
        fig, plot_data = plot_time_series_differences(
            y=data,
            y_label=data_label,
            fig_defaults=fig_defaults,
            model_name=model_names,
            hoverinfo=hoverinfo,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot in ["periodogram", "fft"]:
        fig, plot_data = plot_frequency_components(
            y=data,
            y_label=data_label,
            plot=plot,
            fig_defaults=fig_defaults,
            model_name=model_names,
            hoverinfo=hoverinfo,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "ccf":
        fig, plot_data = plot_ccf(
            y=data,
            y_label=data_label,
            X=X,
            X_labels=X_labels,
            fig_defaults=fig_defaults,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    else:
        raise ValueError(f"Plot: '{plot}' is not supported.")

    return fig, plot_data


def plot_series(
    y: pd.DataFrame,
    y_label: str,
    fig_defaults: Dict[str, Any],
    X: Optional[List[pd.DataFrame]] = None,
    X_labels: Optional[List[str]] = None,
    hoverinfo: Optional[str] = "text",
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the target time series (full or specific splits). Can optionally
    plot exogenous variables if available.

    y can contain multiple columns corresponding to the various splits of the targets
    (full, train, test | original, imputed, transformed)

    X contains 1 dataframe per exogenous variable. Each dataframe can contain multiple
    columns corresponding to the various splits of the exogenous variable
    (original, imputed, transformed)
    """
    fig_kwargs = fig_kwargs or {}

    title = f"Time Series | Target = {y_label}"

    plot_data = [y]
    subplot_titles = [y_label]
    if X is not None:
        # Exogenous Variables present.
        plot_data.extend(X)
        if X_labels is None:
            raise ValueError(
                "X is not None, but X_labels is None. This should not have occurred."
                "\nPlease file a report here with a reproducible example:"
                "\nhttps://github.com/pycaret/pycaret/issues"
            )
        subplot_titles.extend(X_labels)

    rows = len(plot_data)
    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    for i, plot_data_single in enumerate(plot_data):
        fig = time_series_subplot(
            fig=fig,
            data=plot_data_single,
            row=i + 1,
            col=1,
            hoverinfo=hoverinfo,
        )

    with fig.batch_update():
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=True, template=template)

        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {
        "data": plot_data,
    }

    return fig, return_data_dict


def plot_cv(
    data: pd.Series,
    cv,
    fig_defaults: Dict[str, Any],
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the cv splits used on the training split"""
    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    def get_windows(y, cv):
        """
        Generate windows
        Inspired from `https://github.com/alan-turing-institute/sktime`
        """
        train_windows = []
        test_windows = []
        for i, (train, test) in enumerate(cv.split(y)):
            train_windows.append(train)
            test_windows.append(test)
        return train_windows, test_windows

    def plot_windows(data, train_windows, test_windows):
        fig = go.Figure()
        for num_window in reversed(range(len(train_windows))):

            x = (
                data.index.to_timestamp()
                if isinstance(data.index, pd.PeriodIndex)
                else data.index
            )
            time_stamps = x

            y_axis_label = str(num_window)
            [
                fig.add_scattergl(
                    x=(time_stamps[i], time_stamps[i + 1]),
                    y=(y_axis_label, y_axis_label),
                    mode="lines+markers",
                    line_color="#C0C0C0",
                    name="Unchanged",
                    hoverinfo="skip",
                )
                for i in range(len(data) - 1)
            ]
            [
                fig.add_scattergl(
                    x=(time_stamps[i], time_stamps[i + 1]),
                    y=(y_axis_label, y_axis_label),
                    mode="lines+markers",
                    line_color="#1f77b4",
                    name="Train",
                    showlegend=False,
                    hoverinfo="skip",
                )
                for i in train_windows[num_window][:-1]
            ]
            [
                fig.add_scattergl(
                    x=(time_stamps[i], time_stamps[i + 1]),
                    y=(y_axis_label, y_axis_label),
                    mode="lines+markers",
                    line_color="#DE970B",
                    name="ForecastHorizon",
                    hoverinfo="skip",
                )
                for i in test_windows[num_window][:-1]
            ]

            with fig.batch_update():
                fig.update_traces(showlegend=False)

                fig.update_layout(
                    {
                        "title": "Train Cross-Validation Splits",
                        "xaxis": {"title": "Time", "zeroline": False},
                        "yaxis": {"title": "Windows"},
                        "showlegend": True,
                    }
                )
                template = _resolve_dict_keys(
                    dict_=fig_kwargs, key="template", defaults=fig_defaults
                )
                fig.update_layout(template=template)

                fig = _update_fig_dimensions(
                    fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
                )
        return fig

    train_windows, test_windows = get_windows(data, cv)
    fig = plot_windows(data, train_windows, test_windows)
    return_data_dict = {
        "data": data,
        "train_windows": train_windows,
        "test_windows": test_windows,
    }

    return fig, return_data_dict


def plot_xacf(
    y: pd.DataFrame,
    y_label: Optional[str],
    plot: str,
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the ACF or PACF plot for the data provided (could be any data type
    of the data - original, imputed, transformed) or model residuals

    Parameters
    ----------
    data : pd.DataFrame
        Data whose correlation plot needs to be plotted. If based on the original
        data, it can contain multiple columns corresponding to the various splits
        of the targets (full, train, test | original, imputed, transformed).
    y_label : Optional[str]
        The name of the target time series to be used in the plots. Optional when
        data is from residuals. In that case, the user must provide model_name argument
        instead.
    plot : str
        Type of plot, allowed values are ["acf", "pacf"]
    fig_defaults : Dict[str, Any]
        The defaults dictionary containing keys for "width", "height", and "template"
    model_name : Optional[str]
        If the correlation plot is for model residuals, then, model_name must be
        passed for proper display of results. If the correlation plot is for the
        original data, model_name should be left None (name is derived from the
        data passed in this case).
    data_kwargs : Dict[str, Any]
        A dictionary containing options keys for "nlags"
    fig_kwargs : Dict[str, Any]
        A dictionary containing optional keys for "width", "height" and/or "template"

    Returns
    -------
    PlotReturnType
        Returns back the plotly figure along with the correlation data.
    """
    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    ncols = len(y.columns)
    fig = make_subplots(
        rows=1,
        cols=ncols,
        column_titles=y.columns.tolist(),
        shared_yaxes=True,
    )

    all_plot_data = {}
    nlags = data_kwargs.get("nlags", None)
    for i, col_name in enumerate(y.columns):
        fig, plot_data = corr_subplot(
            fig=fig, data=y[col_name], row=1, col=i + 1, plot=plot, nlags=nlags
        )
        all_plot_data.update({col_name: plot_data})

    if plot == "acf":
        title = "Autocorrelation (ACF)"
    elif plot == "pacf":
        title = "Partial Autocorrelation (PACF)"
    else:
        raise ValueError(f"Plot '{plot}' is not supported by plot_xacf().")

    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif y_label is not None:
        title = f"{title} | {y_label}"

    with fig.batch_update():
        for i in np.arange(1, ncols + 1):
            fig.update_xaxes(title_text="Lags", row=1, col=i)

        # Only on first column
        fig.update_yaxes(title_text=plot.upper(), row=1, col=1)
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=False, template=template)
        fig.update_traces(marker={"size": 10})
        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {plot: all_plot_data}

    return fig, return_data_dict


def plot_model_results(
    original_data: pd.Series,
    model_results: List[pd.DataFrame],
    plot: str,
    model_names: List[str],
    fig_defaults: Dict[str, Any],
    hoverinfo: Optional[str] = "text",
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the original target series along with overlaid model results which
    could be out of sample predictions, insample predictions or residuals.

    Parameters
    ----------
    original_data : pd.Series
        The original target series to be plotted
    model_results : List[pd.DataFrame]
        The model results that must be overlaid over the original data. e.g. out
        of sample predictions, insample predictions or residuals from possibly
        multiple models. Each column in the overlay_data is overlaid as a separate
        series over the original_data. The column names are used as labels for the
        overlaid data.
    plot : str
        The type of plot.
            "forecast": Out of Sample Forecasts
            "insample": Insample Forecasts
            "residuals": Model Residuals
    model_names : List[str]
        Name(s) of the models whose results are being plotted
    fig_defaults : Dict[str, Any]
        The defaults dictionary containing keys for "width", "height", and "template"
    hoverinfo : Optional[str], optional
        Whether to display the hoverinfo in the plot, by default "text"
    fig_kwargs : Optional[Dict], optional
        A dictionary containing optional keys for "width", "height" and/or "template"

    Returns
    -------
    PlotReturnType
        Returns back the plotly figure along with the data used to create the plot.
    """

    includes = [
        True if model_result is not None else False for model_result in model_results
    ]

    # Remove None results (produced when insample or residuals can not be obtained)
    model_results = [
        model_result
        for include, model_result in zip(includes, model_results)
        if include
    ]
    model_names = [
        model_name for include, model_name in zip(includes, model_names) if include
    ]

    if plot in ["forecast", "insample"]:
        model_results = [model_result["y_pred"] for model_result in model_results]
    model_results = pd.concat(model_results, axis=1)
    model_results.columns = model_names

    if plot == "forecast":
        key = "Forecast (Out-of-Sample)"
    elif plot == "insample":
        key = "Forecast (In-Sample)"
    elif plot == "residuals":
        key = "Residuals"
    title = f"Actual vs. {key}"

    fig, return_data_dict = plot_original_with_overlays(
        original_data=original_data,
        overlay_data=model_results,
        title=title,
        fig_defaults=fig_defaults,
        hoverinfo=hoverinfo,
        fig_kwargs=fig_kwargs,
    )

    return fig, return_data_dict


def plot_diagnostics(
    y: pd.DataFrame,
    y_label: Optional[str],
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    hoverinfo: Optional[str] = "text",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the diagnostic data such as ACF, Histogram, QQ plot on the data provided

    y_label: The name of the target time series to be used in the plots. Optional when
    data is from residuals. In that case, the user must provide model_name argument
    instead.
    """
    if y.shape[1] != 1:
        raise ValueError(
            "plot_diagnostics() only works on a single time series, "
            f"but {y.shape[1]} target series were provided."
        )
    y_series = y.iloc[:, 0]

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    title = "Diagnostics"
    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif y_label is not None:
        title = f"{title} | {y_label}"

    fig = make_subplots(
        rows=3,
        cols=2,
        row_heights=[0.33, 0.33, 0.33],
        subplot_titles=[
            "Time Plot",
            "Periodogram",
            "Histogram",
            "Q-Q Plot",
            "ACF",
            "PACF",
        ],
    )

    fig.update_layout(showlegend=False)
    template = _resolve_dict_keys(
        dict_=fig_kwargs, key="template", defaults=fig_defaults
    )
    fig.update_layout(template=template)

    fig = _update_fig_dimensions(
        fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
    )

    #### Add diagnostic plots ----

    # ROW 1
    fig = time_series_subplot(fig=fig, data=y, row=1, col=1, hoverinfo=hoverinfo)
    fig, periodogram_data = frequency_components_subplot(
        fig=fig,
        data=y_series,
        row=1,
        col=2,
        hoverinfo=hoverinfo,
        plot="periodogram",
    )

    # ROW 2
    fig = dist_subplot(fig=fig, data=y_series, row=2, col=1)
    fig, qqplot_data = qq_subplot(fig=fig, data=y_series, row=2, col=2)

    # ROW 3
    fig, acf_data = corr_subplot(fig=fig, data=y_series, row=3, col=1, plot="acf")
    fig, pacf_data = corr_subplot(fig=fig, data=y_series, row=3, col=2, plot="pacf")

    with fig.batch_update():
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=False, template=template)
        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {
        "data": y,
        "periodogram": periodogram_data,
        "qqplot": qqplot_data,
        "acf": acf_data,
        "pacf": pacf_data,
    }

    return fig, return_data_dict


def plot_predictions_with_confidence(
    data: pd.Series,
    predictions: List[pd.DataFrame],
    fig_defaults: Dict[str, Any],
    model_names: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the original data and the predictions provided with confidence"""
    if len(predictions) != 1:
        raise ValueError(
            "Plotting with predictions only supports one estimator. Please pass only one estimator to fix"
        )

    preds = predictions[0]["y_pred"]
    upper_interval = predictions[0]["upper"]
    lower_interval = predictions[0]["lower"]
    model_name = model_names[0]

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    title = "Actual vs. 'Out-of-Sample' Forecast"
    time_series_name = data.name
    if time_series_name is not None:
        title = f"{title} | {time_series_name}"

    x = (
        upper_interval.index.to_timestamp()
        if isinstance(upper_interval.index, pd.PeriodIndex)
        else upper_interval.index
    )
    upper_bound = go.Scatter(
        name=f"Prediction Interval | {model_name}",  # Changed since we use only 1 legend
        x=x,
        y=upper_interval,
        mode="lines",
        marker=dict(color="#68BBE3"),
        line=dict(width=0),
        fillcolor="rgba(104,187,227,0.5)",  # eq to # fillcolor="#68BBE3" with alpha,
        showlegend=True,
        fill="tonexty",
    )

    x = (
        preds.index.to_timestamp()
        if isinstance(preds.index, pd.PeriodIndex)
        else preds.index
    )
    mean = go.Scatter(
        name=f"Forecast | {model_name}",
        x=x,
        y=preds,
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        marker=dict(size=5),
        showlegend=True,
    )

    x = (
        data.index.to_timestamp()
        if isinstance(data.index, pd.PeriodIndex)
        else data.index
    )
    original = go.Scatter(
        name="Original",
        x=x,
        y=data,
        mode="lines+markers",
        marker=dict(size=5, color="#3f3f3f"),
        showlegend=True,
    )

    x = (
        lower_interval.index.to_timestamp()
        if isinstance(lower_interval.index, pd.PeriodIndex)
        else lower_interval.index
    )
    lower_bound = go.Scatter(
        name="Lower Interval",
        x=x,
        y=lower_interval,
        marker=dict(color="#68BBE3"),
        line=dict(width=0),
        mode="lines",
        showlegend=False,  # Not outputting since we need only 1 legend for interval
    )

    data = [mean, lower_bound, upper_bound, original]

    layout = go.Layout(
        yaxis=dict(title="Values"), xaxis=dict(title="Time"), title=title
    )

    fig = go.Figure(data=data, layout=layout)

    template = _resolve_dict_keys(
        dict_=fig_kwargs, key="template", defaults=fig_defaults
    )
    fig.update_layout(template=template)
    fig.update_layout(showlegend=True)

    fig = _update_fig_dimensions(
        fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
    )

    return_data_dict = {
        "data": data,
        "predictions": predictions,
        "upper_interval": upper_interval,
        "lower_interval": lower_interval,
    }

    return fig, return_data_dict


def plot_time_series_decomposition(
    y: pd.DataFrame,
    y_label: Optional[str],
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    plot: str = "decomp",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """
    y_label: The name of the target time series to be used in the plots. Optional when
    data is from residuals. In that case, the user must provide model_name argument
    instead.
    """
    if y.shape[1] != 1:
        raise ValueError(
            "plot_time_series_decomposition() only works on a single time series, "
            f"but {y.shape[1]} target series were provided."
        )
    y_series = y.iloc[:, 0]

    fig, return_data_dict = None, None

    if not isinstance(y_series.index, (pd.PeriodIndex, pd.DatetimeIndex)):
        print(
            "Decomposition is currently not supported for pandas dataframes "
            "without a PeriodIndex or DatetimeIndex. Please specify a PeriodIndex "
            "or DatetimeIndex in setup() before plotting decomposition plots."
        )
        return fig, return_data_dict

    data_kwargs = data_kwargs or {}
    period = data_kwargs.get("seasonal_period", None)

    #### Check period ----
    if period is None:
        raise ValueError(
            "Decomposition plot needed seasonal period to be passed through "
            "`data_kwargs`. None was passed."
        )
    if plot == "decomp_stl" and period < 2:
        print(
            "STL Decomposition is not supported for time series that have a "
            f"seasonal period < 2. The seasonal period computed/provided was {period}."
        )
        return fig, return_data_dict

    classical_decomp_type = data_kwargs.get("type", "additive")
    fig_kwargs = fig_kwargs or {}

    if plot == "decomp":
        title = f"Classical Decomposition ({classical_decomp_type})"
    elif plot == "decomp_stl":
        title = "STL Decomposition"

    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif y_label is not None:
        title = f"{title} | {y_label}"
    title = title + f"<br>Seasonal Period = {period}"

    decomp_result = None
    data_ = y.to_timestamp() if isinstance(y.index, pd.PeriodIndex) else y

    if plot == "decomp":
        decomp_result = seasonal_decompose(
            data_, period=period, model=classical_decomp_type
        )
    elif plot == "decomp_stl":
        decomp_result = STL(data_, period=period).fit()

    fig = make_subplots(
        rows=4,
        cols=1,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        row_titles=["Actual", "Seasonal", "Trend", "Residual"],
        shared_xaxes=True,
    )

    x = (
        y_series.index.to_timestamp()
        if isinstance(y_series.index, pd.PeriodIndex)
        else y.index
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_series,
            line=dict(color="#1f77b4", width=2),
            mode="lines+markers",
            name="Actual",
            marker=dict(size=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=decomp_result.seasonal,
            line=dict(color="#1f77b4", width=2),
            mode="lines+markers",
            name="Seasonal",
            marker=dict(size=2),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=decomp_result.trend,
            line=dict(color="#1f77b4", width=2),
            mode="lines+markers",
            name="Trend",
            marker=dict(size=2),
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=decomp_result.resid,
            line=dict(color="#1f77b4", width=2),
            mode="markers",
            name="Residuals",
            marker=dict(
                size=4,
            ),
        ),
        row=4,
        col=1,
    )

    with fig.batch_update():
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=False, template=template)

        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {
        "data": y,
        "seasonal": decomp_result.seasonal,
        "trend": decomp_result.trend,
        "resid": decomp_result.resid,
    }

    return fig, return_data_dict


def plot_time_series_differences(
    y: pd.DataFrame,
    y_label: Optional[str],
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    hoverinfo: Optional[str] = "text",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """y_label: The name of the target time series to be used in the plots. Optional when
    data is from residuals. In that case, the user must provide model_name argument
    instead.
    """
    if y.shape[1] != 1:
        raise ValueError(
            "plot_time_series_differences() only works on a single time series, "
            f"but {y.shape[1]} target series were provided."
        )
    y_series = y.iloc[:, 0]

    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    order_list = data_kwargs.get("order_list", None)
    lags_list = data_kwargs.get("lags_list", None)

    plot_acf = data_kwargs.get("acf", False)
    plot_pacf = data_kwargs.get("pacf", False)
    plot_periodogram = data_kwargs.get("periodogram", False)
    plot_fft = data_kwargs.get("fft", False)

    title = "Difference Plot"
    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif y_label is not None:
        title = f"{title} | {y_label}"

    diff_list, name_list = get_diffs(
        data=y_series, order_list=order_list, lags_list=lags_list
    )

    if len(diff_list) == 0:
        # Issue with reconciliation of orders and diffs
        return fig, return_data_dict

    diff_list = [y_series] + diff_list
    name_list = ["Actual" if model_name is None else "Residuals"] + name_list

    column_titles = ["Time Series"]
    rows = len(diff_list)
    cols = 1
    if plot_acf:
        cols = cols + 1
        column_titles.append("ACF")
    if plot_pacf:
        cols = cols + 1
        column_titles.append("PACF")
    if plot_periodogram:
        cols = cols + 1
        column_titles.append("Periodogram")
    if plot_fft:
        cols = cols + 1
        column_titles.append("FFT")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        row_titles=name_list,
        column_titles=column_titles,
        shared_xaxes=True,
    )

    # Should the following be plotted - Time Series, ACF, PACF, Periodogram, FFT
    plots = [True, plot_acf, plot_pacf, plot_periodogram, plot_fft]

    # Which column should the plots be in
    plot_cols = np.cumsum(plots).tolist()

    for i, subplot_data in enumerate(diff_list):

        #### Add difference data ----

        ts_to_plot = pd.DataFrame(subplot_data)
        ts_to_plot.columns = [name_list[i]]
        fig = time_series_subplot(
            fig=fig,
            data=ts_to_plot,
            row=i + 1,
            col=plot_cols[0],
            hoverinfo=hoverinfo,
        )

        #### Add diagnostics if requested ----
        if plot_acf:
            fig, acf_data = corr_subplot(
                fig=fig,
                data=subplot_data,
                row=i + 1,
                col=plot_cols[1],
                name=name_list[i],
                plot="acf",
            )

        if plot_pacf:
            fig, pacf_data = corr_subplot(
                fig=fig,
                data=subplot_data,
                row=i + 1,
                col=plot_cols[2],
                name=name_list[i],
                plot="pacf",
            )

        if plot_periodogram:
            fig, periodogram_data = frequency_components_subplot(
                fig=fig,
                data=subplot_data,
                row=i + 1,
                col=plot_cols[3],
                hoverinfo=hoverinfo,
                name=name_list[i],
                plot="periodogram",
            )

        if plot_fft:
            fig, fft_data = frequency_components_subplot(
                fig=fig,
                data=subplot_data,
                row=i + 1,
                col=plot_cols[4],
                hoverinfo=hoverinfo,
                name=name_list[i],
                plot="fft",
            )

    with fig.batch_update():
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=False, template=template)

        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {
        "data": y_series,
        "diff_list": diff_list,
        "name_list": name_list,
    }

    if plot_acf:
        return_data_dict.update({"acf": acf_data})
    if plot_pacf:
        return_data_dict.update({"pacf": pacf_data})
    if plot_periodogram:
        return_data_dict.update({"periodogram": periodogram_data})
    if plot_fft:
        return_data_dict.update({"fft": fft_data})

    return fig, return_data_dict


def plot_frequency_components(
    y: pd.Series,
    y_label: Optional[str],
    plot: str,
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    hoverinfo: Optional[str] = "text",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the frequency components in the data
    y_label: The name of the target time series to be used in the plots. Optional when
    data is from residuals. In that case, the user must provide model_name argument
    instead.
    """
    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    ncols = len(y.columns)
    fig = make_subplots(
        rows=1,
        cols=ncols,
        column_titles=y.columns.tolist(),
        shared_yaxes=True,
    )

    all_plot_data = {}
    # nlags = data_kwargs.get("nlags", None)
    for i, col_name in enumerate(y.columns):
        fig, plot_data = frequency_components_subplot(
            fig=fig, data=y[col_name], row=1, col=i + 1, plot=plot, hoverinfo=hoverinfo
        )
        all_plot_data.update({col_name: plot_data})

    if plot == "periodogram":
        title = "Periodogram"
    elif plot == "fft":
        title = "FFT"
    elif plot == "welch":
        title = "Welch"
    else:
        raise ValueError(
            f"Plot '{plot}' is not supported by plot_frequency_components()."
        )

    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif y_label is not None:
        title = f"{title} | {y_label}"

    with fig.batch_update():
        for i in np.arange(1, ncols + 1):
            fig.update_xaxes(title_text="Frequency", row=1, col=i)

        # Only on first column
        fig.update_yaxes(title_text="dB", row=1, col=1)
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=False, template=template)
        fig.update_traces(marker={"size": 10})
        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {plot: all_plot_data}

    return fig, return_data_dict


def plot_ccf(
    y: pd.DataFrame,
    y_label: str,
    X: List[pd.DataFrame],
    X_labels: List[str],
    fig_defaults: Dict[str, Any],
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the Cross Correlation between the data and the exogenous variables X

    X can be a list of dataframes, 1 dataframe per exogenous variable. Each dataframe
    must have only 1 column at a time since ccf does not work on many data types at a time.

    """
    if y.shape[1] != 1:
        raise ValueError(
            "plot_ccf() only works on a single time series, "
            f"but {y.shape[1]} target series were provided."
        )
    if X is not None:
        for dataframe in X:
            if dataframe.shape[1] != 1:
                raise ValueError(
                    f"plot_ccf() only works on a single time series, but {dataframe.shape[1]} "
                    f"exogenous series were provided: {dataframe.columns}."
                )

    y_series = y.iloc[:, 0]
    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    title = f"Cross Correlation Plot(s) | {y_series.name}"

    plot_data = [y_series]
    column_names = [y_label]
    if X is not None:
        plot_data.extend(X)
        column_names.extend(X_labels)

    plot_data = pd.concat(plot_data, axis=1)
    plot_data.columns = column_names

    # Decide the number of rows and columns ----
    num_subplots = plot_data.shape[1]
    rows = _resolve_dict_keys(dict_=fig_kwargs, key="rows", defaults=fig_defaults)
    cols = _resolve_dict_keys(dict_=fig_kwargs, key="cols", defaults=fig_defaults)
    rows, cols = _get_subplot_rows_cols(num_subplots=num_subplots, rows=rows, cols=cols)

    subplot_titles = []
    for i, col_name in enumerate(plot_data.columns):
        subplot_title = f"{y_label} vs. {col_name}"
        subplot_titles.append(subplot_title)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
    )

    all_ccf_data = {}
    for i, col_name in enumerate(plot_data.columns):
        row = int(i / cols) + 1
        col = i % cols + 1
        #### Add CCF plot ----
        fig, ccf_data = corr_subplot(
            fig=fig,
            data=[y_series, plot_data[col_name]],
            row=row,
            col=col,
            plot="ccf",
        )
        all_ccf_data.update({col_name: ccf_data})

    with fig.batch_update():
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=False, template=template)

        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {"ccf": all_ccf_data}

    return fig, return_data_dict
