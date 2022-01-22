from typing import Optional, Any, Union, Dict, Tuple

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

from pycaret.internal.plots.utils.time_series import (
    get_diffs,
    time_series_subplot,
    corr_subplot,
    dist_subplot,
    qq_subplot,
)

__author__ = ["satya-pattnaik", "ngupta23"]


def _plot(
    plot: str,
    data: Optional[pd.Series] = None,
    train: Optional[pd.Series] = None,
    test: Optional[pd.Series] = None,
    predictions: Optional[pd.Series] = None,
    cv: Optional[Union[ExpandingWindowSplitter, SlidingWindowSplitter]] = None,
    model_name: Optional[str] = None,
    return_pred_int: bool = False,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    if plot == "ts":
        fig, plot_data = plot_series(
            data=data, data_kwargs=data_kwargs, fig_kwargs=fig_kwargs,
        )
    elif plot == "train_test_split":
        fig, plot_data = plot_splits_train_test_split(
            train=train, test=test, data_kwargs=data_kwargs, fig_kwargs=fig_kwargs,
        )

    elif plot == "cv":
        fig, plot_data = plot_cv(
            data=data, cv=cv, data_kwargs=data_kwargs, fig_kwargs=fig_kwargs,
        )

    elif plot == "decomp_classical":
        fig, plot_data = plot_time_series_decomposition(
            data=data,
            model_name=model_name,
            plot="decomp_classical",
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )

    elif plot == "decomp_stl":
        fig, plot_data = plot_time_series_decomposition(
            data=data,
            model_name=model_name,
            plot="decomp_stl",
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )

    elif plot == "acf":
        fig, plot_data = plot_acf(
            data=data,
            model_name=model_name,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "pacf":
        fig, plot_data = plot_pacf(
            data=data,
            model_name=model_name,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "diagnostics":
        fig, plot_data = plot_diagnostics(
            data=data,
            model_name=model_name,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "residuals":
        fig, plot_data = plot_series(
            data=data,
            model_name=model_name,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot in ["forecast", "insample"]:
        if return_pred_int:
            fig, plot_data = plot_predictions_with_confidence(
                data=data,
                predictions=predictions["y_pred"],
                upper_interval=predictions["upper"],
                lower_interval=predictions["lower"],
                model_name=model_name,
                data_kwargs=data_kwargs,
                fig_kwargs=fig_kwargs,
            )
        else:
            fig, plot_data = plot_predictions(
                data=data,
                predictions=predictions,
                type_=plot,
                model_name=model_name,
                data_kwargs=data_kwargs,
                fig_kwargs=fig_kwargs,
            )
    elif plot == "diff":
        fig, plot_data = plot_time_series_differences(
            data=data,
            model_name=model_name,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    else:
        raise ValueError(f"Plot: '{plot}' is not supported.")

    return fig, plot_data


def plot_series(
    data: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    """Plots the original time series"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    time_series_name = data.name
    if model_name is not None:
        title = f"Residuals"
        legend = f"Residuals | {model_name}"
    else:
        title = "Time Series"
        if time_series_name is not None:
            title = f"{title} | {time_series_name}"
        legend = f"Time Series"

    x = (
        data.index.to_timestamp()
        if isinstance(data.index, pd.PeriodIndex)
        else data.index
    )
    original = go.Scatter(
        name=legend,
        x=x,
        y=data,
        mode="lines+markers",
        marker=dict(size=5, color="#3f3f3f"),
        showlegend=True,
    )
    plot_data = [original]

    layout = go.Layout(
        yaxis=dict(title="Values"), xaxis=dict(title="Time"), title=title,
    )

    fig = go.Figure(data=plot_data, layout=layout)

    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)
    fig.update_layout(showlegend=True)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    return_data_dict = {
        "data": data,
    }

    return fig, return_data_dict


def plot_splits_train_test_split(
    train: pd.Series,
    test: pd.Series,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    """Plots the train-test split for the time serirs"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    fig = go.Figure()

    x = (
        train.index.to_timestamp()
        if isinstance(train.index, pd.PeriodIndex)
        else train.index
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=train, mode="lines+markers", marker_color="#1f77b4", name="Train",
        )
    )

    x = (
        test.index.to_timestamp()
        if isinstance(test.index, pd.PeriodIndex)
        else test.index
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=test, mode="lines+markers", marker_color="#FFA500", name="Test",
        )
    )
    fig.update_layout(
        {
            "title": "Train Test Split",
            "xaxis": {"title": "Time", "zeroline": False},
            "yaxis": {"title": "Values"},
            "showlegend": True,
        }
    )
    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    return_data_dict = {
        "train": train,
        "test": test,
    }

    return fig, return_data_dict


def plot_cv(
    data: pd.Series,
    cv,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    """Plots the cv splits used on the training split"""
    fig, return_data_dict = None, None

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
                fig.add_scatter(
                    x=(time_stamps[i], time_stamps[i + 1]),
                    y=(y_axis_label, y_axis_label),
                    mode="lines+markers",
                    line_color="#C0C0C0",
                    name=f"Unchanged",
                    hoverinfo="skip",
                )
                for i in range(len(data) - 1)
            ]
            [
                fig.add_scatter(
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
                fig.add_scatter(
                    x=(time_stamps[i], time_stamps[i + 1]),
                    y=(y_axis_label, y_axis_label),
                    mode="lines+markers",
                    line_color="#DE970B",
                    name="ForecastHorizon",
                    hoverinfo="skip",
                )
                for i in test_windows[num_window][:-1]
            ]
            fig.update_traces(showlegend=False)

            fig.update_layout(
                {
                    "title": "Train Cross-Validation Splits",
                    "xaxis": {"title": "Time", "zeroline": False},
                    "yaxis": {"title": "Windows"},
                    "showlegend": True,
                }
            )
            fig_template = fig_kwargs.get("fig_template", "ggplot2")
            fig.update_layout(template=fig_template)

            fig_size = fig_kwargs.get("fig_size", None)
            if fig_size is not None:
                fig.update_layout(
                    autosize=False, width=fig_size[0], height=fig_size[1],
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


def plot_acf(
    data: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    """Plots the ACF on the data provided"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    nlags = data_kwargs.get("nlags", None)
    corr_array = acf(data, alpha=0.05, nlags=nlags)

    time_series_name = data.name
    title = "Autocorrelation (ACF)"
    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif time_series_name is not None:
        title = f"{title} | {time_series_name}"

    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    fig = go.Figure()

    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=corr_array[0],
        mode="markers",
        marker_color="#1f77b4",
        marker_size=10,
        name=f"ACF",
    )

    [
        fig.add_scatter(
            x=(x, x),
            y=(0, corr_array[0][x]),
            mode="lines",
            line_color="#3f3f3f",
            name=f"Lag{ind + 1}",
        )
        for ind, x in enumerate(range(len(corr_array[0])))
    ]

    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=upper_y,
        mode="lines",
        line_color="rgba(255,255,255,0)",
        name="UC",
    )
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=lower_y,
        mode="lines",
        fillcolor="rgba(32, 146, 230,0.3)",
        fill="tonexty",
        line_color="rgba(255,255,255,0)",
        name="LC",
    )
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, 42])

    fig.add_scatter(
        x=(0, len(corr_array[0])), y=(0, 0), mode="lines", line_color="#3f3f3f", name=""
    )
    fig.update_traces(showlegend=False)

    fig.update_yaxes(zerolinecolor="#000000")

    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )
    fig.update_layout(title=title)

    return_data_dict = {
        "acf": corr_array,
    }

    return fig, return_data_dict


def plot_pacf(
    data: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    """Plots the PACF on the data provided"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    nlags = data_kwargs.get("nlags", None)
    corr_array = pacf(data, alpha=0.05, nlags=nlags)

    time_series_name = data.name
    title = "Partial Autocorrelation (PACF)"
    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif time_series_name is not None:
        title = f"{title} | {time_series_name}"

    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    fig = go.Figure()

    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=corr_array[0],
        mode="markers",
        marker_color="#1f77b4",
        marker_size=10,
        name="PACF",
    )

    [
        fig.add_scatter(
            x=(x, x),
            y=(0, corr_array[0][x]),
            mode="lines",
            line_color="#3f3f3f",
            name=f"Lag{ind + 1}",
        )
        for ind, x in enumerate(range(len(corr_array[0])))
    ]

    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=upper_y,
        mode="lines",
        line_color="rgba(255,255,255,0)",
        name="UC",
    )
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=lower_y,
        mode="lines",
        fillcolor="rgba(32, 146, 230,0.3)",
        fill="tonexty",
        line_color="rgba(255,255,255,0)",
        name="LC",
    )
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, 42])

    fig.add_scatter(
        x=(0, len(corr_array[0])), y=(0, 0), mode="lines", line_color="#3f3f3f", name=""
    )
    fig.update_traces(showlegend=False)

    fig.update_yaxes(zerolinecolor="#000000")

    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    fig.update_layout(title=title)

    return_data_dict = {
        "pacf": corr_array,
    }

    return fig, return_data_dict


def plot_predictions(
    data: pd.Series,
    predictions: pd.Series,
    type_: str,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    """Plots the original data and the predictions provided"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    key = "Out-of-Sample" if type_ == "forecast" else "In-Sample"
    title = f"Actual vs. '{key}' Forecast"
    time_series_name = data.name
    if time_series_name is not None:
        title = f"{title} | {time_series_name}"

    x = (
        predictions.index.to_timestamp()
        if isinstance(predictions.index, pd.PeriodIndex)
        else predictions.index
    )
    mean = go.Scatter(
        name=f"Forecast | {model_name}",
        x=x,
        y=predictions,
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        marker=dict(size=5,),
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

    data_for_fig = [mean, original]

    layout = go.Layout(
        yaxis=dict(title="Values"), xaxis=dict(title="Time"), title=title,
    )

    fig = go.Figure(data=data_for_fig, layout=layout)

    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )
    fig.update_layout(showlegend=True)

    return_data_dict = {
        "data": data,
        "predictions": predictions,
    }

    return fig, return_data_dict


def plot_diagnostics(
    data: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    """Plots the diagnostic data such as ACF, Histogram, QQ plot on the data provided"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    time_series_name = data.name
    title = "Diagnostics"
    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif time_series_name is not None:
        title = f"{title} | {time_series_name}"

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.5, 0.5,],
        subplot_titles=[
            "Time Plot",
            "Histogram Plot",
            "ACF Plot",
            "Quantile-Quantile Plot",
        ],
        x_title=title,
    )

    fig.update_layout(showlegend=False)
    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    #### Add diagnostic plots ----
    fig = time_series_subplot(fig=fig, data=data, row=1, col=1, name="Time Plot")
    fig = dist_subplot(fig=fig, data=data, row=1, col=2)
    fig, acf_data = corr_subplot(
        fig=fig, data=data, row=2, col=1, name="ACF", plot_acf=True,
    )
    fig, qqplot_data = qq_subplot(fig=fig, data=data, row=2, col=2)

    return_data_dict = {"data": data, "qqplot": qqplot_data, "acf": acf_data}

    return fig, return_data_dict


def plot_predictions_with_confidence(
    data: pd.Series,
    predictions: pd.Series,
    upper_interval: pd.Series,
    lower_interval: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    """Plots the original data and the predictions provided with confidence"""
    fig, return_data_dict = None, None

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
        predictions.index.to_timestamp()
        if isinstance(predictions.index, pd.PeriodIndex)
        else predictions.index
    )
    mean = go.Scatter(
        name=f"Forecast | {model_name}",
        x=x,
        y=predictions,
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        marker=dict(size=5,),
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
        yaxis=dict(title="Values"), xaxis=dict(title="Time"), title=title,
    )

    fig = go.Figure(data=data, layout=layout)

    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)
    fig.update_layout(showlegend=True)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    return_data_dict = {
        "data": data,
        "predictions": predictions,
        "upper_interval": upper_interval,
        "lower_interval": lower_interval,
    }

    return fig, return_data_dict


def plot_time_series_decomposition(
    data: pd.Series,
    model_name: Optional[str] = None,
    plot: str = "decomp_classical",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    fig, return_data_dict = None, None

    if not isinstance(data.index, (pd.PeriodIndex, pd.DatetimeIndex)):
        print(
            "Decomposition is currently not supported for pandas dataframes "
            "without a PeriodIndex or DatetimeIndex. Please specify a PeriodIndex "
            "or DatetimeIndex in setup() before plotting decomposition plots."
        )
        return fig, return_data_dict

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    classical_decomp_type = data_kwargs.get("type", "additive")

    if plot == "decomp_classical":
        title_name = f"Classical Decomposition ({classical_decomp_type})"
    elif plot == "decomp_stl":
        title_name = "STL Decomposition"

    if model_name is None:
        title = f"{title_name}" if data.name is None else f"{title_name} | {data.name}"
    else:
        title = f"{title_name} | '{model_name}' Residuals"

    decomp_result = None
    data_ = data.to_timestamp() if isinstance(data.index, pd.PeriodIndex) else data

    sp_to_use = data_kwargs.get("sp_to_use", None)
    if plot == "decomp_classical":
        if sp_to_use is None:
            decomp_result = seasonal_decompose(data_, model=classical_decomp_type)
        else:
            decomp_result = seasonal_decompose(
                data_, period=sp_to_use, model=classical_decomp_type
            )
    elif plot == "decomp_stl":
        if sp_to_use is None:
            decomp_result = STL(data_).fit()
        else:
            decomp_result = STL(data_, period=sp_to_use).fit()

    fig = make_subplots(
        rows=4,
        cols=1,
        row_heights=[0.25, 0.25, 0.25, 0.25,],
        row_titles=["Actual", "Seasonal", "Trend", "Residual"],
        shared_xaxes=True,
    )

    x = (
        data.index.to_timestamp()
        if isinstance(data.index, pd.PeriodIndex)
        else data.index
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data,
            line=dict(color="#1f77b4", width=2),
            mode="lines+markers",
            name="Actual",
            marker=dict(size=2,),
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
            marker=dict(size=2,),
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
            marker=dict(size=2,),
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
            marker=dict(size=4,),
        ),
        row=4,
        col=1,
    )
    fig.update_layout(title=title)
    fig.update_layout(showlegend=False)
    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    return_data_dict = {
        "data": data,
        "seasonal": decomp_result.seasonal,
        "trend": decomp_result.trend,
        "resid": decomp_result.resid,
    }

    return fig, return_data_dict


def plot_time_series_differences(
    data: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    order_list = data_kwargs.get("order_list", None)
    lags_list = data_kwargs.get("lags_list", None)

    plot_acf = data_kwargs.get("acf", False)
    plot_pacf = data_kwargs.get("pacf", False)

    title_name = "Difference Plot"
    data_name = data.name if model_name is None else f"'{model_name}' Residuals"

    if model_name is None:
        title = f"{title_name}" if data.name is None else f"{title_name} | {data_name}"
    else:
        title = f"{title_name} | {data_name}"

    diff_list, name_list = get_diffs(
        data=data, order_list=order_list, lags_list=lags_list
    )

    if len(diff_list) == 0:
        # Issue with reconciliation of orders and diffs
        return fig, return_data_dict

    diff_list = [data] + diff_list
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

    fig = make_subplots(
        rows=rows,
        cols=cols,
        row_titles=name_list,
        column_titles=column_titles,
        shared_xaxes=True,
    )

    # Should the following be plotted - Time Series, ACF, PACF
    plots = [True, plot_acf, plot_pacf]

    # Which column should the plots be in
    plot_cols = np.cumsum(plots).tolist()

    for i, subplot_data in enumerate(diff_list):

        #### Add difference data ----
        fig = time_series_subplot(
            fig=fig, data=subplot_data, row=i + 1, col=plot_cols[0], name=name_list[i]
        )

        #### Add diagnostics if requested ----
        if plot_acf:
            fig, acf_data = corr_subplot(
                fig=fig,
                data=subplot_data,
                row=i + 1,
                col=plot_cols[1],
                name=name_list[i],
                plot_acf=True,
            )

        if plot_pacf:
            fig, pacf_data = corr_subplot(
                fig=fig,
                data=subplot_data,
                row=i + 1,
                col=plot_cols[2],
                name=name_list[i],
                plot_acf=False,
            )

    fig.update_layout(title=title)
    fig.update_layout(showlegend=False)
    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", [400 * cols, 200 * rows])
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    return_data_dict = {
        "data": data,
        "diff_list": diff_list,
        "name_list": name_list,
    }

    if plot_acf:
        return_data_dict.update({"acf": acf_data})
    if plot_pacf:
        return_data_dict.update({"pacf": pacf_data})

    return fig, return_data_dict

