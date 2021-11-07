from typing import Optional, Any, Union, Dict
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import pacf, acf
import plotly.express as px
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import matplotlib
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)
from statsmodels.tsa.seasonal import seasonal_decompose, STL

__author__ = ["satya-pattnaik", "ngupta23"]
#################
#### Helpers ####
#################


def plot_(
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
) -> Any:

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

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
    else:
        raise ValueError(f"Plot: '{plot}' is not supported.")

    return fig, plot_data


def plot_series(
    data: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
):
    """Plots the original time series"""
    fig, return_data_dict = None, None

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

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
):
    """Plots the train-test split for the time serirs"""
    fig, return_data_dict = None, None

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}
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
):
    """Plots the cv splits used on the training split"""
    fig, return_data_dict = None, None

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

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
    }

    return fig, return_data_dict


def plot_acf(
    data: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
):
    """Plots the ACF on the data provided"""
    fig, return_data_dict = None, None

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

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
        "acf": corr_array[0],
    }

    return fig, return_data_dict


def plot_pacf(
    data: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
):
    """Plots the PACF on the data provided"""
    fig, return_data_dict = None, None

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

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
        "pacf": corr_array[0],
    }

    return fig, return_data_dict


def plot_predictions(
    data: pd.Series,
    predictions: pd.Series,
    type_: str,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
):
    """Plots the original data and the predictions provided"""
    fig, return_data_dict = None, None

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

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

    data = [mean, original]

    layout = go.Layout(
        yaxis=dict(title="Values"), xaxis=dict(title="Time"), title=title,
    )

    fig = go.Figure(data=data, layout=layout)

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
):
    """Plots the diagnostic data such as ACF, Histogram, QQ plot on the data provided"""
    fig, return_data_dict = None, None

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

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

    def time_plot(fig):
        x = (
            data.index.to_timestamp()
            if isinstance(data.index, pd.PeriodIndex)
            else data.index
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data,
                mode="lines+markers",
                marker_color="#1f77b4",
                marker_size=2,
                name="Time Plot",
            ),
            row=1,
            col=1,
        )

        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)

    def qq(fig):

        matplotlib.use("Agg")
        qqplot_data = qqplot(data, line="s")
        plt.close(qqplot_data)
        qqplot_data = qqplot_data.gca().lines

        fig.add_trace(
            {
                "type": "scatter",
                "x": qqplot_data[0].get_xdata(),
                "y": qqplot_data[0].get_ydata(),
                "mode": "markers",
                "marker": {"color": "#1f77b4"},
                "name": data.name,
            },
            row=2,
            col=2,
        )

        fig.add_trace(
            {
                "type": "scatter",
                "x": qqplot_data[1].get_xdata(),
                "y": qqplot_data[1].get_ydata(),
                "mode": "lines",
                "line": {"color": "#3f3f3f"},
                "name": data.name,
            },
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text="Theoretical Quantities", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantities", row=2, col=2)

    def dist_plot(fig):

        temp_fig = px.histogram(data, color_discrete_sequence=["#1f77b4"])
        fig.add_trace(temp_fig.data[0], row=1, col=2)

        fig.update_xaxes(title_text="Range of Values", row=1, col=2)
        fig.update_yaxes(title_text="PDF", row=1, col=2)

    def plot_acf(fig):
        corr_array = acf(data, alpha=0.05)

        lower_y = corr_array[1][:, 0] - corr_array[0]
        upper_y = corr_array[1][:, 1] - corr_array[0]

        [
            fig.add_scatter(
                x=(x, x),
                y=(0, corr_array[0][x]),
                mode="lines",
                line_color="#3f3f3f",
                row=2,
                col=1,
                name="ACF",
            )
            for x in range(len(corr_array[0]))
        ]
        fig.add_scatter(
            x=np.arange(len(corr_array[0])),
            y=corr_array[0],
            mode="markers",
            marker_color="#1f77b4",
            marker_size=6,
            row=2,
            col=1,
        )

        fig.add_scatter(
            x=np.arange(len(corr_array[0])),
            y=upper_y,
            mode="lines",
            line_color="rgba(255,255,255,0)",
            row=2,
            col=1,
            name="UC",
        )
        fig.add_scatter(
            x=np.arange(len(corr_array[0])),
            y=lower_y,
            mode="lines",
            fillcolor="rgba(32, 146, 230,0.3)",
            fill="tonexty",
            line_color="rgba(255,255,255,0)",
            row=2,
            col=1,
            name="LC",
        )
        fig.update_traces(showlegend=False)
        fig.update_xaxes(range=[-1, 42], row=2, col=1)
        fig.update_yaxes(zerolinecolor="#000000", row=2, col=1)
        fig.update_xaxes(title_text="Lags", row=2, col=1)
        fig.update_yaxes(title_text="ACF", row=2, col=1)

        # fig.update_layout(title=title)

    fig.update_layout(showlegend=False)
    fig_template = fig_kwargs.get("fig_template", "ggplot2")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    qq(fig)
    dist_plot(fig)
    plot_acf(fig)
    time_plot(fig)

    return_data_dict = {
        "data": data,
    }

    return fig, return_data_dict


def plot_predictions_with_confidence(
    data: pd.Series,
    predictions: pd.Series,
    upper_interval: pd.Series,
    lower_interval: pd.Series,
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
):
    """Plots the original data and the predictions provided with confidence"""
    fig, return_data_dict = None, None

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

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
):
    fig, return_data_dict = None, None

    if not isinstance(data.index, (pd.PeriodIndex, pd.DatetimeIndex)):
        print(
            "Decomposition is currently not supported for pandas dataframes "
            "without a PeriodIndex or DatetimeIndex. Please specify a PeriodIndex "
            "or DatetimeIndex in setup() before plotting decomposition plots."
        )
        return fig, return_data_dict

    if data_kwargs is None:
        data_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

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

    if plot == "decomp_classical":
        decomp_result = seasonal_decompose(data_, model=classical_decomp_type)
    elif plot == "decomp_stl":
        decomp_result = STL(data_).fit()

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

