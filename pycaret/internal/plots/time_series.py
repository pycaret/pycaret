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
    return_data: bool = False,
    show: bool = True,
    prediction_interval_flag: bool = False,
    data_kwargs: Dict = {},
    fig_kwargs: Dict = {},
) -> Optional[Any]:
    if plot == "ts":
        plot_data = plot_series(
            data=data,
            return_data=return_data,
            show=show,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "train_test_split":
        plot_data = plot_splits_train_test_split(
            train=train,
            test=test,
            return_data=return_data,
            show=show,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )

    elif plot == "cv":
        plot_data = plot_cv(
            data=data,
            cv=cv,
            return_data=return_data,
            show=show,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )

    elif plot == "acf":
        plot_data = plot_acf(
            data=data,
            model_name=model_name,
            return_data=return_data,
            show=show,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "pacf":
        plot_data = plot_pacf(
            data=data,
            model_name=model_name,
            return_data=return_data,
            show=show,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "diagnostics":
        plot_data = plot_diagnostics(
            data=data,
            model_name=model_name,
            return_data=return_data,
            show=show,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "residuals":
        plot_data = plot_series(
            data=data,
            model_name=model_name,
            return_data=return_data,
            show=show,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "forecast":
        if prediction_interval_flag:
            plot_data = plot_predictions_with_confidence(
                data=data,
                predictions=predictions["y_pred"],
                upper_interval=predictions["upper"],
                lower_interval=predictions["lower"],
                model_name=model_name,
                return_data=False,
                show=True,
                data_kwargs=data_kwargs,
                fig_kwargs=fig_kwargs,
            )
        else:
            plot_data = plot_predictions(
                data=data,
                predictions=predictions,
                model_name=model_name,
                return_data=return_data,
                show=show,
                data_kwargs=data_kwargs,
                fig_kwargs=fig_kwargs,
            )
    else:
        raise ValueError(f"Tests: '{plot}' is not supported.")

    return plot_data if return_data else None


def plot_series(
    data: pd.Series,
    model_name: Optional[str] = None,
    return_data: bool = False,
    show: bool = True,
    data_kwargs: Dict = {},
    fig_kwargs: Dict = {},
):
    """Plots the original time series"""
    title = "Time Series" if model_name is None else f"Residual(s)"
    legend = "Time Series" if model_name is None else f"Residual"

    original = go.Scatter(
        name=f"{legend} | {model_name}",
        x=data.index.to_timestamp(),
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

    fig_template = fig_kwargs.get("fig_template", "simple_white")
    fig.update_layout(template=fig_template)
    fig.update_layout(showlegend=True)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    if show:
        fig.show()

    if return_data:
        return fig, data
    else:
        return fig


def plot_splits_train_test_split(
    train: pd.Series,
    test: pd.Series,
    return_data: bool = False,
    show: bool = True,
    data_kwargs: Dict = {},
    fig_kwargs: Dict = {},
):
    """Plots the train-test split for the time serirs"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train.index.to_timestamp(),
            y=train,
            mode="lines+markers",
            marker_color="#1f77b4",
            name="Train",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test.index.to_timestamp(),
            y=test,
            mode="lines+markers",
            marker_color="#FFA500",
            name="Test",
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
    fig_template = fig_kwargs.get("fig_template", "simple_white")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )
    if show:
        fig.show()

    if return_data:
        return fig, (train, test)
    else:
        return fig


def plot_cv(
    data: pd.Series,
    cv,
    return_data: bool = False,
    show: bool = True,
    data_kwargs: Dict = {},
    fig_kwargs: Dict = {},
):
    """Plots the cv splits used on the training split"""

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
            time_stamps = data.index.to_timestamp()
            [
                fig.add_scatter(
                    x=(time_stamps[i], time_stamps[i + 1]),
                    y=(num_window, num_window),
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
                    y=(num_window, num_window),
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
                    y=(num_window, num_window),
                    mode="lines+markers",
                    line_color="#DE970B",
                    name="ForecastHorizon",
                    hoverinfo="skip",
                )
                for i in test_windows[num_window][:-1]
            ]
            fig.update_traces(showlegend=False)
            fig.update_yaxes(autorange="reversed")

            fig.update_layout(
                {
                    "title": "Train Cross-Validation Splits",
                    "xaxis": {"title": "Time", "zeroline": False},
                    "yaxis": {"title": "Windows"},
                    "showlegend": True,
                }
            )
            fig_template = fig_kwargs.get("fig_template", "simple_white")
            fig.update_layout(template=fig_template)

            fig_size = fig_kwargs.get("fig_size", None)
            if fig_size is not None:
                fig.update_layout(
                    autosize=False, width=fig_size[0], height=fig_size[1],
                )
        return fig

    train_windows, test_windows = get_windows(data, cv)
    fig = plot_windows(data, train_windows, test_windows)
    if show:
        fig.show()

    if return_data:
        return fig, data
    else:
        return fig


def plot_acf(
    data: pd.Series,
    model_name: Optional[str] = None,
    return_data: bool = False,
    show: bool = True,
    data_kwargs: Dict = {},
    fig_kwargs: Dict = {},
):
    """Plots the ACF on the data provided"""

    nlags = data_kwargs.get("nlags", None)
    corr_array = acf(data, alpha=0.05, nlags=nlags)
    title = (
        "Autocorrelation (ACF)"
        if model_name is None
        else f"Autocorrelation (ACF) | {model_name}"
    )

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

    fig_template = fig_kwargs.get("fig_template", "simple_white")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )
    fig.update_layout(title=title)
    if show:
        fig.show()

    if return_data:
        # Return `plotly Figure` object and the ACF values
        return fig, corr_array[0]
    else:
        return fig


def plot_pacf(
    data: pd.Series,
    model_name: Optional[str] = None,
    return_data: bool = False,
    show: bool = True,
    data_kwargs: Dict = {},
    fig_kwargs: Dict = {},
):
    """Plots the PACF on the data provided"""

    nlags = data_kwargs.get("nlags", None)
    corr_array = pacf(data, alpha=0.05, nlags=nlags)
    title = (
        "Partial Autocorrelation (PACF)"
        if model_name is None
        else f"Partial Autocorrelation (PACF) | {model_name}"
    )

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

    fig_template = fig_kwargs.get("fig_template", "simple_white")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    fig.update_layout(title=title)
    if show:
        fig.show()

    if return_data:
        # Return `plotly Figure` object and the PACF values
        return fig, corr_array[0]
    else:
        return fig


def plot_predictions(
    data: pd.Series,
    predictions: pd.Series,
    model_name: Optional[str] = None,
    return_data: bool = False,
    show: bool = True,
    data_kwargs: Dict = {},
    fig_kwargs: Dict = {},
):
    """Plots the original data and the predictions provided"""

    title = "Actual vs. Forecast"
    time_series_name = data.name
    if time_series_name is not None:
        title = time_series_name + " | " + title

    mean = go.Scatter(
        name=f"Forecast | {model_name}",
        x=predictions.index.to_timestamp(),
        y=predictions,
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        marker=dict(size=5,),
        showlegend=True,
    )
    original = go.Scatter(
        name="Original",
        x=data.index.to_timestamp(),
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

    fig_template = fig_kwargs.get("fig_template", "simple_white")
    fig.update_layout(template=fig_template)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )
    fig.update_layout(showlegend=True)
    if show:
        fig.show()

    if return_data:
        return fig, data, predictions
    else:
        return fig


def plot_diagnostics(
    data: pd.Series,
    model_name: Optional[str] = None,
    return_data: bool = False,
    show: bool = True,
    data_kwargs: Dict = {},
    fig_kwargs: Dict = {},
):
    """Plots the diagnostic data such as ACF, Histogram, QQ plot on the data provided"""
    time_series_name = data.name
    title = (
        f"Diagnostics | {time_series_name}"
        if model_name is None
        else f"Diagnostics | '{model_name}' Residuals"
    )

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
        fig.add_trace(
            go.Scatter(
                x=data.index.to_timestamp(),
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
    fig_template = fig_kwargs.get("fig_template", "simple_white")
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
    if show:
        fig.show()

    if return_data:
        return fig, data
    else:
        return fig


def plot_predictions_with_confidence(
    data: pd.Series,
    predictions: pd.Series,
    upper_interval: pd.Series,
    lower_interval: pd.Series,
    model_name: Optional[str] = None,
    return_data: bool = False,
    show: bool = True,
    data_kwargs: Dict = {},
    fig_kwargs: Dict = {},
):
    """Plots the original data and the predictions provided with confidence"""
    title = (
        "Actual and Forecast"
        if data.name is None
        else f"Actual and Forecast | {data.name}"
    )

    upper_bound = go.Scatter(
        name=f"Prediction Interval | {model_name}",  # Changed since we use only 1 legend
        x=upper_interval.index.to_timestamp(),
        y=upper_interval,
        mode="lines",
        marker=dict(color="#68BBE3"),
        line=dict(width=0),
        fillcolor="#68BBE3",
        showlegend=True,
        fill="tonexty",
    )

    mean = go.Scatter(
        name=f"Forecast | {model_name}",
        x=predictions.index.to_timestamp(),
        y=predictions,
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        marker=dict(size=5,),
        fillcolor="#68BBE3",
        fill="tonexty",
        showlegend=True,
    )
    original = go.Scatter(
        name="Original",
        x=data.index.to_timestamp(),
        y=data,
        mode="lines+markers",
        marker=dict(size=5, color="#3f3f3f"),
        showlegend=True,
    )

    lower_bound = go.Scatter(
        name="Lower Interval",
        x=lower_interval.index.to_timestamp(),
        y=lower_interval,
        marker=dict(color="#68BBE3"),
        line=dict(width=0),
        mode="lines",
        showlegend=False,  # Not outputting since we need only 1 legend for interval
    )

    data = [lower_bound, mean, upper_bound, original]

    layout = go.Layout(
        yaxis=dict(title="Values"), xaxis=dict(title="Time"), title=title,
    )

    fig = go.Figure(data=data, layout=layout)

    fig_template = fig_kwargs.get("fig_template", "simple_white")
    fig.update_layout(template=fig_template)
    fig.update_layout(showlegend=True)

    fig_size = fig_kwargs.get("fig_size", None)
    if fig_size is not None:
        fig.update_layout(
            autosize=False, width=fig_size[0], height=fig_size[1],
        )

    if show:
        fig.show()

    if return_data:
        return fig, data, predictions, upper_interval, lower_interval
    else:
        return fig
