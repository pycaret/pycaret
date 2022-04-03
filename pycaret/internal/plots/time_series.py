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
    return_frequency_components,
    _update_fig_dimensions,
    _get_subplot_rows_cols,
    _resolve_hoverinfo,
)

__author__ = ["satya-pattnaik", "ngupta23"]

PlotReturnType = Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]


def _get_plot(
    plot: str,
    fig_defaults: Dict[str, Any],
    data: Optional[pd.Series] = None,
    train: Optional[pd.Series] = None,
    test: Optional[pd.Series] = None,
    X: Optional[pd.DataFrame] = None,
    predictions: Optional[List[pd.DataFrame]] = None,
    cv: Optional[Union[ExpandingWindowSplitter, SlidingWindowSplitter]] = None,
    model_names: Optional[List[str]] = None,
    return_pred_int: bool = False,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """TODO: Fill"""
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
        train=train,
        test=test,
        X=X,
    )

    if plot == "ts":
        fig, plot_data = plot_series(
            data=data,
            fig_defaults=fig_defaults,
            X=X,
            hoverinfo=hoverinfo,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "train_test_split":
        fig, plot_data = plot_splits_train_test_split(
            train=train,
            test=test,
            fig_defaults=fig_defaults,
            data_kwargs=data_kwargs,
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

    elif plot == "decomp":
        fig, plot_data = plot_time_series_decomposition(
            data=data,
            fig_defaults=fig_defaults,
            model_name=model_names,
            plot="decomp",
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )

    elif plot == "decomp_stl":
        fig, plot_data = plot_time_series_decomposition(
            data=data,
            fig_defaults=fig_defaults,
            model_name=model_names,
            plot="decomp_stl",
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )

    elif plot == "acf":
        fig, plot_data = plot_acf(
            data=data,
            fig_defaults=fig_defaults,
            model_name=model_names,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "pacf":
        fig, plot_data = plot_pacf(
            data=data,
            fig_defaults=fig_defaults,
            model_name=model_names,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "diagnostics":
        fig, plot_data = plot_diagnostics(
            data=data,
            fig_defaults=fig_defaults,
            model_name=model_names,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "residuals":
        fig, plot_data = plot_series(
            data=data,
            fig_defaults=fig_defaults,
            X=X,
            hoverinfo=hoverinfo,
            model_name=model_names,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot in ["forecast", "insample"]:
        if return_pred_int:
            fig, plot_data = plot_predictions_with_confidence(
                data=data,
                predictions=predictions,
                fig_defaults=fig_defaults,
                model_names=model_names,
                data_kwargs=data_kwargs,
                fig_kwargs=fig_kwargs,
            )
        else:
            fig, plot_data = plot_predictions(
                data=data,
                predictions=predictions,
                fig_defaults=fig_defaults,
                type_=plot,
                model_names=model_names,
                data_kwargs=data_kwargs,
                fig_kwargs=fig_kwargs,
            )
    elif plot == "diff":
        fig, plot_data = plot_time_series_differences(
            data=data,
            fig_defaults=fig_defaults,
            model_name=model_names,
            hoverinfo=hoverinfo,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "periodogram" or plot == "fft":
        fig, plot_data = plot_frequency_components(
            data=data,
            fig_defaults=fig_defaults,
            model_name=model_names,
            plot=plot,
            hoverinfo=hoverinfo,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "ccf":
        fig, plot_data = plot_ccf(
            data=data,
            X=X,
            fig_defaults=fig_defaults,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    else:
        raise ValueError(f"Plot: '{plot}' is not supported.")

    return fig, plot_data


def plot_series(
    data: pd.Series,
    fig_defaults: Dict[str, Any],
    X: Optional[pd.DataFrame] = None,
    hoverinfo: Optional[str] = "text",
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the original time series or residuals"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    time_series_name = data.name
    if model_name is not None:
        title = "Residuals"
    else:
        title = "Time Series"
        if time_series_name is not None:
            title = f"{title} | Target = {time_series_name}"

    if X is not None:
        # Exogenous Variables present (predictions).
        plot_data = pd.concat([data, X], axis=1)
    else:
        # Exogenous Variables not present (Original Time series or residuals).
        if isinstance(data, pd.Series):
            if model_name is None:
                # Original Time series
                plot_data = pd.DataFrame(data)
            else:
                # Residual
                plot_data = pd.DataFrame(data)
                plot_data.columns = [f"Residuals | {model_name}"]

    rows = plot_data.shape[1]
    subplot_titles = plot_data.columns

    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    for i, col_name in enumerate(plot_data.columns):
        fig = time_series_subplot(
            fig=fig,
            data=plot_data[col_name],
            row=i + 1,
            col=1,
            hoverinfo=hoverinfo,
            name=col_name,
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
        "data": plot_data,
    }

    return fig, return_data_dict


def plot_splits_train_test_split(
    train: pd.Series,
    test: pd.Series,
    fig_defaults: Dict[str, Any],
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
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
            x=x, y=train, mode="lines+markers", marker_color="#1f77b4", name="Train"
        )
    )

    x = (
        test.index.to_timestamp()
        if isinstance(test.index, pd.PeriodIndex)
        else test.index
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=test, mode="lines+markers", marker_color="#FFA500", name="Test"
        )
    )

    with fig.batch_update():
        fig.update_layout(
            {
                "title": "Train Test Split",
                "xaxis": {"title": "Time", "zeroline": False},
                "yaxis": {"title": "Values"},
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

    return_data_dict = {
        "train": train,
        "test": test,
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


def plot_acf(
    data: pd.Series,
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the ACF on the data provided

    Parameters
    ----------
    data : pd.Series
        Data whose correlation plot needs to be plotted
    fig_defaults : Dict[str, Any]
        The defaults dictionary containing keys for "width" and "height" (mandatory)
    model_name : Optional[str]
        If the correlation plot is for model residuals, then, model_name must be 
        passed for proper display of results. If the correlation plot is for the 
        original data, model_name should be left None (name is derived from the 
        data passed in this case).
    data_kwargs : Dict[str, Any]
        A dictionary containing options keys for "nlags"
    fig_kwargs : Dict[str, Any]
        A dictionary containing options keys for "width" and/or "height"

    Returns
    -------
    Tuple[go.Figure, Dict[str, Any]]
        Returns back the plotly figure along with the correlation data.
    """
    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    nlags = data_kwargs.get("nlags", None)

    subplots = make_subplots(rows=1, cols=1)
    fig, acf_data = corr_subplot(
        fig=subplots, data=data, col=1, row=1, plot="acf", nlags=nlags
    )

    time_series_name = data.name
    title = "Autocorrelation (ACF)"
    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif time_series_name is not None:
        title = f"{title} | {time_series_name}"

    with fig.batch_update():
        fig.update_xaxes(title_text="Lags", row=1, col=1)
        fig.update_yaxes(title_text="ACF", row=1, col=1)
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=False, template=template)
        fig.update_traces(marker={"size": 10})
        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {
        "acf": acf_data
    }

    return fig, return_data_dict


def plot_pacf(
    data: pd.Series,
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the PACF on the data provided

    Parameters
    ----------
    data : pd.Series
        Data whose correlation plot needs to be plotted
    fig_defaults : Dict[str, Any]
        The defaults dictionary containing keys for "width" and "height" (mandatory)
    model_name : Optional[str]
        If the correlation plot is for model residuals, then, model_name must be 
        passed for proper display of results. If the correlation plot is for the 
        original data, model_name should be left None (name is derived from the 
        data passed in this case).
    data_kwargs : Dict[str, Any]
        A dictionary containing options keys for "nlags"
    fig_kwargs : Dict[str, Any]
        A dictionary containing options keys for "width" and/or "height"

    Returns
    -------
    Tuple[go.Figure, Dict[str, Any]]
        Returns back the plotly figure along with the correlation data.
    """
    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    nlags = data_kwargs.get("nlags", None)

    subplots = make_subplots(rows=1, cols=1)
    fig, pacf_data = corr_subplot(
        fig=subplots, data=data, col=1, row=1, plot="pacf", nlags=nlags
    )

    time_series_name = data.name
    title = "Partial Autocorrelation (PACF)"
    if model_name is not None:
        title = f"{title} | '{model_name}' Residuals"
    elif time_series_name is not None:
        title = f"{title} | {time_series_name}"

    with fig.batch_update():
        fig.update_xaxes(title_text="Lags", row=1, col=1)
        fig.update_yaxes(title_text="PACF", row=1, col=1)
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=False, template=template)
        fig.update_traces(marker={"size": 10})
        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {
        "pacf": pacf_data
    }

    return fig, return_data_dict


def plot_predictions(
    data: pd.Series,
    predictions: List[pd.DataFrame],
    type_: str,
    fig_defaults: Dict[str, Any],
    model_names: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the original data and the predictions provided"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    key = "Out-of-Sample" if type_ == "forecast" else "In-Sample"
    title = f"Actual vs. '{key}' Forecast"
    time_series_name = data.name
    if time_series_name is not None:
        title = f"{title} | {time_series_name}"

    prediction_plot_data = []
    for i, prediction in enumerate(predictions):
        # Insample predictions can be None for some of the models ----
        if prediction is not None:
            x = (
                prediction.index.to_timestamp()
                if isinstance(prediction.index, pd.PeriodIndex)
                else prediction.index
            )

            mean = go.Scatter(
                name=f"Forecast | {model_names[i]}",
                x=x,
                y=prediction["y_pred"].values,
                mode="lines+markers",
                # line=dict(color="#1f77b4"),
                marker=dict(size=5),
                showlegend=True,
            )
            prediction_plot_data.append(mean)

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

    data_for_fig = prediction_plot_data + [original]

    layout = go.Layout(
        yaxis=dict(title="Values"), xaxis=dict(title="Time"), title=title
    )

    fig = go.Figure(data=data_for_fig, layout=layout)

    with fig.batch_update():
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(template=template)

        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )
        fig.update_layout(showlegend=True)

    return_data_dict = {
        "data": data,
        "predictions": predictions,
    }

    return fig, return_data_dict


def plot_diagnostics(
    data: pd.Series,
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    hoverinfo: Optional[str] = "text",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
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
        x_title=title,
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
    fig = time_series_subplot(fig=fig, data=data, row=1, col=1, hoverinfo=hoverinfo)
    fig, periodogram_data = frequency_components_subplot(
        fig=fig,
        data=data,
        row=1,
        col=2,
        hoverinfo=hoverinfo,
        type="periodogram",
    )

    # ROW 2
    fig = dist_subplot(fig=fig, data=data, row=2, col=1)
    fig, qqplot_data = qq_subplot(fig=fig, data=data, row=2, col=2)

    # ROW 3
    fig, acf_data = corr_subplot(fig=fig, data=data, row=3, col=1, plot="acf")
    fig, pacf_data = corr_subplot(fig=fig, data=data, row=3, col=2, plot="pacf")

    return_data_dict = {
        "data": data,
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
    fig, return_data_dict = None, None

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
    data: pd.Series,
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    plot: str = "decomp",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    fig, return_data_dict = None, None

    if not isinstance(data.index, (pd.PeriodIndex, pd.DatetimeIndex)):
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
        title_name = f"Classical Decomposition ({classical_decomp_type})"
    elif plot == "decomp_stl":
        title_name = "STL Decomposition"

    if model_name is None:
        title = f"{title_name}" if data.name is None else f"{title_name} | {data.name}"
    else:
        title = f"{title_name} | '{model_name}' Residuals"
    title = title + f"<br>Seasonal Period = {period}"

    decomp_result = None
    data_ = data.to_timestamp() if isinstance(data.index, pd.PeriodIndex) else data

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
        "data": data,
        "seasonal": decomp_result.seasonal,
        "trend": decomp_result.trend,
        "resid": decomp_result.resid,
    }

    return fig, return_data_dict


def plot_time_series_differences(
    data: pd.Series,
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    hoverinfo: Optional[str] = "text",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    order_list = data_kwargs.get("order_list", None)
    lags_list = data_kwargs.get("lags_list", None)

    plot_acf = data_kwargs.get("acf", False)
    plot_pacf = data_kwargs.get("pacf", False)
    plot_periodogram = data_kwargs.get("periodogram", False)
    plot_fft = data_kwargs.get("fft", False)

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
        fig = time_series_subplot(
            fig=fig,
            data=subplot_data,
            row=i + 1,
            col=plot_cols[0],
            hoverinfo=hoverinfo,
            name=name_list[i],
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
                type="periodogram",
            )

        if plot_fft:
            fig, fft_data = frequency_components_subplot(
                fig=fig,
                data=subplot_data,
                row=i + 1,
                col=plot_cols[4],
                hoverinfo=hoverinfo,
                name=name_list[i],
                type="fft",
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
        "data": data,
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
    data: pd.Series,
    fig_defaults: Dict[str, Any],
    model_name: Optional[str] = None,
    plot: str = "periodogram",
    hoverinfo: Optional[str] = "text",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the frequency components in the data"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    if plot == "periodogram":
        title = "Periodogram"
    elif plot == "fft":
        title = "FFT"
    elif plot == "welch":
        title = "FFT"

    time_series_name = data.name
    if model_name is not None:
        legend = f"Residuals | {model_name}"
    else:
        if time_series_name is not None:
            title = f"{title} | {time_series_name}"
        legend = "Time Series"

    x, y = return_frequency_components(data=data, type=plot)
    time_period = [round(1 / freq, 4) for freq in x]
    freq_data = pd.DataFrame({"Freq": x, "Amplitude": y, "Time Period": time_period})

    spectral_density = go.Scattergl(
        name=legend,
        x=freq_data["Freq"],
        y=freq_data["Amplitude"],
        customdata=freq_data.to_numpy(),
        hovertemplate="Freq:%{customdata[0]:.4f} <br>Ampl:%{customdata[1]:.4f}<br>Time Period: %{customdata[2]:.4f]}",
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=5),
        showlegend=True,
        hoverinfo=hoverinfo,
    )
    plot_data = [spectral_density]

    layout = go.Layout(
        yaxis=dict(title="dB"), xaxis=dict(title="Frequency"), title=title
    )

    fig = go.Figure(data=plot_data, layout=layout)

    with fig.batch_update():

        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(template=template)
        fig.update_layout(showlegend=True)

        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )

    return_data_dict = {"freq_data": freq_data}

    return fig, return_data_dict


def plot_ccf(
    data: pd.Series,
    X: pd.DataFrame,
    fig_defaults: Dict[str, Any],
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the Cross Correlation between the data and the exogenous variables X"""
    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    title = "Cross Correlation Plot(s)"

    plot_data = pd.concat([data, X], axis=1)

    # Decide the number of rows and columns ----
    num_subplots = plot_data.shape[1]
    rows = _resolve_dict_keys(dict_=fig_kwargs, key="rows", defaults=fig_defaults)
    cols = _resolve_dict_keys(dict_=fig_kwargs, key="cols", defaults=fig_defaults)
    rows, cols = _get_subplot_rows_cols(num_subplots=num_subplots, rows=rows, cols=cols)

    subplot_titles = []
    for i, col_name in enumerate(plot_data.columns):
        subplot_title = f"{data.name} vs. {col_name}"
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
            data=[data, plot_data[col_name]],
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
