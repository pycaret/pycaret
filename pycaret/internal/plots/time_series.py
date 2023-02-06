from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)

from pycaret.internal.plots.utils.time_series import (
    PlotReturnType,
    _clean_model_results_labels,
    _get_subplot_rows_cols,
    _plot_fig_update,
    _resolve_hoverinfo,
    _update_fig_dimensions,
    corr_subplot,
    decomp_subplot,
    dist_subplot,
    frequency_components_subplot,
    plot_original_with_overlays,
    qq_subplot,
    time_series_subplot,
)
from pycaret.utils.generic import _resolve_dict_keys
from pycaret.utils.time_series import get_diffs

__author__ = ["satya-pattnaik", "ngupta23"]


msg1 = (
    "Both model_labels and data_label can not be None. Please specify one based "
    "on where the results are coming from (model or data respectively)."
    "\nIf you believe that this error should not be raised, please file an issue "
    "on GitHub with a reproducible example:"
    "\nhttps://github.com/pycaret/pycaret/issues"
)

msg2 = (
    "model_labels can not be None. Please specify correct value for model_labels."
    "\nIf you believe that this error should not be raised, please file an issue "
    "on GitHub with a reproducible example:"
    "\nhttps://github.com/pycaret/pycaret/issues"
)


def _get_plot(
    plot: str,
    fig_defaults: Dict[str, Any],
    data: Optional[pd.DataFrame] = None,
    data_label: Optional[str] = None,
    X: Optional[List[pd.DataFrame]] = None,
    X_labels: Optional[List[str]] = None,
    cv: Optional[Union[ExpandingWindowSplitter, SlidingWindowSplitter]] = None,
    model_results: Optional[List[pd.DataFrame]] = None,
    model_labels: Optional[List[str]] = None,
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
            fig_kwargs=fig_kwargs,
        )

    elif plot in ["decomp", "decomp_stl"]:
        fig, plot_data = plot_time_series_decomposition(
            data=data,
            plot=plot,
            fig_defaults=fig_defaults,
            data_kwargs=data_kwargs,
            data_label=data_label,
            model_labels=model_labels,
            fig_kwargs=fig_kwargs,
        )
    elif plot in ["acf", "pacf"]:
        fig, plot_data = plot_xacf(
            data=data,
            plot=plot,
            fig_defaults=fig_defaults,
            data_label=data_label,
            model_labels=model_labels,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot in ["periodogram", "fft"]:
        fig, plot_data = plot_frequency_components(
            data=data,
            plot=plot,
            fig_defaults=fig_defaults,
            data_label=data_label,
            model_labels=model_labels,
            hoverinfo=hoverinfo,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "diagnostics":
        fig, plot_data = plot_diagnostics(
            data=data,
            fig_defaults=fig_defaults,
            data_label=data_label,
            model_labels=model_labels,
            hoverinfo=hoverinfo,
            fig_kwargs=fig_kwargs,
        )
    elif plot == "diff":
        fig, plot_data = plot_time_series_differences(
            data=data,
            fig_defaults=fig_defaults,
            data_label=data_label,
            model_labels=model_labels,
            hoverinfo=hoverinfo,
            data_kwargs=data_kwargs,
            fig_kwargs=fig_kwargs,
        )
    elif plot in ["forecast", "insample", "residuals"]:
        if return_pred_int:
            fig, plot_data = plot_predictions_with_confidence(
                data=data,
                predictions=model_results,
                fig_defaults=fig_defaults,
                model_labels=model_labels,
                data_kwargs=data_kwargs,
                fig_kwargs=fig_kwargs,
            )
        else:
            fig, plot_data = plot_model_results(
                original_data=data,
                model_results=model_results,
                model_labels=model_labels,
                plot=plot,
                fig_defaults=fig_defaults,
                hoverinfo=hoverinfo,
                fig_kwargs=fig_kwargs,
            )
    elif plot == "ccf":
        fig, plot_data = plot_ccf(
            y=data,
            y_label=data_label,
            X=X,
            X_labels=X_labels,
            fig_defaults=fig_defaults,
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

    Parameters
    ----------
    y : pd.DataFrame
        A dataframe containing the various plot data types for the target series
        (original, imputed, transformed), the various splits (full, train, test)
        or a combination of these two. Each column in the dataframe is plotted as
        a separate overlaid series in the plot. The names of the columns are used
        in the plot legend.
    y_label : str
        The name of the target time series.
    fig_defaults : Dict[str, Any]
        The default settings for the plotly plot. Must contain keys for "width",
        "height", and "template".
    X : Optional[List[pd.DataFrame]], optional
        The exogenous variables in the data, by default None
        X contains 1 dataframe per exogenous variable. Similar to the target
        series, each dataframe can contain multiple columns corresponding to the
        various plot data types (original, imputed, transformed). Each exogenous
        variable is plotted in a separate subplot and each data type is overlaid
        within that subplot. The names of the columns are used in the plot legend.
    X_labels : Optional[List[str]], optional
        The names of the exogenous variables, by default None
    hoverinfo : Optional[str], optional
        Action when hovering over the plotly plot, by default "text"
    fig_kwargs : Optional[Dict], optional
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".

    Returns
    -------
    PlotReturnType
        The plotly figure object along with a dictionary with the plot data.
        The data can be returned to the user (if requested) so that the plot can
        be and customized by them if not supported by pycaret by default.

    Raises
    ------
    ValueError
        When exogenous variables are passed, but their corresponding names are
        not passed.
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

    fig = _plot_fig_update(fig, title, fig_defaults, fig_kwargs, show_legend=True)

    return_data_dict = {
        "data": plot_data,
    }

    return fig, return_data_dict


def plot_cv(
    data: pd.Series,
    cv: Union[ExpandingWindowSplitter, SlidingWindowSplitter],
    fig_defaults: Dict[str, Any],
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the cv splits used on the training split

    Parameters
    ----------
    data : pd.Series
        The target time series
    cv : Union[ExpandingWindowSplitter, SlidingWindowSplitter]
        The sktime compatible cross validation object to use for the plot
    fig_defaults : Dict[str, Any]
        The default settings for the plotly plot. Must contain keys for "width",
        "height", and "template".
    fig_kwargs : Optional[Dict], optional
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".

    Returns
    -------
    PlotReturnType
        The plotly figure object along with a dictionary with the plot data.
        The data can be returned to the user (if requested) so that the plot can
        be and customized by them if not supported by pycaret by default.
    """
    fig_kwargs = fig_kwargs or {}

    def get_windows(y, cv):
        """
        Generate windows
        Inspired from `https://github.com/sktime/sktime`
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
                fig.add_trace(
                    go.Scattergl(
                        x=(time_stamps[i], time_stamps[i + 1]),
                        y=(y_axis_label, y_axis_label),
                        mode="lines+markers",
                        line_color="#C0C0C0",
                        name="Unchanged",
                        hoverinfo="skip",
                    ),
                )
                for i in range(len(data) - 1)
            ]
            [
                fig.add_trace(
                    go.Scattergl(
                        x=(time_stamps[i], time_stamps[i + 1]),
                        y=(y_axis_label, y_axis_label),
                        mode="lines+markers",
                        line_color="#1f77b4",
                        name="Train",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                )
                for i in train_windows[num_window][:-1]
            ]
            [
                fig.add_trace(
                    go.Scattergl(
                        x=(time_stamps[i], time_stamps[i + 1]),
                        y=(y_axis_label, y_axis_label),
                        mode="lines+markers",
                        line_color="#DE970B",
                        name="ForecastHorizon",
                        hoverinfo="skip",
                    ),
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
    data: pd.DataFrame,
    plot: str,
    fig_defaults: Dict[str, Any],
    data_label: Optional[str] = None,
    model_labels: Optional[List[str]] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the ACF or PACF plot for the target time series (could be any data
    type of the data - original, imputed, transformed) or model residuals.

    Parameters
    ----------
    data : pd.DataFrame
        Data whose correlation plot needs to be plotted. If based on the original
        data, it can contain multiple columns corresponding to the various data
        types of the targets (original, imputed, transformed). If it is based on
        a estimator(s), it can contain residuals from multiple models (one column
        per model). The column names must correspond to the model labels in this
        case.
    plot : str
        Type of plot, allowed values are ["acf", "pacf"]
    fig_defaults : Dict[str, Any]
        The default settings for the plotly plot. Must contain keys for "width",
        "height", and "template".
    data_label : Optional[str], optional
        If data is from the original target series, then the name of the target
        series, by default None.
    model_labels : Optional[List[str]], optional
        If data is from model residuals, then the name(s) of the model(s),
        by default None. At least one of data_label or model_labels must be provided.
    data_kwargs : Dict[str, Any]
        A dictionary containing options keys for "nlags"
    fig_kwargs : Dict[str, Any]
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".

    Returns
    -------
    PlotReturnType
        The plotly figure object along with a dictionary with the plot data.
        The data can be returned to the user (if requested) so that the plot can
        be and customized by them if not supported by pycaret by default.

    Raises
    ------
    ValueError
        (1) When the plot type is not supported
        (2) When both data_label and model_labels are None
    """
    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    ncols = len(data.columns)
    fig = make_subplots(
        rows=1,
        cols=ncols,
        column_titles=data.columns.tolist(),
        shared_yaxes=True,
    )

    all_plot_data = {}
    nlags = data_kwargs.get("nlags", None)
    for i, col_name in enumerate(data.columns):
        fig, plot_data = corr_subplot(
            fig=fig, data=data[col_name], row=1, col=i + 1, plot=plot, nlags=nlags
        )
        all_plot_data.update({col_name: plot_data})

    if plot == "acf":
        title = "Autocorrelation (ACF)"
    elif plot == "pacf":
        title = "Partial Autocorrelation (PACF)"
    else:
        raise ValueError(f"Plot '{plot}' is not supported by plot_xacf().")

    if model_labels is not None:
        title = f"{title} | Model Residuals"
    elif data_label is not None:
        title = f"{title} | {data_label}"
    else:
        # Both model_labels and data_label are None
        raise ValueError(msg1)

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
    model_labels: List[str],
    plot: str,
    fig_defaults: Dict[str, Any],
    hoverinfo: Optional[str] = "text",
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the original target series along with overlaid model results which
    could be out of sample predictions, insample predictions or residuals.

    Parameters
    ----------
    original_data : pd.Series
        The original target series to be plotted.
    model_results : List[pd.DataFrame]
        The model results that must be overlaid over the original data. e.g. out
        of sample predictions, insample predictions or residuals from possibly
        multiple models. Each column in the overlay_data is overlaid as a separate
        series over the original_data. The column names are used as labels for the
        overlaid data.
    model_labels : List[str]
        The labels corresponding to the models whose results are being plotted.
    plot : str
        The type of plot.
            "forecast": Out of Sample Forecasts
            "insample": Insample Forecasts
            "residuals": Model Residuals
    fig_defaults : Dict[str, Any]
        The defaults dictionary containing keys for "width", "height", and "template".
    hoverinfo : Optional[str], optional
        Whether to display the hoverinfo in the plot, by default "text"
    fig_kwargs : Optional[Dict], optional
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".

    Returns
    -------
    PlotReturnType
        The plotly figure object along with a dictionary with the plot data.
        The data can be returned to the user (if requested) so that the plot can
        be and customized by them if not supported by pycaret by default.

    Raises
    ------
    ValueError
        When model_labels is None
    """
    if model_labels is None:
        raise ValueError(msg2)

    model_results, model_labels = _clean_model_results_labels(
        model_results=model_results, model_labels=model_labels
    )

    if plot in ["forecast", "insample"]:
        model_results = [model_result["y_pred"] for model_result in model_results]
    model_results = pd.concat(model_results, axis=1)
    model_results.columns = model_labels

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
    data: pd.DataFrame,
    fig_defaults: Dict[str, Any],
    data_label: Optional[str] = None,
    model_labels: Optional[List[str]] = None,
    hoverinfo: Optional[str] = "text",
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the diagnostic data such as ACF, Histogram, QQ plot on the data
    provided. Data could be the target series (original, imputed or transformed),
    or the residuals from a model. If target series is provided, only one data
    type is supported at a time.

    Parameters
    ----------
    data : pd.DataFrame
        Data whose diagnostic plot needs to be plotted.
    fig_defaults : Dict[str, Any]
        The default settings for the plotly plot. Must contain keys for "width",
        "height", and "template".
    data_label : Optional[str], optional
        If data is from the original target series, then the name of the target
        series, by default None.
    model_labels : Optional[List[str]], optional
        If data is from model residuals, then the name(s) of the model(s),
        by default None. At least one of data_label or model_labels must be provided.
    hoverinfo : Optional[str], optional
        Action when hovering over the plotly plot, by default "text"
    fig_kwargs : Optional[Dict], optional
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".

    Returns
    -------
    PlotReturnType
        The plotly figure object along with a dictionary with the plot data.
        The data can be returned to the user (if requested) so that the plot can
        be and customized by them if not supported by pycaret by default.

    Raises
    ------
    ValueError
        (1) When the data contains more than 1 column indicating more than 1 data type.
        (2) When both data_label and model_labels are None
    """
    if data.shape[1] != 1:
        raise ValueError(
            "plot_diagnostics() only works on a single time series, "
            f"but {data.shape[1]} series were provided."
        )

    title = "Diagnostics"
    if model_labels is not None:
        title = f"{title} | Model Residuals"
        subplot_title = f"{model_labels[0]} Residuals"
    elif data_label is not None:
        title = f"{title} | {data_label}"
        # Do not use data_label for column titles since the actual column name
        # has the more detailed data_type (e.g. with "transformed" at the end.)
        subplot_title = data.columns[0]
    else:
        # Both model_labels and data_label are None
        raise ValueError(msg1)

    data_series = data.iloc[:, 0]

    fig_kwargs = fig_kwargs or {}

    fig = make_subplots(
        rows=3,
        cols=2,
        row_heights=[0.33, 0.33, 0.33],
        subplot_titles=[
            subplot_title,
            "Periodogram",
            "Histogram",
            "Q-Q Plot",
            "ACF",
            "PACF",
        ],
    )

    fig = _plot_fig_update(fig, title, fig_defaults, fig_kwargs)

    # Add diagnostic plots ----

    # ROW 1
    fig = time_series_subplot(fig=fig, data=data, row=1, col=1, hoverinfo=hoverinfo)
    fig, periodogram_data = frequency_components_subplot(
        fig=fig,
        data=data_series,
        row=1,
        col=2,
        hoverinfo=hoverinfo,
        plot="periodogram",
    )

    # ROW 2
    fig = dist_subplot(fig=fig, data=data_series, row=2, col=1)
    fig, qqplot_data = qq_subplot(fig=fig, data=data_series, row=2, col=2)

    # ROW 3
    fig, acf_data = corr_subplot(fig=fig, data=data_series, row=3, col=1, plot="acf")
    fig, pacf_data = corr_subplot(fig=fig, data=data_series, row=3, col=2, plot="pacf")

    fig = _plot_fig_update(fig, title, fig_defaults, fig_kwargs)

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
    model_labels: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the original data and the predictions provided with confidence
    TODO: Refactor and fill docstring
    """
    if len(predictions) != 1:
        raise ValueError(
            "Plotting with predictions only supports one estimator. Please pass only one estimator to fix"
        )

    preds = predictions[0]["y_pred"]
    upper_interval = predictions[0]["upper"]
    lower_interval = predictions[0]["lower"]
    model_label = model_labels[0]

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
        name=f"Prediction Interval | {model_label}",  # Changed since we use only 1 legend
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
        name=f"Forecast | {model_label}",
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

    fig = _plot_fig_update(fig, title, fig_defaults, fig_kwargs, show_legend=True)

    return_data_dict = {
        "data": data,
        "predictions": predictions,
        "upper_interval": upper_interval,
        "lower_interval": lower_interval,
    }

    return fig, return_data_dict


def plot_time_series_decomposition(
    data: pd.DataFrame,
    plot: str,
    fig_defaults: Dict[str, Any],
    data_kwargs: Dict[str, Any],
    data_label: Optional[str] = None,
    model_labels: Optional[List[str]] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the Decomposition for the target time series (could be any data
    type of the data - original, imputed, transformed) or model residuals.

    Parameters
    ----------
    data : pd.DataFrame
        Data whose decomposition plot needs to be plotted. If based on the original
        data, it can contain multiple columns corresponding to the various data
        types of the targets (original, imputed, transformed). If it is based on
        a estimator(s), it can contain residuals from multiple models (one column
        per model). The column names must correspond to the model labels in this
        case.
    plot : str, optional
        Type of plot, allowed values are ["decomp", "decomp_stl"]
    fig_defaults : Dict[str, Any]
        The default settings for the plotly plot. Must contain keys for "width",
        "height", and "template".
    data_kwargs : Dict[str, Any]
        A dictionary containing mandatory keys for "seasonal_period"
    data_label : Optional[str], optional
        If data is from the original target series, then the name of the target
        series, by default None.
    model_labels : Optional[List[str]], optional
        If data is from model residuals, then the name(s) of the model(s),
        by default None. At least one of data_label or model_labels must be provided.
    fig_kwargs : Optional[Dict], optional
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".

    Returns
    -------
    PlotReturnType
        The plotly figure object along with a dictionary with the plot data.
        The data can be returned to the user (if requested) so that the plot can
        be and customized by them if not supported by pycaret by default.

    Raises
    ------
    ValueError
        (1) If seasonal_period is not passed through data_kwargs
        (2) When both data_label and model_labels are None
    """
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

    # Check period ----
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

    ncols = len(data.columns)
    fig = make_subplots(
        rows=4,
        cols=ncols,
        column_titles=data.columns.tolist(),
        row_titles=["Actual", "Seasonal", "Trend", "Residual"],
        shared_xaxes=True,
        shared_yaxes=True,
    )

    classical_decomp_type = data_kwargs.get("type", "additive")
    fig_kwargs = fig_kwargs or {}

    if plot == "decomp":
        title = f"Classical Decomposition ({classical_decomp_type})"
    elif plot == "decomp_stl":
        title = "STL Decomposition"

    if model_labels is not None:
        title = f"{title} | Model Residuals"
    elif data_label is not None:
        title = f"{title} | {data_label}"
    else:
        # Both model_labels and data_label are None
        raise ValueError(msg1)

    title = title + f"<br>Seasonal Period = {period}"

    all_plot_data = {}
    for i, col_name in enumerate(data.columns):
        fig, plot_data = decomp_subplot(
            fig=fig,
            data=data[col_name],
            col=i + 1,
            plot=plot,
            classical_decomp_type=classical_decomp_type,
            period=period,
        )
        all_plot_data.update({col_name: plot_data})

    fig = _plot_fig_update(fig, title, fig_defaults, fig_kwargs)

    return_data_dict = {"data": data, plot: all_plot_data}

    return fig, return_data_dict


def plot_time_series_differences(
    data: pd.DataFrame,
    fig_defaults: Dict[str, Any],
    data_label: Optional[str] = None,
    model_labels: Optional[List[str]] = None,
    hoverinfo: Optional[str] = "text",
    data_kwargs: Optional[Dict] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the differenced data for the target time series (could be any data
    type of the data - original, imputed, transformed) or model residuals. If
    target series is provided, only one data type is supported at a time.

    Parameters
    ----------
    data : pd.DataFrame
        Data whose differenced data needs needs to be plotted.
    fig_defaults : Dict[str, Any]
        The default settings for the plotly plot. Must contain keys for "width",
        "height", and "template".
    data_label : Optional[str], optional
        If data is from the original target series, then the name of the target
        series, by default None.
    model_labels : Optional[List[str]], optional
        If data is from model residuals, then the name(s) of the model(s),
        by default None. At least one of data_label or model_labels must be provided.
    hoverinfo : Optional[str], optional
        Action when hovering over the plotly plot, by default "text"
    data_kwargs : Optional[Dict], optional
        The difference orders to use or lags to use for differencing, by default
        None which plots the original data with first differences. Only one of
        "order_list" or "lags_list" must be provided. e.g.
        "order_list" = [1, 2] will plot original, first and second differences.
        "lags_list" = [1, [1, 12] will plot original, first, and first difference
        with seasonal difference of 12.
        Additionally, user can also ask for diagnostic plots for the differences
        such as "acf", "pacf", "periodogram", "fft" by setting the value of these
        keys to True.
    fig_kwargs : Optional[Dict], optional
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".

    Returns
    -------
    PlotReturnType
        The plotly figure object along with a dictionary with the plot data.
        The data can be returned to the user (if requested) so that the plot can
        be and customized by them if not supported by pycaret by default.

    Raises
    ------
    ValueError
        (1) When the data contains more than 1 column indicating more than 1 data type.
        (2) When both data_label and model_labels are None
    """
    if data.shape[1] != 1:
        raise ValueError(
            "plot_time_series_differences() only works on a single time series, "
            f"but {data.shape[1]} series were provided."
        )

    title = "Difference Plot"
    if model_labels is not None:
        title = f"{title} | Model Residuals"
        column_titles = [f"{model_labels[0]} Residuals"]
    elif data_label is not None:
        title = f"{title} | {data_label}"
        # Do not use data_label for column titles since the actual column name
        # has the more detailed data_type (e.g. with "transformed" at the end.)
        column_titles = [data.columns[0]]
    else:
        # Both model_labels and data_label are None
        raise ValueError(msg1)

    data_series = data.iloc[:, 0]

    fig, return_data_dict = None, None

    data_kwargs = data_kwargs or {}
    fig_kwargs = fig_kwargs or {}

    order_list = data_kwargs.get("order_list", None)
    lags_list = data_kwargs.get("lags_list", None)

    plot_acf = data_kwargs.get("acf", False)
    plot_pacf = data_kwargs.get("pacf", False)
    plot_periodogram = data_kwargs.get("periodogram", False)
    plot_fft = data_kwargs.get("fft", False)

    diff_list, name_list = get_diffs(
        data=data_series, order_list=order_list, lags_list=lags_list
    )

    if len(diff_list) == 0:
        # Issue with reconciliation of orders and diffs
        return fig, return_data_dict

    diff_list = [data_series] + diff_list
    name_list = ["Actual" if model_labels is None else "Residuals"] + name_list

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

        # Add difference data ----

        ts_to_plot = pd.DataFrame(subplot_data)
        ts_to_plot.columns = [name_list[i]]
        fig = time_series_subplot(
            fig=fig,
            data=ts_to_plot,
            row=i + 1,
            col=plot_cols[0],
            hoverinfo=hoverinfo,
        )

        # Add diagnostics if requested ----
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

    fig = _plot_fig_update(fig, title, fig_defaults, fig_kwargs)

    return_data_dict = {
        "data": data_series,
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
    plot: str,
    fig_defaults: Dict[str, Any],
    data_label: Optional[str] = None,
    model_labels: Optional[List[str]] = None,
    hoverinfo: Optional[str] = "text",
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the frequency components for the data provided (could be any data
    type of the data - original, imputed, transformed) or model residuals.

    Parameters
    ----------
    data : pd.Series
        Data whose frequency components needs to be plotted. If based on the original
        data, it can contain multiple columns corresponding to the various data
        types of the targets (original, imputed, transformed). If it is based on
        a estimator(s), it can contain residuals from multiple models (one column
        per model). The column names must correspond to the model labels in this
        case.
    plot : str
        Type of plot, allowed values are ["periodogram", "fft", "welch"]
    fig_defaults : Dict[str, Any]
        The default settings for the plotly plot. Must contain keys for "width",
        "height", and "template".
    data_label : Optional[str], optional
        If data is from the original target series, then the name of the target
        series, by default None.
    model_labels : Optional[List[str]], optional
        If data is from model residuals, then the name(s) of the model(s),
        by default None. At least one of data_label or model_labels must be provided.
    hoverinfo : Optional[str], optional
        Action when hovering over the plotly plot, by default "text"
    fig_kwargs : Optional[Dict], optional
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".

    Returns
    -------
    PlotReturnType
        The plotly figure object along with a dictionary with the plot data.
        The data can be returned to the user (if requested) so that the plot can
        be and customized by them if not supported by pycaret by default.

    Raises
    ------
    ValueError
        (1) When the plot type is not supported.
        (2) When both data_label and model_labels are None
    """
    fig_kwargs = fig_kwargs or {}

    ncols = len(data.columns)
    fig = make_subplots(
        rows=1,
        cols=ncols,
        column_titles=data.columns.tolist(),
        shared_yaxes=True,
    )

    all_plot_data = {}
    for i, col_name in enumerate(data.columns):
        fig, plot_data = frequency_components_subplot(
            fig=fig,
            data=data[col_name],
            row=1,
            col=i + 1,
            plot=plot,
            hoverinfo=hoverinfo,
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

    if model_labels is not None:
        title = f"{title} | Model Residuals"
    elif data_label is not None:
        title = f"{title} | {data_label}"
    else:
        # Both model_labels and data_label are None
        raise ValueError(msg1)

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
    fig_defaults: Dict[str, Any],
    X: List[pd.DataFrame] = None,
    X_labels: List[str] = None,
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the Cross Correlation between the data and the exogenous variables X.
    When no exogenous variables are present, simply plots the self correlation (ACF).

    Parameters
    ----------
    y : pd.DataFrame
        A dataframe containing the a single plot data types for the target series
        (original, imputed, OR transformed).
    y_label : str
        The name of the target time series.
    fig_defaults : Dict[str, Any]
        The default settings for the plotly plot. Must contain keys for "width",
        "height", and "template".
    X : List[pd.DataFrame]
        The exogenous variables in the data, by default None
        X contains 1 dataframe per exogenous variable. Each dataframe must have only
        1 column at a time since ccf does not work on many data types at a time.
    X_labels : List[str]
        The names of the exogenous variables, by default None
    fig_kwargs : Optional[Dict], optional
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".

    Returns
    -------
    PlotReturnType
        The plotly figure object along with a dictionary with the plot data.
        The data can be returned to the user (if requested) so that the plot can
        be and customized by them if not supported by pycaret by default.

    Raises
    ------
    ValueError
        When any dataframe in y or X contains more than 1 column.
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
        # Add CCF plot ----
        fig, ccf_data = corr_subplot(
            fig=fig,
            data=[y_series, plot_data[col_name]],
            row=row,
            col=col,
            plot="ccf",
        )
        all_ccf_data.update({col_name: ccf_data})

    fig = _plot_fig_update(fig, title, fig_defaults, fig_kwargs)

    return_data_dict = {"ccf": all_ccf_data}

    return fig, return_data_dict
