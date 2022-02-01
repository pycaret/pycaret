import warnings
from typing import Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


import plotly.express as px
import plotly.graph_objects as go

from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import pacf, acf


def time_series_subplot(
    fig: go.Figure, data: pd.Series, row: int, col: int, name: Optional[str] = None
) -> go.Figure:
    """Function to add a time series to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the time series needs to be added
    data : pd.Series
        Time Series that needs to be added
    row : int
        Row of the figure where the plot needs to be inserted. Starts from 1.
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.
    name : Optional[str], optional
        Name to show when hovering over plot, by default None

    Returns
    -------
    go.Figure
        Returns back the plotly figure with time series inserted
    """
    x = (
        data.index.to_timestamp()
        if isinstance(data.index, pd.PeriodIndex)
        else data.index
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data.values,
            line=dict(color="#1f77b4", width=2),
            mode="lines+markers",
            name=name,
            marker=dict(size=5,),
        ),
        row=row,
        col=col,
    )
    return fig


def corr_subplot(
    fig: go.Figure,
    data: pd.Series,
    row: int,
    col: int,
    name: Optional[str] = None,
    plot_acf: bool = True,
) -> Tuple[go.Figure, Tuple[np.ndarray, np.ndarray]]:
    """Function to add ACF to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the ACF needs to be added
    data : pd.Series
        Data whose ACF needs to be computed
    row : int
        Row of the figure where the plot needs to be inserted. Starts from 1.
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.
    name : Optional[str], optional
        Name to show when hovering over plot, by default None
    plot_acf : bool, optional
        If True, plots the ACF, else plots the PACF, by default True

    Returns
    -------
    Tuple[go.Figure, Tuple[np.ndarray, np.ndarray]]
        Returns back the plotly figure with ACF inserted along with the ACF data.
    """

    if plot_acf:
        corr_array = acf(data, alpha=0.05)
    else:
        corr_array = pacf(data, alpha=0.05)

    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    default_name = "ACF" if plot_acf else "PACF"
    name = name or default_name

    #### Add corr plot stick lines ----
    [
        fig.add_scatter(
            x=(x, x),
            y=(0, corr_array[0][x]),
            mode="lines",
            line_color="#3f3f3f",
            row=row,
            col=col,
            name=name,
        )
        for x in range(len(corr_array[0]))
    ]

    #### Add corr plot stick endpoints ----
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=corr_array[0],
        mode="markers",
        marker_color="#1f77b4",
        marker_size=6,
        row=row,
        col=col,
        name=name,
    )

    #### Add Upper and Lower Confidence Interval ----
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=upper_y,
        mode="lines",
        line_color="rgba(255,255,255,0)",
        row=row,
        col=col,
        name="UC",
    )
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=lower_y,
        mode="lines",
        fillcolor="rgba(32, 146, 230,0.3)",
        fill="tonexty",
        line_color="rgba(255,255,255,0)",
        row=row,
        col=col,
        name="LC",
    )
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, 42], row=row, col=col)
    fig.update_yaxes(zerolinecolor="#000000", row=row, col=col)
    fig.update_xaxes(title_text="Lags", row=row, col=col)
    fig.update_yaxes(title_text=default_name, row=row, col=col)
    return fig, corr_array


def qq_subplot(
    fig: go.Figure, data: pd.Series, row: int, col: int, name: Optional[str] = None,
) -> Tuple[go.Figure, List[matplotlib.lines.Line2D]]:
    """Function to add QQ plot to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the ACF needs to be added
    data : pd.Series
        Data whose ACF needs to be computed
    row : int
        Row of the figure where the plot needs to be inserted. Starts from 1.
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.
    name : Optional[str], optional
        Name to show when hovering over plot, by default None

    Returns
    -------
    Tuple[go.Figure, List[matplotlib.lines.Line2D]]
        Returns back the plotly figure with QQ plot inserted along with the QQ
        plot data.
    """
    matplotlib.use("Agg")
    qqplot_data = qqplot(data, line="s")
    plt.close(qqplot_data)
    qqplot_data = qqplot_data.gca().lines

    name = name or data.name

    fig.add_trace(
        {
            "type": "scatter",
            "x": qqplot_data[0].get_xdata(),
            "y": qqplot_data[0].get_ydata(),
            "mode": "markers",
            "marker": {"color": "#1f77b4"},
            "name": name,
        },
        row=row,
        col=col,
    )

    fig.add_trace(
        {
            "type": "scatter",
            "x": qqplot_data[1].get_xdata(),
            "y": qqplot_data[1].get_ydata(),
            "mode": "lines",
            "line": {"color": "#3f3f3f"},
            "name": name,
        },
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text="Theoretical Quantities", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantities", row=2, col=2)
    return fig, qqplot_data


def dist_subplot(fig: go.Figure, data: pd.Series, row: int, col: int,) -> go.Figure:
    """Function to add a histogram to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the ACF needs to be added
    data : pd.Series
        Data whose ACF needs to be computed
    row : int
        Row of the figure where the plot needs to be inserted. Starts from 1.
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.
    name : Optional[str], optional
        Name to show when hovering over plot, by default None

    Returns
    -------
    go.Figure
        Returns back the plotly figure with histogram inserted.
    """

    temp_fig = px.histogram(data, color_discrete_sequence=["#1f77b4"])
    fig.add_trace(temp_fig.data[0], row=row, col=col)

    fig.update_xaxes(title_text="Range of Values", row=row, col=col)
    fig.update_yaxes(title_text="PDF", row=row, col=col)
    return fig

