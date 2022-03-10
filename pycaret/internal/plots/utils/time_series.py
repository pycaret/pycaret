from typing import Optional, List, Tuple, Union, Dict, Any
from math import ceil

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


from scipy.signal import periodogram
from scipy.fft import fft, fftfreq

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import pacf, acf, ccf

from pycaret.utils import _resolve_dict_keys


def time_series_subplot(
    fig: go.Figure,
    data: pd.Series,
    row: int,
    col: int,
    hoverinfo: Optional[str],
    name: Optional[str] = None,
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
    hoverinfo : Optional[str]
        Whether hoverinfo should be disabled or not. Options are same as plotly.
        e.g. "text" to display, "skip" or "none" to disable.
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

    name = name or data.name

    # If you add hoverinfo = "text", you must also add the hovertemplate, else no hoverinfo
    # gets displayed. OR alternately, leave it out and it gets plotted by default.
    if hoverinfo == "text":
        # Not specifying the hoverinfo will show it by default
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data.values,
                line=dict(color="#1f77b4", width=2),
                mode="lines+markers",
                name=name,
                marker=dict(size=5),
            ),
            row=row,
            col=col,
        )
    else:
        # Disable hoverinfo
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data.values,
                line=dict(color="#1f77b4", width=2),
                mode="lines+markers",
                name=name,
                marker=dict(size=5),
                hoverinfo=hoverinfo,
            ),
            row=row,
            col=col,
        )

    return fig


def corr_subplot(
    fig: go.Figure,
    data: Union[pd.Series, List[pd.Series]],
    row: int,
    col: int,
    name: Optional[str] = None,
    plot: str = "acf",
    nlags: Optional[int] = None,
) -> Tuple[go.Figure, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Function to add correlation plots to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the ACF needs to be added
    data : Union[pd.Series, List[pd.Series]]
        Data whose correlation plot needs to be plotted.
          - For ACF and PACF, this should be a Pandas Series.
          - For CCF, this should be a list of two pandas Series with the first one
            being the target, and the second one being the exogenous data against
            which the CCF has to be computed. Note that if the second series is
            the same as the first one, then this is equivalent to the ACF plot.
    row : int
        Row of the figure where the plot needs to be inserted. Starts from 1.
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.
    name : Optional[str], optional
        Name to show when hovering over plot, by default None
    plot : str, optional
        Options are
          - "acf": to plot ACF,
          - "pacf": to plot PACF
          - "ccf": to plot CCF
        by default "acf"
    nlags : Optional[int], optional
        Number of lags to plot, by default None which plots 40 lags (same as statsmodels)


    Returns
    -------
    Tuple[go.Figure, Tuple[np.ndarray, np.ndarray]]
        Returns back the plotly figure with Correlation plot inserted along with
        the correlation data.

    Raises
    ------
    ValueError
        If `plot` is not one of the allowed types
    """

    lower, upper = None, None
    if plot in ["acf", "pacf"]:
        if plot == "acf":
            default_name = "ACF"
            corr_array = acf(data, alpha=0.05)
        elif plot == "pacf":
            default_name = "PACF"
            nobs = len(data)
            nlags = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
            corr_array = pacf(data, nlags=nlags, alpha=0.05)
        corr_values = corr_array[0]
        lower = corr_array[1][:, 0] - corr_array[0]
        upper = corr_array[1][:, 1] - corr_array[0]
    elif plot == "ccf":
        default_name = "CCF"
        [target, exog] = data
        corr_values = ccf(target, exog, unbiased=False)
        # No upper and lower bounds available for CCF
    else:
        raise ValueError(
            f"plot must be one of 'acf', 'pacf' or 'ccf'. You passed '{plot}'"
        )

    if not nlags:
        nlags = 40
    corr_values = corr_values[: nlags + 1]
    lags = np.arange(len(corr_values))
    name = name or default_name

    #### Add the correlation plot ----
    fig = _add_corr_stems_subplot(
        fig=fig, corr_values=corr_values, lags=lags, name=name, row=row, col=col
    )
    if upper is not None and lower is not None:
        # Not available for CCF ----
        fig = _add_corr_bounds_subplot(
            fig=fig, lower=lower, upper=upper, row=row, col=col
        )

    with fig.batch_update():
        fig.update_traces(showlegend=False)
        fig.update_xaxes(range=[-1, len(corr_values) + 1], row=row, col=col)
        fig.update_yaxes(range=[-1.1, 1.1], zerolinecolor="#000000", row=row, col=col)
        # For compactness, the following have been removed for now
        # fig.update_xaxes(title_text="Lags", row=row, col=col)
        # fig.update_yaxes(title_text=default_name, row=row, col=col)
    return fig, (corr_values, lags, upper, lower)


def _add_corr_stems_subplot(
    fig: go.Figure,
    corr_values: np.ndarray,
    lags: np.ndarray,
    name: str,
    row: int,
    col: int,
) -> go.Figure:
    """Function to add the correlation stems (sticks) to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the stems needs to be added
    corr_values : np.ndarray
        The correlation values to plot
    lags : np.ndarray
        The lags corresponding to the correlation values
    name : str
        Name to show when hovering over plot
    row : int
        Row of the figure where the plot needs to be inserted. Starts from 1.
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.

    Returns
    -------
    go.Figure
        Returns back the plotly figure with stems inserted
    """
    #### Add corr plot stem lines ----
    [
        fig.add_scattergl(
            x=(lag, lag),
            y=(0, corr_values[lag]),
            mode="lines",
            line_color="#3f3f3f",
            row=row,
            col=col,
            name=name,
        )
        for lag in lags
    ]

    #### Add corr plot stem endpoints ----
    fig.add_scattergl(
        x=lags,
        y=corr_values,
        mode="markers",
        marker_color="#1f77b4",
        marker_size=6,
        row=row,
        col=col,
        name=name,
    )
    return fig


def _add_corr_bounds_subplot(
    fig: go.Figure, lower: np.ndarray, upper: np.ndarray, row: int, col: int
) -> go.Figure:
    """Function to add the correlation confidence bounds to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the correlation confidence bounds need to be added
    lower : np.ndarray
        Lower Confidence Interval values
    upper : np.ndarray
        Upper Confidence Interval values
    row : int
        Row of the figure where the plot needs to be inserted. Starts from 1.
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.

    Returns
    -------
    go.Figure
        Returns back the plotly figure with correlation confidence bounds inserted
    """

    # For some reason scattergl does not work here. Hence switching to scatter.
    # (refer: https://github.com/pycaret/pycaret/issues/2211).

    #### Add the Upper Confidence Interval ----
    fig.add_scatter(
        x=np.arange(len(upper)),
        y=upper,
        mode="lines",
        line_color="rgba(255,255,255,0)",
        row=row,
        col=col,
        name="UC",
    )

    #### Add the Lower Confidence Interval ----
    fig.add_scatter(
        x=np.arange(len(lower)),
        y=lower,
        mode="lines",
        fillcolor="rgba(32, 146, 230,0.3)",
        fill="tonexty",
        line_color="rgba(255,255,255,0)",
        row=row,
        col=col,
        name="LC",
    )
    return fig


def qq_subplot(
    fig: go.Figure, data: pd.Series, row: int, col: int, name: Optional[str] = None
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
            "type": "scattergl",
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
            "type": "scattergl",
            "x": qqplot_data[1].get_xdata(),
            "y": qqplot_data[1].get_ydata(),
            "mode": "lines",
            "line": {"color": "#3f3f3f"},
            "name": name,
        },
        row=row,
        col=col,
    )
    with fig.batch_update():
        fig.update_xaxes(title_text="Theoretical Quantities", row=row, col=col)
        fig.update_yaxes(title_text="Sample Quantities", row=row, col=col)
    return fig, qqplot_data


def dist_subplot(fig: go.Figure, data: pd.Series, row: int, col: int) -> go.Figure:
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

    with fig.batch_update():
        fig.update_xaxes(title_text="Range of Values", row=row, col=col)
        fig.update_yaxes(title_text="PDF", row=row, col=col)
    return fig


def return_frequency_components(
    data: pd.Series, type: str = "periodogram"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Spectral Density using the specified method.
    NOTE: Frequency = 0 is discarded

    Parameters
    ----------
    data : pd.Series
        Data for which the frequency components must be obtained
    type : str, optional
        Type of method to use to get the frequency components, by default "periodogram"
        Allowed methods are: "periodogram", "fft"

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequency, Amplitude
    """

    if type == "periodogram":
        freq, ampl = _return_periodogram(data=data)
    elif type == "fft":
        freq, ampl = _return_fft(data=data)
    else:
        raise ValueError(
            f"Getting frequency component using type: {type} is not supported."
        )

    x = freq[1:]
    y = ampl[1:]

    return x, y


def _return_periodogram(data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the Periodgram data corresponding to the `data`

    Parameters
    ----------
    data : pd.Series
        Data for which the periodgram data must be obtained

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequency, Amplitude
    """
    freq, Pxx = periodogram(
        data.values, return_onesided=True, scaling="density", window="parzen"
    )
    ampl = 10 * np.log10(Pxx)
    return freq, ampl


def _return_fft(data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the FFT data corresponding to the `data`

    Parameters
    ----------
    data : pd.Series
        Data for which the FFT data must be obtained

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequency, Amplitude
    """
    sample_spacing = 1
    fft_values = fft(data.values)
    psd_values = np.abs(fft_values) ** 2
    fft_frequencies = fftfreq(len(psd_values), 1.0 / sample_spacing)
    pos_freq_ix = fft_frequencies > 0
    freq = fft_frequencies[pos_freq_ix]
    ampl = 10 * np.log10(psd_values[pos_freq_ix])
    return freq, ampl


def frequency_components_subplot(
    fig: go.Figure,
    data: pd.Series,
    row: int,
    col: int,
    hoverinfo: Optional[str],
    type: str = "periodogram",
    name: Optional[str] = None,
) -> go.Figure:
    """Function to add a time series to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the time series needs to be added
    freq : pd.Series
        Time Series whose frequency components have to be plotted
    row : int
        Row of the figure where the plot needs to be inserted. Starts from 1.
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.
    hoverinfo : Optional[str]
        Whether hoverinfo should be disabled or not. Options are same as plotly.
        e.g. "text" to display, "skip" or "none" to disable.
    type : str, optional
        Type of method to use to get the frequency components, by default "periodogram"
        Allowed methods are: "periodogram", "fft"
    name : Optional[str], optional
        Name to show when hovering over plot, by default None

    Returns
    -------
    go.Figure
        Returns back the plotly figure with time series inserted
    """
    x, y = return_frequency_components(data=data, type=type)
    time_period = [round(1 / freq, 4) for freq in x]
    freq_data = pd.DataFrame({"Freq": x, "Amplitude": y, "Time Period": time_period})

    name = name or data.name

    # If you add hoverinfo = "text", you must also add the hovertemplate, else no hoverinfo
    # gets displayed. OR alternately, leave it out and it gets plotted by default.
    if hoverinfo == "text":
        hovertemplate = "Freq:%{customdata[0]:.4f} <br>Ampl:%{customdata[1]:.4f}<br>Time Period: %{customdata[2]:.4f]}"
        fig.add_trace(
            go.Scattergl(
                name=name,
                x=freq_data["Freq"],
                y=freq_data["Amplitude"],
                customdata=freq_data.to_numpy(),
                hovertemplate=hovertemplate,
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=5),
                showlegend=True,
                hoverinfo=hoverinfo,
            ),
            row=row,
            col=col,
        )
    else:
        # Disable hoverinfo
        fig.add_trace(
            go.Scattergl(
                name=name,
                x=freq_data["Freq"],
                y=freq_data["Amplitude"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=5),
                showlegend=True,
                hoverinfo=hoverinfo,
            ),
            row=row,
            col=col,
        )

    with fig.batch_update():
        # For compactness, the following have been removed for now
        # fig.update_xaxes(title_text="Frequency", row=row, col=col)

        # X-axis is getting messed up when acf or pacf is being plotted along with this
        # Hence setting explicitly per https://plotly.com/python/subplots/
        fig.update_xaxes(range=[0, 0.5], row=row, col=col)
    return fig, freq_data


def _update_fig_dimensions(
    fig: go.Figure, fig_kwargs: Dict[str, Any], fig_defaults: Dict[str, Any]
) -> go.Figure:
    """Updates the dimensions of the Plotly figure using fig_kwargs.
    If "width" and/or "height" are available in fig_kwargs, they are used, else
    they are picked from the fig_defaults. If both end up being None, figure
    dimensions are not updated.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure whose dimensions need to be updated
    fig_kwargs : Dict[str, Any]
        A dictionary containing options keys for "width" and/or "height".
    fig_defaults : Dict[str, Any]
        The defaults dictionary containing keys for "width" and "height" (mandatory)

    Returns
    -------
    go.Figure
        The Plotly figure with updated dimensions.
    """
    width = _resolve_dict_keys(dict_=fig_kwargs, key="width", defaults=fig_defaults)
    height = _resolve_dict_keys(dict_=fig_kwargs, key="height", defaults=fig_defaults)
    if width is not None or height is not None:
        fig.update_layout(autosize=False, width=width, height=height)
    return fig


def _get_subplot_rows_cols(
    num_subplots: int, rows: Optional[int], cols: Optional[int]
) -> Tuple[int, int]:
    """Returns the number of rows and columns to divide the subplots into

    Parameters
    ----------
    num_subplots : int
        The total number of subplots that need to be divided
    rows : Optional[int]
        The number of rows that the subplots need to be divided into. If none,
        this gets decided automatically based on other parameters.
    cols : Optional[int]
        The number of columns that the subplots need to be divided into. If none,
        this gets decided automatically based on other parameters.

    Returns
    -------
    Tuple[int, int]
        Number of rows and columns to divide the subplots into

    Raises
    ------
    ValueError
        When both `rows` and `cols` are provided, but are not enough to plot all
        subplots.
    """
    if rows is None and cols is not None:
        rows = ceil(num_subplots / cols)
    elif rows is not None and cols is None:
        cols = ceil(num_subplots / rows)
    elif rows is not None and cols is not None:
        available = rows * cols
        if available < num_subplots:
            raise ValueError(
                "Not enough subplots available to plot CCF. "
                f"You provided {rows} * {cols} = {available} subplots. "
                f"Please provide at least {num_subplots} subplots in all."
            )
    else:
        # Decide based on data
        # If 1, then row = 1, col = 1
        # else, try 5, 4, 3, 2, and pick first one which is equally divisible
        # if None, then use 5
        if num_subplots == 1:
            cols = 1
        else:
            cols = 5
            for i in [5, 4, 3, 2]:
                if num_subplots % i == 0:
                    cols = i
                    break
        rows = ceil(num_subplots / cols)
    return rows, cols


def _resolve_hoverinfo(
    hoverinfo: Optional[str],
    threshold: int,
    data: Optional[pd.Series],
    train: Optional[pd.Series],
    test: Optional[pd.Series],
    X: Optional[pd.DataFrame],
) -> str:
    """Decide whether data tip obtained by hovering over a Plotly plot should be
    enabled or disabled based user settings and size of data. If user provides the
    `hoverinfo` option, it is honored, else it gets decided based on size of dataset.

    Parameters
    ----------
    hoverinfo : Optional[str]
        The hoverinfo option selected by the user
    threshold : int
        The number of data points above which the hovering should be disabled.
    data : Optional[pd.Series]
        A series of data
    train : Optional[pd.Series]
        A series of train indices
    test : Optional[pd.Series]
        A series of test indices
    X : Optional[pd.DataFrame]
        A dataframe of exogenous variables

    Returns
    -------
    str
        The hoverinfo option to use for Plotly.
    """
    if hoverinfo is None:
        hoverinfo = "text"
        if data is not None and len(data) > threshold:
            hoverinfo = "skip"
        if train is not None and len(train) > threshold:
            hoverinfo = "skip"
        if test is not None and len(test) > threshold:
            hoverinfo = "skip"
        if X is not None and len(X) * X.shape[1] > threshold:
            hoverinfo = "skip"
    # if hoverinfo is not None, then use as is.
    return hoverinfo


def _resolve_renderer(
    renderer: Optional[str],
    threshold: int,
    data: Optional[pd.Series],
    train: Optional[pd.Series],
    test: Optional[pd.Series],
    X: Optional[pd.DataFrame],
) -> str:
    """Decide the renderer to use for the Plotly plot based user settings and
    size of data. If user provides the `renderer` option, it is honored, else it
    gets decided based on size of dataset.

    Parameters
    ----------
    renderer : Optional[str]
        The renderer option selected by the user
    threshold : int
        The number of data points above which the hovering should be disabled.
    data : Optional[pd.Series]
        A series of data
    train : Optional[pd.Series]
        A series of train indices
    test : Optional[pd.Series]
        A series of test indices
    X : Optional[pd.DataFrame]
        A dataframe of exogenous variables

    Returns
    -------
    str
        The renderer option to use for Plotly.
    """
    if renderer is None:
        renderer = pio.renderers.default
        if data is not None and len(data) > threshold:
            renderer = "png"
        if train is not None and len(train) > threshold:
            renderer = "png"
        if test is not None and len(test) > threshold:
            renderer = "png"
        if X is not None and len(X) * X.shape[1] > threshold:
            renderer = "png"
    # if renderer is not None, then use as is.

    if renderer not in pio.renderers:
        raise ValueError(
            f"Renderer '{renderer}' is not a valid Plotly renderer. "
            f"Valid renderers are:\n {pio.renderers}"
        )

    return renderer
