from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, ccf, pacf

from pycaret.internal.logging import get_logger
from pycaret.utils.generic import _resolve_dict_keys
from pycaret.utils.time_series import TSAllowedPlotDataTypes

logger = get_logger()

PlotReturnType = Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]


# Data Types allowed for each plot type ----
# First one in the list is the default (if requested is None)
ALLOWED_PLOT_DATA_TYPES = {
    "pipeline": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.TRANSFORMED.value,
    ],
    "ts": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.TRANSFORMED.value,
    ],
    "train_test_split": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.TRANSFORMED.value,
    ],
    "cv": [TSAllowedPlotDataTypes.ORIGINAL.value],
    "acf": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "pacf": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "decomp": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "decomp_stl": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "diagnostics": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "diff": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "forecast": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
    ],
    "insample": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
    ],
    "residuals": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
    ],
    "periodogram": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "fft": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "ccf": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
}

# Are multiple plot types allowed at once ----
MULTIPLE_PLOT_TYPES_ALLOWED_AT_ONCE = {
    "ts": True,
    "train_test_split": True,
    "cv": False,
    "acf": True,
    "pacf": True,
    "decomp": True,
    "decomp_stl": True,
    "diagnostics": True,
    "diff": False,
    "forecast": False,
    "insample": False,
    "residuals": False,
    "periodogram": True,
    "fft": True,
    "ccf": False,
}


def time_series_subplot(
    fig: go.Figure,
    data: pd.DataFrame,
    row: int,
    col: int,
    hoverinfo: Optional[str],
) -> go.Figure:
    """Function to add a single or multiple overlaid time series to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the time series needs to be added
    data : pd.DataFrame
        Time Series that needs to be added. If more than one column is present,
        then each column acts as a separate time series that needs to be overlaid.
    row : int
        Row of the figure where the plot needs to be inserted. Starts from 1.
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.
    hoverinfo : Optional[str]
        Whether hoverinfo should be disabled or not. Options are same as plotly.
        e.g. "text" to display, "skip" or "none" to disable.

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

    for i, col_name in enumerate(data.columns):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        # If you add hoverinfo = "text", you must also add the hovertemplate, else no hoverinfo
        # gets displayed. OR alternately, leave it out and it gets plotted by default.
        if hoverinfo == "text":
            # Not specifying the hoverinfo will show it by default
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=data[col_name].values,
                    name=col_name,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=5),
                ),
                row=row,
                col=col,
            )
        else:
            # Disable hoverinfo
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=data[col_name].values,
                    name=col_name,
                    mode="lines+markers",
                    line=dict(width=2),
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

    # Compute n_lags
    if nlags is None:
        if plot in ["acf", "ccf"]:
            nlags = 40
        elif plot in ["pacf"]:
            nobs = len(data)
            nlags = min(int(10 * np.log10(nobs)), nobs // 2 - 1)

    if plot in ["acf", "pacf"]:
        if plot == "acf":
            default_name = "ACF"
            corr_array = acf(data, nlags=nlags, alpha=0.05)
        elif plot == "pacf":
            default_name = "PACF"
            corr_array = pacf(data, nlags=nlags, alpha=0.05)
        corr_values = corr_array[0]
        lower = corr_array[1][:, 0] - corr_array[0]
        upper = corr_array[1][:, 1] - corr_array[0]
    elif plot == "ccf":
        default_name = "CCF"
        target, exog = data
        corr_values = ccf(target, exog, unbiased=False)
        # Default, returns lags = len of data, hence limit it.
        corr_values = corr_values[: nlags + 1]
        # No upper and lower bounds available for CCF
    else:
        raise ValueError(
            f"plot must be one of 'acf', 'pacf' or 'ccf'. You passed '{plot}'"
        )

    lags = np.arange(len(corr_values))
    name = name or default_name

    # Add the correlation plot ----
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
    # Add corr plot stem lines ----
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

    # Add corr plot stem endpoints ----
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

    # Add the Upper Confidence Interval ----
    fig.add_scatter(
        x=np.arange(len(upper)),
        y=upper,
        mode="lines",
        line_color="rgba(255,255,255,0)",
        row=row,
        col=col,
        name="UC",
    )

    # Add the Lower Confidence Interval ----
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


def decomp_subplot(
    fig: go.Figure,
    data: pd.Series,
    col: int,
    plot: str,
    classical_decomp_type: str,
    period: int,
) -> go.Figure:
    """Function to add decomposition to a Plotly subplot

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to which the decomposition plots need to be added
    data : pd.Series
        Data whose decomposition must be added
    col : int
        Column of the figure where the plot needs to be inserted. Starts from 1.
        Note that rows do not need to be specified since there will be 4 rows
        per column. Must be enforced from outside when creating the figure.
    plot : str
        Options are
          - "decomp": for Classical Decomposition
          - "decomp_stl": for STL Decomposition
    classical_decomp_type : str
        The classical decomposition type. Options are: ["additive", "multiplicative"]
    period : int
        The seasonal period to use for decomposition

    Returns
    -------
    go.Figure
        Returns back the plotly figure with the decomposition results inserted.
    """

    data_ = data.to_timestamp() if isinstance(data.index, pd.PeriodIndex) else data

    x = (
        data.index.to_timestamp()
        if isinstance(data.index, pd.PeriodIndex)
        else data.index
    )

    # Plot Original data ----
    row = 1
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=data_,
            line=dict(color=DEFAULT_PLOTLY_COLORS[row - 1], width=2),
            mode="lines+markers",
            name="Actual",
            marker=dict(size=2),
        ),
        row=row,
        col=col,
    )

    if plot == "decomp":
        try:
            decomp_result = seasonal_decompose(
                data_, period=period, model=classical_decomp_type
            )
        except ValueError as exception:
            logger.warning(exception)
            logger.warning(
                "Seasonal Decompose plot failed most likely sue to missing data"
            )
            return fig, None
    elif plot == "decomp_stl":
        decomp_result = STL(data_, period=period).fit()

    row = 2
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=decomp_result.seasonal,
            line=dict(color=DEFAULT_PLOTLY_COLORS[row - 1], width=2),
            mode="lines+markers",
            name="Seasonal",
            marker=dict(size=2),
        ),
        row=row,
        col=col,
    )

    row = 3
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=decomp_result.trend,
            line=dict(color=DEFAULT_PLOTLY_COLORS[row - 1], width=2),
            mode="lines+markers",
            name="Trend",
            marker=dict(size=2),
        ),
        row=row,
        col=col,
    )

    row = 4
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=decomp_result.resid,
            line=dict(color=DEFAULT_PLOTLY_COLORS[row - 1], width=2),
            mode="markers",
            name="Residuals",
            marker=dict(
                size=4,
            ),
        ),
        row=row,
        col=col,
    )

    return fig, decomp_result


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
    plot: str = "periodogram",
    name: Optional[str] = None,
) -> PlotReturnType:
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
    plot : str, optional
        Type of method to use to get the frequency components, by default "periodogram"
        Allowed methods are: "periodogram", "fft"
    name : Optional[str], optional
        Name to show when hovering over plot, by default None

    Returns
    -------
    PlotReturnType
        Returns back the plotly figure along with the data used to create the plot.
    """
    x, y = return_frequency_components(data=data, type=plot)
    time_period = [round(1 / freq, 4) for freq in x]
    freq_data = pd.DataFrame({"Freq": x, "Amplitude": y, "Time Period": time_period})

    name = name or data.name

    # If you add hoverinfo = "text", you must also add the hovertemplate, else no hoverinfo
    # gets displayed. OR alternately, leave it out and it gets plotted by default.
    if hoverinfo == "text":
        # We convert this to hovertext so plotly-resampler can effectively deal with
        # this data modality
        freq_data_str = freq_data.round(4).astype("str")
        hf_hovertext = (
            "Freq: "
            + freq_data_str["Freq"]
            + "<br>Ampl: "
            + freq_data_str["Amplitude"]
            + "<br>Time period: "
            + freq_data_str["Time Period"]
        )

        fig.add_trace(
            go.Scattergl(
                name=name,
                x=freq_data["Freq"],
                y=freq_data["Amplitude"],
                hovertext=hf_hovertext,
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


def plot_original_with_overlays(
    original_data: pd.Series,
    overlay_data: pd.DataFrame,
    title: str,
    fig_defaults: Dict[str, Any],
    hoverinfo: Optional[str] = "text",
    fig_kwargs: Optional[Dict] = None,
) -> PlotReturnType:
    """Plots the original data along with the data to be overlaid in a single plot.

    Parameters
    ----------
    original_data : pd.Series
        The original target series to be plotted
    overlay_data : pd.DataFrame
        The data that must be overlaid over the original data. e.g. prediction
        from possibly multiple models. Each column in the overlay_data is overlaid
        as a separate series over the original_data. The column names are used as
        labels for the overlaid data.
    title : str
        The title to use for the plot (mandatory)
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
    fig, return_data_dict = None, None

    data_to_plot = pd.concat([original_data, overlay_data], axis=1)

    fig = make_subplots(rows=1, cols=1)
    fig = time_series_subplot(
        fig=fig, data=data_to_plot, row=1, col=1, hoverinfo=hoverinfo
    )

    fig_kwargs = fig_kwargs or {}
    with fig.batch_update():
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(template=template)

        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )
        fig.update_layout(showlegend=True)
        fig.update_layout(title=title)

    return_data_dict = {
        "original_data": original_data,
        "overlay_data": overlay_data,
    }

    return fig, return_data_dict


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
    X : Optional[List[pd.DataFrame]]
        A list of dataframe of exogenous variables (1 dataframe per exogenous
        variable; each dataframe containing multiple columns corresponding to the
        plot data types requested)

    Returns
    -------
    str
        The hoverinfo option to use for Plotly.
    """
    if hoverinfo is None:
        hoverinfo = "text"
        if data is not None and len(data) > threshold:
            hoverinfo = "skip"
        if X is not None and len(X) * len(X[0]) * X[0].shape[1] > threshold:
            hoverinfo = "skip"
    # if hoverinfo is not None, then use as is.
    return hoverinfo


def _resolve_renderer(
    renderer: Optional[str],
    threshold: int,
    data: Optional[pd.Series],
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
    X : Optional[List[pd.DataFrame]]
        A list of dataframe of exogenous variables (1 dataframe per exogenous
        variable; each dataframe containing multiple columns corresponding to the
        plot data types requested)

    Returns
    -------
    str
        The renderer option to use for Plotly.
    """
    if renderer is None:
        renderer = pio.renderers.default
        if data is not None and len(data) > threshold:
            renderer = "png"
        if X is not None and len(X) * len(X[0]) * X[0].shape[1] > threshold:
            renderer = "png"
    # if renderer is not None, then use as is.

    return renderer


def _get_data_types_to_plot(
    plot: str, data_types_requested: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """Returns the data types to plot based on the requested ones. If all are allowed
    for the requested plot, they are returned as is, else this function will trim them
    down to the allowed types only.

    NOTE: Some plots only support one data type. If multiple data types are requested
    for such plots, only the first one is used (appropriate warning issued).

    Parameters
    ----------
    plot : str
        The plot for which the data types are being requested
    data_types_requested : Optional[Union[str, List[str]]], optional
        The data types being requested for the plot, by default None
        If None, it picks the default from the internal list.

    Returns
    -------
    List[str]
        The allowed data types for the requested plot based on user inputs

    Raises
    ------
    ValueError
        If none of the requested data types are supported by the plot
    """

    # Get default if not provided ----
    if data_types_requested is None:
        # First one is the default
        data_types_requested = [ALLOWED_PLOT_DATA_TYPES.get(plot)[0]]

    # Convert string to list ----
    if isinstance(data_types_requested, str):
        data_types_requested = [data_types_requested]

    # Is the data type allowed for the requested plot?
    all_plot_data_types = [member.value for member in TSAllowedPlotDataTypes]
    data_types_allowed = [
        True
        if data_type_requested in ALLOWED_PLOT_DATA_TYPES.get(plot)
        and data_type_requested in all_plot_data_types
        else False
        for data_type_requested in data_types_requested
    ]

    # Clean up list based on allowed data types
    cleaned_data_types = []
    for requested, allowed in zip(data_types_requested, data_types_allowed):
        if allowed:
            cleaned_data_types.append(requested)
        else:
            msg = (
                f"Data Type: '{requested}' is not supported for plot: '{plot}'. "
                "This will be ignored."
            )
            logger.warning(msg)
            print(msg)

    if len(cleaned_data_types) == 0:
        raise ValueError(
            "No data to plot. Please check to make sure that you have requested "
            "an allowed data type for plot."
            f"\n Allowed values are: {ALLOWED_PLOT_DATA_TYPES.get(plot)}"
        )

    if (
        not MULTIPLE_PLOT_TYPES_ALLOWED_AT_ONCE.get(plot)
        and len(cleaned_data_types) > 1
    ):
        msg = (
            f"Data Type requested for plot '{plot}' = {cleaned_data_types}, "
            "but this plot only supports a single data type at a time. "
            f"\nThe first one (i.e. '{cleaned_data_types[0]}') will be used."
        )
        logger.warning(msg)
        print(msg)
        cleaned_data_types = [cleaned_data_types[0]]

    return cleaned_data_types


def _reformat_dataframes_for_plots(
    data: List[Union[pd.Series, pd.DataFrame]], labels_suffix: List[str]
) -> List[pd.DataFrame]:
    """Take the input list of dataframes (assuming all dataframes have the same columns)
    and converts them into a list of new dataframes with each new dataframe containing
    the same column from all of the input dataframe.

    e.g. 1
    If input list has 2 dataframes D1 and D2 with columns A, B, and C then the
    output will be a list of 3 dataframes with 2 columns each
        Output dataframe 1 containing D1.A, D2.A
        Output dataframe 2 containing D1.B, D2.B
        Output dataframe 3 containing D1.C, D2.C

    e.g. 2
    If the input list has series, they are just concatenated together to produce one
    output dataframe.

    Parameters
    ----------
    data : List[Union[pd.Series, pd.DataFrame]]
        Input list of dataframes or series
    labels_suffix : List[str]
        The suffix to use for the output dataframes column names.
        Must be the same length as the number of input dataframes

        In the example above, if suffix is ["original", "transformed"], then the
            Output dataframe 1 will have columns ["A (original)", "A (transformed)"]
            Output dataframe 2 will have columns ["B (original)", "B (transformed)"]
            Output dataframe 2 will have columns ["C (original)", "C (transformed)"]

    Returns
    -------
    List[pd.DataFrame]
        Output list of dataframes

    Raises
    ------
    ValueError
        When the number of labels provided does not match the number of input dataframes
    """
    num_labels = len(labels_suffix)
    num_input_dfs = len(data)
    if num_labels != num_input_dfs:
        raise ValueError(
            f"Number of labels provided ({num_labels}) does not match the number of input "
            "dataframes ({num_input_dfs})"
        )

    cols = pd.DataFrame(data[0]).columns

    data = pd.concat(data, axis=1)
    output = []
    for col in cols:
        temp = pd.DataFrame(data[col])
        column_names = [f"{col} ({suffix})" for suffix in labels_suffix]
        temp.columns = column_names
        output.append(temp)

    return output


def _clean_model_results_labels(
    model_results: List[pd.DataFrame], model_labels: List[str]
) -> Tuple[List[pd.DataFrame], List[str]]:
    """Cleans the model results and names to remove models that did not produce
    any results, e.g. no residuals, insample predictions, etc.

    Parameters
    ----------
    model_results : List[pd.DataFrame]
        List of dataframes containing the model results (one dataframe per model)
        Some values might be None if the model did not produce a result. These
        will get dropped by this function.
    model_labels : List[str]
        The labels of the models producing the results.

    Returns
    -------
    Tuple[List[pd.DataFrame], List[str]]
        The cleaned model results and names (after removing those that did not
        produce a result).
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
    model_labels = [
        model_name for include, model_name in zip(includes, model_labels) if include
    ]

    return model_results, model_labels


def _plot_fig_update(
    fig: go.Figure,
    title: str,
    fig_defaults: Dict[str, Any],
    fig_kwargs: Optional[Dict] = None,
    show_legend: bool = False,
) -> go.Figure:
    """Customises the template layout and dimension of plots

    Parameters
    ----------
    fig: go.Figure
        Plotly figure which needs to be customised
    title: str
        Title of the plot
    fig_defaults : Dict[str, Any]
        The default settings for the plotly plot. Must contain keys for "width",
        "height", and "template".
    fig_kwargs : Optional[Dict], optional
        Specific overrides by the user for the plotly figure settings,
        by default None. Can contain keys for "width", "height" and/or "template".
    show_legend: bool, default=False
        If True, displays the legend in the plot when layout is updated.

    Returns
    -------
    go.Figure
        The Plotly figure with updated dimnensions and template layout.
    """
    with fig.batch_update():
        template = _resolve_dict_keys(
            dict_=fig_kwargs, key="template", defaults=fig_defaults
        )
        fig.update_layout(title=title, showlegend=show_legend, template=template)
        fig = _update_fig_dimensions(
            fig=fig, fig_kwargs=fig_kwargs, fig_defaults=fig_defaults
        )
    return fig
