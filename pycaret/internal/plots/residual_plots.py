from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import Layout, widgets
from plotly.basewidget import BaseFigureWidget
from scipy import stats

import pycaret.internal.plots.helper as helper
from pycaret.internal.display import CommonDisplay
from pycaret.internal.display.display_backend import ColabBackend, DatabricksBackend
from pycaret.internal.logging import get_logger
from pycaret.internal.validation import fit_if_not_fitted


class QQPlotWidget(BaseFigureWidget):
    """
    The QQ plot compares the quantiles of the empirical residuals to the theoretical quantiles of a standard normal distribution $N(0, 1)$.
    Assuming that $Y=f(X)+\epsilon$ we can verify with this plot that $\epsilon ~ N(\mu, \sigma^2)$.
    If the error terms actually originate from a normal distribution, then the data points will scatter just slightly
    around the red straight line.
    """

    def __init__(
        self,
        predicted: np.ndarray,
        expected: np.ndarray = None,
        featuresize: int = None,
        split_origin: np.array = None,
        **kwargs,
    ):
        """
        Instantiates a QQ plot

        Parameters
        ----------
        predicted: nd.array
            The predicted values
        expected: np.ndarray
            Optional, the true values. If this attribute is None, the predicted array is assumed to contain the already
            standardized residuals.
        featuresize: int
            number of features
        split_origin: np.ndarray
            Optional, if the data used for the predictions includes unseen test data.
            These residuals can be marked explicitly in the plot. This attribute must have the same dimensionality
            as the predictions and expected array. Each entry in this array must be one of the strings ['train', 'test']
            to denote from which split this observation originates.
        """

        if expected is not None:
            std_res = helper.calculate_standardized_residual(
                predicted, expected=expected, featuresize=featuresize
            )
        else:
            std_res = predicted
        self._plot = self.__qq_plot(std_res, split_origin)
        super(QQPlotWidget, self).__init__(self._plot, **kwargs)

    @staticmethod
    def __get_qq(standardized_residuals: np.ndarray) -> np.ndarray:
        """
        Calculate the theoretical quantiles and the ordered response.

        Parameters
        ----------
        standardized_residuals: np.array
            the standardized residuals of a model for some specific dataset

        Returns
        -------
            (osm, osr) tuple of nd.arrays
                Tuple of theoretical quantiles (osm, or order statistic medians) and ordered responses (osr). osr is
                simply the sorted standardized residuals.

            (slope, intercept) tuple of floats
                Tuple containing the result of the least-squares fit.

        """

        qq = stats.probplot(standardized_residuals, dist="norm", sparams=(1))
        return qq[0], qq[1][:2]

    def __qq_plot(
        self, standardized_residuals: np.ndarray, split_origin: np.array = None
    ) -> go.Figure:
        (osm, osr), (slope, intercept) = self.__get_qq(
            standardized_residuals=standardized_residuals
        )
        fig = go.Figure()

        if split_origin is not None:
            # calculate the sorted split origin list w.r.t to the standardized residuals
            # with this list we know which (theoretical quantile | empirical quantile) point belongs to which origin
            sorted_split_origin = np.array(
                [
                    origin
                    for (_, origin) in sorted(
                        enumerate(split_origin),
                        key=lambda idx_value: standardized_residuals[idx_value[0]],
                    )
                ]
            )
            colors = sorted_split_origin.copy()
            colors[sorted_split_origin == "train"] = "blue"
            colors[sorted_split_origin == "test"] = "green"
            fig.add_scatter(
                x=osm,
                y=osr,
                mode="markers",
                name="quantiles",
                marker=dict(color=colors),
                customdata=sorted_split_origin,
                hovertemplate="%{x},%{y} (%{customdata})",
                opacity=0.7,
            )
        else:
            fig.add_scatter(x=osm, y=osr, mode="markers", name="quantiles", opacity=0.7)

        x = np.array([osm[0], osm[-1]])
        fig.add_scatter(x=x, y=intercept + slope * x, mode="lines", name="OLS")
        fig.layout.update(
            autosize=True,
            showlegend=False,
            title="Normal QQ-Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Standardized Residuals",
        )
        return fig

    def update_values(
        self,
        predicted: np.ndarray,
        expected: np.ndarray = None,
        featuresize: int = None,
        split_origin: np.ndarray = None,
    ):
        """
        Update the QQ plot values

        Parameters
        ----------
        predicted: nd.array
            The predicted values
        expected: np.ndarray
            Optional, the true values. If this attribute is None, the predicted array is assumed to contain the already
            standardized residuals.
        featuresize: int
            number of features
        split_origin: np.ndarray
            Optional, if the data used for the predictions includes unseen test data.
            These residuals can be marked explicitly in the plot. This attribute must have the same dimensionality
            as the predictions and expected array. Each entry in this array must be one of the strings ['train', 'test']
            to denote from which split this observation originates.
        """
        self._plot = self.__qq_plot(
            standardized_residuals=helper.calculate_standardized_residual(
                predicted, expected, featuresize
            ),
            split_origin=split_origin,
        )
        self.update({"data": self._plot.data}, overwrite=True)
        self.update_layout()


class ScaleLocationWidget(BaseFigureWidget):
    """
    The Scale Location plot compares the square root of the absolute standardized residuals $\sqrt{|\tilde r_i|}$
    versus the predicted values $\hat y_i$.
    Assuming that $Y=f(X)+\epsilon$ we can verify with this plot that $Var[\epsilon]=\sigma^2$.
    The fitted trend line should follow a straight line along the axis of predicted values.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        sqrt_abs_standardized_residuals: np.ndarray,
        split_origin: np.ndarray = None,
        **kwargs,
    ):
        """
        Instantiates a Scale Location plot

        Parameters
        ----------
        predictions: np.ndarray
            The predictions on the data
        sqrt_abs_standardized_residuals: np.ndarray
            The square root of the absolute value of the standardized residuals
        split_origin: np.ndarray
            Optional, if the data used for the predictions includes unseen test data.
            These residuals can be marked explicitly in the plot. This attribute must have the same dimensionality
            as the predictions and sqrt_abs_standardized_residuals array. Each entry in this array must be one of the strings ['train', 'test']
            to denote from which split this observation originates.
        """
        self._plot = self.__scale_location_plot(
            predictions, sqrt_abs_standardized_residuals, split_origin
        )
        super(ScaleLocationWidget, self).__init__(self._plot, **kwargs)

    @staticmethod
    def __scale_location_plot(fitted, sqrt_abs_standardized_residuals, split_origin):
        sqrt_abs_standardized_residuals = pd.Series(sqrt_abs_standardized_residuals)

        if split_origin is not None:
            dataframe = pd.DataFrame(
                {
                    "Predictions": fitted,
                    "Split": split_origin,
                    "Standardized Residuals^1/2": sqrt_abs_standardized_residuals,
                }
            )
            fig = px.scatter(
                dataframe,
                x="Predictions",
                y="Standardized Residuals^1/2",
                trendline="lowess",
                color="Split",
                color_discrete_sequence=["blue", "green"],
                title="Scale-Location Plot",
                opacity=0.3,
            )

            fig.update_layout(showlegend=False)
        else:
            dataframe = pd.DataFrame(
                {
                    "Predictions": fitted,
                    "Standardized Residuals^1/2": sqrt_abs_standardized_residuals,
                }
            )
            fig = px.scatter(
                dataframe,
                x="Predictions",
                y="Standardized Residuals^1/2",
                trendline="lowess",
                title="Scale-Location Plot",
                opacity=0.3,
            )

        abs_sq_norm_resid = sqrt_abs_standardized_residuals.sort_values(ascending=False)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3.index:
            fig.add_annotation(
                x=fitted[i],
                y=sqrt_abs_standardized_residuals[i],
                text=f"~r_{i}^1/2",
            )
        fig.update_annotations(
            dict(xref="x", yref="y", showarrow=True, arrowhead=7, ax=0, ay=-40)
        )
        return fig

    def update_values(
        self,
        predicted: np.ndarray,
        sqrt_abs_standardized_residuals: np.ndarray,
        split_origin: np.ndarray = None,
    ):
        """
        Update the Scale Location plot values

        Parameters
        ----------
        predictions: np.ndarray
            The predictions on the data
        sqrt_abs_standardized_residuals: np.ndarray
            The square root of the absolute value of the standardized residuals
        split_origin: np.ndarray
            Optional, if the data used for the predictions includes unseen test data.
            These residuals can be marked explicitly in the plot. This attribute must have the same dimensionality
            as the predictions and sqrt_abs_standardized_residuals array. Each entry in this array must be one of the strings ['train', 'test']
            to denote from which split this observation originates.
        """
        self._plot = self.__scale_location_plot(
            predicted, sqrt_abs_standardized_residuals, split_origin
        )
        self.update({"data": self._plot.data}, overwrite=True)
        self.update_layout()


class CooksDistanceWidget(BaseFigureWidget):
    """
    This widget compares the standardized residuals $\tilde r_i$ versus the leverage $h_i$ of the corresponding observation $x_i$.
    Assuming that $Y=f(X)+\epsilon$ we can verify with this plot that the error terms $\epsilon_i$ are independent.
    The cook's distance is a measure to which extent some points are high leverage points and/or outliers.
    Cook's distance (function of leverage and residual) is a measure of how influential a data point is.
    If a point lies beyond of contour lines corresponding to a Cook's distance larger than 1, then this point should be
    considered as dangerously influential.
    """

    def __init__(
        self,
        model_leverage: np.ndarray,
        cooks_distances: np.ndarray,
        standardized_residuals: np.ndarray,
        n_model_params: int,
        split_origin: np.ndarray = None,
        **kwargs,
    ):
        """
        Instantiates the Cooks Distance widget w.r.t. some model and a dataset $X$ which has $n$ observations.

        Parameters
        ----------
        model_leverage: np.ndarray
            An array of length $n$, containing the leverage of the observations
        cooks_distances: np.ndarray
            An array of length $n$, containing the cooks_distances of the observations
        standardized_residuals: np.ndarray
            An array of length $n$, containing the standardized residuals
        n_model_params: int
            The number of parameters of the used model
        split_origin: np.ndarray
            Optional, if the data used for the predictions includes unseen test data.
            These residuals can be marked explicitly in the plot. This attribute must have the same dimensionality
            as the model_leverage and standardized_residuals array. Each entry in this array must be one of the
            strings ['train', 'test'] to denote from which split this observation originates.
        """
        self._plot = self.__cooks_distance_plot(
            model_leverage,
            cooks_distances,
            standardized_residuals,
            n_model_params,
            split_origin,
        )
        super(CooksDistanceWidget, self).__init__(self._plot, **kwargs)

    @staticmethod
    def __cooks_distance_plot(
        model_leverage,
        cooks_distances,
        standardized_residuals,
        n_model_params,
        split_origin,
    ):
        cooks_distances = pd.Series(cooks_distances)

        if split_origin is not None:
            dataframe = pd.DataFrame(
                {
                    "Leverage": model_leverage,
                    "Standardized Residuals": standardized_residuals,
                    "Split": split_origin,
                }
            )
            fig = px.scatter(
                dataframe,
                x="Leverage",
                y="Standardized Residuals",
                trendline="lowess",
                color="Split",
                color_discrete_sequence=["blue", "green"],
                title="Residuals vs Leverage",
                opacity=0.3,
            )

            fig.update_layout(showlegend=False)
        else:
            dataframe = pd.DataFrame(
                {
                    "Leverage": model_leverage,
                    "Standardized Residuals": standardized_residuals,
                }
            )
            fig = px.scatter(
                dataframe,
                x="Leverage",
                y="Standardized Residuals",
                trendline="lowess",
                title="Residuals vs Leverage",
                opacity=0.3,
            )

        maxmo = max(model_leverage) * 1.05
        fig.update_xaxes(range=[0, maxmo])
        min_r = min(standardized_residuals)
        max_r = max(standardized_residuals)
        fig.update_yaxes(range=[min_r - 0.05 * abs(min_r), max_r + 0.05 * abs(max_r)])
        leverage_top_3 = cooks_distances.sort_values(ascending=False)[:3]
        for i in leverage_top_3.index:
            fig.add_annotation(
                x=model_leverage[i],
                y=standardized_residuals[i],
                text=f"~r_{i}",
            )

        fig.update_annotations(
            dict(xref="x", yref="y", showarrow=True, arrowhead=7, ax=0, ay=-40)
        )

        def graph(formula, x_range, label, c, text):
            x = x_range
            y = formula(x)

            text_list = ["" for _ in range(x_range.shape[0])]
            text_list[x_range.shape[0] // 4] = text
            text_list[(x_range.shape[0] // 4) * 3] = text

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=label,
                    line=dict(color=c, width=2, dash="dash"),
                    mode="lines+text",
                    textposition="bottom center",
                    showlegend=False,
                    text=text_list,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=-y,
                    name=label,
                    line=dict(color=c, width=2, dash="dash"),
                    mode="lines+text",
                    textposition="top center",
                    showlegend=False,
                    text=text_list,
                )
            )

        p = n_model_params
        graph(
            lambda x: np.sqrt(np.abs((0.5 * (p + 1) * (1 - x)) / x)),
            np.linspace(0.001, max(model_leverage), 50),
            "Cook's distance = 0.5",
            "coral",
            "0.5",
        )

        graph(
            lambda x: np.sqrt(np.abs((1 * (p + 1) * (1 - x)) / x)),
            np.linspace(0.001, max(model_leverage), 50),
            "Cook's distance = 1",
            "firebrick",
            "1",
        )
        return fig

    def update_values(
        self,
        model_leverage: np.ndarray,
        cooks_distances: np.ndarray,
        standardized_residuals: np.ndarray,
        n_model_params: int,
        split_origin: np.ndarray = None,
    ):
        """
        Update the Cooks Distance widget values

        Parameters
        ----------
        model_leverage: np.ndarray
            An array of length $n$, containing the leverage of the observations
        cooks_distances: np.ndarray
            An array of length $n$, containing the cooks_distances of the observations
        standardized_residuals: np.ndarray
            An array of length $n$, containing the standardized residuals
        n_model_params: int
            The number of parameters of the used model
        split_origin: np.ndarray
            Optional, if the data used for the predictions includes unseen test data.
            These residuals can be marked explicitly in the plot. This attribute must have the same dimensionality
            as the model_leverage and standardized_residuals array. Each entry in this array must be one of the
            strings ['train', 'test'] to denote from which split this observation originates.
        """
        self._plot = self.__cooks_distance_plot(
            model_leverage,
            cooks_distances,
            standardized_residuals,
            n_model_params,
            split_origin=split_origin,
        )
        self.update({"data": self._plot.data}, overwrite=True)
        self.update_layout()


class TukeyAnscombeWidget(BaseFigureWidget):
    """
    The Tunkey Anscombe Plot compares the residuals $r_i=y_i-\hat y_i$ versus the predicted values $\hat y_i$.
    Assuming that $Y=f(X)+\epsilon$ we can verify with this plot that $E[\epsilon]=0$.
    The fitted trend line should follow a straight line along the axis of predicted values at the position 0 on the
    residual axis.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
        split_origin: np.ndarray = None,
        **kwargs,
    ):
        """
        Instantiates the Tunkey Anscombe plot

        Parameters
        ----------
        predictions: np.ndarray
            The prediction of a model on some data
        residuals: np.ndarray
            The residuals / error of the predictions when compared to the true value
        split_origin: np.ndarray
            Optional, if the data used for the predictions includes unseen test data.
            These residuals can be marked explicitly in the plot. To do this attribute must have the same dimensionality
            as the predictions and residuals array. Each entry in this array must be one of the strings ['train', 'test']
            to denote from which split this observation originates.
        """

        self._plot = self.__tukey_anscombe_plot(predictions, residuals, split_origin)
        super(TukeyAnscombeWidget, self).__init__(self._plot, **kwargs)

    @staticmethod
    def __tukey_anscombe_plot(predictions, residuals, split_origin):
        if split_origin is not None:
            dataframe = pd.DataFrame(
                {
                    "Predictions": predictions,
                    "Residuals": residuals,
                    "Split": split_origin,
                }
            )

            fig = px.scatter(
                dataframe,
                x="Predictions",
                y="Residuals",
                trendline="lowess",
                color="Split",
                color_discrete_sequence=["blue", "green"],
                title="Tukey-Anscombe Plot",
                opacity=0.3,
            )

            fig.update_layout(showlegend=False)
        else:
            dataframe = pd.DataFrame(
                {"Predictions": predictions, "Residuals": residuals}
            )

            fig = px.scatter(
                dataframe,
                x="Predictions",
                y="Residuals",
                trendline="lowess",
                title="Tukey-Anscombe Plot",
                opacity=0.3,
            )

        model_abs_resid = pd.Series(np.abs(residuals))
        abs_resid = model_abs_resid.sort_values(ascending=False)
        abs_resid_top_3 = abs_resid[:3]
        for i in abs_resid_top_3.index:
            fig.add_annotation(x=predictions[i], y=residuals[i], text=f"r_{i}")
        fig.update_annotations(
            dict(xref="x", yref="y", showarrow=True, arrowhead=7, ax=0, ay=-40)
        )
        return fig

    def update_values(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
        split_origin: np.ndarray = None,
    ):
        """
        Update the Tunkey Anscombe plot values

        Parameters
        ----------
        predictions: np.ndarray
            The prediction of a model on some data
        residuals: np.ndarray
            The residuals / error of the predictions when compared to the true value
        split_origin: np.ndarray
            Optional, if the data used for the predictions includes unseen test data.
            These residuals can be marked explicitly in the plot. To do this attribute must have the same dimensionality
            as the predictions and residuals array. Each entry in this array must be one of the strings ['train', 'test']
            to denote from which split this observation originates.
        """
        self._plot = self.__tukey_anscombe_plot(predictions, residuals, split_origin)
        self.update({"data": self._plot.data}, overwrite=True)
        self.update_layout()


class InteractiveResidualsPlot:
    """
    To analyze the residuals of a given model, we are assuming that $Y=f(X)+\epsilon$ for some unknown
    regression function $f(X)$ we further assume that the error terms $\epsilon_i$ are i.i.d. random variables
    with $\epsilon_i~N(0,\sigma^2)$.
    More precisely we assume:
        1. Zero mean $E[e_i]=0$
        2. Constant variance $Var[e_i]=\sigma^2$
        3. A normal distribution $\epsilon_i~N(0,\sigma^2)$
        4. The error terms are independent

    To verify those four assumptions four interactive residual plots will be created.
    If those assumptions are not satisfied, the plausibility of the given model for the given data is to be questioned.
    """

    def __init__(
        self,
        model,
        x: np.ndarray,
        y: np.ndarray,
        x_test: np.ndarray = None,
        y_test: np.ndarray = None,
        display: Optional[CommonDisplay] = None,
    ):
        """
        Instantiates the interactive residual plots for the given data

        Parameters
        ----------
        model
            describes the regression model which is to be evaluated
        x: np.ndarray
            the training data
        y: np.ndarray
            the training labels
        x_test: np.ndarray
            optional, some test data (requires y_test)
        y_test: np.ndarray
            optional, the labels to the provided test data (requires x_test)
        display: CommonDisplay
            this object is required to show the plots
        """

        self.figures: List[BaseFigureWidget] = []
        self.display: CommonDisplay = display or CommonDisplay()
        if isinstance(self.display._general_display, (ColabBackend, DatabricksBackend)):
            raise ValueError(
                "residuals_interactive plot is not supported on Google Colab or Databricks."
            )
        self.plot = self.__create_resplots(model, x, y, x_test, y_test)

    def show(self):
        """
        Show the plots within the provided Display instance
        """
        self.display.display(self.plot)

    def get_html(self):
        """
        Get the HTML representation of the plot.
        """
        style = 'style="width: 50%; height: 50%; float:left;"'
        html = (
            f"<div {style}>{self.figures[0].to_html()}</div><div {style}>{self.figures[1].to_html()}</div>"
            f"<div {style}>{self.figures[2].to_html()}</div><div {style}>{self.figures[3].to_html()}</div>"
        )

        return html

    def write_html(self, plot_filename):
        """
        Write the current plots to a file in HTML format.

        Parameters
        ----------
        plot_filename: str
            name of the file
        """

        html = self.get_html()

        with open(plot_filename, "w") as f:
            f.write(html)

    def __create_resplots(
        self,
        model,
        x: np.ndarray,
        y: np.ndarray,
        x_test: np.ndarray = None,
        y_test: np.ndarray = None,
    ) -> widgets.VBox:
        logger = get_logger()

        with fit_if_not_fitted(model, x, y) as fitted_model:
            fitted = fitted_model.predict(x)
            fitted_residuals = fitted - y

            if x_test is not None and y_test is not None:
                pred = fitted_model.predict(x_test)
                prediction_residuals = pred - y_test

                predictions = np.concatenate((fitted, pred))
                residuals = np.concatenate((fitted_residuals, prediction_residuals))
                split_origin = np.concatenate(
                    (
                        np.repeat("train", fitted.shape[0]),
                        np.repeat("test", pred.shape[0]),
                    )
                )

                x = np.concatenate((x, x_test))
                y = np.concatenate((y, y_test))

            else:
                predictions = fitted
                residuals = fitted_residuals
                split_origin = None

        logger.info("Calculated model residuals")

        tukey_anscombe_widget = TukeyAnscombeWidget(
            predictions, residuals, split_origin=split_origin
        )
        logger.info("Calculated Tunkey-Anscombe Plot")
        self.figures.append(tukey_anscombe_widget)

        qq_plot_widget = QQPlotWidget(
            predictions, y, split_origin=split_origin, featuresize=x.shape[1]
        )
        logger.info("Calculated Normal QQ Plot")
        self.figures.append(qq_plot_widget)

        standardized_residuals = helper.calculate_standardized_residual(
            predictions, y, None
        )
        model_norm_residuals_abs_sqrt = np.sqrt(np.abs(standardized_residuals))
        scale_location_widget = ScaleLocationWidget(
            predictions, model_norm_residuals_abs_sqrt, split_origin=split_origin
        )
        logger.info("Calculated Scale-Location Plot")
        self.figures.append(scale_location_widget)

        leverage = helper.leverage_statistic(np.array(x))

        n_model_params = len(model.get_params())
        distance = helper.cooks_distance(
            standardized_residuals, leverage, n_model_params=n_model_params
        )
        cooks_distance_widget = CooksDistanceWidget(
            leverage,
            distance,
            standardized_residuals,
            n_model_params,
            split_origin=split_origin,
        )
        logger.info("Calculated Residual vs Leverage Plot inc. Cook's distance")
        self.figures.append(cooks_distance_widget)

        items_layout = Layout(width="1000px")
        h0 = widgets.HBox(self.figures[:2], layout=items_layout)
        h1 = widgets.HBox(self.figures[2:], layout=items_layout)
        return widgets.VBox([h0, h1])
