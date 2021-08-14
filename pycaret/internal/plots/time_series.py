from typing import Optional, Any, Union
import pandas as pd

from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)

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
    return_data: bool = False,
) -> Optional[Any]:
    if plot == "ts":
        plot_data = plot_series(data=data, return_data=return_data)
    elif plot == "splits-tt":
        plot_data = plot_splits_tt(train=train, test=test, return_data=return_data)
    elif plot == "splits_cv":
        plot_data = plot_splits_cv(data=data, cv=cv, return_data=return_data)
    elif plot == "acf":
        plot_data = plot_acf(data=data, return_data=return_data)
    elif plot == "pacf":
        plot_data = plot_pacf(data=data, return_data=return_data)
    elif plot == "predictions":
        plot_data = plot_predictions(
            data=train, predictions=predictions, return_data=return_data
        )
    elif plot == "residuals":
        plot_data = plot_diagnostics(data=data, return_data=return_data)
    else:
        raise ValueError(f"Tests: '{plot}' is not supported.")

    return plot_data if return_data else None


def plot_series(data: pd.Series, return_data: bool = False):
    """Plots the original time series"""
    print("Inside plot_series")


def plot_splits_tt(train: pd.Series, test: pd.Series, return_data: bool = False):
    """Plots the train-test split for the time serirs"""
    print("Inside plot_splits_tt")


def plot_splits_cv(data: pd.Series, cv, return_data: bool = False):
    """Plots the cv splits used on the training split"""
    print("Inside plot_splits_cv")


def plot_acf(data: pd.Series, return_data: bool = False):
    """Plots the ACF on the data provided"""
    print("Inside plot_acf")


def plot_pacf(data: pd.Series, return_data: bool = False):
    """Plots the PACF on the data provided"""
    print("Inside plot_pacf")


def plot_predictions(
    data: pd.Series, predictions: pd.Series, return_data: bool = False
):
    """Plots the original data and the predictions provided"""
    print("Inside plot_predictions")


def plot_diagnostics(data: pd.Series, return_data: bool = False):
    """Plots the diagnostic data such as ACF, Histogram, QQ plot on the data provided"""
    print("Inside plot_diagnostics")

