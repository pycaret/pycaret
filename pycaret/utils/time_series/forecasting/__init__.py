from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def get_predictions_with_intervals(
    forecaster, X: pd.DataFrame, fh=None, alpha: float = 0.05
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns the predictions, lower and upper interval values for a
    forecaster. If the forecaster does not support prediction intervals,
    then NAN is returned for lower and upper intervals.

    Parameters
    ----------
    forecaster : sktime compatible forecaster
        Forecaster to be used to get the predictions
    X : pd.DataFrame
        Exogenous Variables
    alpha : float, default = 0.05
        alpha value for prediction interval

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        Predictions, Lower and Upper Interval Values
    """
    # Predict and get lower and upper intervals
    return_pred_int = forecaster.get_tag("capability:pred_int")

    return_values = forecaster.predict(
        fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha
    )

    if return_pred_int:
        y_pred = return_values[0]
        lower = return_values[1]["lower"]
        upper = return_values[1]["upper"]
    else:
        y_pred = return_values
        lower = pd.Series([np.nan] * len(y_pred))
        upper = pd.Series([np.nan] * len(y_pred))
        lower.index = y_pred.index
        upper.index = y_pred.index

    # Prophet with return_pred_int = True returns datetime index.
    for series in [y_pred, lower, upper]:
        if isinstance(series.index, pd.DatetimeIndex):
            series.index = series.index.to_period()

    return y_pred, lower, upper


def update_additional_scorer_kwargs(
    initial_kwargs: Dict[str, Any],
    y_train: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> Dict[str, Any]:
    """Updates the initial kwargs with additional scorer kwargs
    NOTE: Initial kwargs are obtained from experiment, e.g. {"sp": 12} and
    are common to all folds.
    The additional kwargs such as y_train, lower, upper are specific to each
    fold and must be updated dynamically as such.

    Parameters
    ----------
    initial_kwargs : Dict[str, Any]
        Initial kwargs are obtained from experiment, e.g. {"sp": 12} and
        are common to all folds
    y_train : pd.Series
        Training Data. Used in metrics such as MASE
    lower : pd.Series
        Lower Limits of Predictions. Used in metrics such as INPI
    upper : pd.Series
        Upper Limits of Predictions. Used in metrics such as INPI

    Returns
    -------
    Dict[str, Any]
        Updated kwargs dictionary
    """
    additional_scorer_kwargs = initial_kwargs.copy()
    additional_scorer_kwargs.update(
        {"y_train": y_train, "lower": lower, "upper": upper,}
    )
    return additional_scorer_kwargs
