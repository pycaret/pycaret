from typing import Any, Dict, Tuple, Union, Optional, Union, List

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon


PyCaretForecastingHorizonTypes = Union[List[int], int, np.ndarray, ForecastingHorizon]


def get_predictions_with_intervals(
    forecaster,
    X: pd.DataFrame,
    fh=None,
    alpha: float = 0.05,
    merge: bool = False,
    round: Optional[int] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
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
    merge : bool, default = False
        If True, returns a dataframe with 3 columns called
        ["y_pred", "lower", "upper"], else retruns 3 separate series.
    round : Optional[int], default = None
        If set to an integer value, returned values are rounded to as many digits
        If set to None, no rounding is performed.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        Predictions, Lower and Upper Interval Values
    """
    # Predict and get lower and upper intervals
    return_pred_int = forecaster.get_tag("capability:pred_int")

    return_values = forecaster.predict(
        fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha
    )

    if return_pred_int:
        y_pred = pd.DataFrame({"y_pred": return_values[0]})
        lower = pd.DataFrame({"lower": return_values[1]["lower"]})
        upper = pd.DataFrame({"upper": return_values[1]["upper"]})
    else:
        y_pred = pd.DataFrame({"y_pred": return_values})
        lower = pd.DataFrame({"lower": [np.nan] * len(y_pred)})
        upper = pd.DataFrame({"upper": [np.nan] * len(y_pred)})
        lower.index = y_pred.index
        upper.index = y_pred.index

    # PyCaret works on Period Index only when developing models. If user passes
    # DateTimeIndex, it gets converted to PeriodIndex. If the forecaster (such as
    # Prophet) does not support PeriodIndex, then a patched version is created
    # which can support a PeriodIndex input and returns a PeriodIndex prediction.
    # Hence, no casting of index needs to be done here.

    if round is not None:
        # Converting to float since rounding does not support int
        y_pred = y_pred.astype(float).round(round)
        lower = lower.astype(float).round(round)
        upper = upper.astype(float).round(round)

    if merge:
        results = pd.concat([y_pred, lower, upper], axis=1)
        return results
    else:
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
        {"y_train": y_train, "lower": lower, "upper": upper}
    )
    return additional_scorer_kwargs
