from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

PyCaretForecastingHorizonTypes = Union[List[int], int, np.ndarray, ForecastingHorizon]


def _check_and_clean_coverage(coverage: Union[float, List[float]]) -> List[float]:
    """Checks the coverage to make sure it is of the allowed types (float or List).
    Returns the lower and upper quantiles for the coverage.

    If it is of type
    (1) List, it must have a length 2 indicating the lower and upper quantiles.
    (2) float, it is converted to a List of length 2 indicating the lower and
    upper quantiles.

    Parameters
    ----------
    coverage : Union[float, List[float]]
        The coverage value to be checked and cleaned

    Returns
    -------
    List[float]
        A list of length 2, indicating the quantiles corresponding to the lower
        and upper intervals

    Raises
    ------
    TypeError
        coverage is not of the correct type - List or float
    ValueError
        If coverage is of type List but not of length 2
    """
    if not isinstance(coverage, (float, list)):
        raise TypeError(
            f"'coverage' must be of type float or a List of floats of length 2. "
            f"You passed coverage of type: '{type(coverage)}'"
        )
    elif isinstance(coverage, list) and len(coverage) != 2:
        raise ValueError(
            "When coverage is a list, it must be of length 2 corresponding to the "
            f"lower and upper quantile of the prediction. You specified: '{coverage}'"
        )

    if isinstance(coverage, float):
        lower_quantile = (1 - coverage) / 2
        upper_quantile = 1 - lower_quantile
        coverage = [lower_quantile, upper_quantile]
    elif isinstance(coverage, list):
        coverage.sort()

    return coverage


def get_predictions_with_intervals(
    forecaster,
    alpha: Optional[float],
    coverage: Union[float, List[float]],
    X: pd.DataFrame,
    fh: Optional[ForecastingHorizon] = None,
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
    alpha: Optional[float]
        The alpha (quantile) value to use for the point predictions.
    coverage: Union[float, List[float]]
        The coverage to be used for prediction intervals.
    X : pd.DataFrame
        Exogenous Variables
    fh : Optional[ForecastingHorizon], default = None
        The forecasting horizon to use for the predictions
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

    coverage = _check_and_clean_coverage(coverage)

    # Predict and get lower and upper intervals
    return_pred_int = forecaster.get_tag("capability:pred_int")

    if alpha is not None and return_pred_int is False:
        raise ValueError(
            "\nWhen alpha is not None, sktime `predict_quantiles()` is used to get "
            "the predictions instead of `predict()`.\nThis forecaster does not "
            "support `predict_quantiles()`. Please leave `alpha` as `None`."
        )

    # Get Point predictions ----
    if alpha is None:
        y_pred = forecaster.predict(fh=fh, X=X)
        y_pred = pd.DataFrame({"y_pred": y_pred})
    else:
        y_pred = forecaster.predict_quantiles(fh=fh, X=X, alpha=alpha)
        if y_pred.shape[1] != 1:
            raise ValueError(
                "Something wrong happened during point prediction; received values "
                "This should not have happened. Please report on GitHub."
            )
        y_pred = pd.DataFrame({"y_pred": y_pred.iloc[:, 0]})

    # Get Intervals ----
    if return_pred_int:
        pred_intervals = forecaster.predict_quantiles(fh=fh, X=X, alpha=coverage)
        lower = pd.DataFrame({"lower": pred_intervals.iloc[:, 0]})
        upper = pd.DataFrame({"upper": pred_intervals.iloc[:, 1]})
    else:
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
