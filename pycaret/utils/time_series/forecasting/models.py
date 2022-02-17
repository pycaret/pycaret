from pycaret.utils.time_series import TSExogenousPresent


def _disable_pred_int_enforcement(forecaster, enforce_pi: bool) -> bool:
    """Checks to see if prediction interval should be enforced. If it should
    but the forecaster does not support it, the container will be disabled.

    Parameters
    ----------
    forecaster : `sktime` compatible forecaster
        forecaster to check for prediction interval capability.
        Can be a dummy object of the forecasting class
    enforce_pi : bool
        Should prediction interval be enforced?

    Returns
    -------
    bool
        True if user wants to enforce prediction interval and forecaster
        does not supports it. False otherwise.
    """
    if enforce_pi and not forecaster.get_tag("capability:pred_int"):
        return True
    return False


def _disable_exogenous_enforcement(
    forecaster, enforce_exogenous: bool, exp_has_exogenous: TSExogenousPresent
) -> bool:
    """Checks to see if exogenous support should be enforced. If it should
    but the forecaster does not support it, the container will be disabled.
    NOTE: Only enforced if the experiment has exogenous variables. If it does not,
    then no models are disabled.

    Parameters
    ----------
    forecaster : `sktime` compatible forecaster
        forecaster to check for prediction interval capability.
        Can be a dummy object of the forecasting class
    enforce_exogenous : bool
        Should exogenous support be enforced?
    exp_has_exogenous : TSExogenousTypes
        Whether the experiment has exogenous variables or not?

    Returns
    -------
    bool
        True if user wants to enforce exogenous support and forecaster
        does not supports it. False otherwise.
    """

    # Disable models only if the experiment has exogenous variables
    if (
        exp_has_exogenous == TSExogenousPresent.YES
        and enforce_exogenous
        and forecaster.get_tag("ignores-exogeneous-X")
    ):
        return True
    return False


def _check_enforcements(forecaster, globals_dict) -> bool:
    """Checks whether the model supports certain features such as
    (1) Prediction Interval, and (2) support for exogenous variables. The checks
    depend on what features are requested by the user during the experiment setup.

    Parameters
    ----------
    forecaster : sktime compatible forecaster
        The forecaster which needs to be checked for feature support
    globals_dict : dict_
        Used to check what features are requested by the user during setup.
        TODO: To be replaced with experiment object after preprocessing is added.

    Returns
    -------
    bool
        True if the model should remain active in the experiment, False otherwise.
    """

    active = True

    #### Pred Interval Enforcement ----
    disable_pred_int = _disable_pred_int_enforcement(
        forecaster=forecaster, enforce_pi=globals_dict["enforce_pi"]
    )

    #### Exogenous variable support Enforcement ----
    disable_exog_enforcement = _disable_exogenous_enforcement(
        forecaster=forecaster,
        enforce_exogenous=globals_dict["enforce_exogenous"],
        exp_has_exogenous=globals_dict["exogenous_present"],
    )

    if disable_pred_int or disable_exog_enforcement:
        active = False

    return active
