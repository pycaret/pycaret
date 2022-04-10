# Adapted from
# https://github.com/alan-turing-institute/sktime/blob/v0.11.0/sktime/utils/validation/_dependencies.py

from importlib import import_module
from typing import Optional
import logging


def _check_soft_dependencies(
    package: str, error: str = "raise", extra: str = "all_extras"
) -> bool:
    """Check if all soft dependencies are installed and raise appropriate error message
    when not.

    Parameters
    ----------
    package : str
        Package to check
    error : str, optional
        Whether to raise an error ("raise") or just a warning message ("warn"),
        by default "raise"
    extra : str, optional
        The 'extras' that will install this package, by default "all_extras"

    Returns
    -------
    bool
        If error is set to "warn", returns True if package can be imported or False
        if it can not be imported

    Raises
    ------
    ModuleNotFoundError
        User friendly error with suggested action to install all required soft
        dependencies
    """
    try:
        import_module(package)
        package_available = True
    except ModuleNotFoundError as e:
        msg = (
            f"\n{e}."
            f"\n'{package}' is a soft dependency and not included in the "
            f"pycaret installation. Please run: `pip install {package}`. "
            f"\nAlternately, you can install this by running `pip install pycaret[{extra}]`"
        )
        if error == "raise":
            raise ModuleNotFoundError(msg)
        else:
            logging.warning(f"{msg}")
            package_available = False

    return package_available
