# Adapted from
# https://github.com/alan-turing-institute/sktime/blob/v0.11.0/sktime/utils/validation/_dependencies.py

from importlib import import_module
from typing import Optional
import logging


def _check_soft_dependencies(
    package: str,
    severity: str = "error",
    extra: Optional[str] = "all_extras",
    install_name: Optional[str] = None,
) -> bool:
    """Check if all soft dependencies are installed and raise appropriate error message
    when not.

    Parameters
    ----------
    package : str
        Package to check
    severity : str, optional
        Whether to raise an error ("error") or just a warning message ("warning"),
        by default "error"
    extra : Optional[str], optional
        The 'extras' that will install this package, by default "all_extras".
        If None, it means that the dependency is not available in optional
        requirements file and must be installed by the user on their own.
    install_name : Optional[str], optional
        The package name to install, by default None
        If none, the name in `package` argument is used

    Returns
    -------
    bool
        If error is set to "warning", returns True if package can be imported or False
        if it can not be imported

    Raises
    ------
    ModuleNotFoundError
        User friendly error with suggested action to install all required soft
        dependencies
    RuntimeError
        Is the severity argument is not one of the allowed values
    """
    install_name = install_name or package
    try:
        import_module(package)
        package_available = True
    except ModuleNotFoundError as e:
        msg = (
            f"\n{e}."
            f"\n'{package}' is a soft dependency and not included in the "
            f"pycaret installation. Please run: `pip install {install_name}` to install."
        )
        if extra is not None:
            msg = (
                msg
                + f"\nAlternately, you can install this by running `pip install pycaret[{extra}]`"
            )

        if severity == "error":
            raise ModuleNotFoundError(msg)
        elif severity == "warning":
            logging.warning(f"{msg}")
            package_available = False
        else:
            raise RuntimeError(
                "Error in calling _check_soft_dependencies, severity "
                f'argument must be "error" or "warning", found "{severity}".'
            )

    return package_available
