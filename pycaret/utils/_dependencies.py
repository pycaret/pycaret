# Adapted from
# https://github.com/alan-turing-institute/sktime/blob/v0.11.0/sktime/utils/validation/_dependencies.py

import sys
from importlib import import_module
from typing import Optional

from pycaret.internal.logging import get_logger
from pycaret.utils._show_versions import _get_module_version

logger = get_logger()


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
    if package in sys.modules:
        package_available = True
    else:
        try:
            import_module(package)
            package_available = True
        except ModuleNotFoundError as e:
            msg = (
                f"{e}."
                f"\n'{package}' is a soft dependency and not included in the "
                f"pycaret installation. Please run: `pip install {install_name}` to install."
            )
            if extra is not None:
                msg = (
                    msg
                    + f"\nAlternately, you can install this by running `pip install pycaret[{extra}]`"
                )

            if severity == "error":
                logger.exception(f"{msg}")
                raise ModuleNotFoundError(msg)
            elif severity == "warning":
                logger.warning(f"{msg}")
                package_available = False
            else:
                raise RuntimeError(
                    "Error in calling _check_soft_dependencies, severity "
                    f'argument must be "error" or "warning", found "{severity}".'
                )

    if package_available:
        ver = _get_module_version(sys.modules[package])
        logger.info("Sof dependency imported: {k}: {stat}".format(k=package, stat=ver))

    return package_available
