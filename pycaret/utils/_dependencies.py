# Adapted from
# https://github.com/sktime/sktime/blob/v0.11.0/sktime/utils/validation/_dependencies.py

from typing import Optional

from skbase.utils.dependencies import _check_soft_dependencies as _skbase_csd
from skbase.utils.dependencies._dependencies import _get_installed_packages

from pycaret.internal.logging import get_logger

logger = get_logger()


def get_module_version_str(modname: str) -> str:
    """Raises a ValueError if module is not installed"""
    versions = _get_installed_packages()
    return versions.get(modname, "Not installed")


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
    packages : str
        package name to check
        str should be package name and/or package version specifications to check.
        Must be a PEP 440 compatible specifier string, for a single package.
        For instance, the PEP 440 compatible package name such as ``"pandas"``;
        or a package requirement specifier string such as ``"pandas>1.2.3"``.

    severity : str, "error" (default), "warning", "none"
        whether the check should raise an error, a warning, or nothing

        * "error" - raises a ``ModuleNotFoundError`` if one of packages is not installed
        * "warning" - raises a warning if one of packages is not installed
          function returns False if one of packages is not installed, otherwise True
        * "none" - does not raise exception or warning
          function returns False if one of packages is not installed, otherwise True

    extra : Optional[str], optional
        The 'extras' that will install this package, by default "all_extras".
        If None, it means that the dependency is not available in optional
        requirements file and must be installed by the user on their own.

    install_name : ignored, present only for backwards compatibility

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
    msg = (
        f"\n'{package}' is a soft dependency and not included in the "
        f"pycaret installation. Please run: `pip install {package}` to install."
    )
    if extra is not None:
        msg += (
            f"\nAlternately, you can install {package} by running "
            f"`pip install pycaret[{extra}]`"
        )

    package_available = _skbase_csd(package, severity=severity, msg=msg)

    if package_available:
        ver = get_module_version_str(package)
        logger.info(
            "Soft dependency imported: {k}: {stat}".format(k=package, stat=str(ver))
        )

    return package_available
