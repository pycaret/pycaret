# Adapted from
# https://github.com/sktime/sktime/blob/v0.11.0/sktime/utils/validation/_dependencies.py

import sys
from distutils.version import LooseVersion
from importlib import import_module
from typing import Dict, Optional, Union

from importlib_metadata import distributions

from pycaret.internal.logging import get_logger, redirect_output

logger = get_logger()

INSTALLED_MODULES = None


def _try_import_and_get_module_version(
    modname: str,
) -> Optional[Union[LooseVersion, bool]]:
    """Returns False if module is not installed, None if version is not available"""
    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            if logger:
                with redirect_output(logger):
                    mod = import_module(modname)
            else:
                mod = import_module(modname)
        try:
            ver = mod.__version__
        except AttributeError:
            # Version could not be obtained
            ver = None
    except ImportError:
        ver = False
    if ver:
        ver = LooseVersion(ver)
    return ver


# Based on packages_distributions() from importlib_metadata
def get_installed_modules() -> Dict[str, Optional[LooseVersion]]:
    """
    Get installed modules and their versions from pip metadata.
    """
    global INSTALLED_MODULES
    if not INSTALLED_MODULES:
        # Get all installed modules and their versions without
        # needing to import them.
        module_versions = {}
        # top_level.txt contains information about modules
        # in the package. It is not always present, in which case
        # the assumption is that the package name is the module name.
        # https://setuptools.pypa.io/en/latest/deprecated/python_eggs.html
        for dist in distributions():
            for pkg in (dist.read_text("top_level.txt") or "").split():
                try:
                    ver = LooseVersion(dist.metadata["Version"])
                except Exception:
                    ver = None
                module_versions[pkg] = ver
        INSTALLED_MODULES = module_versions
    return INSTALLED_MODULES


def _get_module_version(modname: str) -> Optional[Union[LooseVersion, bool]]:
    """Will cache the version in INSTALLED_MODULES

    Returns False if module is not installed."""
    installed_modules = get_installed_modules()
    if modname not in installed_modules:
        # Fallback. This should never happen unless module is not present
        installed_modules[modname] = _try_import_and_get_module_version(modname)
    return installed_modules[modname]


def get_module_version(modname: str) -> Optional[LooseVersion]:
    """Raises a ValueError if module is not installed"""
    version = _get_module_version(modname)
    if version is False:
        raise ValueError(f"Module '{modname}' is not installed.")
    return version


def is_module_installed(modname: str) -> bool:
    try:
        get_module_version(modname)
        return True
    except ValueError:
        return False


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

    package_available = is_module_installed(package)

    if package_available:
        ver = get_module_version(package)
        logger.info(
            "Soft dependency imported: {k}: {stat}".format(k=package, stat=str(ver))
        )
    else:
        msg = (
            f"\n'{package}' is a soft dependency and not included in the "
            f"pycaret installation. Please run: `pip install {install_name}` to install."
        )
        if extra is not None:
            msg += f"\nAlternately, you can install this by running `pip install pycaret[{extra}]`"

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

    return package_available
