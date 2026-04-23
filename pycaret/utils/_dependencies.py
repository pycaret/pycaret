"""Soft-dependency introspection helpers.

Originally adapted from sktime; rewritten for PyCaret 4.0 to drop:
- `distutils.version.LooseVersion` (removed in Python 3.12)
- the `importlib_metadata` backport (stdlib `importlib.metadata` is enough on >=3.11)

Versions are now returned as `packaging.version.Version` objects, which support
rich comparison against version strings via `Version(...)` and tolerate modern
PEP 440 semantics (including pre/post/dev markers).
"""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.metadata import distributions
from typing import Optional, Union

from packaging.version import InvalidVersion, Version

from pycaret.internal.logging import get_logger, redirect_output

logger = get_logger()

INSTALLED_MODULES: Optional[dict[str, Optional[Version]]] = None


def _parse(v: str) -> Optional[Version]:
    try:
        return Version(v)
    except InvalidVersion:
        return None


def _try_import_and_get_module_version(modname: str) -> Optional[Union[Version, bool]]:
    """Returns False if module is not installed, None if version is not available."""
    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            if logger:
                with redirect_output(logger):
                    mod = import_module(modname)
            else:
                mod = import_module(modname)
        ver_str = getattr(mod, "__version__", None)
    except ImportError:
        return False
    return _parse(ver_str) if ver_str else None


def get_installed_modules() -> dict[str, Optional[Version]]:
    """Map installed top-level module name -> parsed Version."""
    global INSTALLED_MODULES
    if INSTALLED_MODULES is None:
        module_versions: dict[str, Optional[Version]] = {}
        for dist in distributions():
            version_str = dist.metadata.get("Version") if dist.metadata else None
            ver = _parse(version_str) if version_str else None
            for pkg in (dist.read_text("top_level.txt") or "").split():
                module_versions[pkg] = ver
            # Fallback: use normalized distribution name as a module hint.
            if dist.metadata and dist.metadata.get("Name"):
                module_versions.setdefault(
                    dist.metadata["Name"].replace("-", "_"), ver
                )
        INSTALLED_MODULES = module_versions
    return INSTALLED_MODULES


def _get_module_version(modname: str) -> Optional[Union[Version, bool]]:
    installed = get_installed_modules()
    if modname not in installed:
        installed[modname] = _try_import_and_get_module_version(modname)
    return installed[modname]


def get_module_version(modname: str) -> Optional[Version]:
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
    """Check if a soft dependency is installed; raise or warn if not.

    Parameters
    ----------
    package : str
        Module name to check.
    severity : str
        "error" (default) raises ModuleNotFoundError; "warning" logs and returns False.
    extra : str, optional
        Name of the `pip install pycaret[<extra>]` that would install this package.
    install_name : str, optional
        The pip name if it differs from the module name.
    """
    install_name = install_name or package
    package_available = is_module_installed(package)

    if package_available:
        ver = get_module_version(package)
        logger.info("Soft dependency imported: %s: %s", package, ver)
        return True

    msg = (
        f"\n'{package}' is a soft dependency and not included in the "
        f"pycaret installation. Install it with: `pip install {install_name}`."
    )
    if extra is not None:
        msg += f"\nOr install the extras bundle: `pip install pycaret[{extra}]`."

    if severity == "error":
        logger.exception(msg)
        raise ModuleNotFoundError(msg)
    if severity == "warning":
        logger.warning(msg)
        return False
    raise RuntimeError(
        f'severity must be "error" or "warning", got "{severity}".'
    )
