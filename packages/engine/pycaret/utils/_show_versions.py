"""Utility methods to print system info for debugging.
adapted from :func:`sktime.show_versions`
adapted from :func:`sklearn.show_versions`
"""

__author__ = ["Nikhil Gupta"]
__all__ = ["show_versions"]

import logging
import platform
import sys
from typing import Optional

from pycaret.utils._dependencies import get_module_version

required_deps = [
    "pycaret",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "joblib",
    "plotly",
    "tqdm",
    "requests",
    "jinja2",
    "IPython",
]

optional_deps = [
    # [notebook]
    "ipywidgets",
    "nbformat",
    # [export]
    "kaleido",
    # [anomaly]
    "pyod",
    "numba",
    # [timeseries]
    "statsmodels",
    "sktime",
    "pmdarima",
    # [interpret]
    "shap",
]


def _get_sys_info():
    """
    System information.
    Return
    ------
    sys_info : dict
        system and Python version information
    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info(optional: bool = False, logger: Optional[logging.Logger] = None):
    """
    Overview of the installed version of dependencies.

    Parameters
    ----------
    optional : bool, optional
        If False returns the required library versions, if True, returns
        optional library versions, by default False.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """

    deps_info = {}

    if optional:
        deps = optional_deps
    else:
        deps = required_deps

    for modname in deps:
        try:
            ver = get_module_version(modname)
            if not ver:
                ver = "Installed but version unavailable"
        except ValueError:
            ver = "Not installed"
        deps_info[modname] = str(ver)

    return deps_info


def show_versions(optional: bool = True, logger: Optional[logging.Logger] = None):
    """Print useful debugging information (e.g. versions).

    Parameters
    ----------
    optional : bool, optional
        Should optional dependencies be documented, by default True
    logger : Optional[logging.Logger], optional
        The logger to use. If None, then uses print() command to display results,
        by default None
    """

    if logger is None:
        print_func = print
        prefix = "\n"
    else:
        print_func = logger.info
        prefix = ""

    print_func(f"{prefix}System:")  # noqa: T001
    sys_info = _get_sys_info()
    for k, stat in sys_info.items():
        print_func("{k:>10}: {stat}".format(k=k, stat=stat))  # noqa: T001

    print_func(f"{prefix}PyCaret required dependencies:")  # noqa: T001
    optional_deps_info = _get_deps_info(logger=logger)
    for k, stat in optional_deps_info.items():
        print_func("{k:>20}: {stat}".format(k=k, stat=stat))  # noqa: T001

    if optional:
        print_func(f"{prefix}PyCaret optional dependencies:")  # noqa: T001
        optional_deps_info = _get_deps_info(logger=logger, optional=True)
        for k, stat in optional_deps_info.items():
            print_func("{k:>20}: {stat}".format(k=k, stat=stat))  # noqa: T001
