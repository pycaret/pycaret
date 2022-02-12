"""Utility methods to print system info for debugging.
adapted from :func:`sktime.show_versions`
adapted from :func:`sklearn.show_versions`
"""

__author__ = ["Nikhil Gupta"]
__all__ = ["show_versions"]

import importlib
import platform
import sys


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


def _get_deps_info():
    """
    Overview of the installed version of main dependencies.
    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = [
        "pip",
        "setuptools",
        "pycaret",
        "sklearn",
        "sktime",
        "statsmodels",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "plotly",
        "joblib",
        "numba",
        "mlflow",
        "lightgbm",
        "xgboost",
        "pmdarima",
        "tbats",
        "prophet",
        "tsfresh",
    ]

    def get_version(module):
        return module.__version__

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = get_version(mod)
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = "Not installed"
        except AttributeError:
            #### Version could not be obtained
            deps_info[modname] = "Installed but version unavailable"

    return deps_info


def show_versions():
    """Print useful debugging information."""
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")  # noqa: T001
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))  # noqa: T001

    print("\nPython dependencies:")  # noqa: T001
    for k, stat in deps_info.items():
        print("{k:>13}: {stat}".format(k=k, stat=stat))  # noqa: T001
