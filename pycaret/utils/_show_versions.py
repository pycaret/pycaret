"""Utility methods to print system info for debugging.
adapted from :func:`sktime.show_versions`
adapted from :func:`sklearn.show_versions`
"""

__author__ = ["Nikhil Gupta"]
__all__ = ["show_versions"]

import importlib
import logging
import platform
import sys
from typing import Optional

from pycaret.internal.logging import redirect_output

required_deps = [
    "pip",
    "setuptools",
    "pycaret",
    "IPython",
    "ipywidgets",
    "tqdm",
    "numpy",
    "pandas",
    "jinja2",
    "scipy",
    "joblib",
    "sklearn",
    "pyod",
    "imblearn",
    "category_encoders",
    "lightgbm",
    "numba",
    "requests",
    "matplotlib",
    "scikitplot",
    "yellowbrick",
    "plotly",
    "kaleido",
    "statsmodels",
    "sktime",
    "tbats",
    "pmdarima",
    "psutil",
]

optional_deps = [
    "shap",
    "interpret",
    "umap",
    "pandas_profiling",
    "explainerdashboard",
    "autoviz",
    "fairlearn",
    "xgboost",
    "catboost",
    "kmodes",
    "mlxtend",
    "statsforecast",
    "tune_sklearn",
    "ray",
    "hyperopt",
    "optuna",
    "skopt",
    "mlflow",
    "gradio",
    "fastapi",
    "uvicorn",
    "m2cgen",
    "evidently",
    "nltk",
    "pyLDAvis",
    "gensim",
    "spacy",
    "wordcloud",
    "textblob",
    "fugue",
    "streamlit",
    "prophet",
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


def _get_module_version(module) -> str:
    try:
        return module.__version__
    except AttributeError:
        #### Version could not be obtained
        return "Installed but version unavailable"


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
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                with redirect_output(logger):
                    mod = importlib.import_module(modname)
            ver = _get_module_version(mod)
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = "Not installed"

    return deps_info


def show_versions(optional: bool = False, logger: Optional[logging.Logger] = None):
    """Print useful debugging information (e.g. versions).

    Parameters
    ----------
    optional : bool, optional
        Should optional dependencies be documented, by default False
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
