import sys

from pycaret.utils._show_versions import show_versions

version_ = "3.4.0"

__version__ = version_

__all__ = ["show_versions", "__version__"]

# Pycaret only supports python 3.9, 3.10, 3.11, 3.12
# This code is to avoid issues with python 3.7 or other not supported versions
# example (see package versions): https://github.com/pycaret/pycaret/issues/3746

if sys.version_info < (3, 9):
    raise RuntimeError(
        "Pycaret only supports python 3.9, 3.10, 3.11, 3.12. Your actual Python version: ",
        sys.version_info,
        "Please UPGRADE your Python version.",
    )
elif sys.version_info >= (3, 13):
    raise RuntimeError(
        "Pycaret only supports python 3.9, 3.10, 3.11, 3.12. Your actual Python version: ",
        sys.version_info,
        "Please DOWNGRADE your Python version.",
    )
