import platform
import sys

from pycaret.utils._show_versions import show_versions

version_ = "3.4.0"

__version__ = version_

__all__ = ["show_versions", "__version__"]

# Pycaret only supports python 3.10, 3.11, 3.12, 3.13
# This code is to avoid issues with python 3.7 or other not supported versions
# example (see package versions): https://github.com/pycaret/pycaret/issues/3746

_version_info = sys.version_info
_version_tuple = tuple(int(x) for x in platform.python_version_tuple()[:2])

if _version_tuple < (3, 10):  # basedpyright: ignore[reportUnreachable]
    raise RuntimeError(
        "Pycaret only supports python 3.10, 3.11, 3.12, 3.13. Your actual Python version: ",
        _version_info,
        "Please UPGRADE your Python version.",
    )
elif _version_tuple >= (3, 14):  # basedpyright: ignore[reportUnreachable]
    raise RuntimeError(
        "Pycaret only supports python 3.10, 3.11, 3.12, 3.13. Your actual Python version: ",
        _version_info,
        "Please DOWNGRADE your Python version.",
    )
