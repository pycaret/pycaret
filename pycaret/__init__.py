import sys

from pycaret.utils._show_versions import show_versions

# Pycaret only supports python 3.8.x up to 3.10.x
if not (3.8 <= sys.version_info.major < 3.11):
    raise Exception(
        "PyCaret requires Python 3.8 to 3.10. Please upgrade your Python version."
    )

version_ = "3.1.0"

__version__ = version_

__all__ = ["show_versions", "__version__"]
