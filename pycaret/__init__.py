import os
import sys

from pycaret.utils import __version__
from pycaret.utils._show_versions import show_versions

__all__ = ["show_versions", "__version__"]

# Fix for joblib on 3.7
# Otherwise, it will default to cloudpickle which
# uses pickle5, causing exceptions with joblib
if sys.version_info < (3, 8):
    os.environ["LOKY_PICKLER"] = "pickle"
