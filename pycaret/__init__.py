import os
import sys

# Fix for joblib on 3.7
# Otherwise, it will default to cloudpickle which
# uses pickle5, causing exceptions with joblib
# THIS HAS TO BE DONE BEFORE OTHER IMPORTS HERE
if sys.version_info < (3, 8):
    import pycaret.internal.cloudpickle_compat

    os.environ["LOKY_PICKLER"] = pycaret.internal.cloudpickle_compat.__name__

    from joblib.externals.loky.backend.reduction import set_loky_pickler

    set_loky_pickler(pycaret.internal.cloudpickle_compat.__name__)


from pycaret.utils._show_versions import show_versions

version_ = "3.0.3"

__version__ = version_

__all__ = ["show_versions", "__version__"]
