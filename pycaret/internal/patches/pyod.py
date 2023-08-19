import numpy as np
import pandas as pd
from pyod.models.cblof import CBLOF


def convert_to_fp64(X):
    if isinstance(X, pd.DataFrame):
        X = X.astype(
            {
                col: np.float64
                for col in X.columns
                if X.dtypes[col] in (np.float32, np.float16)
            }
        )
    elif X.dtype == np.float32:
        X = X.astype(np.float64)
    return X


# Fixes https://github.com/pycaret/pycaret/issues/3606
class CBLOFForceToDouble(CBLOF):
    """CBLOF with forced float32 -> float64 conversion"""

    def fit(self, X, y=None):
        X = convert_to_fp64(X)
        return super().fit(X, y)

    def decision_function(self, X):
        X = convert_to_fp64(X)
        return super().decision_function(X)
