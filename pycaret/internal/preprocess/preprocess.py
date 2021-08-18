# Author: Mavs (m.524687@gmail.com)
# Last modified: 06/08/2021
# License: MIT


import pandas as pd
import numpy as np
from inspect import signature
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest

from ..utils import to_df, to_series, variable_return


class TransfomerWrapper(BaseEstimator):
    """Meta-estimator for transformers.

    Wrapper for all transformers in preprocess to return a pandas
    dataframe instead of a numpy array. Note that, in order to
    keep the correct column names, the underlying transformer is
    only allowed to add or remove columns, never both.
    From: https://github.com/tvdboom/ATOM/blob/master/atom/utils.py#L815

    """
    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        args = []
        if "X" in signature(self.transformer.fit).parameters:
            args.append(X[X.columns if not self.columns else self.columns])
        if "y" in signature(self.transformer.fit).parameters:
            args.append(y)

        self.transformer.fit(*args, **fit_params)
        return self

    def transform(self, X, y=None):

        def name_cols(array, df):
            """Get the column names after a transformation.

            If the number of columns is unchanged, the original
            column names are returned. Else, give the column a
            default name if the column values changed.

            Parameters
            ----------
            array: np.ndarray
                Transformed dataset.

            df: pd.DataFrame
                Original dataset.

            """
            # If columns were only transformed, return og names
            if array.shape[1] == df.shape[1]:
                return df.columns

            # If columns were added or removed
            temp_cols = []
            for i, col in enumerate(array.T, start=1):
                mask = df.apply(lambda c: all(c == col))
                if any(mask):
                    temp_cols.append(mask[mask].index.values[0])
                else:
                    diff = len(df.columns) - len(X.columns if not self.columns else self.columns)
                    temp_cols.append(f"Feature {str(i + diff)}")

            return temp_cols

        def reorder_cols(df, original_df):
            """Reorder the columns to their original order.

            This function is necessary in case only a subset of the
            columns in the dataset was used. In that case, we need
            to reorder them to their original order.

            Parameters
            ----------
            df: pd.DataFrame
                DataFrame to reorder.

            original_df: pd.DataFrame
                Original dataframe (states the order).

            """
            temp_df = pd.DataFrame()
            for col in list(dict.fromkeys(list(original_df.columns) + list(df.columns))):
                if col in df.columns:
                    temp_df[col] = df[col]
                elif col not in self.columns:
                    temp_df[col] = original_df[col]

                # Derivative cols are added after original
                for col_derivative in df.columns:
                    if col_derivative.startswith(col):
                        temp_df[col_derivative] = df[col_derivative]

            return temp_df

        args = []
        if "X" in signature(self.transformer.transform).parameters:
            args.append(X[X.columns if not self.columns else self.columns])
        if "y" in signature(self.transformer.transform).parameters:
            args.append(y)
        output = self.transformer.transform(*args)

        # Transform can return X, y or both
        if isinstance(output, tuple):
            new_X, new_y = output[0], output[1]
        else:
            if len(output.shape) > 1:
                new_X, new_y = output, None
            else:
                new_X, new_y = None, output

        # Convert to pandas and assign proper column names
        if new_X is not None and not isinstance(new_X, pd.DataFrame):
            new_X = to_df(new_X, columns=name_cols(new_X, X))
        new_X = reorder_cols(new_X, X)
        new_y = to_series(y, name=y.name)

        return variable_return(new_X, new_y)

    def fit_transform(self, *args, **kwargs):
        return self.fit(*args, **kwargs).transform(*args, **kwargs)


class RemoveOutliers(BaseEstimator):
    """Transformer to drop outliers from a dataset."""

    def __init__(self, method="if", threshold=0.05, n_jobs=1, random_state=None):
        self.method = method
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        if self.method == "if":
            self._estimator = IsolationForest(
                n_estimators=100,
                contamination=self.threshold,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            self._estimator.fit(X, y)

    def transform(self, X, y):
        mask = self._estimator.predict(X) != -1
        return X[mask], y[mask]
