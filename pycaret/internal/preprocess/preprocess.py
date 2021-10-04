# Author: Mavs (m.524687@gmail.com)
# Last modified: 06/08/2021
# License: MIT


import pandas as pd
import numpy as np
from inspect import signature
from scipy.sparse import issparse
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..utils import to_df, to_series, variable_return


class TransfomerWrapper(BaseEstimator):
    """Meta-estimator for transformers.

    Wrapper for all transformers in preprocess to return a pandas
    dataframe instead of a numpy array. Note that, in order to
    keep the correct column names, the underlying transformer is
    only allowed to add or remove columns, never both.
    From: https://github.com/tvdboom/ATOM/blob/master/atom/pipeline.py

    """

    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns

    def __repr__(self, N_CHAR_MAX=1400):
        return self.transformer.__repr__(N_CHAR_MAX)

    def fit(self, X, y=None, **fit_params):
        if self.columns is None:
            self.columns = list(X.columns)
        elif not self.columns:
            return self

        args = []
        if "X" in signature(self.transformer.fit).parameters:
            args.append(X[self.columns])
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
            if array.shape[1] == len(self.columns):
                return self.columns

            # If columns were added or removed
            temp_cols = []
            for i, col in enumerate(array.T, start=1):
                mask = df.apply(lambda c: all(c == col))
                if any(mask):
                    temp_cols.append(mask[mask].index.values[0])
                else:
                    diff = len(df.columns) - len(self.columns)
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

        if self.columns is None:
            self.columns = list(X.columns)
        elif not self.columns:
            return variable_return(X, y)

        args = []
        if "X" in signature(self.transformer.transform).parameters:
            args.append(X[self.columns])
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
            # If sparse matrix, convert back to array
            if issparse(new_X):
                new_X = new_X.toarray()

            new_X = to_df(new_X, columns=name_cols(new_X, X))

        new_X = reorder_cols(new_X, X)
        new_y = to_series(y, name=y.name)

        return variable_return(new_X, new_y)

    def fit_transform(self, *args, **kwargs):
        return self.fit(*args, **kwargs).transform(*args, **kwargs)


class ExtractDateTimeFeatures(BaseEstimator):
    """Extract features from datetime columns."""

    def __init__(self, features=["day", "month", "year"]):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in X:
            if col.dtype != "datetime64":
                raise TypeError(f"Column {col} has no dtype datetime64!")

            for fx in self.features:
                values = getattr(col.dt, fx)

                # Only create feature if values contains less than 30% NaTs
                if values.isna().sum() <= 0.3 * len(values):
                    X.insert(X.columns.get_loc(col) + 1, f"{col}_{fx}", values)

            X = X.drop(col, axis=1)  # Drop the original datetime column

        return X


class EmbedTextFeatures(BaseEstimator):
    """Embed text features to an array representation."""

    def __init__(self, method="tf-idf", **kwargs):
        self.method = method
        self.kwargs = kwargs
        self._estimators = {}

    def fit(self, X, y=None):
        if self.method.lower() == "bow":
            estimator = CountVectorizer(**self.kwargs)
        else:
            estimator = TfidfVectorizer(**self.kwargs)

        # Fit every text column in a separate estimator
        for col in X:
            self._estimators[col] = clone(estimator).fit(X[col])

        return self

    def transform(self, X, y=None):
        for col in X:
            matrix = self._estimators[col].transform(X[col]).toarray()
            for i, word in enumerate(self._estimators[col].get_feature_names()):
                X[f"{col}_{word}"] = matrix[:, i]

            X = X.drop(col, axis=1)  # Drop original column

        return X


class RemoveMulticollinearity(BaseEstimator):
    """Drop multicollinear features."""

    def __init__(self, threshold=1):
        self.threshold = threshold
        self._drop = None

    def fit(self, X, y=None):
        mtx = X.corr()  # Pearson correlation coefficient matrix

        # Extract the upper triangle of the correlation matrix
        upper = mtx.where(np.triu(np.ones(mtx.shape).astype(bool), k=1))

        # Select the features with correlations above the threshold
        self._drop = [i for i in upper.columns if any(abs(upper[i] >= self.threshold))]

        return self

    def transform(self, X, y=None):
        return X.drop(self._drop, axis=1)


class RemoveOutliers(BaseEstimator):
    """Transformer to drop outliers from a dataset."""

    def __init__(self, method="iforest", threshold=0.05, n_jobs=1, random_state=None):
        self.method = method
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        if self.method.lower() == "iforest":
            self._estimator = IsolationForest(
                n_estimators=100,
                contamination=self.threshold,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        elif self.method.lower() == "ee":
            self._estimator = EllipticEnvelope(
                contamination=self.threshold,
                random_state=self.random_state,
            )
        elif self.method.lower() == "lof":
            self._estimator = LocalOutlierFactor(
                contamination=self.threshold,
                n_jobs=self.n_jobs,
            )

        return self

    def transform(self, X, y):
        mask = self._estimator.fit_predict(X) != -1
        return X[mask], y[mask]
