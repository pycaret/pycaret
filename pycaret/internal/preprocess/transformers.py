# Author: Mavs (m.524687@gmail.com)
# License: MIT


import pandas as pd
import numpy as np
from inspect import signature
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
    From: https://github.com/tvdboom/ATOM/blob/master/atom/utils.py

    Parameters
    ----------
    transformer: estimator
        Transformer to wrap. Should implement a `fit` and/or `transform`
        method.

    include: list or None
        Columns to apply on the transformer. If specified, only these
        columns are used and the rest ignored. If None, all columns
        are used.

    exclude: list or None
        Columns to NOT apply on the transformer. If None, no columns
        are excluded.

    """

    def __init__(self, transformer, include=None, exclude=None):
        self.transformer = transformer
        self.include = include
        self.exclude = exclude

        self._train_only = getattr(transformer, "_train_only", False)
        self._include = self.include
        self._exclude = self.exclude or []

    def __repr__(self, N_CHAR_MAX=1400):
        return self.transformer.__repr__()

    def fit(self, X=None, y=None, **fit_params):
        args = []
        transformer_params = signature(self.transformer.fit).parameters
        if "X" in transformer_params and X is not None:
            if self._include is None:
                self._include = [c for c in X.columns if c in X and c not in self._exclude]
            elif not self._include:  # Don't fit if empty list
                return self
            else:
                self._include = [c for c in self._include if c in X and c not in self._exclude]
            args.append(X[self._include])
        if "y" in transformer_params and y is not None:
            args.append(y)

        self.transformer.fit(*args, **fit_params)
        return self

    def transform(self, X=None, y=None):

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
            if array.shape[1] == len(self._include):
                return self._include

            # If columns were added or removed
            temp_cols = []
            for i, col in enumerate(array.T, start=2):
                mask = df.apply(lambda c: np.array_equal(c, col, equal_nan=True))
                if any(mask) and mask[mask].index.values[0] not in temp_cols:
                    temp_cols.append(mask[mask].index.values[0])
                else:
                    # If the column is new, use a default name
                    counter = 1
                    while True:
                        n = f"feature {i + counter + df.shape[1] - len(self._include)}"
                        if (n not in df or n in self._include) and n not in temp_cols:
                            temp_cols.append(n)
                            break
                        else:
                            counter += 1

            return temp_cols

        def reorder_cols(df, original_df):
            """Reorder the columns to their original order.

            This function is necessary in case only a subset of the
            columns in the dataset was used. In that case, we need
            to reorder them to their original order.

            Parameters
            ----------
            df: pd.DataFrame
                Dataset to reorder.

            original_df: pd.DataFrame
                Original dataframe (states the order).

            """
            # Check if columns returned by the transformer are already in the dataset
            for col in df.columns:
                if col in original_df.columns and col not in self._include:
                    raise RuntimeError(
                        f"Column '{col}' returned by the transformer "
                        "already exists in the original dataset."
                    )

            temp_df = pd.DataFrame(index=df.index)
            for col in dict.fromkeys(list(original_df.columns) + list(df.columns)):
                if col in df.columns:
                    temp_df[col] = df[col]
                elif col not in self._include:
                    if len(df) != len(original_df):
                        raise ValueError(
                            f"Length of values ({len(df)}) does not match length of "
                            f"index ({len(original_df)}). This usually happens when "
                            "transformations that drop rows aren't applied on all "
                            "the columns."
                        )

                    # Take values to adapt to new index
                    temp_df[col] = original_df[col].values

                # Derivative cols are added after original (e.g. for one-hot encoding)
                for col_derivative in df.columns:
                    if str(col_derivative).startswith(f"{col}_"):
                        temp_df[col_derivative] = df[col_derivative]

            return temp_df

        def prepare_df(out):
            """Convert to df and set correct column names and order."""
            # Convert to pandas and assign proper column names
            if not isinstance(out, pd.DataFrame):
                out = to_df(out, index=X.index, columns=name_cols(out, X))

            # Reorder columns if only a subset was used
            if len(self._include) != X.shape[1]:
                return reorder_cols(out, X)
            else:
                return out

        X = to_df(X, index=getattr(y, "index", None))
        y = to_series(y, index=getattr(X, "index", None))

        args = []
        transform_params = signature(self.transformer.transform).parameters
        if "X" in transform_params:
            if X is not None:
                if self._include is None:
                    self._include = [c for c in X.columns if c in X and c not in self._exclude]
                elif not self._include:  # Don't transform if empty list
                    return variable_return(X, y)
            else:
                return variable_return(X, y)
            args.append(X[self._include])
        if "y" in transform_params:
            if y is not None:
                args.append(y)
            elif "X" not in transform_params:
                return X, y

        output = self.transformer.transform(*args)

        # Transform can return X, y or both
        if isinstance(output, tuple):
            new_X = prepare_df(output[0])
            new_y = to_series(output[1], index=new_X.index, name=y.name)
        else:
            if len(output.shape) > 1:
                new_X = prepare_df(output)
                new_y = y if y is None else y.set_axis(new_X.index)
            else:
                new_y = to_series(output, index=y.index, name=y.name)
                new_X = X if X is None else X.set_index(new_y.index)

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
            if not X[col].dtype.name.startswith("datetime"):
                raise TypeError(f"Column {col} has no dtype datetime64!")

            for fx in self.features:
                values = getattr(X[col].dt, fx)

                # Only create feature if values contains less than 30% NaTs
                if values.isna().sum() <= 0.3 * len(values):
                    X.insert(X.columns.get_loc(col) + 1, f"{col}_{fx}", values)

            X = X.drop(col, axis=1)  # Drop the original datetime column

        return X


class DropImputer(BaseEstimator):
    """Drop rows with missing values."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.dropna(subset=self.columns, axis=0)
        if y is not None:
            y = y[y.index.isin(X.index)]

        return X, y


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

        self._estimator = None
        self._train_only = True

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
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

        mask = self._estimator.fit_predict(X) != -1
        if y is None:
            return X[mask]
        else:
            return X[mask], y[mask]


class FixImbalancer(BaseEstimator):
    """Wrapper for a balancer with a fit_resample method.

    Balancing classes should only be used on the training set,
    therefore this estimator is skipped by the pipeline when
    making new predictions (only used to fit).

    """

    def __init__(self, estimator):
        self.estimator = estimator
        self._train_only = True

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        return self.estimator.fit_resample(X, y)
