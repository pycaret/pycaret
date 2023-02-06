# Author: Mavs (m.524687@gmail.com)
# License: MIT


import re
from collections import defaultdict
from inspect import signature

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor

from pycaret.utils.generic import to_df, to_series, variable_return


class TransformerWrapper(BaseEstimator, TransformerMixin):
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

    include: list or None, default=None
        Columns to apply on the transformer. If specified, only these
        columns are used and the rest ignored. If None, all columns
        are used.

    exclude: list or None, default=None
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
        self._feature_names_in = None

    @property
    def feature_names_in_(self):
        return self._feature_names_in

    def _name_cols(self, array, df):
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

    def _reorder_cols(self, df, original_df):
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
        for col in df:
            if col in original_df and col not in self._include:
                raise ValueError(
                    f"Column '{col}' returned by transformer {self.transformer} "
                    "already exists in the original dataset."
                )

        # Force new indices on old dataset for merge
        try:
            original_df.index = df.index
        except ValueError:  # Length mismatch
            raise IndexError(
                f"Length of values ({len(df)}) does not match length of "
                f"index ({len(original_df)}). This usually happens when "
                "transformations that drop rows aren't applied on all "
                "the columns."
            )

        # Define new column order
        columns = []
        for col in original_df:
            if col in df or col not in self._include:
                columns.append(col)

            # Add all derivative columns: cols that originate from another
            # and start with its progenitor name, e.g. one-hot encoded columns
            columns.extend(
                [
                    c
                    for c in df.columns
                    if c.startswith(f"{col}_") and c not in original_df
                ]
            )

        # Add remaining new columns (non-derivatives)
        columns.extend([col for col in df if col not in columns])

        # Merge the new and old datasets keeping the newest columns
        new_df = df.merge(
            right=original_df[[col for col in original_df if col in columns]],
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=("", "__drop__"),
        )
        new_df = new_df.drop(new_df.filter(regex="__drop__$").columns, axis=1)

        return new_df[columns]

    def _prepare_df(self, X, out):
        """Convert to df and set correct column names and order."""
        # Convert to pandas and assign proper column names
        if not isinstance(out, pd.DataFrame):
            if hasattr(self.transformer, "get_feature_names"):
                columns = self.transformer.get_feature_names()
            elif hasattr(self.transformer, "get_feature_names_out"):
                try:  # Fails for some estimators in Python 3.7
                    # TODO: Remove try after dropping support of Python 3.7
                    columns = self.transformer.get_feature_names_out()
                except AttributeError:
                    columns = self._name_cols(out, X)
            else:
                columns = self._name_cols(out, X)

            out = to_df(out, index=X.index, columns=columns)

        # Reorder columns if only a subset was used
        if len(self._include) != X.shape[1]:
            return self._reorder_cols(out, X)
        else:
            return out

    def fit(self, X=None, y=None, **fit_params):
        # Save the incoming feature names
        feature_names_in = []
        if hasattr(X, "columns"):
            feature_names_in += list(X.columns)
        if hasattr(y, "name"):
            feature_names_in += [y.name]
        if feature_names_in:
            self._feature_names_in = feature_names_in

        args = []
        transformer_params = signature(self.transformer.fit).parameters
        if "X" in transformer_params and X is not None:
            if self._include is None:
                self._include = [
                    c for c in X.columns if c in X and c not in self._exclude
                ]
            elif not self._include:  # Don't fit if empty list
                return self
            else:
                self._include = [
                    c for c in self._include if c in X and c not in self._exclude
                ]
            args.append(X[self._include])
        if "y" in transformer_params and y is not None:
            args.append(y)

        self.transformer.fit(*args, **fit_params)
        return self

    def transform(self, X=None, y=None):
        X = to_df(X, index=getattr(y, "index", None))
        y = to_series(y, index=getattr(X, "index", None))

        args = []
        transform_params = signature(self.transformer.transform).parameters
        if "X" in transform_params:
            if X is not None:
                if self._include is None:
                    self._include = [
                        c for c in X.columns if c in X and c not in self._exclude
                    ]
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
            new_X = self._prepare_df(X, output[0])
            new_y = to_series(output[1], index=new_X.index, name=y.name)
        else:
            if len(output.shape) > 1:
                new_X = self._prepare_df(X, output)
                new_y = y if y is None else y.set_axis(new_X.index)
            else:
                new_y = to_series(output, index=y.index, name=y.name)
                new_X = X if X is None else X.set_index(new_y.index)

        return variable_return(new_X, new_y)


class TransformerWrapperWithInverse(TransformerWrapper):
    def inverse_transform(self, y):
        y = to_series(y, index=getattr(y, "index", None))
        output = self.transformer.inverse_transform(y)
        return to_series(output, index=y.index, name=y.name)


class CleanColumnNames(BaseEstimator, TransformerMixin):
    """Remove weird characters from column names."""

    def __init__(self, match=r"[\]\[\,\{\}\"\:]+"):
        self.match = match

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.rename(columns=lambda x: re.sub(self.match, "", str(x)))


class ExtractDateTimeFeatures(BaseEstimator, TransformerMixin):
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

                # Only create feature if values contains less than 10% NaTs
                if values.isna().sum() <= 0.1 * len(values):
                    X.insert(X.columns.get_loc(col) + 1, f"{col}_{fx}", values)

            X = X.drop(col, axis=1)  # Drop the original datetime column

        return X


class DropImputer(BaseEstimator, TransformerMixin):
    """Drop rows with missing values."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.dropna(subset=self.columns, axis=0)
        if y is not None:
            y = y[y.index.isin(X.index)]

        return variable_return(X, y)


class EmbedTextFeatures(BaseEstimator, TransformerMixin):
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
            data = self._estimators[col].transform(X[col]).toarray()
            columns = [
                f"{col}_{word}"
                for word in self._estimators[col].get_feature_names_out()
            ]

            # Merge the new columns with the dataset
            X = pd.concat(
                [X, pd.DataFrame(data=data, index=X.index, columns=columns)],
                axis=1,
            )

            # Drop original text column
            X = X.drop(col, axis=1)

        return X


class RareCategoryGrouping(BaseEstimator, TransformerMixin):
    """Replace rare categories with the string `other`."""

    def __init__(self, rare_to_value, value="rare"):
        self.rare_to_value = rare_to_value
        self.value = value
        self._to_other = defaultdict(list)

    def fit(self, X, y=None):
        for name, column in X.items():
            for category, count in column.value_counts().items():
                if count < self.rare_to_value * len(X):
                    self._to_other[name].append(category)

        return self

    def transform(self, X, y=None):
        for name, column in X.items():
            if self._to_other[name]:
                X[name] = column.replace(self._to_other[name], self.value)

        return X


class GroupFeatures(BaseEstimator, TransformerMixin):
    """Get statistical properties of similar features.

    Replace a group of features for columns with statistical
    properties of that group.

    """

    def __init__(self, group_features, group_names=None):
        self.group_features = group_features
        self.group_names = group_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.group_names:
            self.group_names = [
                f"group_{i}" for i in range(1, len(self.group_features) + 1)
            ]

        for name, group in zip(self.group_names, self.group_features):
            # Drop columns that are not in the dataframe (can be excluded)
            group = [g for g in group if g in X]

            group_df = X[group]
            X[f"min({name})"] = group_df.apply(np.min, axis=1)
            X[f"max({name})"] = group_df.apply(np.max, axis=1)
            X[f"mean({name})"] = group_df.apply(np.mean, axis=1)
            X[f"std({name})"] = group_df.apply(np.std, axis=1)
            X[f"median({name})"] = group_df.apply(np.median, axis=1)
            X[f"mode({name})"] = stats.mode(group_df, axis=1)[0]
            X = X.drop(group, axis=1)

        return X


class RemoveMulticollinearity(BaseEstimator, TransformerMixin):
    """Drop multicollinear features."""

    def __init__(self, threshold=1):
        self.threshold = threshold
        self._drop = None

    def fit(self, X, y=None):
        # Get the Pearson correlation coefficient matrix
        if y is None:
            corr_X = X.corr()
        else:
            data = X.merge(y.to_frame(), left_index=True, right_index=True)
            corr_matrix = data.corr()
            corr_X, corr_y = corr_matrix.iloc[:-1, :-1], corr_matrix.iloc[:-1, -1]

        self._drop = []
        for col in corr_X:
            # Select columns that are corr
            corr = corr_X[col][corr_X[col] >= self.threshold]

            # Always finds himself with correlation 1
            if len(corr) > 1:
                if y is None:
                    # Drop all but the first one
                    self._drop.extend(list(corr[1:].index))
                else:
                    # Keep feature with the highest correlation with y
                    keep = corr_y[corr.index].idxmax()
                    self._drop.extend(list(corr.index.drop(keep)))

        return self

    def transform(self, X):
        return X.drop(set(self._drop), axis=1)


class RemoveOutliers(BaseEstimator, TransformerMixin):
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


class FixImbalancer(BaseEstimator, TransformerMixin):
    """Wrapper for a balancer with a fit_resample method.

    When oversampling, the newly created samples have an increasing
    integer index for numerical indices, and an index of the form
    [estimator]_N for non-numerical indices, where N stands for the
    N-th sample in the data set.

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
        if "over_sampling" in self.estimator.__module__:
            index = X.index  # Save indices for later reassignment
            X, y = self.estimator.fit_resample(X, y)

            # Create indices for the new samples
            if index.dtype.kind in "ifu":
                new_index = range(max(index) + 1, max(index) + len(X) - len(index) + 1)
            else:
                new_index = [
                    f"{self.estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(X) - len(index) + 1)
                ]

            # Assign the old + new indices
            X.index = list(index) + list(new_index)
            y.index = list(index) + list(new_index)

        elif "under_sampling" in self.estimator.__module__:
            self.estimator.fit_resample(X, y)

            # Select chosen rows (imblearn doesn't return them in order)
            samples = sorted(self.estimator.sample_indices_)
            X, y = X.iloc[samples, :], y.iloc[samples]

        elif "combine" in self.estimator.__module__:
            index = X.index
            X_new, y_new = self.estimator.fit_resample(X, y)

            # Select rows that were kept by the undersampler
            if self.estimator.__class__.__name__ == "SMOTEENN":
                samples = sorted(self.estimator.enn_.sample_indices_)
            elif self.estimator.__class__.__name__ == "SMOTETomek":
                samples = sorted(self.estimator.tomek_.sample_indices_)

            # Select the remaining samples from the old dataframe
            old_samples = [s for s in samples if s < len(X)]
            X, y = X.iloc[old_samples, :], y.iloc[old_samples]

            # Create indices for the new samples
            if index.dtype.kind in "ifu":
                new_index = range(max(index) + 1, max(index) + len(X_new) - len(X) + 1)
            else:
                new_index = [
                    f"{self.estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(X_new) - len(X) + 1)
                ]

            # Select the new samples and assign the new indices
            X_new = X_new.iloc[-len(X_new) + len(old_samples) :, :]
            X_new.index = new_index
            y_new = y_new.iloc[-len(y_new) + len(old_samples) :]
            y_new.index = new_index

            # Add the new samples to the old dataframe
            X, y = X.append(X_new), y.append(y_new)

        return X, y


class TargetTransformer(BaseEstimator):
    """Wrapper for a transformer to be used on target instead."""

    def __init__(self, estimator, enforce_2d: bool = True):
        self.estimator = estimator
        self._train_only = False
        self.enforce_2d = enforce_2d

    def _enforce_2d_on_y(self, y: pd.Series):
        index = y.index
        name = y.name
        if self.enforce_2d:
            if not isinstance(y, pd.DataFrame):
                y = to_df(y, index=index, columns=[name])
        return y, index, name

    def fit(self, y: pd.Series, **fit_params):
        y, _, _ = self._enforce_2d_on_y(y)
        return self.estimator.fit(y, **fit_params)

    def transform(self, y: pd.Series):
        y, index, name = self._enforce_2d_on_y(y)
        output = self.estimator.transform(y)
        return to_series(output, index=index, name=name)

    def inverse_transform(self, y: pd.Series):
        y, index, name = self._enforce_2d_on_y(y)
        output = self.estimator.inverse_transform(y)
        return to_series(output, index=index, name=name)

    def fit_transform(self, y: pd.Series):
        y, index, name = self._enforce_2d_on_y(y)
        output = self.estimator.fit_transform(y)
        return to_series(output, index=index, name=name)
