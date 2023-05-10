import warnings
from time import time
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as SklearnIterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import (
    FLOAT_DTYPES,
    _check_inputs_dtype,
    _get_mask,
    _ImputerTriplet,
    _safe_indexing,
    is_scalar_nan,
    stats,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


# Handle categorical columns. Special cases for some models.
# TODO make "has own cat encoding a container feature"
def prepare_estimator_for_categoricals(
    estimator: BaseEstimator, categorical_indices: List[int], *, preparation_type: str
):
    fit_params = {}
    if not categorical_indices:
        return estimator, fit_params
    if preparation_type.startswith("fit_params_"):
        fit_param_name = preparation_type[len("fit_params_") :]
        fit_params[fit_param_name] = categorical_indices
    elif preparation_type.startswith("params_"):
        param_name = preparation_type[len("params_") :]
        estimator.set_params(**{param_name: categorical_indices})
    elif preparation_type == "ordinal":
        estimator = SklearnPipeline(
            [
                (
                    "encoder",
                    ColumnTransformer(
                        [("encoder", SklearnOrdinalEncoder(), categorical_indices)],
                        remainder="passthrough",
                    ),
                ),
                ("estimator", estimator),
            ]
        )
    elif preparation_type == "one_hot":
        estimator = SklearnPipeline(
            [
                (
                    "encoder",
                    ColumnTransformer(
                        [("encoder", SklearnOneHotEncoder(), categorical_indices)],
                        remainder="passthrough",
                    ),
                ),
                ("estimator", estimator),
            ]
        )
    elif preparation_type:
        raise ValueError(f"Unknown preparation_type {preparation_type}")
    return estimator, fit_params


def _inverse_map_pd(
    Xt: pd.DataFrame, mappings: dict, feature_name_in: list
) -> pd.DataFrame:
    Xt = pd.DataFrame(Xt, columns=feature_name_in)
    Xt = Xt.astype({col: "category" for col in mappings})
    for col in Xt.select_dtypes("category").columns:
        inverse_mapping = {i: k for k, i in mappings[col].items()}
        Xt[col] = Xt[col].cat.rename_categories(inverse_mapping)
    return Xt


class IterativeImputer(SklearnIterativeImputer):
    def __init__(
        self,
        num_estimator=None,
        cat_estimator=None,
        *,
        missing_values=np.nan,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=None,
        num_initial_strategy="mean",
        cat_initial_strategy="most_frequent",
        imputation_order="ascending",
        skip_complete=False,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=0,
        random_state=None,
        add_indicator=False,
        categorical_indices=None,
        num_estimator_fit_params=None,
        cat_estimator_fit_params=None,
        num_estimator_prepare_for_categoricals_type: Optional[str] = None,
        cat_estimator_prepare_for_categoricals_type: Optional[str] = None,
    ):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)

        self.num_estimator = num_estimator
        self.cat_estimator = cat_estimator
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.num_initial_strategy = num_initial_strategy
        self.cat_initial_strategy = cat_initial_strategy
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state
        self.categorical_indices = categorical_indices
        self.num_estimator_fit_params = num_estimator_fit_params
        self.cat_estimator_fit_params = cat_estimator_fit_params
        self.num_estimator_prepare_for_categoricals_type = (
            num_estimator_prepare_for_categoricals_type
        )
        self.cat_estimator_prepare_for_categoricals_type = (
            cat_estimator_prepare_for_categoricals_type
        )

    def _initial_imputation(self, X, in_fit=False):
        """Perform initial imputation for input `X`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        in_fit : bool, default=False
            Whether function is called in :meth:`fit`.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        X_filled : ndarray, shape (n_samples, n_features)
            Input data with the most recent imputations.

        mask_missing_values : ndarray, shape (n_samples, n_features)
            Input data's missing indicator matrix, where `n_samples` is the
            number of samples and `n_features` is the number of features.

        X_missing_mask : ndarray, shape (n_samples, n_features)
            Input data's mask matrix indicating missing datapoints, where
            `n_samples` is the number of samples and `n_features` is the
            number of features.
        """
        if is_scalar_nan(self.missing_values):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True

        X = self._validate_data(
            X,
            dtype=FLOAT_DTYPES,
            order="F",
            reset=in_fit,
            force_all_finite=force_all_finite,
        )
        _check_inputs_dtype(X, self.missing_values)

        X_missing_mask = _get_mask(X, self.missing_values)
        mask_missing_values = X_missing_mask.copy()
        categorical_indices = sorted(self.categorical_indices) or []
        num_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
        if self.initial_imputer_ is None:
            self.initial_imputer_ = ColumnTransformer(
                [
                    (
                        "num_imputer",
                        SimpleImputer(
                            missing_values=self.missing_values,
                            strategy=self.num_initial_strategy,
                        ),
                        num_indices,
                    ),
                    (
                        "cat_imputer",
                        SimpleImputer(
                            missing_values=self.missing_values,
                            strategy=self.cat_initial_strategy,
                        ),
                        categorical_indices,
                    ),
                ],
                remainder="passthrough",
                n_jobs=1,
            )
            X_filled = self.initial_imputer_.fit_transform(X)
        else:
            X_filled = self.initial_imputer_.transform(X)

        reorder_indices_mapping = {
            **{v: i for i, v in enumerate(num_indices)},
            **{v: i + len(num_indices) - 1 for i, v in enumerate(categorical_indices)},
        }
        reorder_indices = [
            reorder_indices_mapping[i] for i in range(len(reorder_indices_mapping))
        ]
        X_filled = X_filled[:, reorder_indices]

        combined_statistics = np.zeros(X.shape[1])
        # Use getattr as the imputers may not have been fitted if there have
        # been no columns with the required dtype
        num_statistics = list(
            getattr(
                self.initial_imputer_.named_transformers_["num_imputer"],
                "statistics_",
                [],
            )
        )
        cat_statistics = list(
            getattr(
                self.initial_imputer_.named_transformers_["cat_imputer"],
                "statistics_",
                [],
            )
        )
        for i in range(len(combined_statistics)):
            if i in categorical_indices:
                combined_statistics[i] = cat_statistics.pop(0)
            else:
                combined_statistics[i] = num_statistics.pop(0)

        valid_mask = np.flatnonzero(np.logical_not(np.isnan(combined_statistics)))
        Xt = X[:, valid_mask]
        mask_missing_values = mask_missing_values[:, valid_mask]

        return Xt, X_filled, mask_missing_values, X_missing_mask

    def fit_transform(self, X, y=None):
        """Fit the imputer on `X` and return the transformed `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
            The imputed input data.
        """
        if self.n_nearest_features is not None:
            raise NotImplementedError("n_nearest_features is not implemented")

        if self.add_indicator:
            raise NotImplementedError("add_indicator is not implemented")

        self.random_state_ = getattr(
            self, "random_state_", check_random_state(self.random_state)
        )

        if self.max_iter < 0:
            raise ValueError(
                "'max_iter' should be a positive integer. Got {} instead.".format(
                    self.max_iter
                )
            )

        if self.tol < 0:
            raise ValueError(
                "'tol' should be a non-negative float. Got {} instead.".format(self.tol)
            )

        if self.num_estimator is None:
            from sklearn.linear_model import BayesianRidge

            self._num_estimator = BayesianRidge()
        else:
            self._num_estimator = clone(self.num_estimator)

        if self.cat_estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            self._cat_estimator = RandomForestClassifier()
        else:
            self._cat_estimator = clone(self.cat_estimator)

        self.mappings_ = {}
        if isinstance(X, pd.DataFrame):
            cat_indices = self.categorical_indices or []
            columns_to_encode = [
                col for i, col in enumerate(X.columns) if i in cat_indices
            ]
            X = X.astype({col: "category" for col in columns_to_encode})
            for col in X.select_dtypes("category").columns:
                self.mappings_[col] = {
                    k: i for i, k in enumerate(X[col].cat.categories)
                }
                X[col] = X[col].cat.rename_categories(self.mappings_[col])

        self.imputation_sequence_ = []

        self.initial_imputer_ = None

        X, Xt, mask_missing_values, complete_mask = self._initial_imputation(
            X, in_fit=True
        )

        super()._fit_indicator(complete_mask)
        X_indicator = super()._transform_indicator(complete_mask)

        if self.max_iter == 0 or np.all(mask_missing_values):
            self.n_iter_ = 0
            return super()._concatenate_indicator(Xt, X_indicator)

        # Edge case: a single feature. We return the initial ...
        if Xt.shape[1] == 1:
            self.n_iter_ = 0
            return super()._concatenate_indicator(Xt, X_indicator)

        self._min_value = self._validate_limit(self.min_value, "min", X.shape[1])
        self._max_value = self._validate_limit(self.max_value, "max", X.shape[1])

        if not np.all(np.greater(self._max_value, self._min_value)):
            raise ValueError("One (or more) features have min_value >= max_value.")

        # order in which to impute
        # note this is probably too slow for large feature data (d > 100000)
        # and a better way would be good.
        # see: https://goo.gl/KyCNwj and subsequent comments
        ordered_idx = self._get_ordered_idx(mask_missing_values)
        self.n_features_with_missing_ = len(ordered_idx)

        abs_corr_mat = self._get_abs_corr_mat(Xt)

        n_samples, n_features = Xt.shape
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s" % (X.shape,))
        start_t = time()
        if not self.sample_posterior:
            Xt_previous = Xt.copy()
            normalized_tol = self.tol * np.max(np.abs(X[~mask_missing_values]))
        for self.n_iter_ in range(1, self.max_iter + 1):
            if self.imputation_order == "random":
                ordered_idx = self._get_ordered_idx(mask_missing_values)

            for feat_idx in ordered_idx:
                neighbor_feat_idx = self._get_neighbor_feat_idx(
                    n_features, feat_idx, abs_corr_mat
                )
                Xt, estimator = self._impute_one_feature(
                    Xt,
                    mask_missing_values,
                    feat_idx,
                    neighbor_feat_idx,
                    estimator=None,
                    fit_mode=True,
                )
                estimator_triplet = _ImputerTriplet(
                    feat_idx, neighbor_feat_idx, estimator
                )
                self.imputation_sequence_.append(estimator_triplet)

            if self.verbose > 1:
                print(
                    "[IterativeImputer] Ending imputation round "
                    "%d/%d, elapsed time %0.2f"
                    % (self.n_iter_, self.max_iter, time() - start_t)
                )

            if not self.sample_posterior:
                inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf, axis=None)
                if self.verbose > 0:
                    print(
                        "[IterativeImputer] Change: {}, scaled tolerance: {} ".format(
                            inf_norm, normalized_tol
                        )
                    )
                if inf_norm < normalized_tol:
                    if self.verbose > 0:
                        print("[IterativeImputer] Early stopping criterion reached.")
                    break
                Xt_previous = Xt.copy()
        else:
            if not self.sample_posterior:
                warnings.warn(
                    "[IterativeImputer] Early stopping criterion not reached.",
                    ConvergenceWarning,
                )
        Xt[~mask_missing_values] = X[~mask_missing_values]
        if self.mappings_:
            Xt = _inverse_map_pd(Xt, self.mappings_, self.feature_names_in_)
        # return super()._concatenate_indicator(Xt, X_indicator)
        return Xt

    def transform(self, X):
        check_is_fitted(self)
        if self.mappings_:
            X = X.astype(
                {
                    col: pd.CategoricalDtype(
                        categories=list(self.mappings_[col].keys())
                    )
                    for col in self.mappings_
                }
            )
            for col in X.select_dtypes("category").columns:
                X[col] = X[col].cat.rename_categories(self.mappings_[col])
        Xt = super().transform(X)
        if self.mappings_:
            Xt = _inverse_map_pd(Xt, self.mappings_, self.feature_names_in_)
        return Xt

    def _impute_one_feature(
        self,
        X_filled,
        mask_missing_values,
        feat_idx,
        neighbor_feat_idx,
        estimator=None,
        fit_mode=True,
    ):
        if estimator is None and fit_mode is False:
            raise ValueError(
                "If fit_mode is False, then an already-fitted "
                "estimator should be passed in."
            )

        is_categorical_feat = feat_idx in self.categorical_indices

        categorical_indices = [
            i - int(i > feat_idx) for i in self.categorical_indices if i != feat_idx
        ]
        prep_fit_params = {}

        if estimator is None:
            if is_categorical_feat:
                estimator = clone(self._cat_estimator)
                if self.cat_estimator_prepare_for_categoricals_type:
                    (estimator, prep_fit_params,) = prepare_estimator_for_categoricals(
                        estimator,
                        categorical_indices,
                        preparation_type=self.cat_estimator_prepare_for_categoricals_type,
                    )
            else:
                estimator = clone(self._num_estimator)
                if self.num_estimator_prepare_for_categoricals_type:
                    (estimator, prep_fit_params,) = prepare_estimator_for_categoricals(
                        estimator,
                        categorical_indices,
                        preparation_type=self.num_estimator_prepare_for_categoricals_type,
                    )

        missing_row_mask = mask_missing_values[:, feat_idx]
        if fit_mode:
            fit_params = (
                self.cat_estimator_fit_params
                if is_categorical_feat
                else self.num_estimator_fit_params
            )
            fit_params = fit_params or {}
            fit_params = {**fit_params, **prep_fit_params}
            X_train = _safe_indexing(X_filled[:, neighbor_feat_idx], ~missing_row_mask)
            y_train = _safe_indexing(X_filled[:, feat_idx], ~missing_row_mask)
            # required for catboost
            X_train = pd.DataFrame(X_train).astype(
                {col: int for col in categorical_indices}
            )
            if is_categorical_feat:
                y_train = y_train.astype(int)

            try:
                estimator.fit(X_train, y_train, **fit_params)
            except Exception:
                # some classifiers raise exceptions if there is only one
                # target value
                if is_categorical_feat and len(np.unique(y_train)) == 1:
                    estimator = DummyClassifier(strategy="most_frequent")
                    estimator.fit(X_train, y_train)
                else:
                    raise

        # if no missing values, don't predict
        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        # get posterior samples if there is at least one missing value
        X_test = _safe_indexing(X_filled[:, neighbor_feat_idx], missing_row_mask)
        # required for catboost
        X_test = pd.DataFrame(X_test).astype({col: int for col in categorical_indices})
        if self.sample_posterior:
            mus, sigmas = estimator.predict(X_test, return_std=True)
            imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
            # two types of problems: (1) non-positive sigmas
            # (2) mus outside legal range of min_value and max_value
            # (results in inf sample)
            positive_sigmas = sigmas > 0
            imputed_values[~positive_sigmas] = mus[~positive_sigmas]
            mus_too_low = mus < self._min_value[feat_idx]
            imputed_values[mus_too_low] = self._min_value[feat_idx]
            mus_too_high = mus > self._max_value[feat_idx]
            imputed_values[mus_too_high] = self._max_value[feat_idx]
            # the rest can be sampled without statistical issues
            inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
            mus = mus[inrange_mask]
            sigmas = sigmas[inrange_mask]
            a = (self._min_value[feat_idx] - mus) / sigmas
            b = (self._max_value[feat_idx] - mus) / sigmas

            truncated_normal = stats.truncnorm(a=a, b=b, loc=mus, scale=sigmas)
            imputed_values[inrange_mask] = truncated_normal.rvs(
                random_state=self.random_state_
            )
        else:
            imputed_values = estimator.predict(X_test)
            imputed_values = np.clip(
                imputed_values, self._min_value[feat_idx], self._max_value[feat_idx]
            )

        # update the feature
        X_filled[missing_row_mask, feat_idx] = imputed_values
        return X_filled, estimator
