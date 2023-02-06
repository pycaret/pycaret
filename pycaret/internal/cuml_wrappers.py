import numpy as np
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array, column_or_1d
from sklearn.utils.validation import _deprecate_positional_args, check_X_y

from pycaret.utils._dependencies import _check_soft_dependencies

if _check_soft_dependencies("cuml", extra=None, severity="warning"):
    from cuml.cluster import DBSCAN as cuMLDBSCAN

    class DBSCAN(cuMLDBSCAN):
        def fit(self, X, y=None, out_dtype="int32"):
            return super().fit(X, out_dtype=out_dtype)

        def fit_predict(self, X, y=None, out_dtype="int32"):
            return super().fit_predict(X, out_dtype=out_dtype)

else:
    DBSCAN = None


def get_dbscan():
    return DBSCAN


if _check_soft_dependencies("cuml", extra=None, severity="warning"):
    from cuml.cluster import KMeans as cuMLKMeans

    class KMeans(cuMLKMeans):
        def fit(self, X, y=None, sample_weight=None):
            return super().fit(X, sample_weight=sample_weight)

        def fit_predict(self, X, y=None, sample_weight=None):
            return super().fit_predict(X, sample_weight=sample_weight)

else:
    KMeans = None


def get_kmeans():

    return KMeans


if _check_soft_dependencies("cuml", extra=None, severity="warning"):
    from cuml.svm import SVC as cuMLSVC

    class SVC(OneVsRestClassifier):
        def __init__(
            self,
            *,
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            nochange_steps=1000,
            random_state=None,
            n_jobs=1,
            handle=None,
            output_type=None,
        ):

            self.kernel = kernel
            self.degree = degree
            self.gamma = gamma
            self.coef0 = coef0
            self.tol = tol
            self.C = C
            self.probability = probability
            self.cache_size = cache_size
            self.class_weight = class_weight
            self.verbose = verbose
            self.max_iter = max_iter
            self.nochange_steps = nochange_steps
            self.random_state = random_state
            self.handle = handle
            self.output_type = output_type

            self.estimator = cuMLSVC(
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                coef0=coef0,
                tol=tol,
                C=C,
                probability=probability,
                cache_size=cache_size,
                class_weight=class_weight,
                verbose=verbose,
                max_iter=max_iter,
                nochange_steps=nochange_steps,
                random_state=random_state,
                handle=handle,
                output_type=output_type,
            )
            self.n_jobs = 1

        def set_params(self, **params):
            """
            Set the parameters of this estimator.

            The method works on simple estimators as well as on nested objects
            (such as pipelines). The latter have parameters of the form
            ``<component>__<parameter>`` so that it's possible to update each
            component of a nested object.

            Parameters
            ----------
            **params : dict
                Estimator parameters.

            Returns
            -------
            self : object
                Estimator instance.
            """
            if "n_jobs" in params:
                params.pop("n_jobs")
            super().set_params(**params)
            estimator_params = self.estimator.get_params()
            self.estimator.set_params(
                **{k: v for k, v in params.items() if k in estimator_params}
            )

            return self

else:
    SVC = None


def get_svc_classifier():
    return SVC


if _check_soft_dependencies("cuml", extra=None, severity="warning"):
    from cuml.linear_model import Ridge

    class RidgeClassifier(OneVsRestClassifier):
        """Classifier using Ridge regression.

        Ridge extends LinearRegression by providing L2 regularization on the
        coefficients when predicting response y with a linear combination of the
        predictors in X. It can reduce the variance of the predictors, and improves
        the conditioning of the problem.

        cuML's Ridge can take array-like objects, either in host as
        NumPy arrays or in device (as Numba or `__cuda_array_interface__`
        compliant), in addition to cuDF objects. It provides 3
        algorithms: SVD, Eig and CD to fit a linear model. In general SVD uses
        significantly more memory and is slower than Eig. If using CUDA 10.1,
        the memory difference is even bigger than in the other supported CUDA
        versions. However, SVD is more stable than Eig (default). CD uses
        Coordinate Descent and can be faster when data is large.

        Parameters
        ----------
        alpha : float (default = 1.0)
            Regularization strength - must be a positive float. Larger values
            specify stronger regularization. Array input will be supported later.
        solver : {'eig', 'svd', 'cd'} (default = 'eig')
            Eig uses a eigendecomposition of the covariance matrix, and is much
            faster.
            SVD is slower, but guaranteed to be stable.
            CD or Coordinate Descent is very fast and is suitable for large
            problems.
        fit_intercept : boolean (default = True)
            If True, Ridge tries to correct for the global mean of y.
            If False, the model expects that you have centered the data.
        normalize : boolean (default = False)
            If True, the predictors in X will be normalized by dividing by it's L2
            norm.
            If False, no scaling will be done.

        Attributes
        ----------
        coef_ : array, shape (n_features)
            The estimated coefficients for the linear regression model.

        intercept_ : array
            The independent term. If `fit_intercept` is False, will be 0.

        n_iter_ : None or ndarray of shape (n_targets,)
            Actual number of iterations for each target. Available only for
            sag and lsqr solvers. Other solvers will return None.

        classes_ : ndarray of shape (n_classes,)
            The classes labels.

        Notes
        -----
        Ridge provides L2 regularization. This means that the coefficients can
        shrink to become very small, but not zero. This can cause issues of
        interpretability on the coefficients.
        Consider using Lasso, or thresholding small coefficients to zero.

        For multi-class classification, n_class classifiers are trained in
        a one-versus-all approach. Concretely, this is implemented by taking
        advantage of the multi-variate response support in Ridge.

        """

        def __init__(
            self,
            *,
            alpha=1.0,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            class_weight=None,
            solver="auto",
            n_jobs=1,
            handle=None,
            output_type=None,
            verbose=False,
        ):

            if solver not in ("auto", "eig", "cd", "svd"):
                raise ValueError(
                    "Known solvers are 'eig', 'cd', 'svd'. Got %s." % solver
                )

            if solver == "auto":
                solver = "eig"

            self.solver = solver
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.normalize = normalize
            self.handle = handle
            self.output_type = output_type
            self.verbose = verbose

            self.estimator = _RidgeClassifierBase(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                solver=self.solver,
                handle=handle,
                output_type=output_type,
                verbose=verbose,
            )
            self.n_jobs = 1
            self.n_iter_ = None

        def set_params(self, **params):
            """
            Set the parameters of this estimator.

            The method works on simple estimators as well as on nested objects
            (such as pipelines). The latter have parameters of the form
            ``<component>__<parameter>`` so that it's possible to update each
            component of a nested object.

            Parameters
            ----------
            **params : dict
                Estimator parameters.

            Returns
            -------
            self : object
                Estimator instance.
            """
            if "n_jobs" in params:
                params.pop("n_jobs")
            super().set_params(**params)
            estimator_params = self.estimator.get_params()
            self.estimator.set_params(
                **{k: v for k, v in params.items() if k in estimator_params}
            )

            return self

    class _RidgeClassifierBase(LinearClassifierMixin, Ridge):
        """Classifier using Ridge regression.

        Does not support multiclass problems. Use RidgeClassifier instead.

        Ridge extends LinearRegression by providing L2 regularization on the
        coefficients when predicting response y with a linear combination of the
        predictors in X. It can reduce the variance of the predictors, and improves
        the conditioning of the problem.

        cuML's Ridge can take array-like objects, either in host as
        NumPy arrays or in device (as Numba or `__cuda_array_interface__`
        compliant), in addition to cuDF objects. It provides 3
        algorithms: SVD, Eig and CD to fit a linear model. In general SVD uses
        significantly more memory and is slower than Eig. If using CUDA 10.1,
        the memory difference is even bigger than in the other supported CUDA
        versions. However, SVD is more stable than Eig (default). CD uses
        Coordinate Descent and can be faster when data is large.

        Parameters
        ----------
        alpha : float (default = 1.0)
            Regularization strength - must be a positive float. Larger values
            specify stronger regularization. Array input will be supported later.
        solver : {'eig', 'svd', 'cd'} (default = 'eig')
            Eig uses a eigendecomposition of the covariance matrix, and is much
            faster.
            SVD is slower, but guaranteed to be stable.
            CD or Coordinate Descent is very fast and is suitable for large
            problems.
        fit_intercept : boolean (default = True)
            If True, Ridge tries to correct for the global mean of y.
            If False, the model expects that you have centered the data.
        normalize : boolean (default = False)
            If True, the predictors in X will be normalized by dividing by it's L2
            norm.
            If False, no scaling will be done.

        Attributes
        ----------
        coef_ : array, shape (n_features)
            The estimated coefficients for the linear regression model.

        intercept_ : array
            The independent term. If `fit_intercept` is False, will be 0.

        n_iter_ : None or ndarray of shape (n_targets,)
            Actual number of iterations for each target. Available only for
            sag and lsqr solvers. Other solvers will return None.

        classes_ : ndarray of shape (n_classes,)
            The classes labels.

        Notes
        -----
        Ridge provides L2 regularization. This means that the coefficients can
        shrink to become very small, but not zero. This can cause issues of
        interpretability on the coefficients.
        Consider using Lasso, or thresholding small coefficients to zero.

        For multi-class classification, n_class classifiers are trained in
        a one-versus-all approach. Concretely, this is implemented by taking
        advantage of the multi-variate response support in Ridge.

        """

        @_deprecate_positional_args
        def __init__(
            self,
            *,
            alpha=1.0,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            class_weight=None,
            solver="eig",
            handle=None,
            output_type="numpy",
            verbose=False,
        ):
            if class_weight:
                raise ValueError("`class_weight` not supported.")

            super().__init__(
                alpha=alpha,
                fit_intercept=fit_intercept,
                normalize=normalize,
                solver=solver,
                handle=handle,
                output_type=output_type,
                verbose=verbose,
            )

            self.n_iter_ = None

        def fit(self, X, y):
            """Fit Ridge classifier model.

            Parameters
            ----------
            X : {ndarray, sparse matrix} of shape (n_samples, n_features)
                Training data.

            y : ndarray of shape (n_samples,)
                Target values.

            sample_weight : float or ndarray of shape (n_samples,), default=None
                Individual weights for each sample. If given a float, every sample
                will have the same weight.

                .. versionadded:: 0.17
                *sample_weight* support to Classifier.

            Returns
            -------
            self : object
                Instance of the estimator.
            """
            X, y = self._validate_data(
                X, y, accept_sparse=False, multi_output=True, y_numeric=False
            )

            self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
            Y = self._label_binarizer.fit_transform(y)
            if not self._label_binarizer.y_type_.startswith("multilabel"):
                y = column_or_1d(y, warn=True)
            else:
                # we don't (yet) support multi-label classification in Ridge
                raise ValueError(
                    "%s doesn't support multi-label classification"
                    % (self.__class__.__name__)
                )

            super().fit(X, Y, convert_dtype=True)
            self.coef_ = np.expand_dims(self.coef_.to_output("numpy"), axis=0)
            try:
                self.intercept_ = self.intercept_.to_output("numpy")
            except AttributeError:
                pass
            return self

        def _check_n_features(self, X, reset):
            """Set the `n_features_in_` attribute, or check against it.

            Parameters
            ----------
            X : {ndarray, sparse matrix} of shape (n_samples, n_features)
                The input samples.
            reset : bool
                If True, the `n_features_in_` attribute is set to `X.shape[1]`.
                Else, the attribute must already exist and the function checks
                that it is equal to `X.shape[1]`.
            """
            n_features = X.shape[1]

            if reset:
                self.n_features_in_ = n_features
            else:
                if not hasattr(self, "n_features_in_"):
                    raise RuntimeError(
                        "The reset parameter is False but there is no "
                        "n_features_in_ attribute. Is this estimator fitted?"
                    )
                if n_features != self.n_features_in_:
                    raise ValueError(
                        "X has {} features, but this {} is expecting {} features "
                        "as input.".format(
                            n_features, self.__class__.__name__, self.n_features_in_
                        )
                    )

        def _validate_data(
            self, X, y=None, reset=True, validate_separately=False, **check_params
        ):
            """Validate input data and set or check the `n_features_in_` attribute.

            Parameters
            ----------
            X : {array-like, sparse matrix, dataframe} of shape \
                    (n_samples, n_features)
                The input samples.
            y : array-like of shape (n_samples,), default=None
                The targets. If None, `check_array` is called on `X` and
                `check_X_y` is called otherwise.
            reset : bool, default=True
                Whether to reset the `n_features_in_` attribute.
                If False, the input will be checked for consistency with data
                provided when reset was last True.
            validate_separately : False or tuple of dicts, default=False
                Only used if y is not None.
                If False, call validate_X_y(). Else, it must be a tuple of kwargs
                to be used for calling check_array() on X and y respectively.
            **check_params : kwargs
                Parameters passed to :func:`sklearn.utils.check_array` or
                :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
                is not False.

            Returns
            -------
            out : {ndarray, sparse matrix} or tuple of these
                The validated input. A tuple is returned if `y` is not None.
            """

            if y is None:
                if self._get_tags()["requires_y"]:
                    raise ValueError(
                        f"This {self.__class__.__name__} estimator "
                        f"requires y to be passed, but the target y is None."
                    )
                X = check_array(X, **check_params)
                out = X
            else:
                if validate_separately:
                    # We need this because some estimators validate X and y
                    # separately, and in general, separately calling check_array()
                    # on X and y isn't equivalent to just calling check_X_y()
                    # :(
                    check_X_params, check_y_params = validate_separately
                    X = check_array(X, **check_X_params)
                    y = check_array(y, **check_y_params)
                else:
                    X, y = check_X_y(X, y, **check_params)
                out = X, y

            if check_params.get("ensure_2d", True):
                self._check_n_features(X, reset=reset)

            return out

        @property
        def classes_(self):
            return self._label_binarizer.classes_

else:
    RidgeClassifier = None


def get_ridge_classifier():

    return RidgeClassifier


def get_random_forest_classifier():
    from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier

    return cuMLRandomForestClassifier


def get_random_forest_regressor():
    from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor

    return cuMLRandomForestRegressor
