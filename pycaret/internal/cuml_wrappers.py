from sklearn.linear_model._ridge import _RidgeClassifierMixin
from sklearn.preprocessing import LabelEncoder

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
    from cuml.svm import SVC
else:
    SVC = None


def get_svc_classifier():
    return SVC


if _check_soft_dependencies("cuml", extra=None, severity="warning"):
    from cuml.linear_model import Ridge

    class RidgeClassifier(Ridge, _RidgeClassifierMixin):
        def decision_function(self, X):
            X = Ridge.predict(self, X)
            try:
                X = X.to_output("numpy")
            except AttributeError:
                pass
            return X.astype(int)

        def fit(self, X, y, sample_weight=None):
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
                *sample_weight* support to RidgeClassifier.

            Returns
            -------
            self : object
                Instance of the estimator.
            """
            self._label_binarizer = LabelEncoder()
            y = self._label_binarizer.fit_transform(y)

            super().fit(X, y, sample_weight=sample_weight)
            return self

        def predict(self, X):
            """Predict class labels for samples in `X`.

            Parameters
            ----------
            X : {array-like, spare matrix} of shape (n_samples, n_features)
                The data matrix for which we want to predict the targets.

            Returns
            -------
            y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
                Vector or matrix containing the predictions. In binary and
                multiclass problems, this is a vector containing `n_samples`. In
                a multilabel problem, it returns a matrix of shape
                `(n_samples, n_outputs)`.
            """
            ret = self.decision_function(X)
            return self._label_binarizer.inverse_transform(ret)

else:
    RidgeClassifier = None


def get_ridge_classifier():
    return RidgeClassifier


def get_random_forest_classifier():
    from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier

    class RandomForestClassifier(cuMLRandomForestClassifier):
        def fit(self, X, y, *args, **kwargs):
            super().fit(X, y, *args, **kwargs)
            self.classes_ = self.classes_.astype(int)
            return self

        def predict(self, X, *args, **kwargs):
            X = super().predict(X, *args, **kwargs)
            try:
                X = X.to_output("numpy")
            except AttributeError:
                pass
            return X.astype(int)

    return RandomForestClassifier


def get_random_forest_regressor():
    from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor

    return cuMLRandomForestRegressor
