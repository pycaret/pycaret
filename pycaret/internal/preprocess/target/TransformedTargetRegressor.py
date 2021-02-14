from sklearn.compose import TransformedTargetRegressor as _TransformedTargetRegressor

from .utils import TargetTransformerMixin


class TransformedTargetRegressor(TargetTransformerMixin, _TransformedTargetRegressor):
    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            Parameters passed to the ``fit`` method of the underlying
            regressor.


        Returns
        -------
        self : object
        """

        # workaround - for some reason, if data is in float32, super().fit() will return an array of 0s
        # this also happens in pure scikit-learn
        y = y.astype("float64")

        r = super().fit(X, y, **fit_params)
        self._carry_over_estimator_fit_vars(
            self.regressor_, ignore=["transformer_", "regressor_"]
        )
        return r
