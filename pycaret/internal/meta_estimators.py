import sklearn.compose
from sklearn.preprocessing import PowerTransformer

from pycaret.internal.utils import get_all_object_vars_and_properties, is_fit_var


class PowerTransformedTargetRegressor(sklearn.compose.TransformedTargetRegressor):
    def __init__(
        self,
        regressor=None,
        *,
        power_transformer_method="box-cox",
        power_transformer_standardize=True,
        **kwargs
    ):
        self.regressor = regressor
        self.power_transformer_method = power_transformer_method
        self.power_transformer_standardize = power_transformer_standardize
        self.transformer = PowerTransformer(
            method=self.power_transformer_method,
            standardize=self.power_transformer_standardize,
        )
        self.func = None
        self.inverse_func = None
        self.check_inverse = False
        self._fit_vars = set()
        self.set_params(**kwargs)

    def _carry_over_regressor_fit_vars(self):
        self._clear_regressor_fit_vars()
        for k, v in get_all_object_vars_and_properties(self.regressor_).items():
            if is_fit_var(k):
                try:
                    setattr(self, k, v)
                    self._fit_vars.add(k)
                except:
                    pass

    def _clear_regressor_fit_vars(self, all: bool = False):
        vars_to_remove = []
        try:
            for var in self._fit_vars:
                if all or var not in get_all_object_vars_and_properties(
                    self.regressor_
                ):
                    vars_to_remove.append(var)
            for var in vars_to_remove:
                try:
                    delattr(self, var)
                    self._fit_vars.remove(var)
                except:
                    pass
        except:
            pass

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

        r = super().fit(X, y, **fit_params)
        self._carry_over_regressor_fit_vars()
        return r

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
        if "power_transformer_method" in params:
            self.power_transformer_method = params["power_transformer_method"]
            params.pop("power_transformer_method")
            self.transformer.set_params(**{"method": self.power_transformer_method})
        if "power_transformer_standardize" in params:
            self.power_transformer_standardize = params["power_transformer_standardize"]
            params.pop("power_transformer_standardize")
            self.transformer.set_params(
                **{"standardize": self.power_transformer_standardize}
            )
        return self.regressor.set_params(**params)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        r = self.regressor.get_params(deep=deep)
        r["power_transformer_method"] = self.power_transformer_method
        r["power_transformer_standardize"] = self.power_transformer_standardize
        r["regressor"] = self.regressor
        return r


def get_estimator_from_meta_estimator(estimator):
    """
    If ``estimator`` is a meta estimator, get estimator inside.
    Otherwise return ``estimator``. Will try to return the fitted
    estimator first.
    """
    try:
        return estimator.regressor_
    except:
        try:
            return estimator.regressor
        except:
            return estimator
