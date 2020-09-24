import sklearn.compose
from sklearn.preprocessing import PowerTransformer


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
        self.set_params(**kwargs)

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
