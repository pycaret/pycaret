# Module: internal.Pipeline
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# Provides a Pipeline supporting partial fitting and several attributes needed for plotting.

import imblearn.pipeline
from sklearn.utils import _print_elapsed_time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import if_delegate_has_method


class Pipeline(imblearn.pipeline.Pipeline):
    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)
        self._carry_over_final_estimator_fit()

    def _carry_over_final_estimator_fit(self):
        from pycaret.internal.utils import is_fitted

        if hasattr(self._final_estimator, "fit") and is_fitted(self._final_estimator):
            for k, v in vars(self._final_estimator).items():
                if k and k.endswith("_") and not k.startswith("__"):
                    try:
                        setattr(self, k, v)
                    except:
                        pass

    def fit(self, X, y=None, **fit_kwargs):
        result = super().fit(X, y=y, **fit_kwargs)

        self._carry_over_final_estimator_fit()
        return result

    def fit_predict(self, X, y=None, **fit_params):
        result = super().fit_predict(X, y=y, **fit_params)

        self._carry_over_final_estimator_fit()
        return result

    def fit_resample(self, X, y=None, **fit_params):
        result = super().fit_resample(X, y=y, **fit_params)

        self._carry_over_final_estimator_fit()
        return result

    @if_delegate_has_method(delegate="_final_estimator")
    def fit_transform(self, X, y=None, **fit_params):
        result = super().fit_transform(X, y=y, **fit_params)

        self._carry_over_final_estimator_fit()
        return result

    @if_delegate_has_method(delegate="_final_estimator")
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Fit the model.

        Fit all the transforms/samplers one after the other and
        transform/sample the data, then fit the transformed/sampled
        data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator.
        """
        try:
            self.Xt_
        except:
            self.Xt_ = None
            self.yt_ = None
        if self.Xt_ is None or self.yt_ is None:
            Xt, yt, _ = self._fit(X, y)
            self.Xt_ = Xt
            self.yt_ = yt
        else:
            Xt = self.Xt_
            yt = self.yt_
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                self._final_estimator.partial_fit(
                    Xt, yt, classes=classes, sample_weight=sample_weight
                )
        self._carry_over_final_estimator_fit()
        return self
