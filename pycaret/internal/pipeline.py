# Module: internal.pipeline
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# Provides a Pipeline supporting partial_fit (needed for tune warm start)
# and copying over fit attributes from the final estimator, so that it can be plotted directly
# and is considered fitted.

# This pipeline is only to be used internally.

from pycaret.internal.utils import get_all_object_vars_and_properties, is_fit_var
import imblearn.pipeline
from sklearn.utils import _print_elapsed_time
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.metaestimators import if_delegate_has_method
import sklearn.pipeline
from pycaret.internal.validation import is_fitted


class Pipeline(imblearn.pipeline.Pipeline):
    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)
        self._fit_vars = set()
        self._carry_over_final_estimator_fit_vars()

    @property
    def inverse_transform(self):
        """Apply inverse transformations in reverse order.

        Parameters
        ----------
        Xt : array-like of shape  (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
        """
        return self._inverse_transform

    def _inverse_transform(self, X):
        Xt = X
        reverse_iter = reversed(list(self._iter()))
        for _, _, transform in reverse_iter:
            try:
                Xt = transform.inverse_transform(Xt)
            except:
                pass
        return Xt

    def _carry_over_final_estimator_fit_vars(self):
        self._clear_final_estimator_fit_vars()
        if hasattr(self._final_estimator, "fit"):
            for k, v in get_all_object_vars_and_properties(
                self._final_estimator
            ).items():
                if is_fit_var(k):
                    try:
                        setattr(self, k, v)
                        self._fit_vars.add(k)
                    except:
                        pass

    def _clear_final_estimator_fit_vars(self, all: bool = False):
        vars_to_remove = []
        try:
            for var in self._fit_vars:
                if (
                    all
                    or var
                    not in get_all_object_vars_and_properties(
                        self._final_estimator
                    ).items()
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

    def get_sklearn_pipeline(self) -> sklearn.pipeline.Pipeline:
        return sklearn.pipeline.Pipeline(self.steps)

    def replace_final_estimator(self, new_final_estimator, name: str = None):
        self._clear_final_estimator_fit_vars(all=True)
        if hasattr(self._final_estimator, "fit"):
            self.steps[-1] = (
                self.steps[-1][0] if not name else name,
                new_final_estimator,
            )
        else:
            self.steps.append(
                (name if name else "actual_estimator", new_final_estimator)
            )
        self._carry_over_final_estimator_fit_vars()

    def set_params(self, **kwargs):
        try:
            result = super().set_params(**kwargs)
        except:
            result = self._final_estimator.set_params(**kwargs)

        self._carry_over_final_estimator_fit_vars()
        return result

    def predict(self, X, **predict_params):
        result = super().predict(X, **predict_params)
        return self.inverse_transform(result)

    def fit(self, X, y=None, **fit_kwargs):
        result = super().fit(X, y=y, **fit_kwargs)

        self._carry_over_final_estimator_fit_vars()
        return result

    def fit_predict(self, X, y=None, **fit_params):
        result = super().fit_predict(X, y=y, **fit_params)

        self._carry_over_final_estimator_fit_vars()
        return self.inverse_transform(result)

    def fit_resample(self, X, y=None, **fit_params):
        result = super().fit_resample(X, y=y, **fit_params)

        self._carry_over_final_estimator_fit_vars()
        return result

    @if_delegate_has_method(delegate="_final_estimator")
    def fit_transform(self, X, y=None, **fit_params):
        result = super().fit_transform(X, y=y, **fit_params)

        self._carry_over_final_estimator_fit_vars()
        return result

    @if_delegate_has_method(delegate="_final_estimator")
    def partial_fit(self, X, y=None, classes=None, **fit_params):
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
                # the try...except block is a workaround until tune-sklearn updates
                try:
                    self._final_estimator.partial_fit(
                        Xt, yt, classes=classes, **fit_params
                    )
                except TypeError:
                    self._final_estimator.partial_fit(Xt, yt, **fit_params)
        self._carry_over_final_estimator_fit_vars()
        return self


class estimator_pipeline(object):
    """
    Context which adds an estimator to pipeline.
    """

    def __init__(self, pipeline: Pipeline, estimator):
        self.pipeline = clone(pipeline)
        self.estimator = estimator

    def __enter__(self):
        add_estimator_to_pipeline(self.pipeline, self.estimator)
        return self.pipeline

    def __exit__(self, type, value, traceback):
        return


def make_internal_pipeline(internal_pipeline_steps: list, memory=None) -> Pipeline:

    if not internal_pipeline_steps:
        memory = None
        internal_pipeline_steps = [("empty_step", "passthrough")]

    return Pipeline(internal_pipeline_steps, memory=memory)


def add_estimator_to_pipeline(pipeline: Pipeline, estimator, name="actual_estimator"):
    try:
        assert hasattr(pipeline._final_estimator, "predict")
        pipeline.replace_final_estimator(estimator, name=name)
    except:
        pipeline.steps.append((name, estimator))
        if hasattr(pipeline, "_carry_over_final_estimator_fit_vars"):
            pipeline._carry_over_final_estimator_fit_vars()


def merge_pipelines(pipeline_to_merge_to: Pipeline, pipeline_to_be_merged: Pipeline):
    pipeline_to_merge_to.steps.extend(pipeline_to_be_merged.steps)
    try:
        pipeline_to_merge_to._carry_over_final_estimator_fit_vars()
    except:
        pass


def get_pipeline_estimator_label(pipeline: Pipeline) -> str:
    try:
        model_step = pipeline.steps[-1]
    except:
        return ""

    return model_step[0]


def get_pipeline_fit_kwargs(pipeline: Pipeline, fit_kwargs: dict) -> dict:
    try:
        model_step = pipeline.steps[-1]
    except:
        return fit_kwargs

    if any(k.startswith(f"{model_step[0]}__") for k in fit_kwargs.keys()):
        return fit_kwargs

    return {f"{model_step[0]}__{k}": v for k, v in fit_kwargs.items()}
