# Module: internal.pipeline
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# Provides a Pipeline supporting partial_fit (needed for tune warm start)
# and copying over fit attributes from the final estimator, so that it can be plotted directly
# and is considered fitted.

# This pipeline is only to be used internally.

import platform
import warnings
from copy import deepcopy
from inspect import signature

import imblearn.pipeline
import sklearn.pipeline
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metadata_routing import _routing_enabled
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory

from pycaret.utils._show_versions import _get_deps_info
from pycaret.utils.generic import get_all_object_vars_and_properties, variable_return

INVERSE_ONLY = False


def _copy_estimator_state(source, target) -> None:
    """Copy the state of source to target."""
    try:
        state = source.__getstate__()
    except Exception:
        state = source.__dict__

    try:
        target.__setstate__(state)
    except Exception:
        target.__dict__ = state

    del source


def _final_estimator_has(attr):
    """Check that final_estimator has attribute `attr`.

    Used together with `available_if` in Pipeline.

    """

    def check(self):
        # Raise original `AttributeError` if `attr` does not exist
        getattr(self._final_estimator, attr)
        return True

    return check


def _fit_one(transformer, X=None, y=None, message=None, params=None):
    """Fit the data using one transformer."""
    with _print_elapsed_time("Pipeline", message):
        if hasattr(transformer, "fit"):
            args = []
            if "X" in signature(transformer.fit).parameters:
                args.append(X)
            if "y" in signature(transformer.fit).parameters:
                args.append(y)
            transformer.fit(*args)
    return transformer


def _transform_one(transformer, X=None, y=None):
    """Transform the data using one transformer."""
    args = []
    if "X" in signature(transformer.transform).parameters:
        args.append(X)
    if "y" in signature(transformer.transform).parameters:
        args.append(y)
    output = transformer.transform(*args)

    if isinstance(output, tuple):
        X, y = output[0], output[1]
    else:
        if len(output.shape) > 1:
            X, y = output, y  # Only X
        else:
            X, y = X, output  # Only y

    return X, y


def _inverse_transform_one(transformer, y=None):
    """Inverse transform the data using one transformer."""
    if not hasattr(transformer, "inverse_transform"):
        return y

    return transformer.inverse_transform(y)


def _fit_transform_one(transformer, X=None, y=None, message=None, params=None):
    """Fit and transform the data using one transformer."""
    transformer = _fit_one(transformer, X, y, message, **params.get("fit", {}))
    X, y = _transform_one(transformer, X, y)

    return X, y, transformer


def _full_transform(pipeline: "Pipeline", X, y, **kwargs):
    for _, _, transformer in pipeline._iter(**kwargs):
        X, y = pipeline._memory_transform(transformer, X, y)
    return X, y


def _noop_transform(pipeline: "Pipeline", X, y, **kwargs):
    return X, y


class Pipeline(imblearn.pipeline.Pipeline):
    def __init__(self, steps, *, memory=None, verbose=False):
        if _routing_enabled():
            raise RuntimeError("PyCaret Pipeline does not support metadata routing.")
        super().__init__(steps, memory=memory, verbose=verbose)
        self._fit_vars = set()
        self._feature_names_in = None
        self._cache_full_transform = True

    def __getattr__(self, name: str):
        # override getattr to allow grabbing of final estimator attrs
        return getattr(self._final_estimator, name)

    def __getstate__(self):
        try:
            state = super().__getstate__()
            state.update(self.__dict__)
        except AttributeError:
            state = self.__dict__

        state = state.copy()
        # Remove memory to avoid pickling issues
        state["_memory"] = None

        return dict(state.items(), _pycaret_versions=self._pycaret_versions)

    def __setstate__(self, state):
        pickle_versions = state.get("_pycaret_versions", {})
        if pickle_versions.get("deps_info") != self._pycaret_versions["deps_info"]:
            warnings.warn(
                "Version mismatch:\ncurrent: {}\npickle: {}".format(
                    self._pycaret_versions, pickle_versions
                )
            )
        try:
            super().__setstate__(state)
        except AttributeError:
            pass

        self.__dict__.update(state)
        # Restore memory
        # This will be replaced with setup-defined value in exp.load_model
        self.memory = state["_memory"]

    @property
    def _pycaret_versions(self):
        return {
            "deps_info": _get_deps_info(optional=False),
            "python": {
                "version": platform.python_version(),
                "machine": platform.machine(),
            },
        }

    @property
    def feature_names_in_(self):
        return self._feature_names_in

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        """Set up cache memory objects."""
        self._memory = check_memory(value)
        self._memory_fit = self._memory.cache(_fit_one)
        self._memory_transform = self._memory.cache(_transform_one)
        self.__memory_full_transform = self._memory.cache(_full_transform)

    @property
    def _memory_full_transform(self):
        if INVERSE_ONLY:
            return _noop_transform
        if self._cache_full_transform:
            return self.__memory_full_transform
        else:
            return _full_transform

    def _iter(
        self,
        with_final=True,
        filter_passthrough=True,
        filter_train_only=True,
        reverse=False,
    ):
        """Generate (idx, name, trans) tuples from self.steps.

        When `filter_passthrough=True`, 'passthrough' and None
        transformers are filtered out. When `filter_train_only=True`,
        the RemoveOutliers and FixImbalancer classes are only used for
        fitting. They are skipped for all prediction methods since we
        want to keep the original distribution of target classes.

        """
        it = super()._iter(with_final, filter_passthrough)
        if reverse:
            it = reversed(list(it))
        if filter_train_only:
            return filter(lambda x: not getattr(x[-1], "_train_only", False), it)
        else:
            return it

    def _fit(self, X=None, y=None, routed_params=None):
        self.steps = list(self.steps)
        self._validate_steps()

        # Save the incoming feature names (if pandas objects)
        if hasattr(X, "columns"):
            self._feature_names_in = list(X.columns) + (
                [y.name] if hasattr(y, "name") else []
            )

        for step_idx, name, transformer in self._iter(False, False, False):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(transformer, "transform"):
                if self._memory_fit.__class__.__name__ == "NotMemorizedFunc":
                    # Don't clone when caching is disabled to
                    # preserve backward compatibility
                    cloned = transformer
                else:
                    cloned = clone(transformer)

                if hasattr(cloned, "_cache_full_transform"):
                    cloned._cache_full_transform = False

                # Fit or load the current transformer from cache
                fitted_transformer = self._memory_fit(
                    transformer=cloned,
                    X=X,
                    y=y,
                    message=self._log_message(step_idx),
                    params=routed_params.get(name, {}),
                )
                X, y = self._memory_transform(
                    transformer=fitted_transformer,
                    X=X,
                    y=y,
                )

            # Replace the transformer of the step with the fitted
            # transformer (necessary when loading from the cache)
            self.steps[step_idx] = (name, fitted_transformer)

        if self._final_estimator == "passthrough":
            return X, y, {}

        return X, y, routed_params.get(self.steps[-1][0], {})

    def fit(self, X=None, y=None, **params):
        routed_params = self._check_method_params(method="fit", props=params)
        X, y, _ = self._fit(X, y, routed_params)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_params = routed_params[self.steps[-1][0]]
                fitted_estimator = self._memory_fit(
                    clone(self.steps[-1][1]), X, y, **last_step_params["fit"]
                )
                # Hacky way to make sure that the state of the estimator
                # loaded from cache is carried over to the estimator
                # in steps
                _copy_estimator_state(fitted_estimator, self.steps[-1][1])

        return self

    def transform(self, X=None, y=None, filter_train_only=True):
        X, y = self._memory_full_transform(
            self,
            X,
            y,
            with_final=hasattr(self._final_estimator, "transform"),
            filter_train_only=filter_train_only,
        )

        return variable_return(X, y)

    def fit_transform(self, X=None, y=None, **params):
        routed_params = self._check_method_params(method="fit_transform", props=params)
        X, y, _ = self._fit(X, y, routed_params)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator == "passthrough":
                return variable_return(X, y)

            last_step_params = routed_params[self.steps[-1][0]]
            fitted_estimator = self._memory_fit(
                clone(self.steps[-1][1]), X, y, **last_step_params["fit_transform"]
            )
            # Hacky way to make sure that the state of the estimator
            # loaded from cache is carried over to the estimator
            # in steps
            _copy_estimator_state(fitted_estimator, self.steps[-1][1])
            X, y = self._memory_transform(self._final_estimator, X, y)

        return variable_return(X, y)

    def inverse_transform(self, y):
        for _, _, transformer in self._iter(with_final=False, reverse=True):
            # Duplicate hasattr check here so we don't cache unnecessarily
            if hasattr(transformer, "inverse_transform"):
                y = _inverse_transform_one(transformer, y)
        return y

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **params):
        X, _ = self._memory_full_transform(self, X, None, with_final=False)

        y = self.steps[-1][-1].predict(X, **params)
        y = self.inverse_transform(y)

        return y

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, **params):
        Xt, _ = self._memory_full_transform(self, X, None, with_final=False)
        return self.steps[-1][1].predict_proba(Xt, **params)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **params):
        X, _ = self._memory_full_transform(self, X, None, with_final=False)

        return self.steps[-1][-1].predict_log_proba(X, **params)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X, **params):
        X, _ = self._memory_full_transform(self, X, None, with_final=False)

        return self.steps[-1][-1].decision_function(X, **params)

    @available_if(_final_estimator_has("score"))
    def score(self, X, y, sample_weight=None):
        X, y = self._memory_full_transform(self, X, y, with_final=False)

        return self.steps[-1][-1].score(X, y, sample_weight=sample_weight)

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
                except Exception:
                    pass
        except Exception:
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

    def set_params(self, **kwargs):
        try:
            result = super().set_params(**kwargs)
        except Exception:
            result = self._final_estimator.set_params(**kwargs)

        return result

    @available_if(_final_estimator_has("partial_fit"))
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
        except Exception:
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
        return self


class TimeSeriesPipeline(Pipeline):
    def _get_fit_params(self, X, y, **fit_params):
        fit_params_steps = {name: {} for name, step in self.steps if step is not None}
        for pname, pval in fit_params.items():
            if "__" not in pname:
                raise ValueError(
                    "Pipeline.fit does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname)
                )
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
        return X, y, fit_params_steps[self.steps[-1][0]]

    @available_if(_final_estimator_has("score"))
    def score(self, X=None, y=None, **score_params):
        Xt = X
        Xt, _ = self._memory_full_transform(self, Xt, None, with_final=False)
        result = self.steps[-1][-1].score(y=y, X=Xt, **score_params)
        return result

    def predict(self, X=None, fh=None, **params):
        Xt = X
        if Xt is not None:
            Xt, _ = self._memory_full_transform(self, Xt, None, with_final=False)
        return self.steps[-1][-1].predict(fh=fh, X=Xt, **params)

    def fit(self, X=None, y=None, **fit_params):
        if X is not None:
            Xt, yt, fit_params = self._fit(X, y, **fit_params)
        else:
            Xt, yt, fit_params = self._get_fit_params(X, y, **fit_params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                self._final_estimator.fit(y=yt, X=Xt, **fit_params)
        return self

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X=None, y=None, **fit_params):
        if X is not None:
            Xt, yt, fit_params = self._fit(X, y, **fit_params)
        else:
            Xt, yt, fit_params = self._get_fit_params(X, y, **fit_params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][-1].fit_predict(y=yt, X=Xt, **fit_params)
        return y_pred

    def fit_resample(self, X=None, y=None, **fit_params):
        last_step = self._final_estimator
        if X is not None:
            Xt, yt, fit_params = self._fit(X, y, **fit_params)
        else:
            Xt, yt, fit_params = self._get_fit_params(X, y, **fit_params)
        result = None
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                result = Xt
            elif hasattr(last_step, "fit_resample"):
                result = last_step.fit_resample(y=yt, X=Xt, **fit_params)
        return result

    def fit_transform(self, X=None, y=None, **fit_params):
        last_step = self._final_estimator
        if X is not None:
            Xt, yt, fit_params = self._fit(X, y, **fit_params)
        else:
            Xt, yt, fit_params = self._get_fit_params(X, y, **fit_params)
        result = None
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                result = Xt
            elif hasattr(last_step, "fit_transform"):
                result = last_step.fit_transform(y=yt, X=Xt, **fit_params)
            else:
                result = last_step.fit(y=yt, X=Xt, **fit_params).transform(Xt)
        return result


class estimator_pipeline(object):
    """
    Context which adds an estimator to pipeline.
    Pipeline created before every cross-validation
    """

    def __init__(self, pipeline: Pipeline, estimator):
        self.pipeline = deepcopy(pipeline)
        self.estimator = estimator

    def __enter__(self):
        if isinstance(self.estimator, Pipeline):
            return self.estimator
        add_estimator_to_pipeline(self.pipeline, self.estimator)
        return self.pipeline

    def __exit__(self, type, value, traceback):
        return


class pipeline_predict_inverse_only(object):
    def __init__(self) -> None:
        pass

    def __enter__(self):
        global INVERSE_ONLY
        INVERSE_ONLY = True

    def __exit__(self, type, value, traceback):
        global INVERSE_ONLY
        INVERSE_ONLY = False
        return


def add_estimator_to_pipeline(pipeline: Pipeline, estimator, name="actual_estimator"):
    """Pipeline created before every cross-validation

    Parameters
    ----------
    pipeline : Pipeline
        [description]
    estimator : [type]
        [description]
    name : str, optional
        [description], by default "actual_estimator"
    """
    try:
        assert hasattr(pipeline._final_estimator, "predict")
        pipeline.replace_final_estimator(estimator, name=name)
    except Exception:
        pipeline.steps.append((name, estimator))


def merge_pipelines(pipeline_to_merge_to: Pipeline, pipeline_to_be_merged: Pipeline):
    pipeline_to_merge_to.steps.extend(pipeline_to_be_merged.steps)


def get_pipeline_estimator_label(pipeline: Pipeline) -> str:
    try:
        model_step = pipeline.steps[-1]
    except Exception:
        return ""

    return model_step[0]


def get_pipeline_fit_kwargs(pipeline: Pipeline, fit_kwargs: dict) -> dict:
    try:
        model_step = pipeline.steps[-1]
    except Exception:
        return fit_kwargs

    if any(k.startswith(f"{model_step[0]}__") for k in fit_kwargs.keys()):
        return fit_kwargs

    return {f"{model_step[0]}__{k}": v for k, v in fit_kwargs.items()}
