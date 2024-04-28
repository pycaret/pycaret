import traceback
import warnings
from typing import Callable

import numpy as np
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics._scorer import _Scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing

from pycaret.utils.generic import get_label_encoder

_fit_failed_message_warning = (
    "Metric '{0}' failed and error score {1} has been returned instead. "
    "If this is a custom metric, this usually means that the error is "
    "in the metric code. "
    "Full exception below:\n{2}"
)


def get_pos_label(globals_dict: dict):
    if globals_dict.get("pipeline"):
        le = get_label_encoder(globals_dict["pipeline"])
        if le:
            return le.classes_
    elif globals_dict.get("y") is not None:
        known_classes = np.unique(globals_dict["y"].values)
        return known_classes
    return None


def _get_response_method(response_method, needs_threshold, needs_proba):
    """Handles deprecation of `needs_threshold` and `needs_proba` parameters in
    favor of `response_method`.
    """
    needs_threshold_provided = needs_threshold != "deprecated"
    needs_proba_provided = needs_proba != "deprecated"
    response_method_provided = response_method is not None

    needs_threshold = False if needs_threshold == "deprecated" else needs_threshold
    needs_proba = False if needs_proba == "deprecated" else needs_proba

    if response_method_provided and (needs_proba_provided or needs_threshold_provided):
        raise ValueError(
            "You cannot set both `response_method` and `needs_proba` or "
            "`needs_threshold` at the same time. Only use `response_method` since "
            "the other two are deprecated in version 1.4 and will be removed in 1.6."
        )

    if needs_proba_provided or needs_threshold_provided:
        warnings.warn(
            (
                "The `needs_threshold` and `needs_proba` parameter are deprecated in "
                "version 1.4 and will be removed in 1.6. You can either let "
                "`response_method` be `None` or set it to `predict` to preserve the "
                "same behaviour."
            ),
            FutureWarning,
        )

    if response_method_provided:
        return response_method

    if needs_proba is True and needs_threshold is True:
        raise ValueError(
            "You cannot set both `needs_proba` and `needs_threshold` at the same "
            "time. Use `response_method` instead since the other two are deprecated "
            "in version 1.4 and will be removed in 1.6."
        )

    if needs_proba is True:
        response_method = "predict_proba"
    elif needs_threshold is True:
        response_method = ("decision_function", "predict_proba")
    else:
        response_method = "predict"

    return response_method


class EncodedDecodedLabelsScoreFunc:
    """Wrapper to handle both encoded and decoded labels."""

    def __init__(self, score_func: Callable, labels: list):
        self.score_func = score_func
        self.labels = tuple(labels) if labels is not None else None
        self.__name__ = score_func.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        if self.labels and _safe_indexing(y_true, 0) in self.labels:
            kwargs["labels"] = self.labels
            kwargs["pos_label"] = self.labels[-1]
        return self.score_func(y_true, y_pred, **kwargs)


class EncodedDecodedLabelsReplaceScoreFunc:
    """Wrapper to encode y_true and y_pred if necessary."""

    def __init__(self, score_func: Callable, labels: list):
        self.score_func = score_func
        self.labels = np.array(labels) if labels is not None else None
        self.__name__ = score_func.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        try:
            return self.score_func(y_true, y_pred, **kwargs)
        except ValueError as e:
            if self.labels is not None and "is not a valid label" in str(e):
                encoder = LabelEncoder()
                encoder.classes_ = self.labels
                return self.score_func(
                    encoder.transform(y_true), encoder.transform(y_pred), **kwargs
                )
            else:
                raise


class BinaryMulticlassScoreFunc:
    """Wrapper to replace call kwargs with preset values if target is binary."""

    def __init__(
        self, score_func: Callable, kwargs_if_binary: dict, response_method=None
    ):
        self.score_func = score_func
        self.kwargs_if_binary = kwargs_if_binary
        self.__name__ = score_func.__name__
        self.response_method = response_method

    def __call__(self, y_true, y_pred, **kwargs):
        if self.kwargs_if_binary:
            labels = kwargs.get("labels", None)
            is_binary = (
                len(labels) <= 2
                if labels is not None
                else ((y_true == 0) | (y_true == 1)).all()
            )
            if is_binary:
                kwargs = {**kwargs, **self.kwargs_if_binary}

        # Use the provided response_method if available
        if self.response_method:
            kwargs["response_method"] = self.response_method

        return self.score_func(y_true, y_pred, **kwargs)


class ScorerWithErrorScore(_Scorer):
    def __init__(
        self, score_func, sign, kwargs, error_score=np.nan, response_method="predict"
    ):
        super().__init__(
            score_func=score_func,
            sign=sign,
            kwargs=kwargs,
            response_method=response_method,
        )
        self.error_score = error_score

    def _score(
        self,
        method_caller,
        estimator,
        X,
        y_true,
        sample_weight=None,
    ):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring.

        X : array-like or sparse matrix
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        try:
            return super()._score(
                method_caller=method_caller,
                estimator=estimator,
                X=X,
                y_true=y_true,
                sample_weight=sample_weight,
            )
        except Exception:
            warnings.warn(
                _fit_failed_message_warning.format(
                    repr(self), self.error_score, traceback.format_exc()
                ),
                FitFailedWarning,
            )
            return self.error_score

    def _factory_args(self):
        return (
            f", response_method={self._response_method}, error_score={self.error_score}"
        )


def make_scorer_with_error_score(
    score_func,
    *,
    response_method=None,
    greater_is_better=True,
    needs_proba="deprecated",
    needs_threshold="deprecated",
    error_score=np.nan,
    **kwargs,
):
    """Make a scorer from a performance metric or loss function.

    A scorer is a wrapper around an arbitrary metric or loss function that is called
    with the signature `scorer(estimator, X, y_true, **kwargs)`.

    It is accepted in all scikit-learn estimators or functions allowing a `scoring`
    parameter.

    The parameter `response_method` allows to specify which method of the estimator
    should be used to feed the scoring/loss function.

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    response_method : {"predict_proba", "decision_function", "predict"} or \
            list/tuple of such str, default=None

        Specifies the response method to use get prediction from an estimator
        (i.e. :term:`predict_proba`, :term:`decision_function` or
        :term:`predict`). Possible choices are:

        - if `str`, it corresponds to the name to the method to return;
        - if a list or tuple of `str`, it provides the method names in order of
          preference. The method returned corresponds to the first method in
          the list and which is implemented by `estimator`.
        - if `None`, it is equivalent to `"predict"`.

        .. versionadded:: 1.4

    greater_is_better : bool, default=True
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `score_func`.

    needs_proba : bool, default=False
        Whether `score_func` requires `predict_proba` to get probability
        estimates out of a classifier.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

        .. deprecated:: 1.4
           `needs_proba` is deprecated in version 1.4 and will be removed in
           1.6. Use `response_method="predict_proba"` instead.

    needs_threshold : bool, default=False
        Whether `score_func` takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example `average_precision` or the area under the roc curve
        can not be computed using discrete predictions alone.

        .. deprecated:: 1.4
           `needs_threshold` is deprecated in version 1.4 and will be removed
           in 1.6. Use `response_method=("decision_function", "predict_proba")`
           instead to preserve the same behaviour.

    **kwargs : additional arguments
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, response_method='predict', beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """

    response_method = _get_response_method(
        response_method, needs_threshold, needs_proba
    )

    sign = 1 if greater_is_better else -1

    # Create an instance of ScorerWithErrorScore
    scorer = ScorerWithErrorScore(
        score_func, sign, kwargs, error_score, response_method
    )

    return scorer
