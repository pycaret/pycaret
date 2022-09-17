import numpy as np
from sklearn.metrics._scorer import _PredictScorer, _ProbaScorer, _ThresholdScorer


class BinaryMulticlassScoreFunc:
    def __init__(self, score_func):
        self.score_func = score_func
        self.__name__ = score_func.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        if "average" in kwargs:
            known_values = kwargs.get("labels", np.unique(y_true))
            if len(known_values) <= 2:
                kwargs["average"] = "binary"
        return self.score_func(y_true, y_pred, **kwargs)


class _ThresholdScorerWithErrorScore(_ThresholdScorer):
    def __init__(self, score_func, sign, kwargs, error_score=np.nan):
        super().__init__(score_func=score_func, sign=sign, kwargs=kwargs)
        self.error_score = error_score

    def _score(self, method_caller, clf, X, y, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y : array-like
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
                clf=clf,
                X=X,
                y=y,
                sample_weight=sample_weight,
            )
        except Exception:
            return self.error_score

    def _factory_args(self):
        return f", needs_threshold=True, error_score={self.error_score}"


class _ProbaScorerWithErrorScore(_ProbaScorer):
    def __init__(self, score_func, sign, kwargs, error_score=np.nan):
        super().__init__(score_func=score_func, sign=sign, kwargs=kwargs)
        self.error_score = error_score

    def _score(self, method_caller, clf, X, y, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        clf : object
            Trained classifier to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

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
                clf=clf,
                X=X,
                y=y,
                sample_weight=sample_weight,
            )
        except Exception:
            return self.error_score

    def _factory_args(self):
        return f", needs_proba=True, error_score={self.error_score}"


class _PredictScorerWithErrorScore(_PredictScorer):
    def __init__(self, score_func, sign, kwargs, error_score=np.nan):
        super().__init__(score_func=score_func, sign=sign, kwargs=kwargs)
        self.error_score = error_score

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

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
            return self.error_score


def make_scorer_with_error_score(
    score_func,
    *,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    error_score=np.nan,
    **kwargs,
):
    """Make a scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in GridSearchCV
    and cross_val_score. It takes a score function, such as ``accuracy_score``,
    ``mean_squared_error``, ``adjusted_rand_index`` or ``average_precision``
    and returns a callable that scores an estimator's output.

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)

    Notes
    -----
    If `needs_proba=False` and `needs_threshold=False`, the score
    function is supposed to accept the output of :term:`predict`. If
    `needs_proba=True`, the score function is supposed to accept the
    output of :term:`predict_proba` (For binary `y_true`, the score function is
    supposed to accept probability of the positive class). If
    `needs_threshold=True`, the score function is supposed to accept the
    output of :term:`decision_function`.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError(
            "Set either needs_proba or needs_threshold to True," " but not both."
        )
    if needs_proba:
        cls = _ProbaScorerWithErrorScore
    elif needs_threshold:
        cls = _ThresholdScorerWithErrorScore
    else:
        cls = _PredictScorerWithErrorScore
    return cls(score_func, sign, kwargs, error_score)
