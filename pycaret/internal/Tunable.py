# Module: internal.Tunable
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# Provides a VotingClassifier which weights can be tuned.

from sklearn.ensemble import VotingClassifier


class TunableVotingClassifier(VotingClassifier):
    """
    A VotingClassifier with weights being kwargs instead of a list, allowing
    for tuning.

    The kwargs need to be in format ``weight_n``, where n is an integer corresponding
    to the index of the estimator.

    If ``weights`` parameter is changed with ``set_params()``, ``weight_n`` parameters 
    will change as well, and vice versa.

    scikit-learn description below:

    Soft Voting/Majority Rule classifier for unfitted estimators.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'``
        using ``set_params``.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted.

        .. deprecated:: 0.22
           Using ``None`` to drop an estimator is deprecated in 0.22 and
           support will be dropped in 0.24. Use the string ``'drop'`` instead.

    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like of shape (n_classifiers,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    flatten_transform : bool, default=True
        Affects shape of transform output only when voting='soft'
        If voting='soft' and flatten_transform=True, transform method returns
        matrix with shape (n_samples, n_classifiers * n_classes). If
        flatten_transform=False, it returns
        (n_classifiers, n_samples, n_classes).

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

    **kwargs:
        Weights in format ``weight_n`` where ``n`` is an integer corresponding
        to the index of the estimator. Will overwrite ``weights``.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    classes_ : array-like of shape (n_predictions,)
        The classes labels.

    See Also
    --------
    VotingRegressor: Prediction voting regressor.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    >>> clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    >>> clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> np.array_equal(eclf1.named_estimators_.lr.predict(X),
    ...                eclf1.named_estimators_['lr'].predict(X))
    True
    >>> eclf2 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1],
    ...        flatten_transform=True)
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>> print(eclf3.transform(X).shape)
    (6, 6)
    """

    def __init__(
        self,
        estimators,
        *,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
        **kwargs,
    ):
        self.weights = weights
        self._weight_kwargs_to_weights(kwargs, estimators)
        super().__init__(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=n_jobs,
            flatten_transform=flatten_transform,
            verbose=verbose,
        )

    def _weight_kwargs_to_weights(self, kwargs, estimators=None):
        if estimators is None:
            estimators = self.estimators
        if not self.weights:
            self.weights = [1 for x in estimators]
        for k, v in kwargs.items():
            if k.startswith("weight_"):
                try:
                    weight = k.split("_")
                    weight = int(weight[1])
                    self.weights[weight] = v
                except:
                    pass
        self._weights_to_weight_kwargs()

    def _weights_to_weight_kwargs(self):
        for i, w in enumerate(self.weights):
            setattr(self, f"weight_{i}", w)

    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the stacking estimator, the individual estimator of
            the stacking estimators can also be set, or can be removed by
            setting them to 'drop'.
        """
        super()._set_params("estimators", **params)
        self._weight_kwargs_to_weights(params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various classifiers and the parameters
            of the classifiers as well.
        """
        r = super()._get_params("estimators", deep=deep)
        if self.weights:
            for i, w in enumerate(self.weights):
                r[f"weight_{i}"] = w
        return r
