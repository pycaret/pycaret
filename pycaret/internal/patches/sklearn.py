# Copyright (C) 2019-2024 PyCaret
# Author: Moez Ali (moez.ali@queensu.ca)
# Contributors (https://github.com/pycaret/pycaret/graphs/contributors)
# License: MIT


from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np
from sklearn.base import is_regressor
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_response_method

from pycaret.internal.pipeline import pipeline_predict_inverse_only

# Monkey patching sklearn.model_selection._search to avoid overflows on windows.


# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/_random.pyx
def _mp_sample_without_replacement(
    n_population: int, n_samples: int, method=None, random_state=None
) -> Any:
    """Sample integers without replacement.

    Select n_samples integers from the set [0, n_population) without
    replacement.


    Parameters
    ----------
    n_population : int
        The size of the set to sample from.

    n_samples : int
        The number of integer to sample.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    out : ndarray of shape (n_samples,)
        The sampled subsets of integer. The subset of selected integer might
        not be randomized, see the method argument.
    """
    if n_population < 0:
        raise ValueError(
            "n_population should be greater than 0, got %s." % n_population
        )

    if n_samples > n_population:
        raise ValueError(
            "n_population should be greater or equal than "
            "n_samples, got n_samples > n_population (%s > %s)"
            % (n_samples, n_population)
        )

    rng = check_random_state(random_state)
    rng_randint = rng.randint
    # The following line of code are heavily inspired from python core,
    # more precisely of random.sample.
    selected = set()
    for _i in range(n_samples):
        j = rng_randint(n_population, dtype=np.uint64)
        while j in selected:
            j = rng_randint(n_population, dtype=np.uint64)
        selected.add(j)
    return [int(x) for x in selected]


def _mp_ParameterGrid_getitem(self, ind):
    """Get the parameters that would be ``ind``th in iteration

    Parameters
    ----------
    ind : int
        The iteration index

    Returns
    -------
    params : dict of str to any
        Equal to list(self)[ind]
    """
    # This is used to make discrete sampling without replacement memory
    # efficient.
    ind = int(ind)
    for sub_grid in self.param_grid:
        # XXX: could memoize information used here
        if not sub_grid:
            if ind == 0:
                return {}
            else:
                ind -= 1
                continue

        # Reverse so most frequent cycling parameter comes first
        keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
        sizes = [len(v_list) for v_list in values_lists]
        total = int(np.product(sizes, dtype=np.uint64))

        if ind >= total:
            # Try the next grid
            ind -= total
        else:
            out = {}
            for key, v_list, n in zip(keys, values_lists, sizes):
                ind, offset = divmod(int(ind), n)
                out[key] = v_list[offset]
            return out

    raise IndexError("ParameterGrid index out of range")


class MultimetricScorerPatched(_MultimetricScorer):
    # Patch use_cache to suppress exception if an estimator
    # doesn't have the required method (this can happen
    # with PyCaret as we just default to error score
    # in that case).
    def _use_cache(self, estimator):
        try:
            return super()._use_cache(estimator)
        except AttributeError:
            return True

    def _score(self, method_caller, estimator, X, y_true, **kwargs):

        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        pos_label = None if is_regressor(estimator) else self._get_pos_label()
        response_method = _check_response_method(estimator, self._response_method)
        y_pred = method_caller(
            estimator, response_method.__name__, X, pos_label=pos_label
        )

        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)


def fit_and_score(*args, **kwargs) -> dict:
    """Wrapper for sklearn's _fit_and_score function.

    Wrap the function sklearn.model_selection._validation._fit_and_score
    to, in turn, path sklearn's _score function to accept pipelines that
    drop samples during transforming, within a joblib parallel context.

    """

    def wrapper(*args, **kwargs) -> dict:
        with patch(
            "sklearn.model_selection._validation._MultimetricScorer",
            MultimetricScorerPatched,
        ), patch("sklearn.model_selection._validation._score", score(_score)):
            return _fit_and_score(*args, **kwargs)

    return wrapper(*args, **kwargs)


def score(f: Callable) -> Callable:
    """Patch decorator for sklearn's _score function.

    Monkey patch for sklearn.model_selection._validation._score
    function to score pipelines that drop samples during transforming.

    """

    def wrapper(*args, **kwargs):
        args = list(args)  # Convert to list for item assignment
        if len(args[0]) > 1:  # Has transformers
            args[1], y_transformed = args[0]._memory_full_transform(
                args[0], args[1], args[2], with_final=False
            )
            args[2] = args[2][args[2].index.isin(y_transformed.index)]

        with pipeline_predict_inverse_only():
            return f(args[0], *tuple(args[1:]), **kwargs)

    return wrapper
