from typing import Any

import numpy as np
from sklearn.utils import check_random_state

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
    for i in range(n_samples):
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
