# Includes code from
# https://github.com/joblib/joblib/blob/6836640abba55611d6e57f20338ea54b3e27f296/joblib/test/test_hashing.py

import gc
import itertools
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from xxhash import xxh128

from pycaret.datasets import get_data
from pycaret.internal.memory import fast_hash as hash
from pycaret.regression import RegressionExperiment


@pytest.fixture(scope="function")
def three_np_arrays():
    rnd = np.random.RandomState(0)
    arr1 = rnd.random_sample((10, 10))
    arr2 = arr1.copy()
    arr3 = arr2.copy()
    arr3[0] += 1
    return arr1, arr2, arr3


# Helper functions for the tests
def time_func(func, *args):
    """Time function func on *args."""
    times = list()
    for _ in range(3):
        t1 = time.time()
        func(*args)
        times.append(time.time() - t1)
    return min(times)


def relative_time(func1, func2, *args):
    """Return the relative time between func1 and func2 applied on
    *args.
    """
    time_func1 = time_func(func1, *args)
    time_func2 = time_func(func2, *args)
    if time_func1 + time_func2 == 0:
        return 0
    relative_diff = 0.5 * (abs(time_func1 - time_func2) / (time_func1 + time_func2))
    return relative_diff


@pytest.mark.parametrize("dtype", ("int", "category", "object"))
def test_pandas_dataframe(dtype):
    a = pd.DataFrame([1, 2, 3, 4]).astype(dtype)
    b = pd.DataFrame([1, 2, 3, 4]).astype(dtype)
    assert hash(a) == hash(b)

    b = pd.DataFrame([1, 2, 4, 4]).astype(dtype)
    assert hash(a) != hash(b)

    a = pd.DataFrame([1, 2, 3, 4], columns=["A"]).astype(dtype)
    b = pd.DataFrame([1, 2, 3, 4], columns=["B"]).astype(dtype)
    assert hash(a) != hash(b)

    a = pd.DataFrame([1, 2, 3, 4], index=[1, 2, 3, 4]).astype(dtype)
    b = pd.DataFrame([1, 2, 3, 4], index=[1, 2, 3, 5]).astype(dtype)
    assert hash(a) != hash(b)


@pytest.mark.parametrize("dtype", ("int", "category", "object"))
def test_pandas_series(dtype):
    a = pd.Series([1, 2, 3, 4]).astype(dtype)
    b = pd.Series([1, 2, 3, 4]).astype(dtype)
    assert hash(a) == hash(b)

    b = pd.Series([1, 2, 4, 4]).astype(dtype)
    assert hash(a) != hash(b)

    a = pd.Series([1, 2, 3, 4], name="A").astype(dtype)
    b = pd.Series([1, 2, 3, 4], name="B").astype(dtype)
    assert hash(a) != hash(b)

    a = pd.Series([1, 2, 3, 4], index=[1, 2, 3, 4]).astype(dtype)
    b = pd.Series([1, 2, 3, 4], index=[1, 2, 3, 5]).astype(dtype)
    assert hash(a) != hash(b)


@pytest.mark.parametrize("dtype", ("int", "category", "object"))
def test_pandas_series_vs_df(dtype):
    a = pd.DataFrame([1, 2, 3, 4]).astype(dtype)
    b = pd.Series([1, 2, 3, 4]).astype(dtype)
    assert hash(a) != hash(b)


@pytest.mark.parametrize("dtype", ("int", "category", "object"))
@pytest.mark.parametrize("dtype2", ("int", "category", "object"))
def test_pandas_different_dtypes(dtype, dtype2):
    a = pd.DataFrame([1, 2, 3, 4]).astype(dtype)
    b = pd.DataFrame([1, 2, 3, 4]).astype(dtype2)
    if dtype == dtype2:
        assert hash(a) == hash(b)
    else:
        assert hash(a) != hash(b)


def test_hash_numpy_arrays(three_np_arrays):
    arr1, arr2, arr3 = three_np_arrays

    for obj1, obj2 in itertools.product(three_np_arrays, repeat=2):
        are_hashes_equal = hash(obj1) == hash(obj2)
        are_arrays_equal = np.all(obj1 == obj2)
        assert are_hashes_equal == are_arrays_equal

    assert hash(arr1) != hash(arr1.T)


def test_hash_numpy_dict_of_arrays(three_np_arrays):
    arr1, arr2, arr3 = three_np_arrays

    d1 = {1: arr1, 2: arr2}
    d2 = {1: arr2, 2: arr1}
    d3 = {1: arr2, 2: arr3}

    assert hash(d1) == hash(d2)
    assert hash(d1) != hash(d3)


@pytest.mark.parametrize("dtype", ["datetime64[s]", "timedelta64[D]"])
def test_numpy_datetime_array(dtype):
    # memoryview is not supported for some dtypes e.g. datetime64
    # see https://github.com/joblib/joblib/issues/188 for more details
    a_hash = hash(np.arange(10))
    array = np.arange(0, 10, dtype=dtype)
    assert hash(array) != a_hash


def test_hash_numpy_noncontiguous():
    a = np.asarray(np.arange(6000).reshape((1000, 2, 3)), order="F")[:, :1, :]
    b = np.ascontiguousarray(a)
    assert hash(a) != hash(b)

    c = np.asfortranarray(a)
    assert hash(a) != hash(c)


@pytest.mark.parametrize("coerce_mmap", [True, False])
def test_hash_memmap(tmpdir, coerce_mmap):
    """Check that memmap and arrays hash identically if coerce_mmap is True."""
    filename = tmpdir.join("memmap_temp").strpath
    try:
        m = np.memmap(filename, shape=(10, 10), mode="w+")
        a = np.asarray(m)
        are_hashes_equal = hash(a, coerce_mmap=coerce_mmap) == hash(
            m, coerce_mmap=coerce_mmap
        )
        assert are_hashes_equal == coerce_mmap
    finally:
        if "m" in locals():
            del m
            # Force a garbage-collection cycle, to be certain that the
            # object is delete, and we don't run in a problem under
            # Windows with a file handle still open.
            gc.collect()


# This is also skipped in joblib tests.
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="This test is not stable under windows" " for some reason",
)
def test_hash_numpy_performance():
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(1000000)

    def get_hash_digest(x):
        hasher = xxh128()
        hasher.update(memoryview(x))
        return hasher.hexdigest()

    relative_diff = relative_time(get_hash_digest, hash, a)
    assert relative_diff < 0.3

    # Check that hashing an tuple of 3 arrays takes approximately
    # 3 times as much as hashing one array
    time_hashlib = 3 * time_func(get_hash_digest, a)
    time_hash = time_func(hash, (a, a, a))
    if time_hash + time_hashlib == 0:
        relative_diff = 0
    else:
        relative_diff = 0.5 * (
            abs(time_hash - time_hashlib) / (time_hash + time_hashlib)
        )
    assert relative_diff < 0.3


def test_hash_object_dtype():
    """Make sure that ndarrays with dtype `object' hash correctly."""

    a = np.array([np.arange(i) for i in range(6)], dtype=object)
    b = np.array([np.arange(i) for i in range(6)], dtype=object)

    assert hash(a) == hash(b)


def test_numpy_scalar():
    # Numpy scalars are built from compiled functions, and lead to
    # strange pickling paths explored, that can give hash collisions
    a = np.float64(2.0)
    b = np.float64(3.0)
    assert hash(a) != hash(b)


def test_numpy_dtype_pickling():
    # numpy dtype hashing is tricky to get right: see #231, #239, #251 #1080,
    # #1082, and explanatory comments inside
    # ``joblib.hashing.NumpyHasher.save``.

    # In this test, we make sure that the pickling of numpy dtypes is robust to
    # object identity and object copy.

    dt1 = np.dtype("f4")
    dt2 = np.dtype("f4")

    # simple dtypes objects are interned
    assert dt1 is dt2
    assert hash(dt1) == hash(dt2)

    dt1_roundtripped = pickle.loads(pickle.dumps(dt1))
    assert dt1 is not dt1_roundtripped
    assert hash(dt1) == hash(dt1_roundtripped)

    assert hash([dt1, dt1]) == hash([dt1_roundtripped, dt1_roundtripped])
    assert hash([dt1, dt1]) == hash([dt1, dt1_roundtripped])

    complex_dt1 = np.dtype([("name", np.str_, 16), ("grades", np.float64, (2,))])
    complex_dt2 = np.dtype([("name", np.str_, 16), ("grades", np.float64, (2,))])

    # complex dtypes objects are not interned
    assert hash(complex_dt1) == hash(complex_dt2)

    complex_dt1_roundtripped = pickle.loads(pickle.dumps(complex_dt1))
    assert complex_dt1_roundtripped is not complex_dt1
    assert hash(complex_dt1) == hash(complex_dt1_roundtripped)

    assert hash([complex_dt1, complex_dt1]) == hash(
        [complex_dt1_roundtripped, complex_dt1_roundtripped]
    )
    assert hash([complex_dt1, complex_dt1]) == hash(
        [complex_dt1_roundtripped, complex_dt1]
    )


def test_hashes_are_different_between_c_and_fortran_contiguous_arrays():
    # We want to be sure that the c-contiguous and f-contiguous versions of the
    # same array produce 2 different hashes.
    rng = np.random.RandomState(0)
    arr_c = rng.random_sample((10, 10))
    arr_f = np.asfortranarray(arr_c)
    assert hash(arr_c) != hash(arr_f)


def test_0d_array():
    hash(np.array(0))


def test_0d_and_1d_array_hashing_is_different():
    assert hash(np.array(0)) != hash(np.array([0]))


def test_hashes_stay_the_same_with_numpy_objects():
    # Note: joblib used to test numpy objects hashing by comparing the produced
    # hash of an object with some hard-coded target value to guarantee that
    # hashing remains the same across joblib versions. However, since numpy
    # 1.20 and joblib 1.0, joblib relies on potentially unstable implementation
    # details of numpy to hash np.dtype objects, which makes the stability of
    # hash values across different environments hard to guarantee and to test.
    # As a result, hashing stability across joblib versions becomes best-effort
    # only, and we only test the consistency within a single environment by
    # making sure:
    # - the hash of two copies of the same objects is the same
    # - hashing some object in two different python processes produces the same
    #   value. This should be viewed as a proxy for testing hash consistency
    #   through time between Python sessions (provided no change in the
    #   environment was done between sessions).

    def create_objects_to_hash():
        rng = np.random.RandomState(42)
        # Being explicit about dtypes in order to avoid
        # architecture-related differences. Also using 'f4' rather than
        # 'f8' for float arrays because 'f8' arrays generated by
        # rng.random.randn don't seem to be bit-identical on 32bit and
        # 64bit machines.
        to_hash_list = [
            rng.randint(-1000, high=1000, size=50).astype("<i8"),
            tuple(rng.randn(3).astype("<f4") for _ in range(5)),
            [rng.randn(3).astype("<f4") for _ in range(5)],
            {
                -3333: rng.randn(3, 5).astype("<f4"),
                0: [
                    rng.randint(10, size=20).astype("<i8"),
                    rng.randn(10).astype("<f4"),
                ],
            },
            # Non regression cases for
            # https://github.com/joblib/joblib/issues/308
            np.arange(100, dtype="<i8").reshape((10, 10)),
            # Fortran contiguous array
            np.asfortranarray(np.arange(100, dtype="<i8").reshape((10, 10))),
            # Non contiguous array
            np.arange(100, dtype="<i8").reshape((10, 10))[:, :2],
        ]
        return to_hash_list

    # Create two lists containing copies of the same objects.  joblib.hash
    # should return the same hash for to_hash_list_one[i] and
    # to_hash_list_two[i]
    to_hash_list_one = create_objects_to_hash()
    to_hash_list_two = create_objects_to_hash()

    e1 = ProcessPoolExecutor(max_workers=1)
    e2 = ProcessPoolExecutor(max_workers=1)

    try:
        for obj_1, obj_2 in zip(to_hash_list_one, to_hash_list_two):
            # testing consistency of hashes across python processes
            hash_1 = e1.submit(hash, obj_1).result()
            hash_2 = e2.submit(hash, obj_1).result()
            assert hash_1 == hash_2

            # testing consistency when hashing two copies of the same objects.
            hash_3 = e1.submit(hash, obj_2).result()
            assert hash_1 == hash_3

    finally:
        e1.shutdown()
        e2.shutdown()


class MyOwnModel(BaseEstimator):
    def fit(self, X, y):
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        return np.array(X.shape[0] * [self.mean_])


def test_using_custom_model():
    insurance = get_data("insurance")

    # init setup
    reg1 = RegressionExperiment()
    reg1 = reg1.setup(data=insurance, target="charges")

    my_own_model = MyOwnModel()
    reg1.create_model(my_own_model)
