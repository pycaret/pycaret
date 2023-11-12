"""Module to benchmark performance of PyCaret joblib.Memory tweaks
"""
import gc
import os
import shutil
from timeit import Timer
from typing import Union

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from joblib import Memory
from joblib.hashing import hash  # noqa

from pycaret.classification import ClassificationExperiment
from pycaret.datasets import get_data
from pycaret.internal.memory import fast_hash  # noqa
from pycaret.regression import RegressionExperiment

pytestmark = [
    pytest.mark.benchmark,
]


data_df = get_data(verbose=False)
supervised_datasets_df = data_df.copy()
supervised_datasets_df["Items"] = (
    supervised_datasets_df["# Instances"] * supervised_datasets_df["# Attributes"]
)
supervised_datasets_df = supervised_datasets_df[
    supervised_datasets_df["Items"] >= 30000
].sort_values("Items", ascending=True)
supervised_datasets_df = supervised_datasets_df[
    supervised_datasets_df["Default Task"].str.startswith("Classification")
    | supervised_datasets_df["Default Task"].str.startswith("Regression")
]
supervised_datasets = supervised_datasets_df[
    ["Dataset", "Default Task", "Target Variable 1"]
].iterrows()


@pytest.fixture
def gc_fixture():
    gc.collect()
    yield


def _test_synthetic_data(data, repeats: int = 100):
    globals_with_data = globals().copy()
    globals_with_data["data"] = data
    pycaret_joblib_time = min(
        Timer(
            "fast_hash(data)",
            setup="gc.collect()",
            globals=globals_with_data,
        ).repeat(repeats, 1)
    )
    original_joblib_time = min(
        Timer("hash(data)", setup="gc.collect()", globals=globals_with_data).repeat(
            repeats, 1
        )
    )
    print(
        f"Original: {original_joblib_time} vs PyCaret: {pycaret_joblib_time} ({original_joblib_time-pycaret_joblib_time})"
    )
    return original_joblib_time, pycaret_joblib_time


def _test_real_data(data_name: str, repeats: int = 20):
    data = get_data(data_name, verbose=False)
    globals_with_data = globals().copy()
    globals_with_data["data"] = data
    pycaret_joblib_time = min(
        Timer(
            "fast_hash(data)", setup="gc.collect()", globals=globals_with_data
        ).repeat(repeats, 1)
    )
    original_joblib_time = min(
        Timer("hash(data)", setup="gc.collect()", globals=globals_with_data).repeat(
            repeats, 1
        )
    )
    print(f"({data_name} {data.shape}")
    print(
        f"({data_name}) Original: {original_joblib_time} vs PyCaret: {pycaret_joblib_time} ({original_joblib_time-pycaret_joblib_time})"
    )
    return original_joblib_time, pycaret_joblib_time


def _test_e2e_setup(
    data: pd.DataFrame, task: str, target: str, memory: str, memory_dir: str
):
    if task.startswith("Classification"):
        exp = ClassificationExperiment()
    else:
        exp = RegressionExperiment()
    shutil.rmtree(memory_dir, ignore_errors=True)
    memory = Memory(memory_dir, verbose=0) if memory == "joblib" else memory_dir
    return exp, data, target, memory


def _test_e2e(
    exp: Union[ClassificationExperiment, RegressionExperiment],
    data: pd.DataFrame,
    target: str,
    memory: str,
):
    exp.setup(
        data,
        target=target,
        memory=memory,
        verbose=False,
        n_jobs=1,
        system_log=False,
        remove_multicollinearity=True,
        feature_selection=True,
        pca=True,
        transformation=True,
        session_id=0,
    )
    for _ in range(4):
        exp.create_model("dummy", verbose=False)
    for _ in range(8):
        exp.dataset_transformed
        exp.train_transformed
        exp.test_transformed


def _test_e2e_timeit(
    data_name: pd.DataFrame, task: str, target: str, memory_dir: str, repeats: int = 3
):
    globals_with_data = globals().copy()
    globals_with_data["data"] = get_data(data_name, verbose=False).dropna(
        subset=[target]
    )
    globals_with_data["task"] = task
    globals_with_data["target"] = target
    globals_with_data["memory_dir"] = os.path.join(memory_dir, "joblib")

    pycaret_joblib_time = min(
        Timer(
            "_test_e2e(*args)",
            setup="args = _test_e2e_setup(data, task, target, None, memory_dir); gc.collect()",
            globals=globals_with_data,
        ).repeat(repeats, 1)
    )
    original_joblib_time = min(
        Timer(
            "_test_e2e(*args)",
            setup="args = _test_e2e_setup(data, task, target, 'joblib', memory_dir); gc.collect()",
            globals=globals_with_data,
        ).repeat(repeats, 1)
    )
    print(
        f"({data_name}) Original: {original_joblib_time} vs PyCaret: {pycaret_joblib_time} ({original_joblib_time-pycaret_joblib_time})"
    )
    return original_joblib_time, pycaret_joblib_time


def test_numpy_hashing_performance(gc_fixture):
    rng = np.random.RandomState(42)
    X_numpy = rng.rand(100000, 100)
    original_joblib_time, pycaret_joblib_time = _test_synthetic_data(X_numpy)
    assert pycaret_joblib_time < original_joblib_time
    assert pycaret_joblib_time < 0.02


def test_numpy_object_hashing_performance(gc_fixture):
    rng = np.random.RandomState(42)
    X_numpy = rng.randint(low=1, high=100, size=(100000, 10)).astype(str).astype(object)
    original_joblib_time, pycaret_joblib_time = _test_synthetic_data(
        X_numpy, repeats=20
    )
    assert pycaret_joblib_time < original_joblib_time
    assert pycaret_joblib_time < 2


@pytest.mark.parametrize("shape", ((10000, 10), (100000, 100)))
def test_pandas_hashing_performance(shape, gc_fixture):
    rng = np.random.RandomState(42)
    X_pandas = pd.DataFrame(rng.rand(*shape))
    original_joblib_time, pycaret_joblib_time = _test_synthetic_data(X_pandas)
    assert pycaret_joblib_time < original_joblib_time
    assert pycaret_joblib_time < 0.1


@pytest.mark.parametrize("shape", ((10000, 10), (100000, 100)))
def test_pandas_categorical_hashing_performance(shape, gc_fixture):
    rng = np.random.RandomState(42)
    shape = (shape[0], shape[1] // 2)
    X_pandas = pd.DataFrame(rng.rand(*shape))
    X_pandas_categorical = pd.DataFrame(
        rng.randint(low=1, high=100, size=X_pandas.shape).astype(str).astype(object)
    )
    X_pandas_categorical = pd.concat((X_pandas, X_pandas_categorical), axis=1)
    original_joblib_time, pycaret_joblib_time = _test_synthetic_data(
        X_pandas_categorical, repeats=20
    )
    assert pycaret_joblib_time < original_joblib_time
    assert pycaret_joblib_time < 2


def test_pandas_series_hashing_performance(gc_fixture):
    rng = np.random.RandomState(42)
    X_pandas_series = pd.Series(rng.rand(10000000))
    original_joblib_time, pycaret_joblib_time = _test_synthetic_data(X_pandas_series)
    assert pycaret_joblib_time < original_joblib_time
    assert pycaret_joblib_time < 0.05


def test_scipy_hashing_performance(gc_fixture):
    rng = np.random.RandomState(42)
    X_csr = scipy.sparse.rand(1000, 10000, random_state=rng)
    original_joblib_time, pycaret_joblib_time = _test_synthetic_data(X_csr)
    assert pycaret_joblib_time < original_joblib_time
    assert pycaret_joblib_time < 0.01


@pytest.mark.parametrize("dataset_name", data_df["Dataset"])
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI as it takes too long")
def test_real_data_performance(dataset_name: str, gc_fixture):
    original_joblib_time, pycaret_joblib_time = _test_real_data(dataset_name)
    # super small differences are fine
    assert (
        pycaret_joblib_time < original_joblib_time
        or abs(pycaret_joblib_time - original_joblib_time) < 0.05
    )


@pytest.mark.parametrize("dataset", supervised_datasets)
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI as it takes too long")
def test_setup_performance(dataset: tuple, gc_fixture, tmpdir):
    dataset = dataset[1]
    original_joblib_time, pycaret_joblib_time = _test_e2e_timeit(
        dataset["Dataset"],
        dataset["Default Task"],
        dataset["Target Variable 1"],
        str(tmpdir),
    )
    # super small differences are fine
    assert (
        pycaret_joblib_time < original_joblib_time
        or abs(pycaret_joblib_time - original_joblib_time) < 0.2
    )
