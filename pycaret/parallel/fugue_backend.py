from typing import Any, Callable, Dict, List, Optional, Union, Iterable

import cloudpickle
import pandas as pd
from fugue import transform
from pycaret.internal.tabular import (
    _append_display_container,
    pull,
    _create_display,
    _get_context_lock,
)
from pycaret.internal.Display import Display
from threading import RLock
from math import ceil
import random
from .parallel_backend import NoDisplay, ParallelBackend

try:
    import fugue_dask
except Exception:
    pass

try:
    import fugue_spark
except Exception:
    pass


class _DisplayUtil:
    def __init__(
        self, display: Optional[Display], progress: int, verbose: bool, sort: str
    ):
        self._lock = RLock()
        self._display = display or _create_display(
            progress, verbose=verbose, monitor_rows=None
        )
        self._sort = sort
        self._df: Optional[pd.DataFrame] = None
        self._display.display_progress()

    def update(self, df: pd.DataFrame) -> None:
        with self._lock:
            if self._df is None:
                self._df = df
            else:
                self._df = pd.concat([self._df, df]).sort_values(
                    self._sort, ascending=False
                )
            self._display.move_progress(df.shape[0])
            self._display.replace_master_display(self._df)
            self._display.display_master_display()

    def finish(self) -> None:
        self._display.display_master_display()


class FugueBackend(ParallelBackend):
    """
    Fugue Backend for PyCaret

    Example
    -------

    >>> from pycaret.datasets import get_data
    >>> from pyspark.sql import SparkSession
    >>> from pycaret.parallel import FugueBackend
    >>> juice = get_data('juice')
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
    >>> spark = SparkSessiong.builder.getOrCreate()
    >>> best_model = compare_models(parallel=FugueBackend(spark))


    engine: Any, default = None
        A ``SparkSession`` instance or "spark" to get the current ``SparkSession``.
        "dask" to get the current Dask client. "native" to test locally using single
        thread. None means using Fugue's Native execution engine to run jobs
        sequentially on local machine, it is for testing/development purpose.

    conf: Any, default = None
        Fugue ExecutionEngine configs.

    batch_size: int, default = 1
        Batch size to partition the tasks. For example if there are 16 tasks,
        and with ``batch_size=3`` we will have 6 batches to be processed
        distributedly. Smaller batch sizes will have better load balance but
        worse overhead, and vise versa.

    display_remote: bool, default = False
        Whether show progress bar and metrics table when using a distributed
        backend. By setting it to True, you must also enable the
        `callback settings <https://fugue-project.github.io/tutorials/tutorials/advanced/rpc.html>`_
        to receive realtime updates.

    top_only: bool, default = False
        Whether only return the top ``n_select`` models from each worker. When top only,
        the overall execution time can be faster.
    """

    def __init__(
        self,
        engine: Any = None,
        conf: Any = None,
        batch_size: int = 1,
        display_remote: bool = False,
        top_only: bool = False,
    ):
        super().__init__()
        self._engine = engine
        self._conf: Dict[str, Any] = conf or {}
        self._batch_size = batch_size
        self._display_remote = display_remote
        self._top_only = top_only
        self._func: Optional[Callable] = None
        self._params: Optional[Dict[str, Any]] = None

    def __getstate__(self) -> Dict[str, Any]:
        res = dict(self.__dict__)
        del res["_engine"]
        return res

    def compare_models(
        self, func: Callable, params: Dict[str, Any]
    ) -> Union[Any, List[Any]]:
        self._func = func
        self._params = dict(params)
        assert "include" in self._params
        assert "sort" in self._params

        shuffled_idx = pd.DataFrame(
            dict(
                idx=random.sample(
                    range(len(self._params["include"])), len(self._params["include"])
                )
            )
        )
        du: Optional[Display] = (
            None
            if not self._display_remote
            else _DisplayUtil(
                self._params.get("display", None),
                progress=shuffled_idx.shape[0],
                verbose=self._params.get("verbose", False),
                sort=self._params["sort"],
            )
        )
        outputs = transform(
            shuffled_idx,
            self._remote_compare_models,
            schema="output:binary",
            partition={
                "num": ceil(shuffled_idx.shape[0] / self._batch_size),
                "algo": "even",
            },
            engine=self._engine,
            engine_conf=self._conf,
            callback=None if du is None else du.update,
            force_output_fugue_dataframe=True,
            as_local=True,
        ).as_array()
        res = pd.concat(cloudpickle.loads(x[0]) for x in outputs)
        res = res.sort_values(self._params["sort"], ascending=False)
        top = res.head(self._params.get("n_select", 1))
        _append_display_container(res.iloc[:, :-1])
        top_models = [cloudpickle.loads(x) for x in top._model]
        if du is not None:
            du.finish()
        return top_models[0] if len(top_models) == 1 else top_models

    def _remote_compare_models(
        self, idx: List[List[Any]], report: Optional[Callable]
    ) -> List[List[Any]]:
        include = [self._params["include"][i[0]] for i in idx]
        self.remote_setup()
        params = dict(self._params)
        params.pop("include")
        params["display"] = NoDisplay()
        results: List[List[Any]] = []
        with _get_context_lock():
            top = (
                min(params.get("n_select", 1), len(include))
                if self._top_only
                else len(include)
            )
            params["n_select"] = top
            m = self._func(include=include, **params)
            if not isinstance(m, list):
                m = [m]
            res = pull()[:top]
            if report is not None:
                report(res)
            results.append(
                [
                    cloudpickle.dumps(
                        res.assign(_model=[cloudpickle.dumps(x) for x in m])
                    )
                ]
            )
        return results
