import random
from math import ceil
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Union

import cloudpickle
import pandas as pd
from fugue import transform

from pycaret.internal.display import CommonDisplay
from pycaret.internal.parallel.parallel_backend import ParallelBackend

_LOCK = RLock()  # noqa


def _get_context_lock():
    # This function may not be necessary, but it's safe
    return globals()["_LOCK"]


class _DisplayUtil:
    def __init__(
        self,
        display: Optional[CommonDisplay],
        progress: int,
        verbose: bool,
        sort: str,
        asc: bool,
    ):
        self._lock = RLock()
        self._display = display or self._create_display(
            progress, verbose=verbose, monitor_rows=None
        )
        self._sort = sort
        self._asc = asc
        self._df: Optional[pd.DataFrame] = None

    def update(self, df: pd.DataFrame) -> None:
        with self._lock:
            if self._df is None:
                self._df = df
            else:
                self._df = pd.concat([self._df, df]).sort_values(
                    self._sort, ascending=self._asc
                )
            self._display.move_progress(df.shape[0])
            self._display.display(self._df, final_display=False)

    def finish(self, df: Any = None) -> None:
        self._display.display(df if df is not None else self._df, final_display=True)

    def _create_display(
        self, progress: int, verbose: bool, monitor_rows: Any
    ) -> CommonDisplay:
        progress_args = {"max": progress}
        return CommonDisplay(
            verbose=verbose,
            progress_args=None if progress < 0 else progress_args,
            monitor_rows=monitor_rows,
        )


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
        self, instance: Any, params: Dict[str, Any]
    ) -> Union[Any, List[Any]]:
        self._params = dict(params)
        assert "include" in self._params
        assert "sort" in self._params

        sort_col, asc = instance._process_sort(self._params["sort"])

        shuffled_idx = pd.DataFrame(
            dict(
                idx=random.sample(
                    range(len(self._params["include"])), len(self._params["include"])
                )
            )
        )
        du = _DisplayUtil(
            self._params.get("display", None),
            progress=-1 if not self._display_remote else shuffled_idx.shape[0],
            verbose=self._params.get("verbose", False),
            sort=sort_col,
            asc=asc,
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
            callback=None if not self._display_remote else du.update,
            force_output_fugue_dataframe=True,
            as_local=True,
        ).as_array()
        res = pd.concat(cloudpickle.loads(x[0]) for x in outputs)
        res = res.sort_values(sort_col, ascending=asc)
        top = res.head(self._params.get("n_select", 1))
        instance._display_container.append(res.iloc[:, :-1])
        top_models = [cloudpickle.loads(x) for x in top._model]
        du.finish(res.iloc[:, :-1])
        return top_models[0] if len(top_models) == 1 else top_models

    def _remote_compare_models(
        self, idx: List[List[Any]], report: Optional[Callable]
    ) -> List[List[Any]]:
        include = [self._params["include"][i[0]] for i in idx]
        instance = self.remote_setup()
        params = dict(self._params)
        params.pop("include")
        params["verbose"] = False
        results: List[List[Any]] = []

        with _get_context_lock():  # protection for non-distributed dask
            top = (
                min(params.get("n_select", 1), len(include))
                if self._top_only
                else len(include)
            )
            params["n_select"] = top
            m = instance.compare_models(include=include, **params)
            if not isinstance(m, list):
                m = [m]
            res = instance.pull()[:top]
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
