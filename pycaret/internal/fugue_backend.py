from typing import Any, Callable, Dict, List, Optional, Union, Iterable

import cloudpickle
import pandas as pd
from fugue import transform
from pycaret.internal.tabular import (
    _append_display_container,
    pull,
    _create_display,
    _get_setup_signature,
    _get_context_lock,
)
from pycaret.internal.Display import Display
from threading import RLock
from math import ceil
import random

try:
    import fugue_dask
except Exception:
    pass

try:
    import fugue_spark
except Exception:
    pass


class _NoDisplay(Display):
    def can_display(self, override):
        return False


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


class _CompareModelsWrapper:
    def __init__(self, setup_call: Dict[str, Any], compare_models_call: Dict[str, Any]):
        self._signature = _get_setup_signature()
        self._setup_func = setup_call["func"]
        # self._setup_params = dict(setup_call["params"])
        self._setup_params = {
            k: v for k, v in setup_call["params"].items() if v is not None
        }
        self._func = compare_models_call["func"]
        self._params = dict(compare_models_call["params"])
        assert "include" in self._params
        assert "sort" in self._params

    def compare_models(
        self, engine: Any, conf: Any, batch_size: int, display_remote: bool
    ) -> Union[Any, List[Any]]:
        """
        This function is a wrapper of the original ``compare_models`` function
        in order to run using Fugue backends.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> from pyspark.sql import SparkSession
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> session = SparkSessiong.builder.getOrCreate()
        >>> best_model = compare_models(engine=session)


        engine: Any
            A ``SparkSession`` instance or "spark" to get the current ``SparkSession``.
            "dask" to get the current Dask client. "native" to test locally using single
            thread.

        conf: Any
            Fugue ExecutionEngine configs.

        batch_size: int
            Batch size to partition the tasks. For example if there are 16 tasks,
            and with ``batch_size=3`` we will have 6 batches to be processed
            distributedly. Smaller batch sizes will have better load balance but
            worse overhead, and vise versa.

        display_remote: bool
            Whether show progress bar and metrics table when using a distributed
            backend. By setting it to True, you must also enable the
            `callback settings <https://fugue-project.github.io/tutorials/tutorials/advanced/rpc.html>`_
            to receive realtime updates.

        Returns:
            Trained model or list of trained models, depending on the ``n_select`` param.
        """
        shuffled_idx = pd.DataFrame(
            dict(
                idx=random.sample(
                    range(len(self._params["include"])), len(self._params["include"])
                )
            )
        )
        du: Optional[Display] = (
            None
            if not display_remote
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
            partition={"num": ceil(shuffled_idx.shape[0] / batch_size), "algo": "even"},
            engine=engine,
            engine_conf=conf,
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

    def _is_remote(self) -> bool:
        return self._signature != _get_setup_signature()

    def _remote_setup(self):
        if self._is_remote():
            params = dict(self._setup_params)
            params["silent"] = True
            params["verbose"] = False
            params["html"] = False
            self._setup_func(**params)

    def _remote_compare_models(
        self, idx: List[List[Any]], report: Optional[Callable]
    ) -> List[List[Any]]:
        include = [self._params["include"][i[0]] for i in idx]
        self._remote_setup()
        params = dict(self._params)
        params.pop("include")
        params["parallel_backend"] = "remote"
        results: List[List[Any]] = []
        with _get_context_lock():
            if report is not None:  # best visual effect for realtime update
                params["n_select"] = 1
                for inc in include:
                    m = self._func(include=[inc], **params)
                    res = pull()
                    report(res)
                    results.append(
                        [cloudpickle.dumps(res.assign(_model=[cloudpickle.dumps(m)]))]
                    )
            else:  # best performance
                params["n_select"] = len(include)
                m = self._func(include=include, **params)
                if not isinstance(m, list):
                    m = [m]
                res = pull()
                results.append(
                    [
                        cloudpickle.dumps(
                            res.assign(_model=[cloudpickle.dumps(x) for x in m])
                        )
                    ]
                )
        return results
