from typing import Any, Callable, Dict, List, Optional, Union

import cloudpickle
import pandas as pd
from fugue import FugueWorkflow
from pycaret.internal.tabular import _append_display_container, pull, _create_display
from pycaret.internal.Display import Display
from threading import RLock
from math import ceil

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

    def update(self, df: pd.DataFrame):
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

    def refresh(self):
        self._display.display_master_display()


class _CompareModelsWrapper:
    def __init__(self, setup_call: Dict[str, Any], compare_models_call: Dict[str, Any]):
        self._setup_func = setup_call["func"]
        self._setup_params = dict(setup_call["params"])
        self._func = compare_models_call["func"]
        self._params = dict(compare_models_call["params"])
        assert "include" in self._params
        assert "sort" in self._params

    def compare_models(
        self, engine: Any, conf: Any, batch_size: int
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
        >>> best_model = compare_models(fugue_engine=session)


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

        Returns:
            Trained model or list of trained models, depending on the ``n_select`` param.
        """

        include = pd.DataFrame(
            dict(models=[cloudpickle.dumps(x) for x in self._params["include"]])
        ).sample(
            frac=1.0
        )  # shuffle
        du = _DisplayUtil(
            self._params.get("display", None),
            progress=include.shape[0],
            verbose=self._params.get("verbose", False),
            sort=self._params["sort"],
        )
        dag = FugueWorkflow()
        dag.df(include).partition(
            num=ceil(include.shape[0] / batch_size), algo="even"
        ).transform(
            self._remote_compare_models,
            schema="output:binary",
            callback=du.update,
        ).persist().yield_dataframe_as(
            "res"
        )
        outputs = dag.run(engine, conf)["res"].as_array()
        res = pd.concat(cloudpickle.loads(x[0]) for x in outputs)
        res = res.sort_values(self._params["sort"], ascending=False)
        top = res.head(self._params.get("n_select", 1))
        _append_display_container(res.iloc[:, :-1])
        top_models = [cloudpickle.loads(x) for x in top._model]
        du.refresh()
        return top_models[0] if len(top_models) == 1 else top_models

    def _remote_setup(self):
        params = dict(self._setup_params)
        params["silent"] = True
        params["verbose"] = False
        params["n_jobs"] = 1
        params["html"] = False
        self._setup_func(**params)

    def _remote_compare_models(
        self, models: List[List[Any]], report: Optional[Callable]
    ) -> List[List[Any]]:
        include = [cloudpickle.loads(x[0]) for x in models]
        self._remote_setup()
        params = dict(self._params)
        params.pop("include")
        params["n_select"] = 1
        params["fugue_engine"] = "remote"
        results: List[pd.DataFrame] = []
        for inc in include:
            m = self._func(include=[inc], **params)
            if report is not None:
                report(pull())
            results.append(pull().assign(_model=[cloudpickle.dumps(m)]))
        return [[cloudpickle.dumps(pd.concat(results))]]
