import pycaret.time_series as pt
from pycaret.datasets import get_data
from pycaret.parallel import FugueBackend


def test_ts_parallel():
    exp = pt.TSForecastingExperiment()

    exp.setup(
        data_func=lambda: get_data("airline", verbose=False),
        fh=12,
        fold=3,
        fig_kwargs={"renderer": "notebook"},
        session_id=42,
    )

    test_models = exp.models().index.tolist()[:2]

    exp.compare_models(include=test_models, n_select=2)
    exp.compare_models(include=test_models, n_select=2, parallel=FugueBackend("dask"))

    fconf = {
        "fugue.rpc.server": "fugue.rpc.flask.FlaskRPCServer",  # keep this value
        "fugue.rpc.flask_server.host": "localhost",  # the driver ip address workers can access
        "fugue.rpc.flask_server.port": "3333",  # the open port on the dirver
        "fugue.rpc.flask_server.timeout": "2 sec",  # the timeout for worker to talk to driver
    }

    be = FugueBackend("dask", fconf, display_remote=True, batch_size=1, top_only=False)
    exp.compare_models(include=test_models, n_select=2, parallel=be)

    exp.pull()


def test_ts_parallel_singleton():
    pt.setup(
        data_func=lambda: get_data("airline", verbose=False),
        fh=12,
        fold=3,
        fig_kwargs={"renderer": "notebook"},
        session_id=42,
    )

    test_models = pt.models().index.tolist()[:2]

    pt.compare_models(include=test_models, n_select=2, parallel=FugueBackend("dask"))
    pt.pull()
