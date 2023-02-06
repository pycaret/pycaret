import pycaret.regression as pr
from pycaret.datasets import get_data


def test_regression_parallel():
    from pycaret.parallel import FugueBackend

    pr.setup(
        data=get_data("insurance", verbose=False),
        target="charges",
        session_id=0,
        n_jobs=1,
        verbose=False,
        html=False,
    )

    test_models = pr.models().index.tolist()[:5]

    pr.compare_models(include=test_models, n_select=2)
    pr.compare_models(include=test_models, n_select=2, parallel=FugueBackend("dask"))

    fconf = {
        "fugue.rpc.server": "fugue.rpc.flask.FlaskRPCServer",  # keep this value
        "fugue.rpc.flask_server.host": "localhost",  # the driver ip address workers can access
        "fugue.rpc.flask_server.port": "3333",  # the open port on the dirver
        "fugue.rpc.flask_server.timeout": "2 sec",  # the timeout for worker to talk to driver
    }

    be = FugueBackend("dask", fconf, display_remote=True, batch_size=3, top_only=False)
    pr.compare_models(n_select=2, parallel=be)

    res = pr.pull()
    assert res.shape[0] > 10
