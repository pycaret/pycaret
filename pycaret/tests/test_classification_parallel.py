from pycaret.datasets import get_data
import pycaret.classification as pc
from pycaret.parallel import FugueBackend


def test():
    pc.setup(
        data=lambda: get_data("juice", verbose=False, profile=False),
        target="Purchase",
        session_id=0,
        n_jobs=1,
        verbose=False,
        silent=True,
        html=False,
    )

    test_models = pc.models().index.tolist()[:5]

    pc.compare_models(include=test_models, n_select=2)
    pc.compare_models(include=test_models, n_select=2, parallel=FugueBackend("dask"))

    fconf = {
        "fugue.rpc.server": "fugue.rpc.flask.FlaskRPCServer",  # keep this value
        "fugue.rpc.flask_server.host": "localhost",  # the driver ip address workers can access
        "fugue.rpc.flask_server.port": "3333",  # the open port on the dirver
        "fugue.rpc.flask_server.timeout": "2 sec",  # the timeout for worker to talk to driver
    }

    be = FugueBackend("dask", fconf, display_remote=True, batch_size=3, top_only=False)
    pc.compare_models(n_select=2, parallel=be)

    res = pc.pull()
    assert res.shape[0] > 10
    assert res.index[0] == "lda"
