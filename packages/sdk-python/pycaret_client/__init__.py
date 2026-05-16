"""PyCaret Control Plane SDK — drive a running ``pycaret-server`` over HTTP.

Distinguishes from the engine package (``pycaret``):

  * ``pycaret``         — runs ML in-process; notebook / script workflow.
  * ``pycaret-client``  — drives a remote Control Plane over HTTP; managed
                          runs with artifact storage, deployment, monitoring.

Both can be used in the same notebook.

Quickstart::

    from pycaret_client import ControlPlane

    cp = ControlPlane.connect("http://localhost:8020", email="me@x.com",
                              password="...")
    ws = cp.workspaces.list()[0]
    proj = cp.projects.create(workspace_id=ws["id"], name="Churn")
    exp = cp.experiments.create(
        project_id=proj["id"],
        name="baseline",
        task="classification",
        target="target",
        setup_params={"session_id": 42, "fold": 5},
    )
    run = cp.runs.submit(
        experiment_id=exp["id"],
        plan="compare",
        sklearn_dataset="iris",
    )
    run = cp.runs.wait(run["id"])
    print(run["leaderboard"])
"""

from pycaret_client.client import ControlPlane

__version__ = "0.1.0a0"

__all__ = ["ControlPlane", "__version__"]
