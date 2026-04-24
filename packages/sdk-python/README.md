# `packages/sdk-python` — Python SDK for the Control Plane API

**Status:** stub. To be generated from `/openapi.json` when the API surface
stabilises (target: after Phase 10 UI completes).

This package will be published to PyPI as `pycaret-client` (name TBD). It
lets you drive the Control Plane from a notebook or a script without going
through the engine directly:

```python
from pycaret_client import ControlPlane

cp = ControlPlane("http://localhost:8000", token="...")
ws = cp.workspaces.create(name="Demo")
proj = cp.projects.create(workspace_id=ws.id, name="Churn")
ds = cp.datasets.upload(workspace_id=ws.id, path="churn.csv")
run = cp.runs.submit(experiment_id=exp.id, plan="compare", data_source_id=ds.id)
run.wait()
print(run.leaderboard)
```

Distinguished from `pycaret` (the engine):
- `pycaret` runs ML **in process** — notebook / script workflow.
- `pycaret-client` talks to a running Control Plane over HTTP — drives
  managed runs with artifact storage, deployment, monitoring.

Both can be used in the same notebook.

Generation plan: `openapi-python-client` against the live `/openapi.json`
emits typed request/response models and a typed client class. We vendor the
output + hand-write thin convenience wrappers (`.wait()`, `.stream_events()`).
