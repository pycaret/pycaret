# `services/worker` — background job runner

**Status:** stub. Current MVP uses an in-process `ThreadPoolExecutor` inside
`services/api/pycaret_server/runs/orchestrator.py`; this directory becomes a
separate deployable once the single-process model stops scaling.

When work starts here, this directory will contain a separate FastAPI-less
Python package (`pycaret-worker`) that:

1. Reads `Job` rows from the platform DB (or a dedicated Redis queue).
2. Dispatches job payloads to the engine (`pycaret.tasks.*Experiment.run(config)`).
3. Writes results + events back through the same DB / event-broker paths the
   in-process orchestrator uses today.

Design constraints:
- Workers are stateless. Every dispatch reads the Job payload fresh.
- Multiple workers can run in parallel; ops picks the backend (Celery / RQ /
  Arq / bare `rq` / Ray / Kubernetes Jobs).
- The worker imports `pycaret-server` only for DB models + schema, never for
  HTTP / auth code.

See `docs/revamp/ARCHITECTURE.md § 15` for the service split.
