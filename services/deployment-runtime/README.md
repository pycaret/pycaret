# `services/deployment-runtime` — inference server for deployed pipelines

**Status:** stub. Current MVP serves predictions inside the API process via
`services/api/pycaret_server/serving.py::DeploymentRegistry`. This directory
becomes a separate deployable for the production serving story.

When work starts here, this directory will contain a minimal FastAPI app that:

1. Accepts an artifact bundle on startup (pickled pipeline + schema.json).
2. Exposes the deployment contract:
   - `GET /health`
   - `GET /schema`
   - `POST /predict`
   - `POST /predict_batch`
   - `GET /metadata`
3. Records request logs and latency back to the control-plane API.

This is what gets deployed to:
- a standalone Docker container (`docker/Dockerfile.deployment`)
- Kubernetes (via Helm)
- AWS ECS / Lambda / GCP Cloud Run / Azure Container Apps (via Terraform)

Each deployment is an **immutable** snapshot of a specific `pipeline_artifact`
— never re-trained in place.

See `docs/revamp/ARCHITECTURE.md § deployment-types` and
`docs/revamp/CONTROL_PLANE_SPEC.md § 10` for the spec.
