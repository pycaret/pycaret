# `infra/helm` — Helm chart for Kubernetes

**Status:** stub. Target for V2 enterprise distribution.

When work starts here, this directory will contain a single Helm chart that
deploys the full Control Plane stack to a Kubernetes cluster:

```
infra/helm/pycaret/
├── Chart.yaml
├── values.yaml            # every knob an operator needs
├── values-dev.yaml        # minimal smoke deploy
├── values-prod.yaml       # opinionated prod
├── templates/
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── web-deployment.yaml
│   ├── web-service.yaml
│   ├── worker-deployment.yaml
│   ├── deployment-runtime.yaml  # templated per user-deployment
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── pvc.yaml
│   └── job-migrate.yaml         # alembic upgrade head on install
└── README.md
```

Chart should be installable by:

```bash
helm repo add pycaret https://charts.pycaret.org
helm install pycaret pycaret/pycaret --version 0.1.0 \
  --set postgresql.enabled=true \
  --set redis.enabled=true \
  --set minio.enabled=true
```

Dependencies on bitnami/postgresql + bitnami/redis + minio/minio charts so
operators can bring their own or use bundled.

See `docs/revamp/CONTROL_PLANE_SPEC.md § 18.4` for target.
