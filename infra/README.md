# `infra/` — deployment infrastructure

Everything non-source-code needed to ship PyCaret Control Plane.

```
infra/
├── docker/                 # local + single-server distribution
│   ├── Dockerfile.api      # FastAPI backend image
│   ├── Dockerfile.ui       # React frontend image
│   ├── docker-compose.yml  # full-stack local compose
│   └── nginx.ui.conf       # nginx config baked into the UI image
├── helm/                   # Kubernetes chart (V2 stub)
└── terraform/              # one-click cloud modules (V2 stubs)
    ├── aws/
    ├── gcp/
    └── azure/
```

Currently active (MVP): `infra/docker/`. Everything else is roadmapped.

## Local compose

```bash
docker compose -f infra/docker/docker-compose.yml up --build
# → http://localhost:3000
```

Reverse-proxies `/api` + `/ws` to the API container via nginx so the browser
sees a single origin.
