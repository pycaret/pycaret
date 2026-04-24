# `infra/terraform` — cloud deployment modules

**Status:** stubs (`aws/`, `gcp/`, `azure/`). Target for V2.

Each cloud gets a self-contained Terraform module that provisions the full
Control Plane from zero:

| Cloud | Compute | DB | Object | Cache | Secrets | Logs |
|-------|---------|----|--------|-------|---------|------|
| AWS   | ECS/Fargate (or EKS) | RDS Postgres | S3 | ElastiCache | Secrets Manager | CloudWatch |
| GCP   | Cloud Run (or GKE)   | Cloud SQL    | GCS | Memorystore | Secret Manager  | Cloud Logging |
| Azure | Container Apps (or AKS) | Azure Database for PostgreSQL | Blob Storage | Azure Cache | Key Vault | App Insights |

Each module is a single `terraform apply` per cloud. Variables expose the
minimum surface (region, domain, tier), everything else is defaulted.

When work starts, expect:

```
infra/terraform/aws/
├── main.tf           # entry module
├── variables.tf      # region, domain, ecs_task_cpu, rds_tier, ...
├── outputs.tf        # endpoint URLs, db host, object bucket name
├── iam.tf
├── ecs.tf
├── rds.tf
├── s3.tf
├── alb.tf
├── secrets.tf
└── README.md         # per-cloud getting-started
```

See `docs/revamp/CONTROL_PLANE_SPEC.md § 18.5` for target.
