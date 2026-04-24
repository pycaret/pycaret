# `infra/terraform/aws` — AWS deployment module (V2 stub)

See `../README.md` for the per-cloud matrix. This module targets ECS/Fargate
(or EKS as an opt-in) + RDS Postgres + S3 + ElastiCache + ALB + Secrets Manager
+ CloudWatch.

Minimum variables an operator provides:

```hcl
module "pycaret" {
  source = "github.com/pycaret/pycaret//infra/terraform/aws?ref=v4"
  region   = "us-east-1"
  domain   = "ml.example.com"
  tier     = "small"         # small | medium | large
  # ... everything else defaulted
}
```
