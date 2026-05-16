# PLATFORM_ARCHITECTURE.md — pluggable-backends foundation

**Status:** Vision document drafted session 57 (2026-05-16). Sets the architectural target the platform is being shaped toward. Not all backends are implemented yet — see § 4 for the matrix.

**Purpose:** Give every future contributor (human or Claude agent) the same mental model of what PyCaret is structurally trying to be. When you're tempted to hardcode a call to `boto3` or to `psycopg`, read this first.

---

## 1. The one rule

> Every external dependency the platform talks to sits behind a `Protocol` with at least two implementations: one for local dev, one (or more) for cloud.

External dependency = anything that isn't pure Python computing on already-in-memory data. Object storage. Relational database. Job queue. Auth. Secrets. Notifications. Compute. Cache. Everything.

This rule is what lets the same codebase run on a laptop via `docker compose up` AND on AWS via Terraform without conditional `if AWS:` branches polluting the handler code. The choice of backend is configuration; the handler code never knows.

---

## 2. The seven backend slots

| Slot | Concern | Local impl | Cloud impl(s) | State today |
|---|---|---|---|---|
| **storage** | Files / pickles / artifacts | `LocalFsObjectStore` | `S3ObjectStore` (AWS), `MinioObjectStore` (self-hosted-S3) | ✅ Both shipped |
| **database** | Relational state | `sqlite:///` | `postgresql+psycopg://` (any provider) | ✅ Both shipped (SQLAlchemy abstracts) |
| **secrets** | Encrypt secrets at rest | `FernetInDb` | `SecretsManagerBackend` (AWS), `VaultBackend` (HashiCorp), `ParameterStoreBackend` (AWS SSM) | ⚠ Only Fernet today; Protocol not extracted |
| **auth** | Verify users + issue tokens | `LocalBcryptJwt` | `CognitoBackend`, `OidcBackend` (Google/Okta/Auth0), `SamlBackend` | ⚠ Only local; Protocol not extracted |
| **queue** | Schedule jobs across processes | `APSchedulerInProcess` | `SqsBackend` (AWS), `CelerySqsBackend`, `RedisRqBackend` | ⚠ Only in-proc; Protocol not extracted |
| **compute** | Run a training job somewhere | `ThreadPoolExecutor` (in-process) | `FargateTaskRunner`, `BatchJobRunner`, `K8sJobRunner` | ⚠ Only in-proc; Protocol not extracted |
| **notifier** | Send a webhook / email / Slack message | `WebhookNotifier` (always available) | `SesNotifier` (AWS), `SlackNotifier`, `SmtpNotifier` | ⚠ Only webhooks; Protocol partially extracted |

✅ = implemented + tested + selectable via config
⚠ = either single implementation OR no Protocol abstraction

Promoting an entry from ⚠ to ✅ is the work the next several sessions need to do — see § 6.

---

## 3. The Backends container

A single `Backends` dataclass the BE constructs once at startup and dependency-injects into every handler that needs to talk to something external. Strawman:

```python
# services/api/pycaret_server/backends/__init__.py (to be created)
from dataclasses import dataclass

@dataclass(frozen=True)
class Backends:
    storage: ObjectStore             # Protocol — has LocalFs / S3 / Minio
    secrets: SecretsBackend          # Protocol — has Fernet / SecretsManager / Vault
    auth: AuthBackend                # Protocol — has Local / Cognito / OIDC / SAML
    queue: JobQueue                  # Protocol — has InProc / SQS / Celery
    compute: ComputeBackend          # Protocol — has ThreadPool / Fargate / Batch / K8s
    notifier: Notifier               # Protocol — has Webhook / SES / SMTP / Slack
```

`from_config(settings: Settings) -> Backends` reads `PYCARET_STORAGE_BACKEND`, `PYCARET_SECRETS_BACKEND`, etc. and assembles the right impls. The FastAPI app exposes `Backends` as a dependency:

```python
@router.post("/data-sources/upload")
def upload(
    user: CurrentUser,
    backends: Annotated[Backends, Depends(get_backends)],
    ...
):
    uri = backends.storage.put_file(key, file.file)  # works against local OR S3
```

Handler code never sees `boto3`. Tests inject a `Backends(...)` with in-memory fakes.

---

## 4. Where the gaps actually are

Out of the 7 slots, only **storage** and **database** are fully pluggable today. The other 5 are hardcoded to their local impls. The gap to close looks like:

| Slot | What's missing | LOC estimate |
|---|---|---|
| secrets | Extract `SecretsBackend` Protocol from `crypto.py`; add `SecretsManagerBackend` | ~300 |
| auth | Extract `AuthBackend` Protocol from `auth/`; add Cognito + OIDC impls | ~600 each |
| queue | Extract `JobQueue` Protocol; build `SqsBackend` (AWS) + keep `APSchedulerBackend` | ~500 |
| compute | Split `services/worker/` into its own container; build `FargateRunner` | ~800 |
| notifier | Extract `Notifier` Protocol from webhooks code; add `SmtpNotifier` + `SesNotifier` | ~400 |

Order of attack (in § 6 phasing) is dictated by dependencies: `secrets` first (auth + connections need it), then `queue`/`compute` (separate worker process unlocks horizontal scaling), then `auth` (SSO matters for enterprise pilots), then `notifier` (last — webhooks alone are usable for V1).

---

## 5. The AWS Terraform module (V2 deliverable)

`infra/terraform/aws/` is the canonical "production deploy" path. It provisions:

```
┌─ AWS account ──────────────────────────────────────────────┐
│                                                             │
│  Route53 ── ACM ── ALB ────────┐                            │
│                                ▼                            │
│                  ┌─────────────┴────────────┐               │
│                  │   ECS Fargate cluster    │               │
│                  │  ┌──────┐ ┌──────────┐   │               │
│                  │  │ api  │ │  worker  │   │               │
│                  │  │ x2   │ │   x2     │   │               │
│                  │  └──┬───┘ └────┬─────┘   │               │
│                  └─────┼──────────┼─────────┘               │
│                        │          │                         │
│       ┌────────────────┴──────────┴─────────────┐           │
│       │                                          │           │
│  ┌────▼─────┐  ┌──────────┐  ┌────────────┐  ┌──▼──────┐    │
│  │ RDS PG   │  │ S3 bkt   │  │ SecretsMgr │  │  SQS    │    │
│  │ (state)  │  │artifacts │  │ JWT+Fernet │  │ job q   │    │
│  └──────────┘  └──────────┘  └────────────┘  └─────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Inputs: domain name, instance sizes, env (dev / staging / prod).
Outputs: API URL, command to create the first admin user.

Org's DevOps flow:
```bash
cd infra/terraform/aws
terraform init && terraform apply -var domain=ml.acme.com -var env=prod
# … 8 min …
terraform output  # → https://ml.acme.com, plus IAM bits for SSO setup
```

**Boundary:** Terraform configures infrastructure. The running BE/UI never configures AWS — too dangerous a security boundary (cross-account write privs from a UI? no thanks). For "Connect to AWS from the UI" parity (Snowflake-style) we use IAM role assumption at the worker layer, never raw access keys in the app DB.

---

## 6. Phasing — local-first, cloud-second, multi-cloud-third

| Phase | Codename | Scope | Status |
|---|---|---|---|
| **0** | Local-only, single-process | What we have today. SQLite, filesystem, in-process worker, Fernet-in-DB. `docker compose up` works. | ✅ Shipped session 57 |
| **1** | Pluggable backends | Extract 5 missing Protocols. Local impls still default. Tests still pass. Zero behaviour change in default config. | 🟡 Next |
| **2** | AWS pack | Implement `S3` (✅), `SecretsManager`, `SqsBackend`, `FargateRunner`, `SesNotifier`. Ship `infra/terraform/aws/`. Document the prod-deploy flow. | 🔴 |
| **3** | Worker/runtime split | Split `services/api` and `services/worker` and `services/deployment-runtime` into separate processes/containers. API becomes stateless HTTP. Workers pull from queue. Deployment-runtime serves predictions independently. Enables horizontal scaling. | 🔴 |
| **4** | Auth pack | Cognito + OIDC + SAML implementations. SSO becomes a config choice, not a fork. | 🔴 |
| **5** | Multi-cloud | GCP pack (Cloud SQL, GCS, Cloud Run, Pub/Sub), Azure pack (Azure SQL, Blob, Container Apps, Service Bus). | 🔴 Far future |

Each phase is publishable independently. Phase 0 is publishable RIGHT NOW. Phase 1 doesn't change user-facing behaviour. Phase 2 unlocks the "one click AWS" story. Phase 3 unlocks scale.

---

## 7. What we explicitly are NOT building

- **A SaaS control plane that provisions customer AWS accounts.** That's what Databricks/Snowflake are. We're self-hosted. Customer brings their own AWS account; Terraform is the boundary.
- **A drag-drop pipeline builder.** PyCaret is config-driven (`RunConfig`). The UI is a wizard over the config, not a graph editor.
- **A general-purpose orchestrator.** No DAGs, no upstream/downstream tasks, no Airflow replacement. Each Run is atomic; Schedules just re-run them. If you need DAGs, plug us into Airflow/Prefect/Dagster as a step.
- **A feature store.** The data layer is "DataSource → Dataset (versioned snapshot)" and that's it. Feature stores are a category and a hard problem; deferred indefinitely.
- **Streaming inference.** Predict is request/response. Streaming is a future deployment-runtime mode (Phase 3+).

The "no" list is part of the architecture. Saying yes to all of these would dilute the core value prop: end-to-end Train → Register → Deploy → Predict, opinionated, self-hosted, AI-advised.

---

## 8. Reading guide for new contributors

1. Read this file (you just did).
2. [`VISION.md`](VISION.md) — 1-page product statement (the *why*).
3. [`CONTROL_PLANE_SPEC.md`](CONTROL_PLANE_SPEC.md) — full feature spec.
4. [`ARCHITECTURE.md`](ARCHITECTURE.md) — current code map (the *what's actually built*).
5. [`ROADMAP.md`](ROADMAP.md) — phase breakdown of what's next.
6. [`DECISIONS.md`](DECISIONS.md) — ADRs for non-obvious choices.

If you're about to add a new external dependency to a handler, **stop and check § 2-3 of this doc.** Add a Protocol slot; ship both impls; wire through the `Backends` container. We've earned this discipline by feeling the pain of every shortcut we took.
