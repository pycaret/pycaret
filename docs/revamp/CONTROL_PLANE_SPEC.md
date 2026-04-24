# PyCaret Control Plane — Technical Specification & Product Vision

> **Status:** canonical product spec as of session 13 (2026-04-24). Supersedes the earlier `PLATFORM_PLAN.md`.
> **Source:** owner-authored; checked in verbatim with minor markdown formatting so tables / code blocks render.

---

## 1. Product Vision

**PyCaret Control Plane** is a self-hosted, enterprise-ready, AI-native machine learning platform built around the PyCaret 4 engine.

The platform allows organizations to:

- manage ML workspaces
- create projects
- upload / connect datasets
- configure experiments
- run AutoML and manual ML pipelines
- compare model runs
- save artifacts
- deploy full pipelines as services
- monitor prediction endpoints
- detect drift
- consult LLM agents for experiment design
- manage users, settings, compute, and integrations

The system should work in three modes:

1. **Local Desktop Mode** — Electron app + local backend + SQLite.
2. **Single-Server Self-Hosted Mode** — Docker Compose with frontend, backend, worker, database, and object storage.
3. **Enterprise Cloud Mode** — Deployable to AWS, GCP, or Azure using Terraform / Kubernetes / Helm.

The long-term goal:

> PyCaret becomes an open-source, self-managed ML control plane for tabular machine learning, combining AutoML, MLOps, deployment, monitoring, and AI-assisted experimentation.

---

## 2. Core Product Philosophy

### 2.1 PyCaret 4 is not just a library

PyCaret 4 has two layers:

```
PyCaret Engine
  - pip installable
  - notebook / script usable
  - stateless
  - config-driven
  - produces artifacts

PyCaret Control Plane
  - web application
  - API backend
  - database
  - workers
  - deployment runtime
  - monitoring
```

### 2.2 The main object is not a model

The main object is a **pipeline artifact**.

A pipeline artifact contains:

- preprocessing
- feature engineering
- trained estimator
- schema
- metrics
- config snapshot
- logs
- plots
- environment metadata
- dependency lock info

### 2.3 Deploy pipelines, not models

Production inference must use the exact same preprocessing pipeline used during training.

The deployment unit is:

> Dataset schema + preprocessing + model + prediction interface

### 2.4 Every action should be reproducible

Every run should store:

- input dataset reference
- target column
- train/test split settings
- preprocessing config
- model search space
- tuning config
- random seed
- package versions
- metrics
- artifacts
- logs

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
React Web App / Electron App
        |
        v
API Backend - FastAPI
        |
        +--> PyCaret Engine
        |
        +--> Job Queue / Worker
        |
        +--> Database
        |
        +--> Artifact Storage
        |
        +--> Deployment Runtime
        |
        +--> LLM Provider Gateway
```

### 3.2 Main Services

**Frontend**

- React + TypeScript
- Vite or Next.js
- Tailwind / shadcn/ui
- TanStack Query
- Zustand or Redux Toolkit
- React Flow (later, for visual pipeline builder)

**Backend**

- Python
- FastAPI
- Pydantic
- SQLAlchemy or SQLModel
- Alembic migrations
- Celery / RQ / Arq for workers
- Redis optional for queue / cache
- Uvicorn / Gunicorn

**Database**

- Local: SQLite
- Self-hosted / enterprise: PostgreSQL

**Artifact Storage**

- Local: filesystem
- Self-hosted: MinIO
- Cloud: S3 / GCS / Azure Blob Storage

**Job Execution**

- Local: in-process worker / background task
- Self-hosted: worker container
- Enterprise: Kubernetes jobs; Ray / Dask optional later

**Deployment Runtime**

Deployment can run as:

- local FastAPI route
- separate process
- Docker container
- Kubernetes service
- serverless function (later)

---

## 4. Main Domain Model

### 4.1 Workspace

Top-level organization container.

```
id
name
slug
description
owner_user_id
default_storage_backend
default_compute_backend
created_at
updated_at
deleted_at
```

### 4.2 User

```
id
email
name
role
status
avatar_url
last_login_at
created_at
updated_at
```

Roles: `owner`, `admin`, `ml_engineer`, `data_scientist`, `viewer`, `service_account`.

### 4.3 Project

```
id
workspace_id
name
slug
description
visibility
created_by
created_at
updated_at
archived_at
```

### 4.4 Dataset

```
id
workspace_id
project_id
name
source_type
source_uri
storage_uri
schema_json
row_count
column_count
target_column
task_type_guess
profile_json
created_by
created_at
updated_at
```

Source types: `upload_csv`, `local_file`, `database_table`, `s3`, `gcs`, `azure_blob`, `snowflake`, `bigquery`, `redshift`, `postgres`, `mysql`, `api`.

### 4.5 Experiment

```
id
workspace_id
project_id
dataset_id
name
description
task_type
target_column
objective_metric
created_by
created_at
updated_at
```

Task types: `classification`, `regression`, `time_series`, `clustering`, `anomaly_detection`, `nlp_later`.

### 4.6 Run

```
id
workspace_id
project_id
experiment_id
dataset_id
name
status
mode
config_json
resolved_config_json
started_at
finished_at
duration_seconds
best_artifact_id
created_by
created_at
updated_at
```

Statuses: `draft`, `queued`, `running`, `completed`, `failed`, `cancelled`.
Modes: `manual`, `assisted`, `automl`, `llm_generated`, `scheduled`, `retraining`.

### 4.7 Trial

A single candidate pipeline/model evaluated inside a run.

```
id
run_id
model_key
pipeline_config_json
hyperparameters_json
metrics_json
rank
status
started_at
finished_at
artifact_id
```

### 4.8 Artifact

Artifact types: `pipeline`, `model`, `plot`, `leaderboard`, `metrics`, `log`, `dataset_profile`, `explanation`, `report`, `deployment_bundle`.

```
id
workspace_id
project_id
experiment_id
run_id
type
name
uri
metadata_json
created_by
created_at
```

### 4.9 Pipeline Artifact

A special artifact representing the deployable trained pipeline. Contains:

```
pipeline.pkl
schema.json
config.json
metrics.json
requirements.txt
environment.json
signature.json
README.md
```

### 4.10 Deployment

```
id
workspace_id
project_id
artifact_id
name
status
deployment_type
endpoint_url
runtime_config_json
replicas
created_by
created_at
updated_at
stopped_at
```

Deployment types: `local`, `docker`, `kubernetes`, `aws_ecs`, `aws_lambda`, `gcp_cloud_run`, `azure_container_apps`.

### 4.11 Prediction Log

```
id
deployment_id
request_id
input_hash
input_schema_valid
prediction_json
latency_ms
status_code
error_message
created_at
```

### 4.12 Drift Report

```
id
deployment_id
baseline_artifact_id
window_start
window_end
drift_score
drift_status
feature_drift_json
prediction_drift_json
created_at
```

### 4.13 Model Library

```
id
model_key
display_name
task_types
library
enabled
default_hyperparameters_json
search_space_json
tags
created_at
updated_at
```

### 4.14 LLM Provider Setting

```
id
workspace_id
provider
api_key_encrypted
base_url
model_name
enabled
created_by
created_at
updated_at
```

Providers: `openai`, `anthropic`, `google`, `azure_openai`, `ollama`, `custom_openai_compatible`.

### 4.15 LLM Consultation

```
id
workspace_id
project_id
experiment_id
run_id
type
prompt
response_json
generated_config_json
created_by
created_at
```

Types: `dataset_analysis`, `experiment_design`, `metric_selection`, `pipeline_recommendation`, `run_summary`, `failure_debugging`, `deployment_risk_review`, `drift_analysis`.

---

## 5. Core Functional Modules

(see spec source for full module-by-module feature + page breakdown)

- **5.1 Workspace Management** — create / update / archive, members, integrations, billing, security, artifact retention.
- **5.2 User & Admin Management** — first-admin setup, invite, deactivate, roles, reset password, service accounts, API keys, audit logs.
- **5.3 Project Management** — dashboard, members, datasets, experiments, deployments, settings, archive.
- **5.4 Dataset Management** — upload, connect DB, profile, preview, schema infer, target detect, quality, versions.
- **5.5 Experiment Management** — create, clone, compare runs, task-type + target + objective metric, validation strategy.

---

## 6. Run Configuration System

The most important engineering component. Every UI interaction generates a valid RunConfig.

### 6.1 RunConfig Schema

```json
{
  "dataset": {
    "dataset_id": "ds_123",
    "target": "churn",
    "train_size": 0.8,
    "random_state": 42
  },
  "task": {
    "type": "classification",
    "positive_class": "yes",
    "objective_metric": "recall"
  },
  "preprocessing": {
    "missing_values": {
      "enabled": true,
      "numeric_strategy": "median",
      "categorical_strategy": "most_frequent"
    },
    "encoding": { "enabled": true, "method": "one_hot", "handle_unknown": "ignore" },
    "scaling": { "enabled": false, "method": "standard" },
    "transformation": { "enabled": false, "method": "yeo_johnson" },
    "feature_selection": { "enabled": false, "method": "mutual_info", "max_features": null },
    "class_imbalance": { "enabled": false, "method": "smote" }
  },
  "model_selection": {
    "mode": "include",
    "include": ["logistic_regression", "random_forest", "xgboost"],
    "exclude": []
  },
  "evaluation": {
    "cv_strategy": "stratified_kfold",
    "folds": 10,
    "metrics": ["accuracy", "auc", "recall", "precision", "f1"],
    "primary_metric": "auc"
  },
  "automl": {
    "enabled": true,
    "max_trials": 50,
    "max_runtime_minutes": 60,
    "parallelism": 4,
    "early_stopping": true
  },
  "tuning": { "enabled": false, "backend": "optuna", "max_trials": 30 },
  "explainability": { "enabled": true, "feature_importance": true, "shap": false }
}
```

### 6.2 Configuration Modes

- **Manual** — user explicitly selects everything.
- **Assisted** — LLM suggests a config, user approves.
- **Auto** — system searches pipeline space under budget.
- **Expert** — raw JSON / YAML editor.

---

## 7. AutoML System

### 7.1 Scope

Search includes: imputation, encoding, scaling, transformations, feature selection, class imbalance, model family, hyperparameters, threshold tuning, probability calibration.

### 7.2 Budget Controls

```
number of trials
max runtime
parallelism
max memory
preferred metric
latency constraint
interpretability constraint
model allowlist / blocklist
```

### 7.3 Output

- ranked leaderboard
- best pipeline artifact
- full trial history
- importance of choices
- failed trials
- suggested next experiments

---

## 8. Model Library

Admin can enable/disable models, set default hyperparameters, set search spaces, mark as experimental / production-approved, define dependency requirements.

Each entry has: `key`, `name`, `task_type`, `library`, `supports_probability`, `supports_feature_importance`, `supports_gpu`, `supports_sparse_input`, `default_params`, `search_space`, `dependency`, `status`.

---

## 9. Artifact System

### 9.1 Layout

```
artifacts/
  workspace_id/
    project_id/
      experiment_id/
        run_id/
          config.json
          resolved_config.json
          leaderboard.json
          metrics.json
          logs.txt
          plots/
            confusion_matrix.png
            roc_curve.png
            feature_importance.png
          pipelines/
            trial_001.pkl
            trial_002.pkl
            best_pipeline.pkl
          reports/
            run_summary.md
            model_card.md
```

### 9.2 Features

list · download · compare · promote · delete · pin · tag · export · deploy

---

## 10. Deployment System

### 10.1 Types

- **Local** — inside backend process (desktop / demo).
- **Docker** — standalone container (self-hosted / staging).
- **Kubernetes** — deployment + service + ingress + secret + configmap.
- **Cloud** — one-click for AWS ECS/Fargate/Lambda, GCP Cloud Run, Azure Container Apps.

### 10.2 Features

create · stop · restart · rollback · scale · test prediction · view request logs · view latency / error rate / drift · configure monitoring / input validation / endpoint auth.

### 10.3 Runtime Contract

Each deployed pipeline exposes:

```
GET /health
GET /schema
POST /predict
POST /predict_batch
GET /metadata
```

---

## 11. Monitoring & Drift

- **11.1 Deployment monitoring:** rpm, latency p50/p95/p99, error rate, prediction distribution, input schema failures, missing feature frequency, uptime.
- **11.2 Drift monitoring:** feature drift, prediction drift, missing-value drift, schema drift, target drift (when labels arrive), performance decay.
- **11.3 Alerts:** `schema_drift`, `feature_drift`, `prediction_drift`, `latency_spike`, `error_rate_spike`, `deployment_down`, `performance_drop`. Channels: UI, email, Slack, webhook.

---

## 12. LLM / AI Assistant System

### 12.1 Settings

Workspace admin configures: provider, API key, model, base URL, max tokens, temperature, usage limits, allowed features.

### 12.2 Features

- **Dataset Consultant** — analyzes dataset profile; suggests task type, target risks, leakage risks, preprocessing strategy, metric.
- **Experiment Designer** — generates RunConfig.
- **Run Explainer** — why best model won, metric tradeoffs, suspicious behavior, recommended next runs.
- **Failure Debugger** — explains failed-run logs.
- **Deployment Reviewer** — checks artifact safety.
- **Drift Analyst** — explains drift reports, suggests retraining.

### 12.3 Constraint

> LLM should not directly execute dangerous actions.

It proposes `suggested_config_json`, `suggested_action`, `reasoning_summary`, `risk_flags`. User approves.

---

## 13. UI / Navigation Specification

### 13.1 Sidebar

```
Home
Workspaces
Projects
Datasets
Experiments
Runs
Artifacts
Deployments
Monitoring
Model Library
AI Assistant
Integrations
Settings
Admin
```

### 13.2–13.6 — detailed screen specs (see original spec for full widget lists per page)

---

## 14. API Surface

The final system may easily reach **300+ endpoints**, organised by module:

- **14.1 Auth / Setup** — `/setup/*`, `/auth/*`, API keys.
- **14.2 Workspaces** — CRUD + members + settings.
- **14.3 Users** — CRUD + preferences.
- **14.4 Projects** — CRUD + overview + activity + settings.
- **14.5 Datasets** — upload + connect + preview + schema + profile + quality + validate.
- **14.6 Experiments** — CRUD + runs + leaderboard + artifacts + clone.
- **14.7 Runs** — CRUD + start / cancel / clone / retry + status / logs / metrics / leaderboard / plots / config / resolved-config / artifacts.
- **14.8 Trials** — per-run trial list + details.
- **14.9 Artifacts** — CRUD + download / promote / tag / export / deploy + metadata + schema.
- **14.10 Model Library** — CRUD + search-space + enable / disable.
- **14.11 Deployments** — CRUD + start / stop / restart / rollback / scale + health / schema / predict / predict-batch / logs / metrics / requests / drift.
- **14.12 Monitoring** — overview + deployments + alerts + drift runs + reports.
- **14.13 LLM** — settings + test-connection + 6 advisory endpoints + consultations history.
- **14.14 Integrations** — CRUD for database / object_storage / llm_provider / notification / compute / git / container_registry.
- **14.15 Admin / System** — health / info / storage / jobs / workers / audit-logs / settings / backup / restore / migrations.

(See original spec for every endpoint under each module.)

---

## 15. Backend Internal Services

### 15.1 Core Services

WorkspaceService · UserService · ProjectService · DatasetService · ExperimentService · RunService · TrialService · ArtifactService · DeploymentService · MonitoringService · DriftService · LLMService · ModelLibraryService · IntegrationService · AuditLogService · SettingsService

### 15.2 Engine Services

PipelineBuilder · PreprocessingBuilder · ModelRegistry · Evaluator · AutoMLRunner · TuningRunner · PlotGenerator · ArtifactWriter · DeploymentPackager · PredictionService

---

## 16. Background Jobs

Types: `dataset_profile_job`, `run_training_job`, `automl_trial_job`, `tuning_job`, `plot_generation_job`, `artifact_packaging_job`, `deployment_start_job`, `deployment_stop_job`, `drift_detection_job`, `report_generation_job`, `llm_consultation_job`, `cleanup_job`, `backup_job`.

Fields: `id`, `type`, `status`, `payload_json`, `result_json`, `error_message`, `created_at`, `started_at`, `finished_at`, `created_by`.

---

## 17. Security & Enterprise Readiness

### 17.1 Authentication

MVP: local admin, email/password, API keys. Later: SSO, SAML, OAuth, LDAP.

### 17.2 Authorization

Roles: `owner`, `admin`, `project_admin`, `ml_engineer`, `data_scientist`, `viewer`, `service_account`.

Permissions: `workspace.manage`, `users.manage`, `projects.create`, `projects.read`, `datasets.upload`, `experiments.create`, `runs.execute`, `artifacts.deploy`, `deployments.manage`, `admin.access`.

### 17.3 Secrets

Encrypted storage for: LLM keys, cloud credentials, database passwords, registry credentials.

### 17.4 Audit Logs

Track: user login, dataset upload, run start/cancel, artifact deletion, deployment creation, settings changes, LLM key changes, user role changes.

---

## 18. Deployment Strategy

- **18.1 Local Developer** — `make dev`: frontend + backend + worker + SQLite + local artifact storage.
- **18.2 Docker Compose** — `docker compose up`: frontend + api + worker + postgres + redis + minio + deployment-runner.
- **18.3 Electron Desktop** — React UI + local FastAPI + SQLite + local artifact directory.
- **18.4 Enterprise Kubernetes** — Helm chart + manifests + Terraform + container images.
- **18.5 Cloud One-Click** — `infra/aws`, `infra/gcp`, `infra/azure` Terraform modules (see infra/README.md).

---

## 19. Repository Structure

```
pycaret-platform/
├── apps/
│   ├── web/
│   └── desktop/
├── services/
│   ├── api/
│   ├── worker/
│   └── deployment-runtime/
├── packages/
│   ├── engine/
│   ├── sdk-python/
│   └── shared-schemas/
├── infra/
│   ├── docker/
│   ├── helm/
│   └── terraform/
│       ├── aws/
│       ├── gcp/
│       └── azure/
├── docs/
│   ├── architecture/
│   ├── api/
│   ├── deployment/
│   ├── user-guide/
│   └── admin-guide/
├── scripts/
└── tests/
```

(Implemented as of session 13. See `docs/revamp/ARCHITECTURE.md` for the live mapping.)

---

## 20. MVP Scope

- **MVP 1: Engine** — config-driven supervised ML runs, classification/regression, preprocessing, comparison, leaderboard, metrics, plots, artifact writing, local prediction from artifact.
- **MVP 2: Backend** — workspace / project / dataset / experiment / run / artifact / deployment / prediction logs.
- **MVP 3: UI** — login/setup, project dashboard, dataset upload, new-experiment wizard, run details, leaderboard, plots, artifact deploy button, deployment test page.
- **MVP 4: Self-hosted** — Docker Compose, Postgres, local artifact or MinIO, one-command startup.

---

## 21. V2 Scope

user roles · audit logs · model library UI · LLM assistant · AutoML full pipeline search · drift monitoring · deployment rollback · cloud deployment templates · API keys · backup/restore · scheduled retraining.

---

## 22. V3 Scope

SSO/SAML · Kubernetes-native execution · distributed AutoML · approval workflows · model cards · governance reports · multi-environment deployments · feature store integrations · advanced monitoring · plugin system · marketplace for models / preprocessors.

---

## 23. Critical Engineering Principles

- **23.1 Engine is stateless.** `result = engine.run(config)`, not `setup() + compare_models()`.
- **23.2 Config is the contract.** Same config powers notebook, API, UI, reproducibility, LLM-generated experiments.
- **23.3 Backend owns persistence.** Engine doesn't know about users / workspaces / databases.
- **23.4 Artifacts are immutable.** A completed artifact is never mutated — create a new one.
- **23.5 Deployment is versioned.** Every deployment points to a specific artifact version.
- **23.6 LLM is advisory.** LLM proposes configs + explanations; the deterministic engine executes.

---

## 24. Final Product Statement

> A self-hosted, enterprise-ready, AI-native machine learning platform for building, comparing, deploying, and monitoring production ML pipelines.

Differentiators:

- simple local-first experience
- pip-installable engine
- full SaaS-style control plane
- self-managed enterprise deployment
- AutoML over complete pipelines
- AI-assisted experiment design
- deployment and monitoring built in
- no dependency on MLflow or closed enterprise tools

**The first beautiful product loop:**

```
Create project
Upload dataset
Create experiment
Run AutoML
Compare pipelines
Save artifact
Deploy endpoint
Monitor predictions
Ask AI what to improve next
```

That is the core vision.
