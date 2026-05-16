"""SQLAlchemy data model for the PyCaret platform.

Hierarchy (per docs/revamp/PLATFORM_PLAN.md § 3):

    Workspace
    ├── members (User × role)
    ├── pipelines (shareable across projects)
    ├── deployments (in-house serving)
    └── projects
        ├── data_sources (CSV upload / S3 / Postgres)
        └── experiments
            └── runs
                ├── events (append-only)
                ├── artifacts (run.ipynb, fitted_pipeline.pkl, etc.)
                └── fold_metrics (per-fold × per-model × per-metric)

14 tables total. Every row has `id: UUID`, `created_at`, `updated_at`; most
have `created_by` (FK to users.id).
"""

from pycaret_server.db.base import Base, TimestampMixin, UUIDMixin
from pycaret_server.db.models import (
    AlertRule,
    Analysis,
    ApiKey,
    ApprovalWorkflow,
    Artifact,
    AuditLog,
    Connection,
    DataSource,
    Dataset,
    Deployment,
    DriftReport,
    Event,
    Experiment,
    ExperimentTemplate,
    FoldMetric,
    GitRepository,
    Job,
    Lineage,
    LLMConsultation,
    LLMProviderSetting,
    MetricPoint,
    ModelLibrary,
    Notebook,
    NotebookSession,
    Pipeline,
    PipelineProjectLink,
    PredictionLog,
    Project,
    RegisteredModel,
    RegisteredModelVersion,
    Run,
    ScheduledJob,
    Secret,
    Session,
    Trial,
    WebhookSubscription,
    User,
    Workspace,
    WorkspaceMember,
)
from pycaret_server.db.session import (
    engine,
    get_db,
    get_session,
    session_factory,
)

__all__ = [
    "AlertRule",
    "Analysis",
    "ApiKey",
    "ApprovalWorkflow",
    "Artifact",
    "AuditLog",
    "Base",
    "Connection",
    "DataSource",
    "Dataset",
    "Deployment",
    "DriftReport",
    "Event",
    "Experiment",
    "ExperimentTemplate",
    "FoldMetric",
    "GitRepository",
    "Job",
    "Lineage",
    "LLMConsultation",
    "LLMProviderSetting",
    "MetricPoint",
    "ModelLibrary",
    "Notebook",
    "NotebookSession",
    "Pipeline",
    "PipelineProjectLink",
    "PredictionLog",
    "Project",
    "RegisteredModel",
    "RegisteredModelVersion",
    "Run",
    "ScheduledJob",
    "Secret",
    "Session",
    "TimestampMixin",
    "Trial",
    "UUIDMixin",
    "User",
    "WebhookSubscription",
    "Workspace",
    "WorkspaceMember",
    "engine",
    "get_db",
    "get_session",
    "session_factory",
]
