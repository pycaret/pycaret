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
    ApiKey,
    Artifact,
    AuditLog,
    DataSource,
    Deployment,
    DriftReport,
    Event,
    Experiment,
    FoldMetric,
    LLMConsultation,
    LLMProviderSetting,
    Pipeline,
    PipelineProjectLink,
    Project,
    Run,
    Session,
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
    "ApiKey",
    "Artifact",
    "AuditLog",
    "Base",
    "DataSource",
    "Deployment",
    "DriftReport",
    "Event",
    "Experiment",
    "FoldMetric",
    "LLMConsultation",
    "LLMProviderSetting",
    "Pipeline",
    "PipelineProjectLink",
    "Project",
    "Run",
    "Session",
    "TimestampMixin",
    "UUIDMixin",
    "User",
    "Workspace",
    "WorkspaceMember",
    "engine",
    "get_db",
    "get_session",
    "session_factory",
]
