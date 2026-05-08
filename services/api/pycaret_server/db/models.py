"""The 14-table data model for the PyCaret platform.

See docs/revamp/PLATFORM_PLAN.md § 3 for the design. Every table has a UUID
v4 string `id`, `created_at`, and `updated_at`. Most rows also carry
`created_by` (FK to users.id) — added explicitly (not via mixin) so the FK
target is resolvable at mapper-configuration time.

Tables (v1):

- users                     local user store
- workspaces                top-level containers
- workspace_members         user × workspace × role
- data_sources              CSV upload / S3 / Postgres connection
- projects                  inside a workspace
- experiments               configured Experiment
- runs                      one invocation of an experiment
- events                    append-only engine event stream per run
- artifacts                 run outputs (pickle / notebook / leaderboard)
- fold_metrics              per-fold × per-model × per-metric
- pipelines                 workspace-scoped fitted sklearn Pipeline registry
- pipeline_project_links    many-to-many: pipelines ↔ projects
- deployments               in-house serving record
- api_keys                  programmatic-access tokens
- sessions                  refresh-token storage for auth
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pycaret_server.db.base import Base, TimestampMixin, UUIDMixin

# -----------------------------------------------------------------------------
# users / auth
# -----------------------------------------------------------------------------


class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(128))
    password_hash: Mapped[str | None] = mapped_column(String(256))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    memberships: Mapped[list[WorkspaceMember]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    sessions: Mapped[list[Session]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Session(UUIDMixin, TimestampMixin, Base):
    """Refresh-token session. `refresh_token_hash` stored; plaintext held by client."""

    __tablename__ = "sessions"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    refresh_token_hash: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str | None] = mapped_column(String(256))
    ip_address: Mapped[str | None] = mapped_column(String(64))

    user: Mapped[User] = relationship(back_populates="sessions")


class ApiKey(UUIDMixin, TimestampMixin, Base):
    """Programmatic-access tokens: for CI / scripts / deployments."""

    __tablename__ = "api_keys"

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    # hashed token; plaintext returned once on creation, never stored
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    prefix: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True
    )  # first N chars for display
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    workspace_id: Mapped[str | None] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scopes: Mapped[list[str] | None] = mapped_column(JSON)


# -----------------------------------------------------------------------------
# workspace / project / experiment
# -----------------------------------------------------------------------------


class Workspace(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "workspaces"

    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    config: Mapped[dict | None] = mapped_column(JSON)  # theme, default compute, etc.

    members: Mapped[list[WorkspaceMember]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    projects: Mapped[list[Project]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    pipelines: Mapped[list[Pipeline]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    deployments: Mapped[list[Deployment]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    data_sources: Mapped[list[DataSource]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )


class WorkspaceMember(UUIDMixin, TimestampMixin, Base):
    """User × Workspace × role. Role is 'admin' or 'member' in v1."""

    __tablename__ = "workspace_members"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id", name="uq_member"),)

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="member")

    workspace: Mapped[Workspace] = relationship(back_populates="members")
    user: Mapped[User] = relationship(back_populates="memberships")


class DataSource(UUIDMixin, TimestampMixin, Base):
    """Registered CSV / S3 / Postgres source a Project can point at.

    Type-specific config goes in `config` (JSON). E.g. for S3:
    ``{"bucket": "…", "key": "…", "region": "us-east-1"}``.
    """

    __tablename__ = "data_sources"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_datasource_name"),)

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # csv_upload | s3 | postgres
    config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    workspace: Mapped[Workspace] = relationship(back_populates="data_sources")


class Project(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_project_name"),)

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str] | None] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    workspace: Mapped[Workspace] = relationship(back_populates="projects")
    experiments: Mapped[list[Experiment]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    pipeline_links: Mapped[list[PipelineProjectLink]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Experiment(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "experiments"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_experiment_name"),)

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)  # TaskType.value
    target: Mapped[str | None] = mapped_column(String(128))
    setup_params: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    data_source_id: Mapped[str | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL"), index=True
    )
    description: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="experiments")
    runs: Mapped[list[Run]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


# -----------------------------------------------------------------------------
# run / events / artifacts / fold_metrics
# -----------------------------------------------------------------------------


class Run(UUIDMixin, TimestampMixin, Base):
    """One invocation of an experiment — captures status + timings."""

    __tablename__ = "runs"

    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), default="queued", nullable=False, index=True
    )  # queued | running | succeeded | failed | cancelled
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    leaderboard: Mapped[dict | None] = mapped_column(JSON)  # CompareResult.leaderboard shape
    metrics_summary: Mapped[dict | None] = mapped_column(JSON)  # aggregated leaderboard
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    # Inputs captured at dispatch time (immutable snapshot of experiment config)
    snapshot: Mapped[dict | None] = mapped_column(JSON)

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    events: Mapped[list[Event]] = relationship(back_populates="run", cascade="all, delete-orphan")
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    fold_metrics: Mapped[list[FoldMetric]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    produced_pipelines: Mapped[list[Pipeline]] = relationship(back_populates="origin_run")


class Event(UUIDMixin, TimestampMixin, Base):
    """Append-only engine Event captured per Run.

    Mirrors ``pycaret.logging.events.Event`` — kind + message + payload + duration.
    """

    __tablename__ = "events"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict | None] = mapped_column(JSON)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    # Event timestamp from the engine (may precede the row's created_at by fractions)
    emitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    run: Mapped[Run] = relationship(back_populates="events")


class Artifact(UUIDMixin, TimestampMixin, Base):
    """Run output file (pickle / notebook / html preview / json leaderboard)."""

    __tablename__ = "artifacts"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    # pipeline_pickle | notebook_ipynb | notebook_html | leaderboard_json | events_jsonl | plot_png
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    content_type: Mapped[str | None] = mapped_column(String(128))

    run: Mapped[Run] = relationship(back_populates="artifacts")


class FoldMetric(Base):
    """Per-fold × per-model × per-metric value. Composite primary key."""

    __tablename__ = "fold_metrics"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    fold_idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped[Run] = relationship(back_populates="fold_metrics")


# -----------------------------------------------------------------------------
# pipelines + cross-project links + deployments
# -----------------------------------------------------------------------------


class Pipeline(UUIDMixin, TimestampMixin, Base):
    """Workspace-scoped fitted sklearn Pipeline registry entry.

    Shareable across projects via `pipeline_project_links`.
    """

    __tablename__ = "pipelines"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str] | None] = mapped_column(JSON)
    model_id: Mapped[str | None] = mapped_column(String(64))  # pycaret id: "lr", "rf", ...
    origin_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), index=True
    )
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    params: Mapped[dict | None] = mapped_column(JSON)  # estimator.get_params(deep=False)
    # Versioning lineage (Spec § 4.7): pipelines that share a ``family_id``
    # are revisions of the same logical model. ``version`` increments within
    # a family. The first promotion creates a new family; subsequent
    # promotions of the same name within a workspace bump the version and
    # reuse the family_id so deployment rollback is just "select an earlier
    # row in the same family".
    family_id: Mapped[str | None] = mapped_column(String(36), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    workspace: Mapped[Workspace] = relationship(back_populates="pipelines")
    origin_run: Mapped[Run | None] = relationship(back_populates="produced_pipelines")
    project_links: Mapped[list[PipelineProjectLink]] = relationship(
        back_populates="pipeline", cascade="all, delete-orphan"
    )
    deployments: Mapped[list[Deployment]] = relationship(back_populates="pipeline")


class PipelineProjectLink(Base):
    """Many-to-many: a Pipeline can be used by multiple Projects."""

    __tablename__ = "pipeline_project_links"

    pipeline_id: Mapped[str] = mapped_column(
        ForeignKey("pipelines.id", ondelete="CASCADE"), primary_key=True
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )

    pipeline: Mapped[Pipeline] = relationship(back_populates="project_links")
    project: Mapped[Project] = relationship(back_populates="pipeline_links")


class Deployment(UUIDMixin, TimestampMixin, Base):
    """In-house serving record for a fitted Pipeline.

    See PLATFORM_PLAN § decision 4. Each active deployment is loaded into
    `DeploymentRegistry` at server boot; `POST /api/v1/deployments/{slug}/predict`
    dispatches by slug.
    """

    __tablename__ = "deployments"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_id: Mapped[str] = mapped_column(
        ForeignKey("pipelines.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    endpoint_slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(16), default="active", nullable=False, index=True
    )  # active | paused | archived
    auth_mode: Mapped[str] = mapped_column(
        String(16), default="workspace", nullable=False
    )  # workspace | api-key | public
    inference_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_inference_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    p50_latency_ms: Mapped[float | None] = mapped_column(Float)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float)
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    workspace: Mapped[Workspace] = relationship(back_populates="deployments")
    pipeline: Mapped[Pipeline] = relationship(back_populates="deployments")


# -----------------------------------------------------------------------------
# LLM provider settings + advisory consultations
# -----------------------------------------------------------------------------


class LLMProviderSetting(UUIDMixin, TimestampMixin, Base):
    """Per-workspace LLM provider configuration.

    Unique on (workspace_id, provider) so a workspace can have one active
    Anthropic entry + one OpenAI entry side-by-side; the `enabled` flag picks
    which one actually runs. ``api_key_encrypted`` is stored raw for v1 — the
    spec (§ 17.3) requires KMS/Vault wrapping before V2 SSO ships, tracked as
    a roadmap item.
    """

    __tablename__ = "llm_provider_settings"
    __table_args__ = (UniqueConstraint("workspace_id", "provider", name="uq_llm_provider"),)

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # anthropic | openai | google | azure_openai | ollama | custom_openai_compatible
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    api_key_encrypted: Mapped[str | None] = mapped_column(Text)  # TODO: KMS wrap (V2)
    base_url: Mapped[str | None] = mapped_column(String(512))
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    config: Mapped[dict | None] = mapped_column(JSON)  # max_tokens, temperature, …
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class LLMConsultation(UUIDMixin, TimestampMixin, Base):
    """Append-only audit record of every LLM advisory request.

    Every consultation produces the same envelope shape: ``suggested_config_json``,
    ``suggested_action``, ``reasoning_summary``, ``risk_flags``. This lets the
    UI + server treat Claude / OpenAI output uniformly — the `response_json`
    column stores that envelope as-is.
    """

    __tablename__ = "llm_consultations"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    # dataset_analysis | experiment_design | metric_selection | …
    type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    # Extracted from response_json for convenience / indexing.
    generated_config_json: Mapped[dict | None] = mapped_column(JSON)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


# -----------------------------------------------------------------------------
# drift reports + audit logs (session 21)
# -----------------------------------------------------------------------------


class DriftReport(UUIDMixin, TimestampMixin, Base):
    """Snapshot of distribution drift for a Deployment over a time window.

    Spec § 4.12. In v1 there's no scheduled job yet (needs the Job queue that
    lands post-4.0.0); drift reports are created by explicit POST — either
    from the UI or from CI/cron hitting the API with a ``X-PyCaret-Key``.
    The row stores:

    - ``drift_score`` — overall 0..1 PSI-weighted score (higher = more drift).
    - ``drift_status`` — bucketed label ``none | mild | moderate | severe``.
    - ``feature_drift_json`` — per-feature drift values + kind
      (``{feature: {score, kind}}``, kind ∈ ``psi | ks | chi2 | missing_rate``).
    - ``prediction_drift_json`` — prediction distribution shift
      (``{kind, score, ...details}`` — e.g. JS-divergence on prediction histogram).

    The LLM ``drift_analysis`` consultation reads these rows + suggests
    ``RETRAIN NOW`` / ``INVESTIGATE`` / ``MONITOR`` / ``NO ACTION``.
    """

    __tablename__ = "drift_reports"

    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    baseline_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), index=True
    )
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    drift_score: Mapped[float] = mapped_column(Float, nullable=False)
    # none | mild | moderate | severe
    drift_status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    feature_drift_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    prediction_drift_json: Mapped[dict | None] = mapped_column(JSON)
    sample_size: Mapped[int | None] = mapped_column(Integer)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class AuditLog(UUIDMixin, Base):
    """Append-only audit trail. Spec § 17.4.

    One row per mutating API call (POST/PATCH/PUT/DELETE) + selected read-only
    events like login. Intentionally *no* ``updated_at`` — rows are immutable.
    Written by a FastAPI middleware; read via ``/admin/audit-logs`` or
    ``/workspaces/{id}/audit-logs``.

    ``action`` is a dotted namespace: ``workspace.create``, ``run.cancel``,
    ``deployment.delete``, ``llm.analyze-drift``. Derived from route path +
    method. ``target_type`` / ``target_id`` let us filter "everything that
    touched deployment X".

    ``payload`` is the scrubbed request body (passwords / tokens redacted).
    Response status stored so we can audit 4xx/5xx attempts too.
    """

    __tablename__ = "audit_logs"

    # Both nullable: workspace-less events (login) + unauth events (failed
    # login attempts — useful for intrusion forensics).
    workspace_id: Mapped[str | None] = mapped_column(
        ForeignKey("workspaces.id", ondelete="SET NULL"), index=True
    )
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(8), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    target_type: Mapped[str | None] = mapped_column(String(32), index=True)
    target_id: Mapped[str | None] = mapped_column(String(36), index=True)
    status_code: Mapped[int | None] = mapped_column(Integer)
    payload: Mapped[dict | None] = mapped_column(JSON)
    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=lambda: datetime.now(UTC),
    )


class PredictionLog(UUIDMixin, Base):
    """Append-only log of every prediction served by a deployment.

    Spec § 4.11 / § 11.2 (drift). Written by the ``/deployments/{slug}/predict``
    handler after each successful (or failed) inference. Drift detection,
    rate-limit forensics, and post-hoc auditing all read from this table.

    Sampling: ``request_sample`` and ``response_sample`` capture *up to*
    ``MAX_LOG_ROWS`` records to bound storage; the full ``n_rows`` count is
    always exact. Set ``request_sample = None`` to disable input retention
    on a per-deployment basis (future privacy switch).

    No ``updated_at`` — rows are immutable.
    """

    __tablename__ = "prediction_logs"

    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="ok", nullable=False, index=True)
    error: Mapped[str | None] = mapped_column(Text)
    request_sample: Mapped[list | None] = mapped_column(JSON)
    response_sample: Mapped[list | None] = mapped_column(JSON)
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=lambda: datetime.now(UTC),
    )


class ModelLibrary(UUIDMixin, TimestampMixin, Base):
    """Workspace-scoped, editable view of the engine's model registry.

    The engine ships a hardcoded set of models per task in
    ``pycaret.api.list_models(task)``. ``ModelLibrary`` mirrors that set into
    DB rows the workspace admin can enable/disable and override params on.

    v1 semantics: rows are populated lazily on first read of a (workspace,
    task) pair via ``sync_from_engine``. Subsequent reads serve the DB.
    Engine-side enforcement (filtering ``compare_models`` by enabled rows)
    is V2; for now this is a UI-driven catalog.

    Unique on (workspace_id, task_type, model_id).
    """

    __tablename__ = "model_library"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "task_type", "model_id", name="uq_model_library_ws_task_model"
        ),
    )

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    custom_params: Mapped[dict | None] = mapped_column(JSON)
    created_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )


class ScheduledJob(UUIDMixin, TimestampMixin, Base):
    """Cron / interval-scheduled background job (drift monitor, retrain, …).

    Spec § 4.13 / § 11.2. Hydrated into the in-process scheduler at startup
    (see ``pycaret_server.scheduler``). v1: each row's ``kind`` picks one of
    a small set of built-in handlers. Workspace-scoped so each tenant
    schedules its own jobs.

    ``schedule`` is one of:
      * ``{"interval_seconds": int}`` — fixed interval
      * ``{"cron": "0 */6 * * *"}``    — cron expression (UTC)

    ``target_id`` is the FK kind-specific:
      * ``kind="drift_monitor"``  → deployment_id
      * ``kind="retrain"``        → experiment_id
    """

    __tablename__ = "scheduled_jobs"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_id: Mapped[str | None] = mapped_column(String(36), index=True)
    schedule: Mapped[dict] = mapped_column(JSON, nullable=False)
    spec: Mapped[dict | None] = mapped_column(JSON)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str | None] = mapped_column(String(16))
    last_error: Mapped[str | None] = mapped_column(Text)
    last_run_run_id: Mapped[str | None] = mapped_column(String(36))
    created_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )


class WebhookSubscription(UUIDMixin, TimestampMixin, Base):
    """Outgoing webhook target — fired on platform events.

    Spec § 12.4. Each row matches a workspace + a list of event types
    (e.g. ``run.succeeded``, ``run.failed``, ``deployment.created``,
    ``drift.alert``). Payloads are JSON-POSTed to ``url``; an HMAC of
    the body using ``secret`` is sent in ``X-PyCaret-Signature``.

    ``filters`` is a free-form match dict (e.g. ``{"experiment_id": "..."}``);
    rows whose filters subset-match the event's payload are fired.
    """

    __tablename__ = "webhook_subscriptions"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    event_types: Mapped[list] = mapped_column(JSON, nullable=False)
    secret_encrypted: Mapped[str | None] = mapped_column(Text)
    filters: Mapped[dict | None] = mapped_column(JSON)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_fired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status_code: Mapped[int | None] = mapped_column(Integer)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )


class ExperimentTemplate(UUIDMixin, TimestampMixin, Base):
    """Saved experiment configuration that can be reused as a starting point.

    Spec § 4.14. A workspace admin captures a known-good ``setup_params``
    dict + plan defaults (``plan_params``) and surfaces it on the New
    Experiment screen so users can pick a template instead of filling out
    the dynamic form from scratch.
    """

    __tablename__ = "experiment_templates"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    task: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    setup_params: Mapped[dict] = mapped_column(JSON, nullable=False)
    plan_params: Mapped[dict | None] = mapped_column(JSON)
    created_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )


class Trial(UUIDMixin, TimestampMixin, Base):
    """One AutoML candidate from a ``compare_models`` / ``automl`` run.

    Spec § 4.6: promote the JSON ``Run.leaderboard`` rows into queryable
    entities so the UI's Trials tab can sort, filter, link to fitted
    pipelines, and rank cross-run. Written by ``RunOrchestrator`` at the
    same time it persists the leaderboard JSON; the JSON column on Run
    stays for backwards-compat callers but Trial rows are the source of
    truth going forward.

    ``model_id`` is the engine model registry id (``"lr"``, ``"rf"``, …).
    ``rank`` is 1-based on the leaderboard's primary sort metric.
    ``metrics`` is the row's per-metric values as written by
    ``CompareResult.leaderboard``.
    """

    __tablename__ = "trials"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fitted_pipeline_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipelines.id", ondelete="SET NULL"), index=True
    )
