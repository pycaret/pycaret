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
    business_context: Mapped[str | None] = mapped_column(Text)
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
    business_context: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="experiments")
    trials: Mapped[list[Trial]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    runs: Mapped[list[Run]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


# -----------------------------------------------------------------------------
# Phase 0 model:
#   Experiment ──► Trial (logical pipeline candidate)
#                    └── Run (execution instance — 1..N per Trial)
#
# Compare produces N Trials (one per algorithm), each with Run 1, all sharing
# the same ``created_by_action_id``. Tune / Ensemble / Blend / Stack spawn a
# single new Trial + Run 1, linked to source(s) via ``parent_trial_ids``.
# Retraining the same Trial appends a new Run with the next ``sequence``.
# -----------------------------------------------------------------------------


class Run(UUIDMixin, TimestampMixin, Base):
    """A user-visible execution event inside an Experiment.

    Phase 0-v2: a Run is "I clicked Compare" (or "I clicked Tune", or
    a schedule fired) — a single event. It **contains** N Trials (one
    per algorithm candidate). Each Trial is its own scheduled unit of
    work with its own status / metrics / artifact; the Run's status is
    derived from the aggregate of its Trials' statuses.

    Why this shape (vs. v1's "Trial owns Runs"):

    * Matches how people think — Compare produces *one* event that
      yields *many* models.
    * Makes parallelization natural — the orchestrator decomposes a
      ``compare`` Run into N ``create_model`` Jobs (one per Trial)
      dispatched independently to workers.
    * Each Trial has its own lifecycle, so the UI can render rows
      lighting up as workers complete instead of waiting for the
      whole compare to finish.
    """

    __tablename__ = "runs"

    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Aggregate status derived from the Trials this Run contains.
    # ``queued`` while every trial is queued, ``running`` once any is
    # running, ``succeeded`` when all are terminal-OK, ``failed`` /
    # ``cancelled`` when any are in those states.
    status: Mapped[str] = mapped_column(
        String(32), default="queued", nullable=False, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    # Pre-aggregation cache of the leaderboard for legacy UI surfaces
    # (the live leaderboard is computed from the Trial rows directly).
    leaderboard: Mapped[dict | None] = mapped_column(JSON)
    metrics_summary: Mapped[dict | None] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    # Inputs captured at dispatch time (immutable snapshot of the
    # experiment config that the per-Trial workers all share).
    snapshot: Mapped[dict | None] = mapped_column(JSON)

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    trials: Mapped[list[Trial]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
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
    pipeline_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipelines.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    # Phase 7: a Deployment now references a RegisteredModel + version
    # (the named, governed thing). The legacy ``pipeline_id`` FK stays
    # for migrated rows; new Deployments leave it NULL and rely on the
    # registry pair below. Future cuts will drop ``pipeline_id``
    # entirely once every install has migrated.
    registered_model_id: Mapped[str | None] = mapped_column(
        ForeignKey("registered_models.id", ondelete="RESTRICT"), index=True
    )
    registered_model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("registered_model_versions.id", ondelete="RESTRICT"), index=True
    )
    # Phase 0: every Deployment also points at the exact ``(trial_id, run_id)``
    # pair that produced it. Kept alongside the registry pair for
    # fine-grained reproducibility.
    trial_id: Mapped[str | None] = mapped_column(
        ForeignKey("trials.id", ondelete="SET NULL"), index=True
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), index=True
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


class Notebook(UUIDMixin, TimestampMixin, Base):
    """Phase 8: a Jupyter notebook tracked by the platform.

    The actual ``.ipynb`` file lives on object storage (Phase 2) at
    ``object_uri``; we never inline its bytes. ``kernel`` is the kernel
    spec to launch (e.g. ``python3``, ``ir``, ``julia-1.9``); ``path`` is
    the path-in-repo for users who've enabled Git sync (Phase 5).
    """

    __tablename__ = "notebooks"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    # Path inside the project's repo (drives Git sync layout). Defaults
    # to ``notebooks/<name>.ipynb`` when not provided.
    path: Mapped[str | None] = mapped_column(String(512))
    kernel: Mapped[str] = mapped_column(
        String(64), default="python3", nullable=False
    )
    # ``file://`` or ``s3://`` URI pointing at the saved .ipynb bytes.
    object_uri: Mapped[str | None] = mapped_column(String(1024))
    description: Mapped[str | None] = mapped_column(Text)
    last_executed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_modified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    tags: Mapped[list[str] | None] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class NotebookSession(UUIDMixin, TimestampMixin, Base):
    """Phase 8: a live JupyterLab container hosting one notebook.

    One Session per (user, notebook) typically. The Notebook Manager
    spawns a container, allocates a port, and stamps ``container_id`` +
    ``port`` so the proxy can forward to it. Idle sessions are
    reaped by the manager's housekeeping loop after
    ``idle_timeout_seconds``.

    ``token`` is the per-session JupyterLab token — embedded into the
    iframe URL so the user authenticates without seeing a prompt.
    Stored unhashed for v1 (only the user who owns the session sees
    it); a future cut wraps it through the Secret store.
    """

    __tablename__ = "notebook_sessions"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    notebook_id: Mapped[str] = mapped_column(
        ForeignKey("notebooks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # ``starting`` | ``running`` | ``stopping`` | ``stopped`` | ``failed``.
    status: Mapped[str] = mapped_column(
        String(16), default="starting", nullable=False, index=True
    )
    # Container handle from the Docker / Kubernetes manager — opaque
    # string the manager understands. Empty for the in-memory test
    # backend used in CI.
    container_id: Mapped[str | None] = mapped_column(String(128))
    # Host port the platform proxy forwards to. NULL until ``running``.
    port: Mapped[int | None] = mapped_column(Integer)
    token: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_active_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    idle_timeout_seconds: Mapped[int] = mapped_column(
        Integer, default=1800, nullable=False
    )
    # Resource caps the manager applies to the container.
    cpu_limit: Mapped[float | None] = mapped_column(Float)
    memory_mb_limit: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)


class Analysis(UUIDMixin, TimestampMixin, Base):
    """Phase 11: a Statistical-computing peer to Experiment.

    Each Analysis is one *typed* statistical procedure (t-test, ANOVA,
    Chi-square, OLS diagnostics, Kaplan–Meier, ARIMA, …) over a
    DataSource. The ``kind`` discriminator drives both the validator
    and the result-renderer.

    Like Experiment, an Analysis can have many Runs (Phase 0 Run
    semantics — every execution captures inputs in ``snapshot`` and
    stamps results into the Run row). For analyses, the metrics dict
    on the Run carries (test_statistic, p_value, effect_size, ci_low,
    ci_high, interpretation) and the params dict carries the inputs
    (grouping_column, measure_column, …).
    """

    __tablename__ = "analyses"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    # ``ttest`` | ``welch_ttest`` | ``paired_ttest`` | ``mannwhitney`` |
    # ``anova_oneway`` | ``anova_twoway`` | ``kruskal`` | ``chi2`` |
    # ``fisher`` | ``cramers_v`` | ``ols`` | ``kaplan_meier`` |
    # ``logrank`` | ``cox_ph`` | ``arima`` | ``prophet`` ; free-form so
    # adding a new procedure is one new key + one new module.
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Persistent inputs the user picked (column names, group keys,
    # alpha, etc.). Each Run captures a snapshot at dispatch.
    params: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    data_source_id: Mapped[str | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL"),
        index=True,
    )
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class AlertRule(UUIDMixin, TimestampMixin, Base):
    """Phase 10: declarative threshold over a deployment metric.

    Rules evaluate on a rolling window. When the predicate trips, the
    worker emits to a Destination (Slack webhook, email recipient list,
    webhook URL). De-duplication is by (rule_id, deployment_id, window)
    so a long-running breach only pages once per window.
    """

    __tablename__ = "alert_rules"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    # Metric key the rule reads (e.g. ``p95_latency_ms``, ``error_rate``,
    # ``drift_score``, ``requests_per_minute``).
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    # ``gt`` | ``gte`` | ``lt`` | ``lte`` | ``eq`` — the rule fires when
    # ``metric <comparator> threshold`` over ``window_seconds``.
    comparator: Mapped[str] = mapped_column(String(8), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    window_seconds: Mapped[int] = mapped_column(
        Integer, default=300, nullable=False
    )
    # ``slack`` | ``email`` | ``webhook`` (free-form for future
    # destinations: PagerDuty, MS Teams, …).
    destination_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    # JSON config — Slack ``{"webhook_url": "..."}``, email
    # ``{"to": ["..."]}``, webhook ``{"url": "..."}``.
    destination_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_fired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str | None] = mapped_column(String(32))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class MetricPoint(UUIDMixin, TimestampMixin, Base):
    """Phase 10: simple roll-up time-series row.

    Deliberately not TimescaleDB-shaped — Phase 10 ships a portable
    "good enough" table that works on SQLite + Postgres. When volume
    demands it, a future cut swaps in the Timescale extension on
    Postgres; the read API stays the same.

    One row per (deployment_id, metric, ts_bucket). The bucket size is
    settable per metric (default 60s for latency, 300s for drift, etc.).
    """

    __tablename__ = "metric_points"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"), index=True
    )
    metric: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ts_bucket: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)
    count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    extra: Mapped[dict | None] = mapped_column(JSON)


class ApprovalWorkflow(UUIDMixin, TimestampMixin, Base):
    """Phase 12: required-approval gate on a sensitive action.

    Configurable per (workspace, target_kind, action) — e.g. require N
    approvals on ``promote_to_production`` for any
    ``registered_model_version``. The actual approval rows are stored
    inline on the workflow; once N approvers have signed off, the
    backend executes the gated action.
    """

    __tablename__ = "approval_workflows"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # ``registered_model_version`` | ``deployment`` | ``project_delete`` | …
    target_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_id: Mapped[str | None] = mapped_column(String(36), index=True)
    # ``promote_to_production`` | ``create`` | ``delete`` | ``rollback``.
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="pending", nullable=False, index=True
    )  # pending | approved | rejected | executed | cancelled
    required_approvals: Mapped[int] = mapped_column(
        Integer, default=1, nullable=False
    )
    # ``[{"user_id": "...", "approved_at": "...", "comment": "..."}, ...]``
    approvals: Mapped[list | None] = mapped_column(JSON)
    request_payload: Mapped[dict | None] = mapped_column(JSON)
    requested_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class RegisteredModel(UUIDMixin, TimestampMixin, Base):
    """Phase 7: the named, governed thing a Deployment references.

    A RegisteredModel is workspace-scoped and stable across retrains.
    The actual bytes + lineage live on the linked
    :class:`RegisteredModelVersion` rows; the Deployment FK points at
    a specific version so rollback is a one-column update with no
    re-training.

    Separating this from the prior ``pipelines`` table fixes the
    ambiguity where "promoting a trial" both *created* a Pipeline AND
    mutated the trial. In v2 the trial stays clean; promotion creates a
    new Version row attached to a (new or existing) RegisteredModel.
    """

    __tablename__ = "registered_models"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_registered_model_name"),
    )

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    # The "live" version a Deployment defaults to when created without
    # an explicit version override. ``use_alter=True`` lives on the FK
    # so SQLAlchemy emits an ALTER TABLE rather than a circular CREATE
    # (registered_models references versions which references models).
    current_version_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "registered_model_versions.id",
            ondelete="SET NULL",
            use_alter=True,
            name="fk_registered_models_current_version_id",
        ),
        index=True,
    )
    tags: Mapped[list[str] | None] = mapped_column(JSON)
    owner_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class RegisteredModelVersion(UUIDMixin, TimestampMixin, Base):
    """Phase 7: one promoted Run, scoped to a RegisteredModel.

    Each row is immutable once written: same Run produced the same
    bytes; same bytes get the same sha. Re-promoting the same Trial
    bumps ``version`` and writes a new row, preserving the prior one
    for rollback.

    The promotion lifecycle (``staging`` → ``production`` → ``archived``)
    is the user-facing thing; rollbacks just flip a Deployment to point
    at an older version.
    """

    __tablename__ = "registered_model_versions"

    registered_model_id: Mapped[str] = mapped_column(
        ForeignKey("registered_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Lineage back to the exact execution that produced these bytes.
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), index=True
    )
    trial_id: Mapped[str | None] = mapped_column(
        ForeignKey("trials.id", ondelete="SET NULL"), index=True
    )
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    params: Mapped[dict | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    # ``staging`` | ``production`` | ``archived``.
    status: Mapped[str] = mapped_column(
        String(16), default="staging", nullable=False, index=True
    )
    promoted_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)


class GitRepository(UUIDMixin, TimestampMixin, Base):
    """Phase 5: a Git repo a project syncs Experiments / Trials / Runs to.

    One Project ↔ at most one Git repo (v1). Auth is via a workspace
    Secret (PAT for now; GitHub-app + GitLab-app installations land in
    a future cut). Sync is one-way: backend export → repo on every Run
    completion. Repo as source-of-truth is explicitly out of scope.
    """

    __tablename__ = "git_repositories"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    # ``github`` | ``gitlab`` | ``gitea`` | ``bitbucket``.
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # The clone URL the worker uses (``https://github.com/owner/repo``).
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    default_branch: Mapped[str] = mapped_column(
        String(128), default="main", nullable=False
    )
    # Subdir under which exported YAML lands (default repo root).
    path_prefix: Mapped[str | None] = mapped_column(String(256))
    # PAT or app-install credential (encrypted at rest via Secret).
    secret_id: Mapped[str | None] = mapped_column(
        ForeignKey("secrets.id", ondelete="SET NULL"), index=True
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # When set, the orchestrator enqueues a ``git_publish`` Job each
    # time a Run in this repo's linked project flips to ``succeeded``.
    # No-op for repos without ``project_id``.
    auto_publish: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )
    last_push_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_push_status: Mapped[str | None] = mapped_column(String(32))
    last_push_sha: Mapped[str | None] = mapped_column(String(64))
    last_push_error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class Connection(UUIDMixin, TimestampMixin, Base):
    """Phase 4: a credentialed link to an external data source.

    Splits **how to connect** (Connection — db host/port/secret) from
    **what to register** (DataSource — a specific table/file/path inside
    that Connection). A workspace typically has one Postgres Connection
    and many DataSource rows pointing at different tables.

    The actual credentials live in ``secrets`` (encrypted at rest) and
    are referenced by ``secret_id`` — never inlined in ``config``.
    """

    __tablename__ = "connections"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    # ``postgres`` | ``snowflake`` | ``bigquery`` | ``s3`` | ``http`` | …
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Driver-specific connection params (host, port, database name, region…).
    # Anything secret lives in the linked Secret instead.
    config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    secret_id: Mapped[str | None] = mapped_column(
        ForeignKey("secrets.id", ondelete="SET NULL"), index=True
    )
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_test_status: Mapped[str | None] = mapped_column(String(32))
    last_test_error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class Secret(UUIDMixin, TimestampMixin, Base):
    """Phase 4: workspace-scoped encrypted secret store.

    Generic enough to back any credential — Postgres password, S3 keys,
    Snowflake token, OpenAI key — without a per-kind table per phase.
    ``value_encrypted`` carries the ciphertext (Fernet, same key as
    ``LLMProviderSetting.api_key_encrypted``); plaintext is never logged.
    """

    __tablename__ = "secrets"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    # Discriminator so we can list "all DB passwords" or "all API keys"
    # in the secret picker. ``opaque`` covers everything else.
    kind: Mapped[str] = mapped_column(
        String(32), default="opaque", nullable=False, index=True
    )
    value_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    # Stable hint for the UI ("**********xyz") without ever decrypting.
    last4: Mapped[str | None] = mapped_column(String(8))
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class Dataset(UUIDMixin, TimestampMixin, Base):
    """Phase 4: a *versioned* snapshot of a DataSource.

    A DataSource is the abstract pointer ("the orders table on prod
    Postgres"). A Dataset is one materialised cut of that pointer at a
    point in time — schema, row count, byte count, optional snapshot
    URI. Re-running the same DataSource produces a new Dataset version,
    which is what Experiment/Run rows actually reference for lineage.

    Snapshot URIs land in object storage (Phase 2) and are optional —
    small DataSources can be re-pulled on demand without materialising.
    """

    __tablename__ = "datasets"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    data_source_id: Mapped[str] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 1-based per-DataSource version number; the orchestrator bumps it
    # on every successful refresh.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    name: Mapped[str | None] = mapped_column(String(128))
    # Column list + dtypes captured at introspection time.
    schema_json: Mapped[dict | None] = mapped_column(JSON)
    row_count: Mapped[int | None] = mapped_column(Integer)
    byte_count: Mapped[int | None] = mapped_column(Integer)
    # Pointer to a frozen copy in object storage. NULL when the snapshot
    # wasn't materialised (e.g. cheap small CSVs we re-read each run).
    snapshot_uri: Mapped[str | None] = mapped_column(String(1024))
    sample_uri: Mapped[str | None] = mapped_column(String(1024))
    created_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )


class Lineage(UUIDMixin, TimestampMixin, Base):
    """Phase 4: append-only edge log for the DAG viewer.

    One row per directed edge: Dataset → Experiment, Run →
    RegisteredModelVersion, Deployment → Run, etc. Lets the UI render
    a graph by walking outbound from any node.

    ``source_kind`` / ``target_kind`` are open strings (no FK polymorphism)
    so future entities slot in without a schema change. ``relation`` is a
    short verb (``trained_on``, ``produced``, ``deployed_from``) that
    drives edge labels in the viewer.
    """

    __tablename__ = "lineage"

    workspace_id: Mapped[str | None] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    target_kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    relation: Mapped[str] = mapped_column(String(32), nullable=False)
    # Free-form extras: metric values, version numbers, hashes the UI
    # surfaces on hover.
    metadata_json: Mapped[dict | None] = mapped_column(JSON)


class Job(UUIDMixin, TimestampMixin, Base):
    """Phase 1: queueable unit of work.

    Every async operation (training a Run, refreshing a Dataset sample,
    running a batch-predict) becomes a Job row. The current in-process
    executor and the future Redis/RQ worker pool both read from this
    table — so we can flip the executor backend by env var without
    touching call sites.

    Lifecycle: ``queued`` → ``running`` → ``succeeded`` | ``failed`` |
    ``cancelled``. Retries bump ``attempts``; ``locked_by`` carries the
    worker id while a Job is in-flight so a crashed worker's Jobs can
    be re-queued cleanly.
    """

    __tablename__ = "jobs"

    # What kind of unit-of-work this row represents. Used to route to
    # the right worker handler. ``run`` is the only kind in Phase 1;
    # ``dataset_refresh``, ``drift_check``, ``batch_predict`` land in
    # later phases without schema changes.
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="queued", index=True
    )
    # Optional FK back to the Run this Job executes. Nullable because
    # future Job kinds (drift, batch-predict) don't necessarily map 1:1
    # to a Run row.
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    # Free-form payload — the worker reads this to drive execution.
    # For ``kind=run``, this is the orchestrator's RunSpec serialised
    # as JSON; future kinds dictate their own shape.
    payload: Mapped[dict | None] = mapped_column(JSON)
    # Retry bookkeeping. ``attempts`` increments on every dispatch
    # attempt; ``max_attempts`` clamps it; ``next_retry_at`` lets the
    # worker pull due-now retries with a single query.
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Worker that currently owns this Job. NULL when queued or terminal.
    locked_by: Mapped[str | None] = mapped_column(String(128), index=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    # Which queue this Job belongs to. Phase 14 uses this to route to
    # worker classes (cpu-heavy / gpu / inference); Phase 1 default is
    # always ``default``.
    queue: Mapped[str] = mapped_column(
        String(32), default="default", nullable=False, index=True
    )
    # Optional grouping key — multiple Jobs belonging to one user click
    # share a ``correlation_id`` so logs/audits can join them.
    correlation_id: Mapped[str | None] = mapped_column(String(36), index=True)
    # Free-form resource hints the dispatcher writes and a GPU-aware
    # worker pool reads. Today the worker still serves every queue,
    # but a "gpu" worker can filter for ``requested_resources.gpu == 1``
    # before claiming the row. Phase 14.
    requested_resources: Mapped[dict | None] = mapped_column(JSON)


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
    """One fitted-pipeline candidate inside a Run.

    Phase 0-v2 model (matches PyCaret 3.x convention):

        Run ──▶ contains 1..N Trials (one per algorithm candidate)

    A Trial is its own scheduled unit of work. When the orchestrator
    receives a ``compare`` Run, it decomposes it into N Trial rows
    (status=queued) + N Job rows (kind="trial") — one per algorithm.
    Workers pick the Trial-Jobs up independently, run
    ``create_model(model_id)``, and stamp the matching Trial row with
    its metrics + artifact + status. The Run's status is derived from
    the aggregate of its Trials' statuses.

    Each Trial owns its artifact pointer (``stored_path``) and metrics
    JSON; lineage between Trials across Runs (tune of X, blend of [Y,
    Z]) lives in ``parent_trial_ids``.
    """

    __tablename__ = "trials"

    # Parent Run — required for compare/standard trials; nullable
    # only because retired-from-Phase-0 rows survived the migration.
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    # Denorm so listing all trials in an Experiment doesn't need a join.
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Stable display name. Defaults to ``model_id`` when the
    # orchestrator creates the row; user-editable post-hoc.
    name: Mapped[str | None] = mapped_column(String(128))
    # Engine model id (``"lr"``, ``"rf"``, …). For ensembles / blends
    # / stacks, holds the synthetic id of the meta-pipeline
    # (e.g. ``stacked``) so the UI can still group / colour by it.
    model_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # ``compare | tuned | ensembled | blended | stacked | manual``.
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="compare", index=True
    )
    # Per-Trial lifecycle. Each Trial is its own scheduled unit of
    # work — workers flip this independently. The Run's status is the
    # aggregate of every Trial under it.
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="succeeded", server_default="succeeded", index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    # Leaderboard position within the parent Run (1 = best). Cached by
    # the orchestrator after all trials in the Run finish.
    rank: Mapped[int | None] = mapped_column(Integer, index=True)
    is_best: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    # Per-Trial metrics + fitted-artifact pointer. Workers stamp these
    # when they finish ``create_model``.
    metrics: Mapped[dict | None] = mapped_column(JSON)
    stored_path: Mapped[str | None] = mapped_column(String(1024))
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    params: Mapped[dict | None] = mapped_column(JSON)
    # Lineage — non-empty for tune/ensemble (1 source) and blend/stack (N).
    parent_trial_ids: Mapped[list[str] | None] = mapped_column(JSON)
    # Groups Trials spawned by one user action (e.g. a compare_models
    # click that emits 22 Trials, OR a tune action that produces 1).
    created_by_action_id: Mapped[str | None] = mapped_column(String(36), index=True)
    # Once a Trial is promoted to the registry, this points at the
    # resulting Pipeline row. Migrated to RegisteredModel in Phase 7.
    fitted_pipeline_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipelines.id", ondelete="SET NULL"), index=True
    )
    # Free-form annotation editable from the trial detail page.
    notes: Mapped[str | None] = mapped_column(Text)

    run: Mapped[Run | None] = relationship(back_populates="trials")
    experiment: Mapped[Experiment | None] = relationship(back_populates="trials")
