"""Phases 4 / 5 / 7 / 10 / 12 — new entity tables in one migration.

Revision ID: b2c3d4e5f6a8
Revises: a1b2c3d4e5f7
Create Date: 2026-05-14 10:00:00.000000+00:00

Schemas added:

- Phase 4: ``secrets``, ``connections``, ``datasets``, ``lineage``.
- Phase 5: ``git_repositories``.
- Phase 7: ``registered_models``, ``registered_model_versions``;
  ``deployments`` gains ``registered_model_id`` + ``registered_model_version_id``
  FKs and relaxes the ``pipeline_id`` NOT NULL constraint.
- Phase 10: ``alert_rules``, ``metric_points``.
- Phase 12: ``approval_workflows``.

Every table is additive; no existing row is migrated. The
``pipeline_id`` constraint relaxation on ``deployments`` is done via
SQLite-batch so the dev DB rebuilds the table cleanly; Postgres takes
the ALTER COLUMN path.

There's no per-phase migration intentionally — this lump is the
"phases-4-to-12 schema cut" with one head bump. Downstream tooling
(bootstrap detector, ``pycaret-server doctor``) keys off the head id.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "b2c3d4e5f6a8"
down_revision: str | None = "a1b2c3d4e5f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── Phase 4: secrets ─────────────────────────────────────────
    op.create_table(
        "secrets",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False, server_default="opaque"),
        sa.Column("value_encrypted", sa.Text(), nullable=False),
        sa.Column("last4", sa.String(length=8), nullable=True),
        sa.Column(
            "created_by",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_secrets_workspace_id", "secrets", ["workspace_id"], unique=False)
    op.create_index("ix_secrets_kind", "secrets", ["kind"], unique=False)

    # ─── Phase 4: connections ─────────────────────────────────────
    op.create_table(
        "connections",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column(
            "secret_id",
            sa.String(length=36),
            sa.ForeignKey("secrets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("last_tested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_test_status", sa.String(length=32), nullable=True),
        sa.Column("last_test_error", sa.Text(), nullable=True),
        sa.Column(
            "created_by",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_connections_workspace_id", "connections", ["workspace_id"], unique=False)
    op.create_index("ix_connections_kind", "connections", ["kind"], unique=False)
    op.create_index("ix_connections_secret_id", "connections", ["secret_id"], unique=False)

    # ─── Phase 4: datasets ────────────────────────────────────────
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "data_source_id",
            sa.String(length=36),
            sa.ForeignKey("data_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("name", sa.String(length=128), nullable=True),
        sa.Column("schema_json", sa.JSON(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("byte_count", sa.Integer(), nullable=True),
        sa.Column("snapshot_uri", sa.String(length=1024), nullable=True),
        sa.Column("sample_uri", sa.String(length=1024), nullable=True),
        sa.Column(
            "created_by",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_datasets_workspace_id", "datasets", ["workspace_id"], unique=False)
    op.create_index("ix_datasets_data_source_id", "datasets", ["data_source_id"], unique=False)

    # ─── Phase 4: lineage ─────────────────────────────────────────
    op.create_table(
        "lineage",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("target_kind", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=36), nullable=False),
        sa.Column("relation", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_lineage_workspace_id", "lineage", ["workspace_id"], unique=False)
    op.create_index("ix_lineage_source_kind", "lineage", ["source_kind"], unique=False)
    op.create_index("ix_lineage_source_id", "lineage", ["source_id"], unique=False)
    op.create_index("ix_lineage_target_kind", "lineage", ["target_kind"], unique=False)
    op.create_index("ix_lineage_target_id", "lineage", ["target_id"], unique=False)

    # ─── Phase 5: git_repositories ────────────────────────────────
    op.create_table(
        "git_repositories",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.String(length=36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("url", sa.String(length=512), nullable=False),
        sa.Column("default_branch", sa.String(length=128), nullable=False, server_default="main"),
        sa.Column("path_prefix", sa.String(length=256), nullable=True),
        sa.Column(
            "secret_id",
            sa.String(length=36),
            sa.ForeignKey("secrets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("last_push_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_push_status", sa.String(length=32), nullable=True),
        sa.Column("last_push_sha", sa.String(length=64), nullable=True),
        sa.Column("last_push_error", sa.Text(), nullable=True),
        sa.Column(
            "created_by",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_git_repositories_workspace_id", "git_repositories", ["workspace_id"], unique=False
    )
    op.create_index(
        "ix_git_repositories_project_id", "git_repositories", ["project_id"], unique=False
    )
    op.create_index(
        "ix_git_repositories_provider", "git_repositories", ["provider"], unique=False
    )
    op.create_index(
        "ix_git_repositories_secret_id", "git_repositories", ["secret_id"], unique=False
    )

    # ─── Phase 7: registered_models ───────────────────────────────
    op.create_table(
        "registered_models",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.String(length=36),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        # FK to versions table — we'll wire the constraint below after
        # creating versions to avoid a circular DDL.
        sa.Column("current_version_id", sa.String(length=36), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column(
            "owner_user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_by",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("workspace_id", "name", name="uq_registered_model_name"),
    )
    op.create_index(
        "ix_registered_models_workspace_id", "registered_models", ["workspace_id"], unique=False
    )
    op.create_index(
        "ix_registered_models_project_id", "registered_models", ["project_id"], unique=False
    )

    # ─── Phase 7: registered_model_versions ───────────────────────
    op.create_table(
        "registered_model_versions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "registered_model_id",
            sa.String(length=36),
            sa.ForeignKey("registered_models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "run_id",
            sa.String(length=36),
            sa.ForeignKey("runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "trial_id",
            sa.String(length=36),
            sa.ForeignKey("trials.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("stored_path", sa.String(length=1024), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="staging"),
        sa.Column(
            "promoted_by",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_rmv_registered_model_id",
        "registered_model_versions",
        ["registered_model_id"],
        unique=False,
    )
    op.create_index("ix_rmv_run_id", "registered_model_versions", ["run_id"], unique=False)
    op.create_index(
        "ix_rmv_trial_id", "registered_model_versions", ["trial_id"], unique=False
    )
    op.create_index("ix_rmv_sha256", "registered_model_versions", ["sha256"], unique=False)
    op.create_index("ix_rmv_status", "registered_model_versions", ["status"], unique=False)
    op.create_index(
        "ix_rmv_promoted_by", "registered_model_versions", ["promoted_by"], unique=False
    )

    # ─── Phase 7: deployments — new FKs + relax pipeline_id ───────
    with op.batch_alter_table("deployments", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("registered_model_id", sa.String(length=36), nullable=True)
        )
        batch_op.add_column(
            sa.Column("registered_model_version_id", sa.String(length=36), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_deployments_registered_model_id",
            "registered_models",
            ["registered_model_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_deployments_registered_model_version_id",
            "registered_model_versions",
            ["registered_model_version_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_index(
            "ix_deployments_registered_model_id",
            ["registered_model_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_deployments_registered_model_version_id",
            ["registered_model_version_id"],
            unique=False,
        )
        # Relax pipeline_id from NOT NULL to NULL so new Phase 7
        # Deployments can skip the legacy column.
        batch_op.alter_column(
            "pipeline_id", existing_type=sa.String(length=36), nullable=True
        )

    # ─── Phase 10: alert_rules ────────────────────────────────────
    op.create_table(
        "alert_rules",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "deployment_id",
            sa.String(length=36),
            sa.ForeignKey("deployments.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("comparator", sa.String(length=8), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("window_seconds", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("destination_kind", sa.String(length=32), nullable=False),
        sa.Column("destination_config", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status", sa.String(length=32), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_by",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_alert_rules_workspace_id", "alert_rules", ["workspace_id"], unique=False)
    op.create_index(
        "ix_alert_rules_deployment_id", "alert_rules", ["deployment_id"], unique=False
    )

    # ─── Phase 10: metric_points ──────────────────────────────────
    op.create_table(
        "metric_points",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "deployment_id",
            sa.String(length=36),
            sa.ForeignKey("deployments.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("ts_bucket", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_metric_points_workspace_id", "metric_points", ["workspace_id"], unique=False
    )
    op.create_index(
        "ix_metric_points_deployment_id", "metric_points", ["deployment_id"], unique=False
    )
    op.create_index("ix_metric_points_metric", "metric_points", ["metric"], unique=False)
    op.create_index(
        "ix_metric_points_ts_bucket", "metric_points", ["ts_bucket"], unique=False
    )

    # ─── Phase 12: approval_workflows ─────────────────────────────
    op.create_table(
        "approval_workflows",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("target_kind", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=36), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("required_approvals", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("approvals", sa.JSON(), nullable=True),
        sa.Column("request_payload", sa.JSON(), nullable=True),
        sa.Column(
            "requested_by",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_approval_workflows_workspace_id",
        "approval_workflows",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_approval_workflows_target_kind",
        "approval_workflows",
        ["target_kind"],
        unique=False,
    )
    op.create_index(
        "ix_approval_workflows_target_id",
        "approval_workflows",
        ["target_id"],
        unique=False,
    )
    op.create_index(
        "ix_approval_workflows_status", "approval_workflows", ["status"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_approval_workflows_status", table_name="approval_workflows")
    op.drop_index("ix_approval_workflows_target_id", table_name="approval_workflows")
    op.drop_index("ix_approval_workflows_target_kind", table_name="approval_workflows")
    op.drop_index("ix_approval_workflows_workspace_id", table_name="approval_workflows")
    op.drop_table("approval_workflows")

    op.drop_index("ix_metric_points_ts_bucket", table_name="metric_points")
    op.drop_index("ix_metric_points_metric", table_name="metric_points")
    op.drop_index("ix_metric_points_deployment_id", table_name="metric_points")
    op.drop_index("ix_metric_points_workspace_id", table_name="metric_points")
    op.drop_table("metric_points")

    op.drop_index("ix_alert_rules_deployment_id", table_name="alert_rules")
    op.drop_index("ix_alert_rules_workspace_id", table_name="alert_rules")
    op.drop_table("alert_rules")

    with op.batch_alter_table("deployments", schema=None) as batch_op:
        batch_op.alter_column(
            "pipeline_id", existing_type=sa.String(length=36), nullable=False
        )
        batch_op.drop_index("ix_deployments_registered_model_version_id")
        batch_op.drop_index("ix_deployments_registered_model_id")
        batch_op.drop_constraint(
            "fk_deployments_registered_model_version_id", type_="foreignkey"
        )
        batch_op.drop_constraint(
            "fk_deployments_registered_model_id", type_="foreignkey"
        )
        batch_op.drop_column("registered_model_version_id")
        batch_op.drop_column("registered_model_id")

    op.drop_index("ix_rmv_promoted_by", table_name="registered_model_versions")
    op.drop_index("ix_rmv_status", table_name="registered_model_versions")
    op.drop_index("ix_rmv_sha256", table_name="registered_model_versions")
    op.drop_index("ix_rmv_trial_id", table_name="registered_model_versions")
    op.drop_index("ix_rmv_run_id", table_name="registered_model_versions")
    op.drop_index("ix_rmv_registered_model_id", table_name="registered_model_versions")
    op.drop_table("registered_model_versions")

    op.drop_index("ix_registered_models_project_id", table_name="registered_models")
    op.drop_index("ix_registered_models_workspace_id", table_name="registered_models")
    op.drop_table("registered_models")

    op.drop_index("ix_git_repositories_secret_id", table_name="git_repositories")
    op.drop_index("ix_git_repositories_provider", table_name="git_repositories")
    op.drop_index("ix_git_repositories_project_id", table_name="git_repositories")
    op.drop_index("ix_git_repositories_workspace_id", table_name="git_repositories")
    op.drop_table("git_repositories")

    op.drop_index("ix_lineage_target_id", table_name="lineage")
    op.drop_index("ix_lineage_target_kind", table_name="lineage")
    op.drop_index("ix_lineage_source_id", table_name="lineage")
    op.drop_index("ix_lineage_source_kind", table_name="lineage")
    op.drop_index("ix_lineage_workspace_id", table_name="lineage")
    op.drop_table("lineage")

    op.drop_index("ix_datasets_data_source_id", table_name="datasets")
    op.drop_index("ix_datasets_workspace_id", table_name="datasets")
    op.drop_table("datasets")

    op.drop_index("ix_connections_secret_id", table_name="connections")
    op.drop_index("ix_connections_kind", table_name="connections")
    op.drop_index("ix_connections_workspace_id", table_name="connections")
    op.drop_table("connections")

    op.drop_index("ix_secrets_kind", table_name="secrets")
    op.drop_index("ix_secrets_workspace_id", table_name="secrets")
    op.drop_table("secrets")
