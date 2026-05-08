"""session 24: scheduled jobs, webhooks, templates, pipeline versioning

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-08 15:00:00.000000+00:00

Adds:
  - ``scheduled_jobs`` table (drift monitor + retraining)
  - ``webhook_subscriptions`` table (outgoing event hooks)
  - ``experiment_templates`` table (saved experiment configs)
  - ``pipelines.family_id`` + ``pipelines.version`` (deployment rollback)
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "scheduled_jobs",
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=36), nullable=True),
        sa.Column("schedule", sa.JSON(), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status", sa.String(length=16), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_run_run_id", sa.String(length=36), nullable=True),
        sa.Column("created_by", sa.String(length=36), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("scheduled_jobs", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_scheduled_jobs_workspace_id"),
            ["workspace_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_scheduled_jobs_kind"), ["kind"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_scheduled_jobs_target_id"), ["target_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_scheduled_jobs_created_by"), ["created_by"], unique=False
        )

    op.create_table(
        "webhook_subscriptions",
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("url", sa.String(length=512), nullable=False),
        sa.Column("event_types", sa.JSON(), nullable=False),
        sa.Column("secret_encrypted", sa.Text(), nullable=True),
        sa.Column("filters", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status_code", sa.Integer(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=36), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("webhook_subscriptions", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_webhook_subscriptions_workspace_id"),
            ["workspace_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_webhook_subscriptions_created_by"),
            ["created_by"],
            unique=False,
        )

    op.create_table(
        "experiment_templates",
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("setup_params", sa.JSON(), nullable=False),
        sa.Column("plan_params", sa.JSON(), nullable=True),
        sa.Column("created_by", sa.String(length=36), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("experiment_templates", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_experiment_templates_workspace_id"),
            ["workspace_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_experiment_templates_task"), ["task"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_experiment_templates_created_by"),
            ["created_by"],
            unique=False,
        )

    # pipelines.family_id + pipelines.version (deployment rollback)
    with op.batch_alter_table("pipelines", schema=None) as batch_op:
        batch_op.add_column(sa.Column("family_id", sa.String(length=36), nullable=True))
        batch_op.add_column(
            sa.Column("version", sa.Integer(), nullable=False, server_default="1")
        )
        batch_op.create_index(
            batch_op.f("ix_pipelines_family_id"), ["family_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("pipelines", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_pipelines_family_id"))
        batch_op.drop_column("version")
        batch_op.drop_column("family_id")

    with op.batch_alter_table("experiment_templates", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_experiment_templates_created_by"))
        batch_op.drop_index(batch_op.f("ix_experiment_templates_task"))
        batch_op.drop_index(batch_op.f("ix_experiment_templates_workspace_id"))
    op.drop_table("experiment_templates")

    with op.batch_alter_table("webhook_subscriptions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_webhook_subscriptions_created_by"))
        batch_op.drop_index(batch_op.f("ix_webhook_subscriptions_workspace_id"))
    op.drop_table("webhook_subscriptions")

    with op.batch_alter_table("scheduled_jobs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_scheduled_jobs_created_by"))
        batch_op.drop_index(batch_op.f("ix_scheduled_jobs_target_id"))
        batch_op.drop_index(batch_op.f("ix_scheduled_jobs_kind"))
        batch_op.drop_index(batch_op.f("ix_scheduled_jobs_workspace_id"))
    op.drop_table("scheduled_jobs")
