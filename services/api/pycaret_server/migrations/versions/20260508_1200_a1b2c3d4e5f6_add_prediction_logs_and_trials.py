"""add prediction_logs, trials, and model_library

Revision ID: a1b2c3d4e5f6
Revises: 0cd9d5ea2e17
Create Date: 2026-05-08 12:00:00.000000+00:00

Session 22: control plane progress.

- ``prediction_logs`` — append-only log of every served prediction. Powers
  drift detection, latency forensics, audit.
- ``trials`` — first-class row per AutoML candidate. Promotes the JSON
  ``Run.leaderboard`` into queryable entities.
- ``model_library`` — workspace-scoped, editable mirror of the engine model
  registry. Lazy-seeded from ``pycaret.api.list_models`` on first read.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "0cd9d5ea2e17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("deployment_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("n_rows", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("request_sample", sa.JSON(), nullable=True),
        sa.Column("response_sample", sa.JSON(), nullable=True),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(["deployment_id"], ["deployments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("prediction_logs", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_prediction_logs_deployment_id"), ["deployment_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_prediction_logs_workspace_id"), ["workspace_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_prediction_logs_request_id"), ["request_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_prediction_logs_status"), ["status"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_prediction_logs_user_id"), ["user_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_prediction_logs_created_at"), ["created_at"], unique=False
        )

    op.create_table(
        "trials",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("is_best", sa.Boolean(), nullable=False),
        sa.Column("fitted_pipeline_id", sa.String(length=36), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["fitted_pipeline_id"], ["pipelines.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_trials_run_id"), ["run_id"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_trials_workspace_id"), ["workspace_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_trials_model_id"), ["model_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_trials_rank"), ["rank"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_trials_fitted_pipeline_id"), ["fitted_pipeline_id"], unique=False
        )

    op.create_table(
        "model_library",
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("custom_params", sa.JSON(), nullable=True),
        sa.Column("created_by", sa.String(length=36), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "workspace_id", "task_type", "model_id", name="uq_model_library_ws_task_model"
        ),
    )
    with op.batch_alter_table("model_library", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_model_library_workspace_id"), ["workspace_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_model_library_task_type"), ["task_type"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_model_library_model_id"), ["model_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_model_library_created_by"), ["created_by"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("model_library", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_model_library_created_by"))
        batch_op.drop_index(batch_op.f("ix_model_library_model_id"))
        batch_op.drop_index(batch_op.f("ix_model_library_task_type"))
        batch_op.drop_index(batch_op.f("ix_model_library_workspace_id"))
    op.drop_table("model_library")

    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_trials_fitted_pipeline_id"))
        batch_op.drop_index(batch_op.f("ix_trials_rank"))
        batch_op.drop_index(batch_op.f("ix_trials_model_id"))
        batch_op.drop_index(batch_op.f("ix_trials_workspace_id"))
        batch_op.drop_index(batch_op.f("ix_trials_run_id"))
    op.drop_table("trials")

    with op.batch_alter_table("prediction_logs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_prediction_logs_created_at"))
        batch_op.drop_index(batch_op.f("ix_prediction_logs_user_id"))
        batch_op.drop_index(batch_op.f("ix_prediction_logs_status"))
        batch_op.drop_index(batch_op.f("ix_prediction_logs_request_id"))
        batch_op.drop_index(batch_op.f("ix_prediction_logs_workspace_id"))
        batch_op.drop_index(batch_op.f("ix_prediction_logs_deployment_id"))
    op.drop_table("prediction_logs")
