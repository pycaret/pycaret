"""Phases 8 + 11 — notebooks + analyses tables.

Revision ID: c3d4e5f6a8b9
Revises: b2c3d4e5f6a8
Create Date: 2026-05-15 11:00:00.000000+00:00

Schemas added:

- Phase 8: ``notebooks``, ``notebook_sessions``.
- Phase 11: ``analyses``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "c3d4e5f6a8b9"
down_revision: str | None = "b2c3d4e5f6a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── Phase 8: notebooks ───────────────────────────────────────
    op.create_table(
        "notebooks",
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
            nullable=False,
        ),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=True),
        sa.Column("kernel", sa.String(length=64), nullable=False, server_default="python3"),
        sa.Column("object_uri", sa.String(length=1024), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("last_executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_modified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column(
            "created_by",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_notebooks_workspace_id", "notebooks", ["workspace_id"], unique=False)
    op.create_index("ix_notebooks_project_id", "notebooks", ["project_id"], unique=False)

    # ─── Phase 8: notebook_sessions ───────────────────────────────
    op.create_table(
        "notebook_sessions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(length=36),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "notebook_id",
            sa.String(length=36),
            sa.ForeignKey("notebooks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="starting"),
        sa.Column("container_id", sa.String(length=128), nullable=True),
        sa.Column("port", sa.Integer(), nullable=True),
        sa.Column("token", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("idle_timeout_seconds", sa.Integer(), nullable=False, server_default="1800"),
        sa.Column("cpu_limit", sa.Float(), nullable=True),
        sa.Column("memory_mb_limit", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_notebook_sessions_workspace_id", "notebook_sessions", ["workspace_id"], unique=False
    )
    op.create_index(
        "ix_notebook_sessions_notebook_id", "notebook_sessions", ["notebook_id"], unique=False
    )
    op.create_index(
        "ix_notebook_sessions_user_id", "notebook_sessions", ["user_id"], unique=False
    )
    op.create_index(
        "ix_notebook_sessions_status", "notebook_sessions", ["status"], unique=False
    )

    # ─── Phase 11: analyses ───────────────────────────────────────
    op.create_table(
        "analyses",
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
            nullable=False,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column(
            "data_source_id",
            sa.String(length=36),
            sa.ForeignKey("data_sources.id", ondelete="SET NULL"),
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
    )
    op.create_index("ix_analyses_workspace_id", "analyses", ["workspace_id"], unique=False)
    op.create_index("ix_analyses_project_id", "analyses", ["project_id"], unique=False)
    op.create_index("ix_analyses_kind", "analyses", ["kind"], unique=False)
    op.create_index(
        "ix_analyses_data_source_id", "analyses", ["data_source_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_analyses_data_source_id", table_name="analyses")
    op.drop_index("ix_analyses_kind", table_name="analyses")
    op.drop_index("ix_analyses_project_id", table_name="analyses")
    op.drop_index("ix_analyses_workspace_id", table_name="analyses")
    op.drop_table("analyses")

    op.drop_index("ix_notebook_sessions_status", table_name="notebook_sessions")
    op.drop_index("ix_notebook_sessions_user_id", table_name="notebook_sessions")
    op.drop_index("ix_notebook_sessions_notebook_id", table_name="notebook_sessions")
    op.drop_index("ix_notebook_sessions_workspace_id", table_name="notebook_sessions")
    op.drop_table("notebook_sessions")

    op.drop_index("ix_notebooks_project_id", table_name="notebooks")
    op.drop_index("ix_notebooks_workspace_id", table_name="notebooks")
    op.drop_table("notebooks")
