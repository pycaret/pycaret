"""Phase 1: jobs table — queueable unit-of-work for the worker pool.

Revision ID: a1b2c3d4e5f7
Revises: f0a1b2c3d4e5
Create Date: 2026-05-13 09:00:00.000000+00:00

Adds the ``jobs`` table that backs the executor backend toggle
(``RUNS_BACKEND=inprocess|redis``). With ``inprocess`` the in-process
``RunOrchestrator`` keeps running as today and Jobs are a passive
audit trail. With ``redis``, the worker process polls this table (and
Redis) for queued Jobs and executes them out-of-band.

The schema is intentionally generic — ``kind`` carries the discriminator
so future Phase 9 job types (drift_check, batch_predict, dataset_refresh)
can land without another migration.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "a1b2c3d4e5f7"
down_revision: str | None = "f0a1b2c3d4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column(
            "run_id",
            sa.String(length=36),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=128), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("queue", sa.String(length=32), nullable=False, server_default="default"),
        sa.Column("correlation_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_kind", "jobs", ["kind"], unique=False)
    op.create_index("ix_jobs_status", "jobs", ["status"], unique=False)
    op.create_index("ix_jobs_run_id", "jobs", ["run_id"], unique=False)
    op.create_index("ix_jobs_locked_by", "jobs", ["locked_by"], unique=False)
    op.create_index("ix_jobs_queue", "jobs", ["queue"], unique=False)
    op.create_index(
        "ix_jobs_correlation_id", "jobs", ["correlation_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_jobs_correlation_id", table_name="jobs")
    op.drop_index("ix_jobs_queue", table_name="jobs")
    op.drop_index("ix_jobs_locked_by", table_name="jobs")
    op.drop_index("ix_jobs_run_id", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_kind", table_name="jobs")
    op.drop_table("jobs")
