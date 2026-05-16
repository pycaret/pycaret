"""Session 33 — Git auto-publish on Run complete + Phase 14 GPU queue

Revision ID: e5f6a8b9c0d1
Revises: d4e5f6a8b9c0
Create Date: 2026-05-17 13:00:00.000000+00:00

Two small additive columns; no data migration needed.

1. ``git_repositories.auto_publish`` (bool, default False) — when set,
   the orchestrator's ``reconcile_run_status`` enqueues a
   ``Job(kind="git_publish")`` row each time a Run in the linked
   project flips to ``succeeded``. The worker handler reuses the
   existing ``publish_project`` service, so the publish path is
   identical to the manual button.

2. ``jobs.queue`` already exists from Phase 1 but ``jobs.requested_resources``
   does not. Add it as a free-form JSON column so a dispatcher can stash
   ``{"gpu": 1, "vram_gb": 16}`` on a Trial-Job and a GPU-aware worker
   can route by that — without us shipping a queue-name convention yet.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "e5f6a8b9c0d1"
down_revision = "d4e5f6a8b9c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # GitRepository.auto_publish
    with op.batch_alter_table("git_repositories") as batch:
        batch.add_column(
            sa.Column(
                "auto_publish",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )

    # Job.requested_resources — opaque JSON so we don't have to pick a
    # vocabulary up-front. The dispatcher writes; the worker reads.
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(
            sa.Column("requested_resources", sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_column("requested_resources")
    with op.batch_alter_table("git_repositories") as batch:
        batch.drop_column("auto_publish")
